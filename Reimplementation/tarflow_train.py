from gc import enable

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import dataclasses
import math
import os
import pathlib

from TARFlow import TARFlow

@dataclasses.dataclass
class TrainConfig:

    # --- Paths ---
    data_dir:   str = './data'
    output_dir: str = './runs'
 
    # --- Image / patch geometry ---
    img_size:     int = 28   # MNIST is 28×28
    channel_size: int = 1    # greyscale
    patch_size:   int = 4    # → 7×7 = 49 patches, each of dim 1*4*4 = 16

    # --- Model architecture ---
    channels:         int  = 128   # Transformer internal dimension
    blocks:           int  = 4     # Number of AF blocks (T)
    layers_per_block: int  = 2     # AttentionBlocks per MetaBlock (K)
    nvp:              bool = True  # True = NVP (learn scale+shift); False = VP (shift only)
    unconditional:    bool = False # True = no class conditioning

    # --- Optimiser ---
    epochs:       int   = 50
    batch_size:   int   = 128
    lr:           float = 1e-4
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0    # max gradient norm; 0 disables clipping

    noise_std:  float = 0.05
    noise_type: str   = 'gaussian'   # 'gaussian' | 'uniform'

    # --- Classifier-free guidance training (Zhai et al., 2024, §2.6) ---
    # Randomly replacing a label with -1 trains the conditional and
    # unconditional paths simultaneously with a single model.
    # MetaBlock.forward uses the mean class embedding for dropped labels.
    drop_label: float = 0.1   # probability of replacing a label with -1
 
    # --- Logging ---
    log_freq: int = 50   # print loss every N batches
 
    # --- Checkpoint ---
    resume: str = ''   # path to a model checkpoint to resume from

    @property
    def num_classes(self) -> int:
        return 0 if self.unconditional else 10
 
    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2
 
    @property
    def patch_dim(self) -> int:
        return self.channel_size * self.patch_size ** 2
    
def build_model(cfg, device = 'cpu'):
    model = TARFlow(
    in_channels      = cfg.channel_size,
    img_size         = cfg.img_size,
    patch_size       = cfg.patch_size,
    channels         = cfg.channels,
    num_blocks       = cfg.blocks,
    layers_per_block = cfg.layers_per_block,
    nvp              = cfg.nvp,
    num_classes      = cfg.num_classes,
    ).to(device)
    return model


def _apply_noise(x, cfg):
    if cfg.noise_type == 'gaussian':
        return x + cfg.noise_std * torch.randn_like(x)
    elif cfg.noise_type == 'uniform':
        x_int = (x + 1) * (255 / 2)
        x = (x_int + torch.rand_like(x_int)) / 256
        return x * 2 - 1
    return x

def _apply_label_drop(y, drop_prob):
    mask = (torch.rand(y.size(0), device=y.device) < drop_prob).int()
    return (1 - mask) * y - mask

class CosineLRSchedule:
    def __init__(self, optimizer, warmup_steps, total_steps,
        lr_min, lr_max):

        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.lr_min       = lr_min
        self.lr_max       = lr_max
        self._step        = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr
    
    def _get_lr(self) -> float:
        s = self._step
        if s <= self.warmup_steps:
            return self.lr_min + (self.lr_max - self.lr_min) * s / max(1, self.warmup_steps)
        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine
    
    def state_dict(self) -> dict:
        return {'_step': self._step}
 
    def load_state_dict(self, d: dict):
        self._step = d['_step']

class Metrics:
    def __init__(self):
        self._sums:   dict = {}
        self._counts: dict = {}

    def update(self, d):
        for k, v in d.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            self._sums[k]   = self._sums.get(k, 0.0) + val
            self._counts[k] = self._counts.get(k, 0)  + 1
    
    def compute(self):
        return {k: self._sums[k] / self._counts[k] for k in self._sums}
    
    def reset(self):
        self._sums   = {}
        self._counts = {}



class Trainer:
    def __init__(self, model, train_loader, cfg, device):
        self.model = model
        self.train_loader = train_loader
        self.cfg = cfg
        self.device  = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = cfg.lr,
            betas = (0.9, 0.95),
            weight_decay= cfg.weight_decay,
        )

        # LR schedule
        steps_per_epoch = len(train_loader)
        total_steps     = cfg.epochs * steps_per_epoch
        self.lr_schedule = CosineLRSchedule(
            self.optimizer,
            warmup_steps = steps_per_epoch,
            total_steps  = total_steps,
            lr_min       = 1e-6,
            lr_max       = cfg.lr,
        )

        self._use_amp = (cfg.noise_type == 'gaussian') and (self.device == 'cuda')
        self.scaler   = torch.amp.GradScaler(enabled=self._use_amp)
        #self.scaler = torch.amp.grad_scaler(enable=self._use_amp)
        
        # Checkpoint paths
        out = pathlib.Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._model_ckpt = out / 'model.pth'
        self._opt_ckpt   = out / 'optimizer.pth'

        # Starting epoch (updated if resuming)
        self.start_epoch = 0
        if cfg.resume:
            self._load_checkpoint(cfg.resume)
 
        # Training history — populated by .train(), readable from notebook
        self.history: dict = {
            'loss':[], # mean NLL loss per epoch
            'nll':[], # mean prior term  0.5||z||²
            'logdet':[], # mean log-det term
            'lr':[], # learning rate at end of each epoch
        }

    def train(self):
        print(f'{" TARFlow Training ":-^70}')
        print(f'  Device         : {self.device}')
        print(f'  Parameters     : {self._count_params():,}')
        print(f'  Epochs         : {self.cfg.epochs}')
        print(f'  Batch size     : {self.cfg.batch_size}')
        print(f'  Noise type     : {self.cfg.noise_type}  (σ={self.cfg.noise_std})')
        print(f'  Label drop     : {self.cfg.drop_label}')
        print()

        for epoch in range(self.start_epoch, self.cfg.epochs):
            epoch_metrics = self._train_epoch(epoch)
            self._log_epoch(epoch, epoch_metrics)
            self._save_checkpoint(epoch)
            # Append to history for notebook access
            self.history['loss'].append(epoch_metrics['loss'])
            self.history['nll'].append(epoch_metrics['loss/nll_prior'])
            self.history['logdet'].append(epoch_metrics['loss/logdet'])
            self.history['lr'].append(epoch_metrics['lr'])
 
        print(f'\n{" Training complete ":-^70}')
        print(f'  Checkpoint : {self._model_ckpt}')
        return self.history
    
    def _train_epoch(self, epoch):
        self.model.train()
        metrics = Metrics()
        steps_per_epoch = len(self.train_loader)
        current_lr = self.cfg.lr

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            x = _apply_noise(x, self.cfg)

            if self.cfg.num_classes:
                y = y.to(self.device)
                y = _apply_label_drop(y, self.cfg.drop_label)

            else:
                y = None

            self.optimizer.zero_grad()

            with torch.autocast(
                device_type = self.device,
                dtype       = torch.bfloat16
            ):
                z, outputs, logdets = self.model(x, y)
                loss = self.model.get_loss(z, logdets)

            if not self.cfg.nvp:
                self.model.update_prior(z)

            self.scaler.scale(loss).backward()
 
            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            current_lr = self.lr_schedule.step()

            metrics.update({
                'loss':           loss,
                'loss/nll_prior': 0.5 * (z ** 2).mean(),
                'loss/logdet':    logdets.mean(),
            })

            if (batch_idx + 1) % self.cfg.log_freq == 0:
                m = metrics.compute()
                print(
                    f'  Ep [{epoch+1:3d}/{self.cfg.epochs}] '
                    f'Step [{batch_idx+1:4d}/{steps_per_epoch}] '
                    f'loss={m["loss"]:.4f}  '
                    f'nll={m["loss/nll_prior"]:.4f}  '
                    f'logdet={m["loss/logdet"]:.4f}  '
                    f'lr={current_lr:.2e}'
                )

        m = metrics.compute()
        m['lr']          = current_lr
        m['block_norms'] = [float(z.pow(2).mean()) for z in outputs]
        return m
    
    def _log_epoch(self, epoch, m):
        norms = ' '.join(f'{v:.4f}' for v in m['block_norms'])
        print(
            f'\nEpoch {epoch+1:3d}/{self.cfg.epochs} | '
            f'loss={m["loss"]:.4f} | '
            f'nll={m["loss/nll_prior"]:.4f} | '
            f'logdet={m["loss/logdet"]:.4f} | '
            f'block_norms=[{norms}]\n'
        )

    def _save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), self._model_ckpt)
        torch.save({
            'optimizer':   self.optimizer.state_dict(),
            'lr_schedule': self.lr_schedule.state_dict(),
            'epoch':       epoch + 1,
        }, self._opt_ckpt)

    def _load_checkpoint(self, path):
        model_state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(model_state)

        opt_path = path.replace('model', 'optimizer')
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location='cpu')
            self.optimizer.load_state_dict(opt_state['optimizer'])
            self.lr_schedule.load_state_dict(opt_state['lr_schedule'])
            self.start_epoch = opt_state.get('epoch', 0)
 
        print(f'  Resumed from {path}  (epoch {self.start_epoch})')

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)






