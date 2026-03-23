import dataclasses
 
import torch
import torch.nn.functional as F

from tarflow_train import build_model

@dataclasses.dataclass
class SampleConfig:

    # Guidance
    guidance:          float = 0.0
    guide_what:        str   = 'ab'
    annealed_guidance: bool  = False
    attn_temp:         float = 1.0

    # Denoising
    denoise:       bool  = False
    noise_std:     float = 0.05
    denoise_steps: int   = 1

    # Output
    return_sequence: bool = False
    clip_output:     bool = True

def load_model(checkpoint_path, train_cfg, device = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model(train_cfg, device=device)
    state  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def score_based_denoise(model, y, noise_std, steps=1,):
    x = y.detach().clone().requires_grad_(True)
    for _ in range(steps):
        z, _, logdets = model(x)
        log_p = -(0.5 * z.pow(2).mean() - logdets.mean())

        grad  = torch.autograd.grad(log_p, x)[0]
        x = (x + noise_std ** 2 * grad).detach().requires_grad_(True)
        return x.detach()
    

class Sampler:
    def __init__(self, model, sample_cfg = None,):
        self.model = model
        self.cfg = sample_cfg or SampleConfig()
        self.device = next(model.parameters()).device
        self._num_patches = model.num_patches
        self._patch_dim = model.var.shape[-1]

    def _make_noise(self, n):
        return torch.randn(n, self._num_patches, self._patch_dim, device=self.device)

    def _make_labels(self, labels):
        if labels is None:
            return None
        if isinstance(labels, torch.Tensor):
            return labels.to(self.device)
        return torch.tensor(labels, dtype=torch.long, device=self.device)
    
    def _reverse(self, noise, labels, guidance, guide_what, annealed_guidance, 
                 attn_temp, return_sequence):
        
        with torch.no_grad():
            with torch.autocast(
                device_type = str(self.device).split(':')[0],
                dtype       = torch.bfloat16,
                enabled     = (str(self.device) != 'cpu'),
            ):
                result = self.model.reverse(
                    noise,
                    y = labels,
                    guidance = guidance,
                    guide_what = guide_what,
                    attn_temp = attn_temp,
                    annealed_guidance = annealed_guidance,
                    return_sequence = return_sequence)
        return result

    def _postprocess(self, images):
        if self.cfg.clip_output:
            return images.clip(-1.0, 1.0)
        return images
    
    def sample_plain(self, n):
        noise  = self._make_noise(n)
        images = self._reverse(
            noise, labels=None, guidance=0.0, guide_what='ab',
            annealed_guidance=False, attn_temp=1.0, return_sequence=False,
        )
        return self._postprocess(images)
    
    def sample_conditional(self, labels, guidance = None, guide_what = None):
        labels_t = self._make_labels(labels)
        n        = labels_t.shape[0]
        noise    = self._make_noise(n)
        g        = guidance   if guidance   is not None else self.cfg.guidance
        gw       = guide_what if guide_what is not None else self.cfg.guide_what
 
        images = self._reverse(
            noise, labels=labels_t, guidance=g, guide_what=gw,
            annealed_guidance=False, attn_temp=1.0, return_sequence=False,
        )
        return self._postprocess(images)
    
    def sample_temperature_guided(self, n, guidance = None, attn_temp = None):
        noise = self._make_noise(n)
        g     = guidance  if guidance  is not None else self.cfg.guidance
        t     = attn_temp if attn_temp is not None else self.cfg.attn_temp
 
        images = self._reverse(
            noise, labels=None, guidance=g, guide_what=self.cfg.guide_what,
            annealed_guidance=False, attn_temp=t, return_sequence=False,
        )
        return self._postprocess(images)
    
    def sample_annealed(self, labels, guidance = None):
        labels_t = self._make_labels(labels)
        n = labels_t.shape[0]
        noise = self._make_noise(n)
        g = guidance if guidance is not None else self.cfg.guidance
 
        images = self._reverse(
            noise, labels=labels_t, guidance=g, guide_what=self.cfg.guide_what,
            annealed_guidance=True, attn_temp=1.0, return_sequence=False,
        )
        return self._postprocess(images)

    def sample_denoised(self, n, labels=None, guidance=None, noise_std=None, steps=None):
        labels_t  = self._make_labels(labels)
        noise     = self._make_noise(n)
        g         = guidance  if guidance  is not None else self.cfg.guidance
        sigma     = noise_std if noise_std is not None else self.cfg.noise_std
        n_steps   = steps     if steps     is not None else self.cfg.denoise_steps
 
        # Step 1: reverse flow → raw noisy sample
        raw = self._reverse(
            noise, labels=labels_t, guidance=g, guide_what=self.cfg.guide_what,
            annealed_guidance=self.cfg.annealed_guidance, attn_temp=self.cfg.attn_temp,
            return_sequence=False,
        )
 
        # Step 2: Tweedie denoising — requires grad, so done outside no_grad
        self.model.eval()
        denoised = score_based_denoise(self.model, raw, noise_std=sigma, steps=n_steps)
 
        return self._postprocess(denoised)
    
    def sample_trajectory(self, n, labels=None, guidance=None):
        labels_t = self._make_labels(labels)
        noise = self._make_noise(n)
        g = guidance if guidance is not None else self.cfg.guidance
 
        seq = self._reverse(
            noise, labels=labels_t, guidance=g, guide_what=self.cfg.guide_what,
            annealed_guidance=self.cfg.annealed_guidance, attn_temp=self.cfg.attn_temp,
            return_sequence=True,
        )
 
        if self.cfg.clip_output:
            seq = [s.clip(-1.0, 1.0) for s in seq]
        return seq
        
    def sample(self, n, labels=None):
        if self.cfg.return_sequence:
            return self.sample_trajectory(n, labels)
 
        if self.cfg.denoise:
            return self.sample_denoised(n, labels)
 
        if self.cfg.annealed_guidance and labels is not None:
            return self.sample_annealed(labels)
 
        if self.cfg.attn_temp != 1.0:
            return self.sample_temperature_guided(n)
 
        if labels is not None:
            return self.sample_conditional(labels)
 
        return self.sample_plain(n)
