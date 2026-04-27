import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader



def get_mnist_loaders(cfg):
    """
    Build MNIST DataLoaders with preprocessing matching the original train.py:
      - Resize to cfg.img_size (no-op for standard 28×28 MNIST)
      - Rescale pixels from [0, 1] to [-1, 1]
 
    Patch compatibility check:
        img_size must be divisible by patch_size.
        28 % 4 == 0  ✓   → 49 patches of dimension 16
 
    Args:
        cfg: TrainConfig instance.
 
    Returns:
        train_loader, val_loader: standard PyTorch DataLoader objects.
    """
    assert cfg.img_size % cfg.patch_size == 0, (
        f'img_size ({cfg.img_size}) must be divisible by patch_size ({cfg.patch_size})'
    )
 
    transform = T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),   # [0,1] → [-1,1]
    ])
 
    train_dataset = torchvision.datasets.MNIST(
        cfg.data_dir, train=True,  download=True, transform=transform
    )
    val_dataset = torchvision.datasets.MNIST(
        cfg.data_dir, train=False, download=True, transform=transform
    )
 
    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True,
        drop_last   = True,   # keeps batch size constant; avoids recompilation
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )
    return train_loader, val_loader


def get_fmnist_loaders(cfg):
    """
    Fashion-MNIST: 28×28 greyscale, 60,000 training / 10,000 test.
    10 classes: T-shirt, Trouser, Pullover, Dress, Coat,
                Sandal, Shirt, Sneaker, Bag, Ankle boot.
    Preprocessing identical to MNIST — centre values to [-1, 1].
    """
    transform = T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_dataset = torchvision.datasets.FashionMNIST(
        cfg.data_dir, train=True,  download=True, transform=transform
    )
    val_dataset = torchvision.datasets.FashionMNIST(
        cfg.data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )
    return train_loader, val_loader