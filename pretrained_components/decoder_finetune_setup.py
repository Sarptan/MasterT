"""
decoder_finetune_setup.py
--------------------------
Sets up the VAE decoder for the finetuning stage described in
Gu et al. (2025, Appendix B.3).

Context
-------
After main STARFlow training, the flow model produces noisy latents
x̃ ~ N(E(x), 0.3^2 * I) at inference time. The original VAE decoder
was trained on clean latents and produces blurry outputs when given
noisy ones. The fix is to finetune ONLY the decoder to learn to
reconstruct clean pixels from noisy latents:

    min_ϕ  L( D(E(x + σε); ϕ), x )
    where  L = L_L2 + L_LPIPS + β*L_GAN    (Gu et al., 2025 Eq. 7)

Training setup (Appendix B.3):
    - 200K update steps
    - Batch size 64
    - 8 GPUs (single node)
    - Dataset: ImageNet 256x256
    - Encoder weights: frozen throughout
    - Decoder weights: updated

This file provides:
    1. setup_decoder_finetuning()  — isolates trainable parameters
    2. noisy_latent_from_image()   — generates the training targets
    3. decoder_reconstruction_loss() — the L2 part of the loss
       (LPIPS and GAN require additional libraries; stubs are provided)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

SIGMA_L = 0.3   # noise level used in STARFlow (Gu et al., 2025 §3.3)


def setup_decoder_finetuning(vae: AutoencoderKL) -> tuple:
    """
    Isolate which parameters are trainable vs frozen for decoder finetuning.

    The encoder is ALWAYS frozen. Only decoder + post_quant_conv are trained.

    Returns
    -------
    trainable_params : list of parameters to pass to the optimiser
    frozen_count     : total number of frozen parameters
    trainable_count  : total number of trainable parameters
    """
    # Freeze encoder completely
    for param in vae.encoder.parameters():
        param.requires_grad = False
    for param in vae.quant_conv.parameters():
        param.requires_grad = False

    # Make decoder trainable
    for param in vae.decoder.parameters():
        param.requires_grad = True
    for param in vae.post_quant_conv.parameters():
        param.requires_grad = True

    trainable_params = [p for p in vae.parameters() if p.requires_grad]
    frozen_count     = sum(p.numel() for p in vae.parameters() if not p.requires_grad)
    trainable_count  = sum(p.numel() for p in vae.parameters() if p.requires_grad)

    print("Decoder finetuning setup:")
    print(f"  Frozen params    : {frozen_count/1e6:.1f}M  (encoder + quant_conv)")
    print(f"  Trainable params : {trainable_count/1e6:.1f}M  (decoder + post_quant_conv)")

    return trainable_params, frozen_count, trainable_count


def noisy_latent_from_image(
    vae: AutoencoderKL,
    x: torch.Tensor,
    sigma: float = SIGMA_L,
) -> torch.Tensor:
    """
    Generate the noisy latent that the decoder will be trained to decode.

    Implements Eq. 7 of Gu et al. (2025):
        noisy_latent = E(x + σ*ε),   ε ~ N(0, I)

    Note: noise is added to the PIXEL image before encoding, not to
    the latent directly. This matches Appendix B.3 exactly.

    Parameters
    ----------
    x     : (B, 3, H, W) clean image in [-1, 1]
    sigma : noise standard deviation (default 0.3)

    Returns
    -------
    noisy_latent : (B, 4, H/8, W/8), scaled, with no gradient
    """
    with torch.no_grad():
        eps = torch.randn_like(x) * sigma
        x_noisy = x + eps
        latent = vae.encode(x_noisy).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    return latent


def decode_latent(vae: AutoencoderKL, latent: torch.Tensor) -> torch.Tensor:
    """
    Decode a scaled latent through the (possibly finetuned) decoder.
    Gradients flow through the decoder — used during finetuning.

    Parameters
    ----------
    latent : (B, 4, H/8, W/8) scaled latent

    Returns
    -------
    recon : (B, 3, H, W) reconstructed image in [-1, 1]
    """
    recon = vae.decode(latent / vae.config.scaling_factor).sample
    return recon.clamp(-1, 1)


def decoder_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute the L2 reconstruction loss (the primary term in Eq. 7).

    LPIPS and GAN losses are also used in the full training setup but
    require additional libraries (lpips, a discriminator network).
    Those are left as stubs here — the L2 term alone is enough to
    verify that the decoder is learning.

    Parameters
    ----------
    recon  : (B, 3, H, W) decoded image
    target : (B, 3, H, W) clean ground-truth image

    Returns
    -------
    dict with individual loss terms and total loss
    """
    l2_loss   = F.mse_loss(recon, target)

    # Stub: LPIPS perceptual loss (requires pip install lpips)
    # from lpips import LPIPS
    # lpips_fn  = LPIPS(net='vgg').to(recon.device)
    # lpips_loss = lpips_fn(recon, target).mean()
    lpips_loss = torch.tensor(0.0)   # placeholder

    # Stub: GAN loss requires a discriminator — not included here
    gan_loss  = torch.tensor(0.0)    # placeholder

    total = l2_loss + lpips_loss + gan_loss

    return {
        "loss":       total,
        "l2":         l2_loss,
        "lpips":      lpips_loss,   # 0 until lpips is installed
        "gan":        gan_loss,     # 0 until discriminator is added
    }


def build_decoder_optimizer(trainable_params: list, lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Build the AdamW optimiser for decoder finetuning.
    Learning rate mirrors the main STARFlow training config (Appendix B.2).
    """
    return torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )


if __name__ == "__main__":
    from load_vae import load_vae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(device=device, freeze_encoder=False)   # load all unfrozen first

    trainable_params, frozen, trainable = setup_decoder_finetuning(vae)
    optimizer = build_decoder_optimizer(trainable_params)

    # Simulate one decoder finetuning step
    x_clean  = torch.randn(2, 3, 256, 256).to(device)   # fake clean images
    noisy_z  = noisy_latent_from_image(vae, x_clean)
    recon    = decode_latent(vae, noisy_z)
    losses   = decoder_reconstruction_loss(recon, x_clean)

    optimizer.zero_grad()
    losses["loss"].backward()
    optimizer.step()

    print(f"\nOne decoder finetuning step:")
    print(f"  noisy_z shape : {tuple(noisy_z.shape)}")
    print(f"  recon shape   : {tuple(recon.shape)}")
    print(f"  L2 loss       : {losses['l2'].item():.4f}")
    print("Decoder finetuning setup OK.")
