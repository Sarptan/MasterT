"""
load_vae.py
-----------
Loads the STARFlow VAE (stabilityai/sd-vae-ft-mse).

The VAE has two parts with different training roles:
  - encoder  : FROZEN throughout all training (Gu et al., 2025 §3.3)
  - decoder  : finetuned separately AFTER main flow training
               to reconstruct from noisy latents (Appendix B.3)

Source in model_setup.py:
    self.vae = AutoencoderKL.from_pretrained(model_name)
    self.scaling_factor = self.vae.config.scaling_factor
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from diffusers.models import AutoencoderKL

VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
SIGMA_L = 0.3  # STARFlow training noise (Gu et al., 2025 §3.3)


def load_vae(device: str = "cpu", freeze_encoder: bool = True) -> AutoencoderKL:
    """
    Download (first run) or load from cache the SD VAE.

    Parameters
    ----------
    device        : 'cuda' or 'cpu'
    freeze_encoder: if True, encoder weights have requires_grad=False
                    (matches STARFlow training setup)

    Returns
    -------
    vae : AutoencoderKL, moved to device, eval mode
    """
    print(f"Loading VAE from {VAE_MODEL_ID} ...")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID)
    vae = vae.to(device).eval()

    if freeze_encoder:
        for param in vae.encoder.parameters():
            param.requires_grad = False
        for param in vae.quant_conv.parameters():
            param.requires_grad = False
        print("  Encoder frozen  (requires_grad=False)")

    # Decoder stays trainable so decoder finetuning can update it
    for param in vae.decoder.parameters():
        param.requires_grad = True
    for param in vae.post_quant_conv.parameters():
        param.requires_grad = True

    enc_params   = sum(p.numel() for p in vae.encoder.parameters())
    dec_params   = sum(p.numel() for p in vae.decoder.parameters())
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"  Encoder params : {enc_params/1e6:.1f}M")
    print(f"  Decoder params : {dec_params/1e6:.1f}M  (trainable)")
    print(f"  Total VAE      : {total_params/1e6:.1f}M")
    print(f"  Scaling factor : {vae.config.scaling_factor}")
    print(f"  Latent channels: {vae.config.latent_channels}")
    ds = 2 ** (len(vae.config.down_block_types) - 1)
    print(f"  Spatial downsample: {ds}x  (256px image → {256//ds}x{256//ds} latent)")
    return vae


def encode(vae: AutoencoderKL, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
    """
    Encode image tensor to (scaled, optionally noisy) latent.

    Replicates STARFlow training:
        x̃ ~ N(E(x), sigma_L^2 * I)   (Gu et al., 2025 Eq. 6)

    Parameters
    ----------
    x         : (B, 3, H, W) in [-1, 1]
    add_noise : if True, adds sigma_L=0.3 Gaussian noise (training mode)
                if False, returns clean latent (eval/inspection mode)

    Returns
    -------
    latent : (B, 4, H/8, W/8) scaled latent
    """
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        latent = posterior.sample() * vae.config.scaling_factor
        if add_noise:
            latent = latent + torch.randn_like(latent) * SIGMA_L
    return latent


def decode(vae: AutoencoderKL, latent: torch.Tensor) -> torch.Tensor:
    """
    Decode a (scaled) latent back to pixel space.

    Returns
    -------
    image : (B, 3, H, W) clamped to [-1, 1]
    """
    with torch.no_grad():
        image = vae.decode(latent / vae.config.scaling_factor).sample
    return image.clamp(-1, 1)


# ── Image utilities ────────────────────────────────────────────────────────────

def pil_to_tensor(img: Image.Image, size: int = 256) -> torch.Tensor:
    """PIL image → (1, 3, size, size) tensor in [-1, 1]."""
    img = img.convert("RGB").resize((size, size))
    t = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return t(img).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) tensor in [-1, 1] → PIL image."""
    arr = ((t.squeeze(0).permute(1, 2, 0).cpu().float().numpy() + 1) / 2 * 255)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def print_latent_shapes():
    """Print the STARFlow latent sequence lengths for each resolution."""
    print("\nSTARFlow latent sequence lengths (patch_size=1, Appendix B.1):")
    for res in [256, 512, 1024]:
        h = res // 8
        print(f"  {res:4d}x{res} → {h}x{h} latent → {h*h:5d} tokens x 4 channels")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(device=device)
    print_latent_shapes()

    # Sanity check with a random image
    dummy = torch.randn(1, 3, 256, 256).to(device)
    latent = encode(vae, dummy, add_noise=True)
    recon  = decode(vae, latent)
    print(f"\nRound-trip check:")
    print(f"  Input  : {tuple(dummy.shape)}")
    print(f"  Latent : {tuple(latent.shape)}")
    print(f"  Output : {tuple(recon.shape)}")
    print("VAE OK.")
