"""
Loading the STARFlow VAE in VSCode — step by step
===================================================
The VAE used in STARFlow is stabilityai/sd-vae-ft-mse, loaded via
diffusers.AutoencoderKL. This is confirmed in Gu et al. (2025, §3.3,
footnote 1): "when using SD-1.4 autoencoder" with the HuggingFace path
https://huggingface.co/stabilityai/sd-vae-ft-mse.

In model_setup.py the same call appears as:
    self.vae = AutoencoderKL.from_pretrained(model_name)
    self.scaling_factor = self.vae.config.scaling_factor

Run this file with:
    python load_vae.py
"""

import torch
from diffusers.models import AutoencoderKL
from PIL import Image
import torchvision.transforms as T
import numpy as np


# ── Step 1: Load the pretrained VAE ───────────────────────────────────────────
# from_pretrained() downloads weights from HuggingFace on first run
# (~335 MB) and caches them in ~/.cache/huggingface/
# On subsequent runs it loads from the local cache — no internet needed.

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
print("Done.")

# Move to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = vae.to(device)

# Freeze all parameters — this is what STARFlow does.
# No gradient will ever flow through the encoder.
vae.requires_grad_(False)
vae.eval()


# ── Step 2: Inspect the architecture ──────────────────────────────────────────
print("\n── VAE config ──────────────────────────────────────")
print(f"  Latent channels:   {vae.config.latent_channels}")   # 4
print(f"  Scaling factor:    {vae.config.scaling_factor}")    # 0.18215
print(f"  Down block types:  {vae.config.down_block_types}")
print(f"  Spatial downsample: {2 ** (len(vae.config.down_block_types) - 1)}×")
# A 256×256 image becomes 256/8 = 32×32 latent (4 channels)
# A 512×512 image becomes 64×64 latent (4 channels)
# STARFlow uses patch_size=1, so the flow sequence length = 32*32 = 1024
# for 256×256 images (Gu et al., 2025, Appendix B.1)

total_params = sum(p.numel() for p in vae.parameters())
print(f"  Total parameters:  {total_params / 1e6:.1f}M")


# ── Step 3: Encode an image to a noisy latent ─────────────────────────────────
# We simulate what happens during STARFlow training:
# x̃ ~ N(E(x), σ_L² I)  with σ_L = 0.3  (Gu et al., 2025, §3.3)

def load_image(path: str, size: int = 256) -> torch.Tensor:
    """Load an image as a (1, 3, size, size) tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size))
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return transform(img).unsqueeze(0)   # (1, 3, H, W)


def encode_with_noise(vae, x: torch.Tensor, sigma_L: float = 0.3) -> torch.Tensor:
    """
    Replicate the STARFlow encoding step:
        x̃ ~ N(E(x), sigma_L² I)

    The VAE encoder returns a diagonal Gaussian posterior.
    We call .sample() on it, which draws x̃ = mean + std * eps.
    We then add the STARFlow training noise on top.
    """
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        # Sample from the encoder posterior
        latent = posterior.sample()
        # Apply the VAE scaling factor (matches model_setup.py line:
        #     z = self._encode(x).latent_dist.sample()
        #     z = z * self.scaling_factor
        latent = latent * vae.config.scaling_factor
        # Add STARFlow training noise σ_L = 0.3
        noise = torch.randn_like(latent) * sigma_L
        noisy_latent = latent + noise
    return noisy_latent


def decode_latent(vae, noisy_latent: torch.Tensor) -> torch.Tensor:
    """
    Decode a (noisy) latent back to pixel space.
    In a fully deployed STARFlow this would be the finetuned decoder —
    here we use the standard pretrained one for demonstration.
    """
    with torch.no_grad():
        latent_unscaled = noisy_latent / vae.config.scaling_factor
        image = vae.decode(latent_unscaled).sample
        # Clamp to [-1, 1] then convert to [0, 255]
        image = image.clamp(-1, 1)
    return image


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) tensor in [-1, 1] to a PIL image."""
    arr = ((t.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Step 4: Run a round-trip encode → decode on a test image ──────────────────
# To run this part, provide a path to any .jpg or .png file.
# Example: change TEST_IMAGE_PATH to a real file on your machine.

TEST_IMAGE_PATH = "IMG_3088.JPG"  # e.g. "/path/to/your/image.jpg"

if TEST_IMAGE_PATH is not None:
    x = load_image(TEST_IMAGE_PATH).to(device)
    print(f"\n── Round-trip test ──────────────────────────────────")
    print(f"  Input shape:   {tuple(x.shape)}")       # (1, 3, 256, 256)

    noisy_latent = encode_with_noise(vae, x, sigma_L=0.3)
    print(f"  Latent shape:  {tuple(noisy_latent.shape)}")  # (1, 4, 32, 32)
    print(f"  Latent mean:   {noisy_latent.mean().item():.4f}")
    print(f"  Latent std:    {noisy_latent.std().item():.4f}")

    reconstructed = decode_latent(vae, noisy_latent)
    print(f"  Output shape:  {tuple(reconstructed.shape)}")

    # Save both images side by side for visual comparison
    orig_pil  = tensor_to_pil(x)
    recon_pil = tensor_to_pil(reconstructed)
    combined  = Image.new("RGB", (512, 256))
    combined.paste(orig_pil,  (0, 0))
    combined.paste(recon_pil, (256, 0))
    combined.save("vae_roundtrip.png")
    print("  Saved side-by-side comparison → vae_roundtrip.png")
    print("  Left = original, Right = reconstructed from noisy latent")
else:
    print("\nSkipping round-trip test — set TEST_IMAGE_PATH to run it.")


# ── Step 5: Show latent shapes for all STARFlow resolutions ───────────────────
print("\n── Latent sequence lengths in STARFlow (patch_size=1) ──")
for res in [256, 512, 1024]:
    latent_h = res // 8   # VAE downsamples 8×
    latent_w = res // 8
    seq_len  = latent_h * latent_w
    channels = 4          # VAE latent channels
    print(f"  {res}×{res} image → {latent_h}×{latent_w} latent "
          f"→ {seq_len} tokens × {channels} channels")
# Expected output (Gu et al., 2025, Appendix B.1):
#   256×256 →  32× 32 →  1024 tokens × 4 channels
#   512×512 →  64× 64 →  4096 tokens × 4 channels
# 1024×1024 → 128×128 → 16384 tokens × 4 channels
