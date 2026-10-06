"""Live denoising preview: turns the model's current guess of the final video (x0 estimate) into a small picture.
A linear map (fitted by fit_latent_rgb.py) sends each latent cell to a colour, so this is a colour/composition sketch
(first / middle / last frame side by side), NOT a real decode - but it shows the picture forming step by step.
"""
import os
import numpy as np
import torch
from PIL import Image

_W = None


def _map():
    global _W
    if _W is None:
        _W = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_rgb.npy"))
    return _W


def save_preview(x0_packed, F, h, w, pipe_cls, vae, patch, patch_t, path, height=168):
    """x0_packed: [B, N, C] model estimate of the clean latents (normalised, packed). Writes a PNG strip to `path`."""
    z = x0_packed.float()
    if z.ndim == 3:
        z = pipe_cls._unpack_latents(z, F, h, w, patch, patch_t)                  # packed tokens -> [B, C, F, h, w]
    z = pipe_cls._denormalize_latents(z, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)   # [B, C, F, h, w]
    Wm = _map()
    tiles = []
    for k in sorted({0, F // 2, F - 1}):
        cell = z[0, :, k].permute(1, 2, 0).float().cpu().numpy()                    # [h, w, C]
        rgb = np.clip(cell @ Wm[:128] + Wm[128], 0, 1)
        im = Image.fromarray((rgb * 255).astype("uint8")).resize((int(height * w / h), height), Image.BICUBIC)
        tiles.append(im)
    gap = 6
    strip = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), height), (20, 20, 20))
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0)); x += t.width + gap
    os.makedirs(os.path.dirname(path), exist_ok=True)
    strip.save(path + ".tmp.png"); os.replace(path + ".tmp.png", path)
