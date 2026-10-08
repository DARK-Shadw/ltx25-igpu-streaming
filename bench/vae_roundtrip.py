"""Where does detail get lost? (1) VAE encode->decode of the source image alone, (2) decode of a saved stage-1 latent."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stream"))
from PIL import Image
from diffusers import AutoencoderKLLTX2Video
from dev import DEV, acc, MODELS
vae = AutoencoderKLLTX2Video.from_pretrained(f"{MODELS}/vae", torch_dtype=torch.bfloat16).to(DEV); vae.enable_tiling()
src = Image.open("episodes/ye-chen-ep1/frames/Shot 1.png").convert("RGB")
def rt(w, h):
    im = src.resize((w, h), Image.LANCZOS); x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).unsqueeze(2).to(DEV, torch.bfloat16)  # [1,3,1,H,W]
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        ts = torch.tensor([0.0], device=DEV, dtype=torch.bfloat16) if vae.config.timestep_conditioning else None
        y = vae.decode(z, ts, return_dict=False)[0]
    out = ((y[0, :, 0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype("uint8"); acc.synchronize(); return out
os.makedirs("frames", exist_ok=True)
for (w, h) in ((1600, 896), (800, 448)):
    Image.fromarray(rt(w, h)).save(f"frames/ep1_vae_roundtrip_{w}x{h}.png"); print("roundtrip", w, h, "ok")
lat = torch.load("latents/ye-chen-ep1_shot01_a0_custom_seed501.pt.stage1")["video"].to(DEV, torch.bfloat16)
print("stage-1 latent", tuple(lat.shape))
with torch.no_grad():
    ts = torch.tensor([0.0], device=DEV, dtype=torch.bfloat16) if vae.config.timestep_conditioning else None
    y = vae.decode(lat[:, :, :2], ts, return_dict=False)[0]
fr = ((y[0, :, 0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype("uint8")
Image.fromarray(fr).save("frames/ep1_stage1_decoded_f0.png"); print("stage-1 frame0", fr.shape)
