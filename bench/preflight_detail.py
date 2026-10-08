"""Pre-flight: how much of a start frame survives the video model's working size? (VAE round trip at stage-1 size, plus the loss from shrinking.)"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stream"))
from PIL import Image
from diffusers import AutoencoderKLLTX2Video
from dev import DEV, acc, MODELS
vae = AutoencoderKLLTX2Video.from_pretrained(f"{MODELS}/vae", torch_dtype=torch.bfloat16).to(DEV); vae.enable_tiling()
ep = sys.argv[1] if len(sys.argv) > 1 else "ye-chen-ep1"; W, H = 800, 448
plan = json.load(open(f"episodes/{ep}/shots.json", encoding="utf-8"))
psnr = lambda a, b: 10 * np.log10(255 ** 2 / ((a.astype(float) - b.astype(float)) ** 2).mean())
def rt(im):
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).unsqueeze(2).to(DEV, torch.bfloat16)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode(); ts = torch.tensor([0.0], device=DEV, dtype=torch.bfloat16) if vae.config.timestep_conditioning else None
        y = vae.decode(z, ts, return_dict=False)[0]
    return ((y[0, :, 0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype("uint8")
rows = []
for s in plan["shots"]:
    full = Image.open(f"episodes/{ep}/frames/{s['frame']}").convert("RGB"); small = full.resize((W, H), Image.LANCZOS)
    r = Image.fromarray(rt(small)).resize(full.size, Image.LANCZOS)
    lap = lambda im: float(np.abs(np.diff(np.array(im.convert("L")).astype("float32"), axis=1)).mean())
    rows.append((s["id"], round(psnr(np.array(full), np.array(r)), 1), round(lap(full), 1)))
print("shot  PSNR(full-res source vs its 800x448 video-encoder round trip, higher=safer)  detail(edge energy)")
for r in sorted(rows, key=lambda x: x[1]): print(f"{r[0]:>4}  {r[1]:5.1f}   {r[2]:5.1f}")
