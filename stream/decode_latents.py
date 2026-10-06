import sys, torch, numpy as np
from PIL import Image
from diffusers import AutoencoderKLLTX2Video
from diffusers.video_processor import VideoProcessor
dev = torch.device("xpu")
vae = AutoencoderKLLTX2Video.from_pretrained("models/ltx25/vae", torch_dtype=torch.bfloat16).to(dev)
vae.enable_tiling()
lat = torch.load(sys.argv[2] if len(sys.argv) > 2 else "latents.pt")["video"].to(dev, torch.bfloat16)
print("latents", tuple(lat.shape), "finite", torch.isfinite(lat).all().item())
ts = torch.tensor([0.0], device=dev, dtype=torch.bfloat16) if vae.config.timestep_conditioning else None
with torch.no_grad():
    v = vae.decode(lat, ts, return_dict=False)[0]
frames = VideoProcessor(vae_scale_factor=32).postprocess_video(v, output_type="np")[0]
fr = (frames * 255).round().astype("uint8")
print("frames", fr.shape, "mean", fr.mean().round(1), "std", fr.std().round(1))
Image.fromarray(np.concatenate([fr[0], fr[8], fr[16], fr[-1]], axis=1)).save(sys.argv[1])
