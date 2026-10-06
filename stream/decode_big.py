"""Memory-lean, staged decode of saved latents to mp4 (video + audio) for large clips.
Frames are converted to uint8 on the GPU, so no large float32 arrays are built on the CPU."""
import os, sys, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dev import DEV, acc, MODELS
from diffusers import AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from diffusers.utils import encode_video

lat_path, out = sys.argv[1], sys.argv[2]
REPEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # 2 = show every generated frame twice (12 fps -> 24 fps "on twos")
dev, M, T0 = DEV, MODELS, time.perf_counter()
def log(*a): print(f"[{time.perf_counter() - T0:6.1f}s]", *a, flush=True)

d = torch.load(lat_path)
log("latents", tuple(d["video"].shape))
vae = AutoencoderKLLTX2Video.from_pretrained(f"{M}/vae", torch_dtype=torch.bfloat16).to(dev); vae.enable_tiling()
log("vae loaded; tiling:", {k: getattr(vae, k, None) for k in ("tile_sample_min_height", "tile_sample_min_width", "tile_sample_min_num_frames")})
with torch.no_grad():
    ts = torch.tensor([0.0], device=dev, dtype=torch.bfloat16) if vae.config.timestep_conditioning else None
    v = vae.decode(d["video"].to(dev, torch.bfloat16), ts, return_dict=False)[0]      # (1, 3, F, H, W) in [-1, 1]
    acc.synchronize(); log("vae decoded", tuple(v.shape))
    frames = []
    for i in range(0, v.shape[2], 16):                                                  # chunked GPU -> uint8 -> CPU
        c = ((v[:, :, i:i + 16].float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        frames.append(c[0].permute(1, 2, 3, 0).cpu().numpy())
    frames = np.concatenate(frames, 0)
if REPEAT > 1:
    frames = np.repeat(frames, REPEAT, axis=0)
log("frames on cpu", frames.shape, frames.dtype)
del v; acc.empty_cache()

av_ = AutoencoderKLLTX2Audio.from_pretrained(f"{M}/audio_vae", torch_dtype=torch.float32).to(dev)
voc = LTX2VocoderWithBWE.from_pretrained(f"{M}/vocoder", torch_dtype=torch.float32).to(dev)
with torch.no_grad():
    wav = voc(av_.decode(d["audio"].to(dev, torch.float32), return_dict=False)[0])
log("audio decoded", tuple(wav.shape))
encode_video(frames, fps=24, output_path=out, audio=wav[0].float().cpu(), audio_sample_rate=voc.config.output_sampling_rate)
log("wrote", out)
