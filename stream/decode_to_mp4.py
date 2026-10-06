"""Decode saved latents (video + audio) to an mp4 with sound."""
import sys, torch
from diffusers import AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from diffusers.utils import encode_video
from diffusers.video_processor import VideoProcessor
lat_path, out = sys.argv[1], sys.argv[2]
dev, M = torch.device("xpu"), "models/ltx25"
d = torch.load(lat_path)
vae = AutoencoderKLLTX2Video.from_pretrained(f"{M}/vae", torch_dtype=torch.bfloat16).to(dev); vae.enable_tiling()
with torch.no_grad():
    ts = torch.tensor([0.0], device=dev, dtype=torch.bfloat16) if vae.config.timestep_conditioning else None
    v = vae.decode(d["video"].to(dev, torch.bfloat16), ts, return_dict=False)[0]
    frames = (VideoProcessor(vae_scale_factor=32).postprocess_video(v, output_type="np")[0] * 255).round().astype("uint8")
del vae, v; torch.xpu.empty_cache()
av_ = AutoencoderKLLTX2Audio.from_pretrained(f"{M}/audio_vae", torch_dtype=torch.float32).to(dev)
voc = LTX2VocoderWithBWE.from_pretrained(f"{M}/vocoder", torch_dtype=torch.float32).to(dev)
with torch.no_grad():
    wav = voc(av_.decode(d["audio"].to(dev, torch.float32), return_dict=False)[0])
encode_video(frames, fps=24, output_path=out, audio=wav[0].float().cpu(), audio_sample_rate=voc.config.output_sampling_rate)
print("wrote", out, frames.shape, f"{frames.shape[0] / 24:.2f}s")
