"""Decode audio latents with an fp32 vocoder and replace the audio track of existing mp4s (video stream copied untouched)."""
import os, sys, numpy as np, torch, av
from diffusers import AutoencoderKLLTX2Audio
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
M, dev = "models/ltx25", torch.device("xpu")
_av = _voc = None
def decode_audio(lat_path):
    global _av, _voc
    if _av is None:
        _av = AutoencoderKLLTX2Audio.from_pretrained(f"{M}/audio_vae", torch_dtype=torch.float32).to(dev)
        _voc = LTX2VocoderWithBWE.from_pretrained(f"{M}/vocoder", torch_dtype=torch.float32).to(dev)
    a = torch.load(lat_path)["audio"].to(dev, torch.float32)
    with torch.no_grad():
        wav = _voc(_av.decode(a, return_dict=False)[0])
    return wav[0].float().cpu().numpy(), _voc.config.output_sampling_rate      # (2, N), sr

def replace_audio(video_in, wav, sr, video_out):
    inp = av.open(video_in); out = av.open(video_out, "w")
    vin = inp.streams.video[0]; vout = out.add_stream_from_template(vin)
    aout = out.add_stream("aac", rate=sr); aout.layout = "stereo"
    for pkt in inp.demux(vin):
        if pkt.dts is None: continue
        pkt.stream = vout; out.mux(pkt)
    wav = np.ascontiguousarray(np.clip(wav, -1, 1).astype(np.float32)); n = 1024; pts = 0
    for i in range(0, wav.shape[1], n):
        chunk = wav[:, i:i + n]
        fr = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo"); fr.sample_rate = sr; fr.pts = pts; pts += chunk.shape[1]
        for pkt in aout.encode(fr): out.mux(pkt)
    for pkt in aout.encode(None): out.mux(pkt)
    out.close(); inp.close()

if __name__ == "__main__":
    pairs = [a.split("=", 1) for a in sys.argv[1:]]            # video=latent pairs
    for vid, lat in pairs:
        wav, sr = decode_audio(lat); tmp = vid[:-4] + ".audiofix.mp4"
        replace_audio(vid, wav, sr, tmp)
        c = av.open(tmp); nv = sum(1 for _ in c.decode(video=0)); c.close()
        os.replace(tmp, vid)
        print(f"fixed {vid}: {nv} frames kept, audio {wav.shape[1]/sr:.2f}s peak {np.abs(wav).max():.3f} rms {np.sqrt((wav**2).mean()):.4f}", flush=True)
