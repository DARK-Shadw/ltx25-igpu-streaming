import sys, torch, numpy as np
from diffusers import AutoencoderKLLTX2Audio
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
M = "models/ltx25"
def st(name, x):
    x = x.detach().float().cpu()
    print(f"  {name:28s} shape={tuple(x.shape)} finite={torch.isfinite(x).all().item()} min={x.min():.4g} max={x.max():.4g} mean={x.mean():.4g} std={x.std():.4g} |abs|max={x.abs().max():.4g}")
for path in sys.argv[1:]:
    d = torch.load(path); a = d["audio"]
    print("==", path)
    st("audio latents (stored)", a)
    for dev, dt in (("xpu", torch.bfloat16), ("xpu", torch.float32), ("cpu", torch.float32)):
        try:
            av_ = AutoencoderKLLTX2Audio.from_pretrained(f"{M}/audio_vae", torch_dtype=dt).to(dev)
            voc = LTX2VocoderWithBWE.from_pretrained(f"{M}/vocoder", torch_dtype=dt).to(dev)
            print(f" decode on {dev} {str(dt)}:")
            if dev == "xpu" and dt == torch.bfloat16:
                print("  audio_vae latents_mean/std (first 4):", av_.latents_mean[:4].tolist(), av_.latents_std[:4].tolist())
            with torch.no_grad():
                mel = av_.decode(a.to(dev, dt), return_dict=False)[0]; st("mel", mel)
                wav = voc(mel); st("wav", wav)
            del av_, voc
        except Exception as e:
            print(f" decode on {dev} {dt}: FAILED {type(e).__name__}: {str(e)[:150]}")
