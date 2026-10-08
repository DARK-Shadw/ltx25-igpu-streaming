import os, sys, numpy as np, torch, av
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stream"))
from PIL import Image
from diffusers import AutoencoderKLLTX2Video
from dev import DEV, acc, MODELS
vae = AutoencoderKLLTX2Video.from_pretrained(f"{MODELS}/vae", torch_dtype=torch.bfloat16).to(DEV); vae.enable_tiling()
lat = torch.load("latents/ye-chen-ep1_shot01_a0_custom_seed501.pt.stage1")["video"].to(DEV, torch.bfloat16)
with torch.no_grad():
    ts = torch.tensor([0.0], device=DEV, dtype=torch.bfloat16) if vae.config.timestep_conditioning else None
    y = vae.decode(lat, ts, return_dict=False)[0]            # [1,3,F,448,800]
fr = ((y[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).cpu().numpy().astype("uint8"); acc.synchronize()
print("stage-1 decoded", fr.shape)
up = np.stack([np.array(Image.fromarray(f).resize((1600, 896), Image.LANCZOS)) for f in fr])
np.save("frames/ep1_stage1_up_f0.npy", up[0]); 
def flicker(fr_list):
    g = [x.astype("float32").mean(2)[::2, ::2] for x in fr_list]; H, W = g[0].shape; th, tw = H // 8, W // 8; acc_ = np.zeros((8, 8))
    for i in range(len(g) - 1): acc_ += np.abs(g[i + 1] - g[i])[:th * 8, :tw * 8].reshape(8, th, 8, tw).mean((1, 3))
    acc_ /= len(g) - 1; return float(np.sort(acc_.ravel())[:16].mean()), float(np.median(acc_))
fin = [f.to_ndarray(format="rgb24") for f in av.open("videos/episodes/ye-chen-ep1/shots/shot01_a0_custom_1600x896_24fps_6.8s_seed501.mp4").decode(video=0)][::2]
print("temporal noise (quiet tiles / median): stage-1 only enlarged x2 %.2f / %.2f | final FLASH %.2f / %.2f" % (*flicker(list(up)), *flicker(fin)))
src = np.array(Image.open("episodes/ye-chen-ep1/frames/Shot 1.png").convert("RGB").resize((1600, 896), Image.LANCZOS))
def psnr(a, b): return 10 * np.log10(255 ** 2 / ((a.astype(float) - b.astype(float)) ** 2).mean())
for k in (0, 40, 80): print("frame", k, "PSNR vs source: stage-1 enlarged %.1f | final %.1f" % (psnr(src, up[k]), psnr(src, fin[k])))
box = (560, 420, 960, 645)
S = Image.new("RGB", (400 * 3 + 16, 225), (255, 0, 0))
for i, a in enumerate((src, up[40], fin[40])): S.paste(Image.fromarray(a[box[1]:box[3], box[0]:box[2]]), (i * 408, 0))
S.save("frames/ep1_stage1_vs_final_f40.png")
