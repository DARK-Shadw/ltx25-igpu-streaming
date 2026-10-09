"""Build the showcase media in docs/media/: a 16:9 highlight film, a 9:16 reel (blurred-background fit + captions), a preview GIF and a speed/quality comparison image.

  python make_showcase.py
"""
import glob, os, subprocess, sys
import av, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(ROOT, "stream")); import stitch
OUT = os.path.join(ROOT, "docs", "media"); os.makedirs(OUT, exist_ok=True)
V1 = "videos/episodes/ye-chen-ep1/shots/"; V2 = "videos/episodes/ye-chen-ep1-v2/shots/"
g = lambda pat: sorted(glob.glob(pat))[0]
FONT = lambda sz: ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", sz)
TOP = "LTX-2.5 (22B) · fully unquantized bf16 · Intel Core Ultra 5 iGPU · 15.6 GB RAM · no discrete GPU"

# (path, trim_end_s, audio_delay_s, caption)
SEG = [
 (g(V1 + "shot01_a0s2_custom_*.mp4"), 3.0, 0, "Image → video, audio generated too"),
 (g(V1 + "shot03_a0_custom_*.mp4"), 3.4, 0, "Start frame: my own artwork"),
 (g(V1 + "shot10_a0_custom_*seed510.mp4"), 3.2, 0, "5 s clips take ~14-30 min each on the iGPU"),
 (g(V1 + "shot04_a0_custom_*.mp4"), 3.4, 0, "Weights stream from NVMe layer by layer"),
 (g(V1 + "shot05A_a0_custom_*.mp4"), 3.4, 0, "1600×896 after a ×2 latent upsample"),
 (g(V2 + "shot25_a0_custom_*.mp4"), 4.0, 0, "Speech, sound effects and picture in one pass"),
 (g(V2 + "shot28_a0_custom_*.mp4"), 4.4, 0, "No quantization, no cloud"),
 (g(V2 + "shot30_a1_custom_*.mp4"), 6.6, 1.2, "Local. Offline. Integrated graphics."),
]

def caption(img, text, top=True):
    im = Image.fromarray(img); d = ImageDraw.Draw(im, "RGBA"); W, H = im.size
    f = FONT(max(14, W // 62)) if top else FONT(max(18, W // 36))
    tw = d.textlength(text, font=f); pad = 12
    y = 14 if top else H - int(H * 0.11)
    d.rounded_rectangle((W / 2 - tw / 2 - pad, y - 6, W / 2 + tw / 2 + pad, y + f.size + 8), 10, fill=(0, 0, 0, 150))
    d.text((W / 2 - tw / 2, y), text, font=f, fill=(255, 255, 255, 255)); return np.asarray(im)

def active_rms(p):
    c = av.open(p); rs = av.AudioResampler(format="fltp", layout="mono", rate=16000); o = []
    for f in c.decode(c.streams.audio[0]):
        for r in rs.resample(f): o.append(r.to_ndarray().reshape(-1))
    x = np.concatenate(o); w = x[:len(x) // 800 * 800].reshape(-1, 800); r = np.sqrt((w ** 2).mean(1)); a = r[r > 0.3 * r.max()]
    return float(np.sqrt((a ** 2).mean())) if len(a) else 0.01
gain = {p: 10 ** (float(np.clip(20 * np.log10(0.05 / active_rms(p)), -12, 14)) / 20) for p, *_ in SEG}

# ---- 16:9 highlight film
shots = [dict(path=p, trim_end_s=t, audio_delay_s=d, gain=gain[p], raw=True, frame_fn=(lambda cap: (lambda x, k: caption(caption(x, TOP, True), cap, False)))(c)) for p, t, d, c in SEG]
hl = os.path.join(OUT, "ltx25_igpu_highlights_1600x896.mp4")
print(stitch.stitch(shots, hl, size=(1600, 896), crf=27), hl)

# ---- 9:16 reel (1080x1920): clip fitted to the middle, blurred copy behind, captions above/below
RW, RH = 1080, 1920
def reel_frame(x, cap):
    im = Image.fromarray(x); sm = im.resize((RW // 4, RH // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6)).resize((RW, RH), Image.BILINEAR)
    bg = np.asarray(sm).astype("float32") * 0.45
    fit = im.resize((RW, int(RW * im.height / im.width)), Image.LANCZOS); y0 = (RH - fit.height) // 2
    canvas = Image.fromarray(bg.astype("uint8")); canvas.paste(fit, (0, y0)); d = ImageDraw.Draw(canvas, "RGBA")
    def wrap(t, f, wmax):
        words, lines, cur = t.split(), [], ""
        for w in words:
            if d.textlength((cur + " " + w).strip(), font=f) > wmax: lines.append(cur); cur = w
            else: cur = (cur + " " + w).strip()
        return lines + [cur]
    f1 = FONT(46); y = 150
    for ln in wrap("A 22B video model, fully unquantized, on a laptop's integrated GPU", f1, RW - 120):
        tw = d.textlength(ln, font=f1); d.text(((RW - tw) / 2, y), ln, font=f1, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255)); y += 62
    f2 = FONT(50); yb = y0 + fit.height + 70
    for ln in wrap(cap, f2, RW - 120):
        tw = d.textlength(ln, font=f2); d.text(((RW - tw) / 2, yb), ln, font=f2, fill=(255, 224, 140, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255)); yb += 66
    f3 = FONT(34); t3 = "Intel Core Ultra 5 125H · 15.6 GB RAM · ~14-30 min per 5 s clip"; tw = d.textlength(t3, font=f3)
    d.text(((RW - tw) / 2, RH - 170), t3, font=f3, fill=(255, 255, 255, 230), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return np.asarray(canvas)
shots = [dict(path=p, trim_end_s=t, audio_delay_s=d, gain=gain[p], raw=True, frame_fn=(lambda cap: (lambda x, k: reel_frame(x, cap)))(c)) for p, t, d, c in SEG]
reel = os.path.join(OUT, "ltx25_igpu_reel_1080x1920.mp4")
print(stitch.stitch(shots, reel, size=(RW, RH), crf=28), reel)

# ---- preview GIF (first 10 s, 640 px wide, 12 fps) via ffmpeg
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe(); gif = os.path.join(OUT, "preview.gif")
subprocess.run([ff, "-y", "-i", hl, "-t", "8", "-vf", "fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer", gif], capture_output=True)
print("gif", os.path.getsize(gif) // 1024, "KB")
