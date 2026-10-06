"""Build ltx_igpu_pack.zip: everything a friend needs (no models, videos, latents or venv)."""
import os, zipfile
ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = ["setup_ltx.py", "render.py", "friend_probe.py", "README_FRIEND.md", "BIBLE.md"]
STREAM = ["dev.py", "blockstream.py", "runtime.py", "run_ltx.py", "decode_big.py", "selftest.py", "stitch.py", "fix_audio.py", "transcribe.py"]
out = os.path.join(ROOT, "ltx_igpu_pack.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(os.path.join(ROOT, f), f"ltx_igpu/{f}")
    for f in STREAM:
        z.write(os.path.join(ROOT, "stream", f), f"ltx_igpu/stream/{f}")
    for f in sorted(os.listdir(os.path.join(ROOT, "prompts"))):
        p = os.path.join(ROOT, "prompts", f)
        if os.path.isfile(p):
            z.write(p, f"ltx_igpu/prompts/{f}")
        else:
            for g in os.listdir(p):
                z.write(os.path.join(p, g), f"ltx_igpu/prompts/{f}/{g}")
print(out, round(os.path.getsize(out) / 1e6, 2), "MB")
