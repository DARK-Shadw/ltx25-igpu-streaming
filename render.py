"""One-command renderer: picks the settings for a mode, runs generation + decode, and files everything neatly.

  python render.py --mode flash --name ronin-entrance --prompt-file prompts/anime_ronin-entrance.txt --seed 77
  python render.py --mode preview ...        # stage 1 only (7 min) to check a prompt/seed before a full render
  python render.py --mode flash --name X --prompt-file P --seed 77 --reuse-stage1   # reuse saved stage-1 latents

Output:  videos/<category>/<name>/<mode>_<WxH>_<fps>_<seconds>s_seed<N>.mp4  (+ .json with settings and timings)
Latents: latents/<name>_<mode>_seed<N>.pt (+ .pt.stage1)      Log: logs/<name>_<mode>_seed<N>.log
"""
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
_V = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
PY = _V if os.path.exists(_V) else sys.executable

# W,H = stage-1 size; two-stage doubles it. s2 = refinement steps; fps = generation fps; repeat = frame duplication at export
MODES = {
    "flash":     dict(two=True,  W=768, H=448, s2=1, fps=12, repeat=2, note="12 fps doubled, 1 refinement step -> 1536x896"),
    "flash2":    dict(two=True,  W=768, H=448, s2=2, fps=12, repeat=2, note="12 fps doubled, 2 refinement steps -> 1536x896"),
    "preview8":  dict(two=False, W=768, H=448, s2=0, fps=8, repeat=3, s1="1.0,0.9875,0.975,0.909375,0.725,0.421875", note="EXPERIMENT: 8 fps tripled, 6 stage-1 steps -> 768x448"),
    "nativehd":  dict(two=False, W=1536, H=896, s2=0, fps=12, repeat=2, note="EXPERIMENT: native 1536x896, all 8 distilled steps, no upscaler stage (use --seconds 1.4)"),
    "smoke":     dict(two=False, W=384, H=256, s2=0, fps=12, repeat=2, note="SETUP SMOKE TEST: tiny 384x256 clip to prove the whole pipeline runs (use --seconds 1.4)"),
    "custom":    dict(two=True,  W=768, H=448, s2=1, fps=12, repeat=2, note="custom settings from the command line / studio UI"),
    "turbo":     dict(two=True,  W=640, H=352, s2=1, fps=12, repeat=2, note="quick look -> 1280x704"),
    "preview":   dict(two=False, W=768, H=448, s2=0, fps=12, repeat=2, note="stage 1 only -> 768x448"),
    "audiotest": dict(two=False, W=512, H=320, fps=12, repeat=2, note="small stage 1 only (~4 min) for audio/dialogue tests -> 512x320"),
    "audiohd":   dict(two=True,  W=512, H=320, s2=1, fps=12, repeat=2, note="refine a saved audiotest stage 1 (use --reuse-stage1) -> 1024x640"),
    "reference": dict(two=True,  W=768, H=448, s2=3, fps=24, repeat=1, note="24 fps, 3 refinement steps -> 1536x896 (slow)"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=MODES, default="flash")
ap.add_argument("--name", required=True, help="scene name, used for folders and file names")
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--seconds", type=float, default=5.4)
ap.add_argument("--category", default="anime")
ap.add_argument("--reuse-stage1", action="store_true")
ap.add_argument("--prefix", default="", help="file-name prefix, e.g. shot3_")
ap.add_argument("--subdir", default="", help="sub-folder inside videos/<category>/<name>/")
ap.add_argument("--image", default=None, help="first-frame conditioning image (image-to-video)")
ap.add_argument("--w", type=int, help="stage-1 width (multiple of 32)")
ap.add_argument("--h", type=int, help="stage-1 height (multiple of 32)")
ap.add_argument("--fps", type=int, help="generation fps (12 = frames doubled on export, 24 = true 24 fps)")
ap.add_argument("--s2", type=int, help="refinement steps at 2x resolution (0 = stage 1 only, no upscaling)")
ap.add_argument("--s1sigmas", help="custom stage-1 sigma schedule, comma separated (default: the 8-step distilled recipe)")
a = ap.parse_args()

m = dict(MODES[a.mode])
if a.w: m["W"] = a.w
if a.h: m["H"] = a.h
if a.fps: m["fps"] = a.fps; m["repeat"] = max(1, 24 // a.fps)
if a.s2 is not None: m["s2"] = a.s2; m["two"] = a.s2 > 0
if a.s1sigmas: m["s1"] = a.s1sigmas
frames = max(9, int(round(a.seconds * m["fps"] / 8)) * 8 + 1)          # frame counts must be 8k+1
FW, FH = (m["W"] * 2, m["H"] * 2) if m["two"] else (m["W"], m["H"])
secs = frames / m["fps"]
tag = f"{a.name}_{a.prefix}{a.mode}_seed{a.seed}"
out_dir = os.path.join(ROOT, "videos", a.category, a.name, a.subdir)
for d in (out_dir, os.path.join(ROOT, "latents"), os.path.join(ROOT, "logs")):
    os.makedirs(d, exist_ok=True)
lat = os.path.join("latents", f"{tag}.pt")
log = os.path.join(ROOT, "logs", f"{tag}.log")
video = os.path.join(out_dir, f"{a.prefix}{a.mode}_{FW}x{FH}_{int(m['fps'] * m['repeat'])}fps_{secs:.1f}s_seed{a.seed}.mp4")
prompt = open(os.path.join(ROOT, a.prompt_file), encoding="utf-8").read().strip()

prev_dir = os.path.join(ROOT, "previews", tag)
shutil.rmtree(prev_dir, ignore_errors=True)
env = dict(os.environ, PREVIEW_DIR=prev_dir, SEED=str(a.seed), SKIPDEC="1", LAT=lat, W=str(m["W"]), H=str(m["H"]), FRAMES=str(frames), FPS=str(m["fps"]),
           PROMPT=prompt, BOS="1", MAXSEQ="512", PYTHONUNBUFFERED="1")
if m.get("s1"):
    env["S1SIGMAS"] = m["s1"]
if a.image:
    env["IMAGE"] = os.path.abspath(a.image)
if m["two"]:
    env.update(TWOSTAGE="1", S2STEPS=str(m["s2"]))
    if a.reuse_stage1 and os.path.exists(os.path.join(ROOT, lat + ".stage1")):
        env["RESUME"] = "1"

print(f"[render] {a.name} | mode {a.mode} ({m['note']}) | {frames} frames @ {m['fps']} fps = {secs:.1f} s | seed {a.seed}")
print(f"[render] output: {video}")
t0 = time.perf_counter()
with open(log, "w", encoding="utf-8") as lf:
    r = subprocess.run([PY, "stream/run_ltx.py"], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
t_gen = time.perf_counter() - t0
if r.returncode != 0:
    print(f"[render] GENERATION FAILED (exit {r.returncode}); see {log}"); sys.exit(r.returncode)
print(f"[render] generation done in {t_gen / 60:.1f} min; decoding...")
t1 = time.perf_counter()
with open(log, "a", encoding="utf-8") as lf:
    r = subprocess.run([PY, "stream/decode_big.py", lat, video, str(m["repeat"])], cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                       stdout=lf, stderr=subprocess.STDOUT)
t_dec = time.perf_counter() - t1
if r.returncode != 0:
    print(f"[render] DECODE FAILED (exit {r.returncode}); latents are saved at {lat}; see {log}"); sys.exit(r.returncode)
meta = dict(name=a.name, mode=a.mode, note=m["note"], seed=a.seed, frames=frames, gen_fps=m["fps"], frame_repeat=m["repeat"], seconds=round(secs, 2),
            resolution=f"{FW}x{FH}", generation_minutes=round(t_gen / 60, 1), decode_minutes=round(t_dec / 60, 1),
            total_minutes=round((t_gen + t_dec) / 60, 1), image=a.image, prompt=prompt, latents=lat, log=os.path.relpath(log, ROOT))
json.dump(meta, open(video[:-4] + ".json", "w", encoding="utf-8"), indent=2)
print(f"[render] DONE in {(t_gen + t_dec) / 60:.1f} min -> {video}")
