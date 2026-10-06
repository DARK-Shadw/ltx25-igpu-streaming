"""Render the 30-second short "The Lantern Keeper": 6 shots in FLASH mode (each chained from a frame of an earlier shot),
stitch them with audio crossfades, transcribe the result, and write a timing report.

  python story.py            # resumes: finished shots are skipped
"""
import glob, json, os, subprocess, sys, time
import av
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
NAME, MODE = "lantern-keeper", "flash"
OUT = os.path.join(ROOT, "videos", "anime", NAME)
FR = os.path.join(ROOT, "frames", NAME)
os.makedirs(FR, exist_ok=True)

# id, seed, image source (shot id whose LAST frame is the first frame), frames to drop at the join with the previous shot
SHOTS = [
    dict(id=1, seed=101, image_from=None, drop=0),
    dict(id=2, seed=102, image_from=1, drop=2),     # continuous with shot 1 (first frame == last frame of shot 1)
    dict(id=3, seed=103, image_from=2, drop=2),     # continuous with shot 2
    dict(id=4, seed=104, image_from=1, drop=0),     # cut back to the girl
    dict(id=5, seed=105, image_from=3, drop=0),     # cut to the swordsman
    dict(id=6, seed=106, image_from=5, drop=0),     # continuous with 5 only if 5 directly precedes: it does -> drop 2 below
]
SHOTS[5]["drop"] = 2

def shot_video(s):
    g = glob.glob(os.path.join(OUT, "shots", f"shot{s['id']}_{MODE}_*_seed{s['seed']}.mp4"))
    return g[0] if g else None

def last_frame(video, png):
    fr = None
    for f in av.open(video).decode(video=0): fr = f
    Image.fromarray(fr.to_ndarray(format="rgb24")).save(png)

t_story = time.perf_counter()
times = {}
for s in SHOTS:
    v = shot_video(s)
    if v and os.path.exists(v[:-4] + ".json"):
        print(f"[story] shot {s['id']} already rendered: {os.path.basename(v)}", flush=True)
    else:
        cmd = [PY, "render.py", "--mode", MODE, "--category", "anime", "--name", NAME, "--subdir", "shots", "--prefix", f"shot{s['id']}_",
               "--prompt-file", f"prompts/{NAME}/shot{s['id']}.txt", "--seed", str(s["seed"]), "--seconds", "5.4", "--reuse-stage1"]
        if s["image_from"]:
            png = os.path.join(FR, f"shot{s['image_from']}_last.png")
            if not os.path.exists(png):
                last_frame(shot_video(next(x for x in SHOTS if x["id"] == s["image_from"])), png)
            cmd += ["--image", png]
        print(f"[story] rendering shot {s['id']} ({' '.join(cmd[2:])})", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[story] shot {s['id']} FAILED (exit {r.returncode}); fix and rerun story.py to resume"); sys.exit(1)
        v = shot_video(s)
    meta = json.load(open(v[:-4] + ".json"))
    times[s["id"]] = meta
    png = os.path.join(FR, f"shot{s['id']}_last.png")
    if not os.path.exists(png):
        last_frame(v, png)

# ---- stitch ----
sys.path.insert(0, os.path.join(ROOT, "stream"))
import stitch
final = os.path.join(OUT, f"{NAME}_30s_FLASH_1536x896.mp4")
t0 = time.perf_counter()
info = stitch.stitch([dict(path=shot_video(s), drop_first_frames=s["drop"]) for s in SHOTS], final, xfade_ms=300)
t_stitch = time.perf_counter() - t0
print("[story] stitched:", info, f"in {t_stitch:.0f}s -> {final}", flush=True)

# ---- timing report ----
gen = sum(m["generation_minutes"] for m in times.values()); dec = sum(m["decode_minutes"] for m in times.values())
report = dict(title="The Lantern Keeper", final=os.path.relpath(final, ROOT), final_seconds=info["seconds"], resolution=info["size"], mode="FLASH (12 fps doubled, 768x448 -> 1536x896, 1 refinement step)",
              per_shot={str(i): dict(total_min=m["total_minutes"], gen_min=m["generation_minutes"], decode_min=m["decode_minutes"], seed=m["seed"], image=m.get("image")) for i, m in times.items()},
              generation_minutes=round(gen, 1), decode_minutes=round(dec, 1), stitch_minutes=round(t_stitch / 60, 2),
              total_render_minutes=round(gen + dec + t_stitch / 60, 1), wall_clock_minutes_this_run=round((time.perf_counter() - t_story) / 60, 1))
json.dump(report, open(os.path.join(OUT, "story_report.json"), "w"), indent=2)
print(json.dumps(report, indent=2))
