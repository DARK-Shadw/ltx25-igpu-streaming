"""Snow Bridge (~59 s): 11 FLASH shots with a planned mix of text-to-video (hard cuts to new subjects) and
image-to-video (same subject continues / cut back to a character already shown).

  python story_snowbridge.py     # resumable: finished shots are skipped

Smart parts:
  * the first frame for an image-to-video shot is the SHARPEST clean frame of the source shot inside a chosen window
    (tail = last ~8 frames for direct continuation, wider window for cut-backs) - not blindly the last frame
  * motion check: shots marked `motion` must show real movement (median frame change >= 8); otherwise one retry with a new seed
  * a "film so far" cut is stitched after shots 3, 6 and 9, and the final film + speech transcript at the end
"""
import glob, json, os, subprocess, sys, time
import av, numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
NAME, MODE = "snowbridge", "flash"
OUT = os.path.join(ROOT, "videos", "anime", NAME)
FR = os.path.join(ROOT, "frames", NAME)
os.makedirs(os.path.join(OUT, "shots"), exist_ok=True); os.makedirs(FR, exist_ok=True)

# src = (shot id whose frame starts this shot, window as fraction of that clip) ; None = text-to-video
SHOTS = [
    dict(id=1, seed=301, src=None),
    dict(id=2, seed=302, src=(1, "tail")),            # continues shot 1 (camera glides in on the same hero)
    dict(id=3, seed=303, src=None),                   # hard cut: rival enters (new subject)
    dict(id=4, seed=304, src=(3, "tail")),            # continues shot 3 (close-up on the same rival)
    dict(id=5, seed=305, src=(2, (0.45, 1.0))),       # cut back to the hero, anchored on shot 2's hero frame
    dict(id=6, seed=306, src=None),                   # wide standoff with both (new framing)
    dict(id=7, seed=307, src=None, motion=True),      # action beat 1
    dict(id=8, seed=308, src=(7, "tail"), motion=True),   # action beat 2 continues the clash
    dict(id=9, seed=309, src=None),                   # insert: the black blade shatters
    dict(id=10, seed=310, src=(5, (0.5, 1.0))),       # hero again, anchored on shot 5
    dict(id=11, seed=311, src=None),                  # ending: wide, rising camera
]


def gray_frames(path):
    return [f.to_ndarray(format="gray").astype("float32") for f in av.open(path).decode(video=0)]


def sharp(g):
    s = g[::4, ::4]
    return float(np.abs(np.diff(s, axis=1)).mean() + np.abs(np.diff(s, axis=0)).mean())


def pick_frame(video, window, png):
    g = gray_frames(video); n = len(g)
    lo, hi = ((max(0, n - 16) / n, 1.0) if window == "tail" else window)
    a, b = int(lo * n), n
    b = max(a + 1, int(hi * n))
    chg = [np.abs(g[i] - g[i - 1]).mean() if i else 0.0 for i in range(n)]
    med = float(np.median(chg)) + 1e-6
    best, bi = -1, b - 1
    for i in range(a, b):
        sc = sharp(g[i]) * (0.5 if chg[i] > 4 * med else 1.0)      # avoid frames in a sudden jump/smear
        if sc > best: best, bi = sc, i
    for k, f in enumerate(av.open(video).decode(video=0)):
        if k == bi:
            Image.fromarray(f.to_ndarray(format="rgb24")).save(png); break
    return bi, n


def motion_median(video):
    g = gray_frames(video)[::2]
    return float(np.median([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)]))


def vid(seed, sid):
    g = glob.glob(os.path.join(OUT, "shots", f"shot{sid}_{MODE}_*_seed{seed}.mp4"))
    return g[0] if g else None


chosen, report_shots, t_all = {}, {}, time.perf_counter()
sys.path.insert(0, os.path.join(ROOT, "stream"))
import stitch


def render(s, seed, alt=False):
    cmd = [PY, "render.py", "--mode", MODE, "--category", "anime", "--name", NAME, "--subdir", "shots", "--prefix", f"shot{s['id']}_",
           "--prompt-file", (f"prompts/{NAME}/shot{s['id']}_alt.txt" if alt and os.path.exists(os.path.join(ROOT, "prompts", NAME, f"shot{s['id']}_alt.txt")) else f"prompts/{NAME}/shot{s['id']}.txt"), "--seed", str(seed), "--seconds", "5.4", "--reuse-stage1"]
    if s["src"]:
        sid, win = s["src"]
        png = os.path.join(FR, f"shot{s['id']}_start_from_shot{sid}.png")
        if not os.path.exists(png):
            bi, n = pick_frame(chosen[sid], win, png)
            print(f"[story] shot {s['id']}: start frame = frame {bi}/{n} of shot {sid}", flush=True)
        cmd += ["--image", png]
    print(f"[story] rendering shot {s['id']} seed {seed}" + (f" (image-to-video from shot {s['src'][0]})" if s["src"] else " (text-to-video)"), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[story] shot {s['id']} FAILED (exit {r.returncode}); rerun story_snowbridge.py to resume", flush=True); sys.exit(1)
    return vid(seed, s["id"])


def progress_cut(upto):
    p = os.path.join(OUT, f"{NAME}_PROGRESS_shots1-{upto}.mp4")
    for old in glob.glob(os.path.join(OUT, f"{NAME}_PROGRESS_*.mp4")): os.remove(old)
    info = stitch.stitch([dict(path=chosen[i]) for i in range(1, upto + 1)], p, xfade_ms=300)
    print(f"[story] film so far ({info['seconds']} s): {p}", flush=True)


for s in SHOTS:
    seeds = [s["seed"], s["seed"] + 1000] if s.get("motion") else [s["seed"]]
    best = None
    for k, seed in enumerate(seeds):
        v = vid(seed, s["id"])
        if not (v and os.path.exists(v[:-4] + ".json")):
            v = render(s, seed, alt=(k > 0))
        m = motion_median(v)
        ok = (not s.get("motion")) or (8 <= m <= 38)          # <8 = static, >38 = chaotic smear (calibrated: liked clip 27, smeared clip 47)
        print(f"[story] shot {s['id']} seed {seed}: median motion {m:.1f} -> {'OK' if ok else 'REJECTED'}", flush=True)
        dist = 0 if ok else (8 - m if m < 8 else m - 38)
        if best is None or dist < best[3]: best = (v, m, seed, dist)
        if ok:
            break
        print(f"[story] shot {s['id']} outside the motion range 8-38; retrying once with a calmer prompt and a new seed", flush=True)
    chosen[s["id"]] = best[0]
    report_shots[s["id"]] = dict(seed=best[2], motion=round(best[1], 1), i2v_from=(s["src"][0] if s["src"] else None),
                                 minutes=json.load(open(best[0][:-4] + ".json"))["total_minutes"])
    if s["id"] in (3, 6, 9):
        progress_cut(s["id"])

final = os.path.join(OUT, f"{NAME}_60s_FLASH_1536x896.mp4")
info = stitch.stitch([dict(path=chosen[i]) for i in range(1, 12)], final, xfade_ms=300)
for old in glob.glob(os.path.join(OUT, f"{NAME}_PROGRESS_*.mp4")): os.remove(old)
tr = subprocess.run([PY, "stream/transcribe.py", final, "small.en"], cwd=ROOT, capture_output=True, text=True)
open(os.path.join(OUT, "transcript.txt"), "w", encoding="utf-8").write(tr.stdout)
report = dict(title="Snow Bridge", final=os.path.relpath(final, ROOT), seconds=info["seconds"], resolution=info["size"], shots=report_shots,
              render_minutes=round(sum(v["minutes"] for v in report_shots.values()), 1), wall_clock_minutes=round((time.perf_counter() - t_all) / 60, 1))
json.dump(report, open(os.path.join(OUT, "story_report.json"), "w"), indent=2)
print(json.dumps(report, indent=2)); print(tr.stdout)
