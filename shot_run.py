"""Render one shot of an episode from its card in shots.json.

  python shot_run.py ye-chen-ep1 01              # attempt 0, seed from the card id
  python shot_run.py ye-chen-ep1 01 --attempt 1  # retry with a different seed
  python shot_run.py ye-chen-ep1 01 --seed 777   # explicit seed

Prepares the start image (crop-to-fill to the stage-1 size), writes the prompt, runs render.py in custom mode,
appends the attempt to episodes/<episode>/runs.json and writes a contact sheet next to the video. No judgement is made here.
"""
import argparse, json, os, subprocess, sys, time
import av
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

ap = argparse.ArgumentParser()
ap.add_argument("episode"); ap.add_argument("shot")
ap.add_argument("--attempt", type=int, default=0); ap.add_argument("--seed", type=int)
ap.add_argument("--refine", type=int, help="override refinement steps")
ap.add_argument("--size", help="override stage-1 size, e.g. 1024x576")
ap.add_argument("--extra", default="", help="extra sentence appended to the prompt (retry strategies)")
ap.add_argument("--tag", default="", help="suffix for the output name so variants never overwrite each other")
a = ap.parse_args()

ep_dir = os.path.join(ROOT, "episodes", a.episode)
plan = json.load(open(os.path.join(ep_dir, "shots.json"), encoding="utf-8"))
card = next(s for s in plan["shots"] if s["id"] == a.shot)
W, H = (int(x) for x in (a.size or card.get("render_override", {}).get("stage1_size") or plan["render"]["stage1_size"]).split("x"))
fps = plan["render"]["gen_fps"]
refine = a.refine if a.refine is not None else card.get("render_override", {}).get("refinement_steps", plan["render"]["refinement_steps"])
digits = "".join(c for c in a.shot if c.isdigit())
seed = a.seed if a.seed is not None else 500 + int(digits) + plan["render"].get("seed_offset", 0) + 1000 * a.attempt

# start image: crop-to-fill to the stage-1 size (no stretching)
src = Image.open(os.path.join(ep_dir, "frames", card["frame"])).convert("RGB")
s = max(W / src.width, H / src.height)
src = src.resize((max(W, round(src.width * s)), max(H, round(src.height * s))), Image.LANCZOS)
x, y = (src.width - W) // 2, (src.height - H) // 2
os.makedirs(os.path.join(ep_dir, "start"), exist_ok=True)
img_path = os.path.join(ep_dir, "start", f"shot{a.shot}_{W}x{H}.png")
src.crop((x, y, x + W, y + H)).save(img_path)

prompt_path = os.path.join(ep_dir, "start", f"shot{a.shot}_prompt.txt")
open(prompt_path, "w", encoding="utf-8").write(card["prompt"] + (" " + a.extra if a.extra else "") + "\n")

secs = card["frames"] / fps
cmd = [PY, "render.py", "--mode", "custom", "--category", "episodes", "--name", a.episode, "--subdir", "shots", "--prefix", f"shot{a.shot}_a{a.attempt}{a.tag}_",
       "--prompt-file", os.path.relpath(prompt_path, ROOT), "--seed", str(seed), "--seconds", f"{secs:.4f}", "--w", str(W), "--h", str(H),
       "--fps", str(fps), "--s2", str(refine), "--image", os.path.relpath(img_path, ROOT)]
print("[shot_run]", a.episode, a.shot, f"attempt {a.attempt} seed {seed}", f"{card['frames']} frames = {secs:.2f} s, refine {refine}", flush=True)
t0 = time.time()
r = subprocess.run(cmd, cwd=ROOT)
if r.returncode != 0:
    print("[shot_run] render FAILED", r.returncode); sys.exit(r.returncode)

vd = os.path.join(ROOT, "videos", "episodes", a.episode, "shots")
video = next(os.path.join(vd, f) for f in sorted(os.listdir(vd), key=lambda f: -os.path.getmtime(os.path.join(vd, f)))
             if f.startswith(f"shot{a.shot}_a{a.attempt}{a.tag}_") and f.endswith(".mp4"))
fr = [f.to_ndarray(format="rgb24") for f in av.open(video).decode(video=0)]
idx = [round(i * (len(fr) - 1) / 11) for i in range(12)]
sheet = Image.new("RGB", (4 * 400, 3 * 225))
for k, i in enumerate(idx):
    sheet.paste(Image.fromarray(fr[i]).resize((400, 225), Image.LANCZOS), ((k % 4) * 400, (k // 4) * 225))
sheet_path = video[:-4] + "_contact.png"; sheet.save(sheet_path)
c = av.open(video)
info = dict(shot=a.shot, attempt=a.attempt, seed=seed, video=os.path.relpath(video, ROOT), contact_sheet=os.path.relpath(sheet_path, ROOT),
            frames=len(fr), size=f"{fr[0].shape[1]}x{fr[0].shape[0]}", seconds=round(float(c.duration) / 1e6, 2), audio=len(c.streams.audio) > 0,
            minutes=round((time.time() - t0) / 60, 1), status="awaiting_user_review")
runs_p = os.path.join(ep_dir, "runs.json")
runs = json.load(open(runs_p)) if os.path.exists(runs_p) else []
runs.append(info); json.dump(runs, open(runs_p, "w"), indent=2)
print("[shot_run] DONE", json.dumps(info, indent=2))
