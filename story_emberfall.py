"""Emberfall (~59 s): the existing ronin opening (2 clips) + 9 new FLASH shots, all hard cuts (text-to-video, no frame chaining).

  python story_emberfall.py     # resumes: finished shots are skipped
"""
import glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
NAME, MODE = "emberfall", "flash"
OUT = os.path.join(ROOT, "videos", "anime", NAME)
OPENING = [os.path.join(ROOT, "videos", "anime", "ronin-entrance", "flash_1536x896_24fps_5.4s_seed77.mp4"),
           os.path.join(ROOT, "videos", "tests", "i2v-test", "preview_768x448_24fps_5.4s_seed9.mp4")]
SHOTS = [dict(id=i, seed=200 + i) for i in range(1, 10)]


def shot_video(s):
    g = glob.glob(os.path.join(OUT, "shots", f"shot{s['id']}_{MODE}_*_seed{s['seed']}.mp4"))
    return g[0] if g else None


t_story = time.perf_counter()
times = {}
for s in SHOTS:
    v = shot_video(s)
    if v and os.path.exists(v[:-4] + ".json"):
        print(f"[story] shot {s['id']} already rendered: {os.path.basename(v)}", flush=True)
    else:
        cmd = [PY, "render.py", "--mode", MODE, "--category", "anime", "--name", NAME, "--subdir", "shots", "--prefix", f"shot{s['id']}_",
               "--prompt-file", f"prompts/{NAME}/shot{s['id']}.txt", "--seed", str(s["seed"]), "--seconds", "5.4"]
        print(f"[story] rendering shot {s['id']}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[story] shot {s['id']} FAILED (exit {r.returncode}); rerun story_emberfall.py to resume"); sys.exit(1)
        v = shot_video(s)
    times[s["id"]] = json.load(open(v[:-4] + ".json"))

sys.path.insert(0, os.path.join(ROOT, "stream"))
import stitch
final = os.path.join(OUT, f"{NAME}_60s_FLASH_1536x896.mp4")
info = stitch.stitch([dict(path=p) for p in OPENING] + [dict(path=shot_video(s)) for s in SHOTS], final, xfade_ms=300)
gen = sum(m["generation_minutes"] for m in times.values()); dec = sum(m["decode_minutes"] for m in times.values())
report = dict(title="Emberfall", final=os.path.relpath(final, ROOT), final_seconds=info["seconds"], resolution=info["size"],
              opening="existing clips: ronin-entrance flash + i2v-test preview (768x448, upscaled)",
              per_shot={str(i): dict(total_min=m["total_minutes"], seed=m["seed"]) for i, m in times.items()},
              new_shots_render_minutes=round(gen + dec, 1), wall_clock_minutes_this_run=round((time.perf_counter() - t_story) / 60, 1))
json.dump(report, open(os.path.join(OUT, "story_report.json"), "w"), indent=2)
print(json.dumps(report, indent=2))
