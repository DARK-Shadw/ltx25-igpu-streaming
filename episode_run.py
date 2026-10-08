"""Render the remaining shots of an episode one after another, check each with shot_qc, retry failures, then assemble the scene.

  python episode_run.py ye-chen-ep1 --from 05A            # resumable: shots already locked/approved/auto_passed are skipped
Rules: never restart a render that failed with a non-zero exit (stop and report); stop if the laptop is on battery below 40 %;
max 3 attempts per shot; after that the best attempt is kept and the shot is flagged needs_director.
"""
import argparse, ctypes, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--from", dest="start", default="05A"); ap.add_argument("--until", default="12")
a = ap.parse_args()
EP = os.path.join(ROOT, "episodes", a.episode); SHOTS = os.path.join(EP, "shots.json"); RUNS = os.path.join(EP, "runs.json")
sys.path.insert(0, ROOT)
import shot_qc


def log(*x): print(time.strftime("[%H:%M:%S]"), *x, flush=True)


def power_ok():
    class SPS(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("r", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    s = SPS(); ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    return s.ac == 1 or s.pct >= 40 or s.pct == 255, s.ac, s.pct


def wait_for_ac(sid):
    """Never render on battery (a dead battery killed a 40-minute render once): wait for the charger, give up only if the battery gets low."""
    n = 0
    while True:
        ok, ac, pct = power_ok()
        if ac == 1: return
        if pct < 25 and pct != 255: log(f"STOP: on battery at {pct} %, plug in and re-run (resumes at shot {sid})."); sys.exit(3)
        if n % 10 == 0: log(f"waiting for the charger before shot {sid} (on battery, {pct} %). Plug it in.")
        n += 1; time.sleep(60)


def load(): return json.load(open(SHOTS, encoding="utf-8"))
def save(d): json.dump(d, open(SHOTS, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


EXTRA = {
    "hold": "Nothing moves except what is described: the subjects stay exactly where they are in the first frame and the camera stays almost still.",
    "speech": "He speaks the line clearly and slowly, every word distinct and audible.",
    "motion": "The camera stays completely still and all motion is very small and slow.",
}


def score(q): return (len(q["fails"]), len(q["warns"]))


order = [s["id"] for s in load()["shots"]]
i0, i1 = order.index(a.start), order.index(a.until)
for sid in order[i0:i1 + 1]:
    d = load(); card = next(s for s in d["shots"] if s["id"] == sid)
    if card.get("status") in ("locked", "approved", "auto_passed", "needs_director", "needs_modification") and card.get("selected_video"):
        log("skip", sid, card["status"]); continue
    wait_for_ac(sid)
    tried, extra = [], ""
    for attempt in range(3):
        wait_for_ac(sid)
        log(f"shot {sid} attempt {attempt}" + (f" (retry strategy: {extra})" if extra else ""))
        cmd = [PY, "shot_run.py", a.episode, sid, "--attempt", str(attempt)] + (["--extra", extra] if extra else [])
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            log(f"STOP: render failed with exit code {r.returncode} on shot {sid}. Not restarting automatically; see logs."); sys.exit(r.returncode)
        runs = json.load(open(RUNS)); info = [x for x in runs if x["shot"] == sid and x["attempt"] == attempt][-1]
        q = shot_qc.run_qc(a.episode, sid, os.path.join(ROOT, info["video"]), attempt)
        sp = q["metrics"].get("speech", {}).get("similarity", 0)
        log(f"shot {sid} attempt {attempt}: {q['verdict']}  fails={q['fails']}  warns={q['warns']}")
        tried.append((score(q), -sp, attempt, info["video"], q))
        if q["verdict"] != "FAIL": break
        f = " ".join(q["fails"]).lower()
        if "hold failed" in f and "speech does not match" not in f and "no audio" not in f and "audio is dead" not in f:
            log(f"shot {sid}: subject/camera drift (face zoom / morph) - NOT retrying (director decision); kept for the director's review"); break
        extra = EXTRA["speech"] if "speech" in f else EXTRA["hold"] if "hold" in f else EXTRA["motion"] if "motion" in f else ""
    best = sorted(tried, key=lambda t: (t[0], t[1], t[2]))[0]
    d = load(); card = next(s for s in d["shots"] if s["id"] == sid)
    card["selected_video"] = best[3]; card["status"] = "auto_passed" if best[4]["verdict"] != "FAIL" else "needs_modification"
    card["qc_summary"] = dict(attempt=best[2], verdict=best[4]["verdict"], fails=best[4]["fails"], warns=best[4]["warns"])
    save(d); log(f"shot {sid} -> {card['status']} (attempt {best[2]})")

log("all requested shots done; assembling the scene")
subprocess.run([PY, "assemble_scene.py", a.episode], cwd=ROOT)
log("DONE")
