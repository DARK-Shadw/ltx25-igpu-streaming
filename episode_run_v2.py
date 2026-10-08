"""Render several TAKES per shot (free compute), run the QC on each, never retry-by-rule: the director/assistant picks the best take by eye.

  python episode_run_v2.py ye-chen-ep1-v2 30 28 09 12 14 06 18 26 25 19     # shot order = priority
Skips shots whose status is not 'ready'; resumable (takes already rendered are skipped); waits for the charger; stops on a render error.
"""
import ctypes, glob, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__)); PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
sys.path.insert(0, ROOT); import shot_qc
ep = sys.argv[1]; order = sys.argv[2:]; EP = os.path.join(ROOT, "episodes", ep); SH = os.path.join(EP, "shots.json"); RUNS = os.path.join(EP, "runs.json")
def log(*x): print(time.strftime("[%H:%M:%S]"), *x, flush=True)
def ac():
    class S(ctypes.Structure): _fields_ = [("ac", ctypes.c_ubyte), ("f", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("r", ctypes.c_ubyte), ("l", ctypes.c_ulong), ("fu", ctypes.c_ulong)]
    s = S(); ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)); return s.ac == 1, s.pct
def wait_ac(tag):
    n = 0
    while True:
        ok, pct = ac()
        if ok: return
        if pct < 25 and pct != 255: log(f"STOP: battery {pct}% while waiting for the charger ({tag})"); sys.exit(3)
        if n % 10 == 0: log(f"waiting for the charger before {tag} (battery {pct}%)")
        n += 1; time.sleep(60)
def load(): return json.load(open(SH, encoding="utf-8"))
def save(d): json.dump(d, open(SH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
for sid in order:
    card = next(s for s in load()["shots"] if s["id"] == sid)
    if card.get("status") not in ("ready", "takes_ready", "rendering"):
        log(f"skip {sid}: status {card.get('status')}"); continue
    takes = card.get("takes", 1); results = card.get("takes_results", [])
    for t in range(takes):
        if any(r["attempt"] == t for r in results): continue
        wait_ac(f"shot {sid} take {t}"); log(f"shot {sid} take {t + 1}/{takes}")
        d = load(); next(s for s in d["shots"] if s["id"] == sid)["status"] = "rendering"; save(d)
        r = subprocess.run([PY, "shot_run.py", ep, sid, "--attempt", str(t)], cwd=ROOT)
        if r.returncode != 0: log(f"STOP: render failed (exit {r.returncode}) on shot {sid} take {t}; not restarting automatically"); sys.exit(r.returncode)
        info = [x for x in json.load(open(RUNS)) if x["shot"] == sid and x["attempt"] == t][-1]
        q = shot_qc.run_qc(ep, sid, os.path.join(ROOT, info["video"]), t)
        m = q["metrics"]; sp = m.get("speech", {})
        res = dict(attempt=t, video=info["video"], verdict=q["verdict"], fails=q["fails"], warns=q["warns"], zoom_max=m.get("zoom_max"), speech_similarity=sp.get("similarity"), speech_onset=sp.get("onset_s"), minutes=info["minutes"])
        results.append(res); d = load(); c = next(s for s in d["shots"] if s["id"] == sid); c["takes_results"] = results; c["status"] = "takes_ready" if len(results) >= takes else "rendering"; save(d)
        log(f"shot {sid} take {t}: {q['verdict']} zoom {res['zoom_max']} speech {res['speech_similarity']} onset {res['speech_onset']} fails={q['fails']} warns={q['warns']}")
log("all requested shots rendered; review the takes")
