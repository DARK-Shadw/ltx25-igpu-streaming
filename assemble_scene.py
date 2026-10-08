"""Assemble every selected shot of an episode into one scene: colour-matched (60 % toward the scene average) and loudness-matched.
No music/ambience bed yet (open decision in edit_plan.md). Output: videos/episodes/<ep>/assembly/scene_matched_<n>shots.mp4

  python assemble_scene.py ye-chen-ep1
"""
import json, os, sys
import av, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "stream")); import stitch

ep = sys.argv[1]; plan = json.load(open(os.path.join(ROOT, "episodes", ep, "shots.json"), encoding="utf-8"))
items = [(s["id"], os.path.join(ROOT, s["selected_video"]), s.get("trim_end_s")) for s in plan["shots"] if s.get("selected_video") and os.path.exists(os.path.join(ROOT, s["selected_video"]))]
print("assembling", [i[0] for i in items])


def sample_stats(p):
    fr = np.stack([f.to_ndarray(format="rgb24").astype("float32") for k, f in enumerate(av.open(p).decode(video=0)) if k % 12 == 0]); return fr.mean((0, 1, 2)), fr.std((0, 1, 2))


def active_rms(p):
    c = av.open(p); rs = av.AudioResampler(format="fltp", layout="mono", rate=16000); o = []
    for f in c.decode(c.streams.audio[0]):
        for r in rs.resample(f): o.append(r.to_ndarray().reshape(-1))
    x = np.concatenate(o); w = x[:len(x) // 800 * 800].reshape(-1, 800); r = np.sqrt((w ** 2).mean(1)); act = r[r > 0.3 * r.max()]
    return float(np.sqrt((act ** 2).mean())) if len(act) else 0.01


st = [sample_stats(p) for _, p, _t in items]; tm = np.mean([s[0] for s in st], 0); ALPHA = 0.6
def mk(mu):
    return lambda x: np.clip((x.astype("float32") - mu) * 1.0 + (mu + (tm - mu) * ALPHA), 0, 255).astype("uint8")
gains = [float(np.clip(20 * np.log10(0.05 / active_rms(p)), -12, 14)) for _, p, _t in items]
print("loudness gains (dB):", [round(g, 1) for g in gains])
shots = [dict(path=p, frame_fn=mk(s[0]), gain=10 ** (g / 20), trim_end_s=t) for (_, p, t), s, g in zip(items, st, gains)]
os.makedirs(os.path.join(ROOT, "videos", "episodes", ep, "assembly"), exist_ok=True)
out = os.path.join(ROOT, "videos", "episodes", ep, "assembly", f"scene_matched_{len(items)}shots.mp4")
print(stitch.stitch(shots, out), out)
