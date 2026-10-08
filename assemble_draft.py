"""Assemble a draft cut of ye-chen-ep1 v2 from the beat sheet with my picks. Memory-lean (runs next to a render).

  python assemble_draft.py <draft_name>      # picks/trims/delays live in PLAN below
"""
import glob, json, os, sys
import av, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream")); import stitch
ROOT = os.path.dirname(os.path.abspath(__file__)); V1 = "videos/episodes/ye-chen-ep1/shots/"; V2 = "videos/episodes/ye-chen-ep1-v2/shots/"
def old(shot, att): return sorted(glob.glob(f"{V1}shot{shot}_a{att}_custom_*.mp4"))[0]
def new(shot, att):
    g = sorted(glob.glob(f"{V2}shot{shot}_a{att}_custom_*.mp4")); return g[0] if g else None
# (beat id, path, trim_end_s, audio_delay_s, note)
PLAN = [
 ("01", sorted(glob.glob(V1 + "shot01_a0s2_custom_*.mp4"))[0], None, 0, "hall, 2-step"),
 ("02", old("02", 1), None, 0, "whispering gallery"),
 ("03", old("03", 0), None, 0, "the Saintess"),
 ("04", old("04", 0), None, 0, "teacup plant"),
 ("05", new("05", 0), None, 0, "PLANT 2: disciple glances back"),
 ("06", new("06", 0), None, 0, "PLANT 3: the throne, empty circle"),
 ("07", old("05A", 0), None, 0, "Ye Chen from behind"),
 ("08", new("08", 0), 2.2, 0, "crowd turns (cut before they show their backs)"),
 ("09", new("09", 1), None, 0.8, "Ye Chen line 1 (take 2, sound delayed 0.8 s)"),
 ("10", old("05B", 1), 4.7, 0, "Ye Chen line 2 (your pick, cut at 4.7 s)"),
 ("11", old("10", 0), None, 0, "the Saintess reacts (your pick)"),
 ("12", new("12", 0), None, 0, "Ye Chen line 3 (take 1)"),
 ("13", old("06", 0), 1.7, 0, "the Elder glares (first 1.7 s, SOUND MUTED)"),
 ("14", new("14", 1), None, 0, "the Elder roars (take 2)"),
 ("15", new("15", 1), 3.6, 0, "gossip (take 2)"),
 ("16", new("16", 0), None, 0, "THE KEY LINE: 'young master' (take 1)"),
 ("17", new("17", 1), None, 0, "faces freeze (take 2)"),
 ("18", new("18", 0), 3.0, 0, "throne again (first 3 s)"),
 ("19", new("19", 0), None, 0, "the hush"),
 ("20", new("20", 1), None, 0, "the Archbishop opens his eyes (take 2)"),
 ("21", old("08", 0), None, 0.6, "the Archbishop's aura + line (sound delayed 0.6 s)"),
 ("22", old("09", 0), None, 0, "Ye Chen under pressure"),
 ("23", new("23", 0), None, 0, "'T-To the devil?' (take 1)"),
 ("24", old("07", 0), 1.8, 0, "sneering whisper (first 1.8 s)"),
 ("25", new("25", 0), None, 0, "THE CLICK (take 1)"),
 ("26", new("26", 1), None, 0, "Ye Chen senses the silence (take 2)"),
 ("27", new("27", 0), None, 0, "the crowd turns and bows"),
 ("28", new("28", 0), None, 0, "REVEAL: Gu Changge (take 1)"),
 ("29", old("12", 0), 2.7, 0, "Gu sips his tea (first 2.7 s, SOUND MUTED)"),
 ("30", new("30", 1), None, 1.2, "Gu looks at us + narration (take 2, sound delayed 1.2 s)"),
]
PLAN = [p for p in PLAN if p[1]]
name = sys.argv[1] if len(sys.argv) > 1 else "draft1"
# title/black card at the end (2 s) generated once
black = os.path.join(ROOT, "videos", "episodes", "ye-chen-ep1-v2", "assembly", "_black2s.mp4")
os.makedirs(os.path.dirname(black), exist_ok=True)
if not os.path.exists(black):
    o = av.open(black, "w"); vs = o.add_stream("libx264", rate=24); vs.width, vs.height, vs.pix_fmt = 1600, 896, "yuv420p"
    asr = o.add_stream("aac", rate=48000); asr.layout = "stereo"
    for _ in range(48):
        for pk in vs.encode(av.VideoFrame.from_ndarray(np.zeros((896, 1600, 3), np.uint8), format="rgb24")): o.mux(pk)
    for pk in vs.encode(None): o.mux(pk)
    pts = 0
    for i in range(0, 96000, 1024):
        n = min(1024, 96000 - i); fr = av.AudioFrame.from_ndarray(np.zeros((2, n), np.float32), format="fltp", layout="stereo"); fr.sample_rate = 48000; fr.pts = pts; pts += n
        for pk in asr.encode(fr): o.mux(pk)
    for pk in asr.encode(None): o.mux(pk)
    o.close()
PLAN.append(("31", black, None, 0, "fade to black"))

def mean_rgb(p):                                  # streaming, uint8 -> tiny memory
    acc, n = np.zeros(3), 0
    for k, f in enumerate(av.open(p).decode(video=0)):
        if k % 12 == 0: acc += f.to_ndarray(format="rgb24")[::4, ::4].reshape(-1, 3).mean(0); n += 1
    return acc / max(n, 1)
def active_rms(p):
    c = av.open(p); rs = av.AudioResampler(format="fltp", layout="mono", rate=16000); o = []
    for f in c.decode(c.streams.audio[0]):
        for r in rs.resample(f): o.append(r.to_ndarray().reshape(-1))
    x = np.concatenate(o); w = x[:len(x) // 800 * 800].reshape(-1, 800); r = np.sqrt((w ** 2).mean(1)); act = r[r > 0.3 * r.max()]
    return float(np.sqrt((act ** 2).mean())) if len(act) else 0.0
real = [p for p in PLAN if p[0] != "31"]
mus = {p[0]: mean_rgb(p[1]) for p in real}; tm = np.mean(list(mus.values()), 0); ALPHA = 0.6
def mk(mu): return lambda x: np.clip(x.astype("float32") - mu + (mu + (tm - mu) * ALPHA), 0, 255).astype("uint8")
gains = {p[0]: float(np.clip(20 * np.log10(0.05 / max(active_rms(p[1]), 1e-4)), -12, 14)) for p in real}
MUTE = {"13", "29"}      # reused clips whose original line must not play twice (old shot 6 says "Blasphemous!", old shot 12 starts the narration)
for b in MUTE:
    if b in gains: gains[b] = -120.0
shots = []; t = 0.0; log = []
for bid, path, trim, delay, note in PLAN:
    c = av.open(path); dur = float(c.streams.video[0].frames) / 24 if c.streams.video[0].frames else float(c.duration) / 1e6; c.close()
    length = min(dur, trim) if trim else dur
    log.append((bid, round(t, 2), round(length, 2), note, round(gains.get(bid, 0), 1))); t += length
    d = dict(path=path, trim_end_s=trim, audio_delay_s=delay)
    if bid != "31": d.update(frame_fn=mk(mus[bid]), gain=10 ** (gains[bid] / 20))
    shots.append(d)
out = os.path.join(ROOT, "videos", "episodes", "ye-chen-ep1-v2", "assembly", f"{name}.mp4")
print(stitch.stitch(shots, out), out)
with open(out[:-4] + "_cutlist.txt", "w", encoding="utf-8") as f:
    for b, st, ln, note, g in log: f.write(f"{b:>3}  start {st:6.2f}s  len {ln:5.2f}s  gain {g:+5.1f} dB  {note}\n")
print(open(out[:-4] + "_cutlist.txt", encoding="utf-8").read())
