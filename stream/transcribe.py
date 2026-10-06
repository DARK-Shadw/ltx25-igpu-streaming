"""Transcribe the audio of an mp4 (or wav) with faster-whisper to check whether generated speech is intelligible."""
import sys, numpy as np, av
from faster_whisper import WhisperModel
def load_mono16k(path):
    c = av.open(path); st = c.streams.audio[0]
    rs = av.AudioResampler(format="flt", layout="mono", rate=16000); out = []
    for fr in c.decode(st):
        for r in rs.resample(fr): out.append(r.to_ndarray().reshape(-1))
    for r in rs.resample(None): out.append(r.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32)
model = WhisperModel(sys.argv[2] if len(sys.argv) > 2 else "base.en", device="cpu", compute_type="int8")
x = load_mono16k(sys.argv[1]); print(f"audio {len(x)/16000:.2f}s rms {np.sqrt((x**2).mean()):.4f}")
segs, info = model.transcribe(x, language="en", vad_filter=False, beam_size=5, condition_on_previous_text=False)
n = 0
for s in segs:
    n += 1; print(f"  [{s.start:5.2f}-{s.end:5.2f}] (no-speech p={s.no_speech_prob:.2f}) {s.text.strip()}")
if n == 0: print("  (no speech detected)")
