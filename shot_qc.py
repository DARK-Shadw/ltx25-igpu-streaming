"""Automated per-shot checks (built from what actually went wrong on Ye Chen ep1).

  python shot_qc.py ye-chen-ep1 04 videos/episodes/ye-chen-ep1/shots/shot04_a0_....mp4

Checks and why each exists:
  start_frame   frame 0 must match the start image (3-step refinement re-drew the set: 14 dB; good: 20+ dB)
  motion        median frame change inside the card's band (static / smear detection)
  hold          "nobody leaves": peak short-window change of the people layout (shot 2: 24.6 when they stood up and left, 9.6 when they held)
  sharpness     no collapse to mush (min/median edge energy)
  audio         finite, audible (shot 2 had rms 0.006 = inaudible), not only a low rumble
  speech        ASR transcript vs the exact line, onset time, not cut off at the end
Verdict FAIL = re-render is worth it; WARN = report to the director (audio level WARNs are fixed in the sound pass, not by re-rendering).
"""
import argparse, difflib, json, os, re, sys
import av, numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))


def decode(video):
    gray, tiny, rgb_small, first = [], [], [], None
    for k, f in enumerate(av.open(video).decode(video=0)):
        if k == 0:
            first = f.to_ndarray(format="rgb24")
        im = f.to_ndarray(format="rgb24")
        g = im.astype("float32").mean(2)
        h, w = g.shape; s = max(1, h // 224)
        g4 = g[:h // s * s, :w // s * s].reshape(h // s, s, w // s, s).mean((1, 3))      # ~224x400 gray
        gray.append(g4)
        gh, gw = g4.shape; tiny.append(g4[:gh // 8 * 8, :gw // 8 * 8].reshape(gh // 8, 8, gw // 8, 8).mean((1, 3)))   # ~28x50 layout
        rgb_small.append(np.array(Image.fromarray(im).resize((400, 225), Image.LANCZOS)))
    return first, gray, tiny, rgb_small


def audio_mono16k(video):
    c = av.open(video)
    if not c.streams.audio:
        return None
    rs = av.AudioResampler(format="fltp", layout="mono", rate=16000); out = []
    for fr in c.decode(c.streams.audio[0]):
        for r in rs.resample(fr): out.append(r.to_ndarray().reshape(-1))
    return np.concatenate(out) if out else np.zeros(1, "float32")


def norm_words(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower().replace("-", " ")).split()


def run_qc(episode, shot, video, attempt=0, do_asr=True):
    ep = os.path.join(ROOT, "episodes", episode)
    plan = json.load(open(os.path.join(ep, "shots.json"), encoding="utf-8"))
    card = next(s for s in plan["shots"] if s["id"] == shot)
    q = card.get("qc", {}) or {}
    res, fails, warns = {}, [], []
    first, gray, tiny, rgb_small = decode(video)
    n_frames = len(gray); uniq = gray[::2]
    # --- start frame fidelity
    src = Image.open(os.path.join(ep, "frames", card["frame"])).convert("RGB"); H, W = first.shape[:2]
    s = max(W / src.width, H / src.height); src = src.resize((max(W, round(src.width * s)), max(H, round(src.height * s))), Image.LANCZOS)
    x0, y0 = (src.width - W) // 2, (src.height - H) // 2; src = np.array(src.crop((x0, y0, x0 + W, y0 + H)))
    psnr = float(10 * np.log10(255 ** 2 / ((src.astype(float) - first.astype(float)) ** 2).mean()))
    res["start_frame_psnr_db"] = round(psnr, 1)
    if psnr < 15: fails.append(f"frame 0 does not match the start image ({psnr:.1f} dB)")
    elif psnr < 18: warns.append(f"frame 0 only loosely matches the start image ({psnr:.1f} dB)")
    # --- motion
    ch = np.array([np.abs(uniq[i + 1] - uniq[i]).mean() for i in range(len(uniq) - 1)])
    med, p90, mx = float(np.median(ch)), float(np.percentile(ch, 90)), float(ch.max())
    res["motion"] = dict(median=round(med, 1), p90=round(p90, 1), max=round(mx, 1))
    lo, hi = q.get("motion_median", [1, 14])
    if med > hi * 1.5: fails.append(f"too much motion: median {med:.1f} (band {lo}-{hi})")
    elif med > hi or med < lo * 0.5: warns.append(f"motion median {med:.1f} outside the band {lo}-{hi}")
    # --- hold check (people must not leave / stand up): peak layout change over 4 unique frames
    tu = tiny[::2]; jump = float(max(np.abs(tu[i + 4] - tu[i]).mean() for i in range(len(tu) - 4))) if len(tu) > 4 else 0.0
    res["layout_jump"] = round(jump, 1)
    if q.get("must_hold"):
        hmax = q.get("hold_jump_max", 12)        # crowd shots: 12 (people leaving = 23-25). Single-subject inserts with a moving hand/object may allow more.
        if jump > hmax + 4: fails.append(f"hold failed: layout changed fast (jump {jump:.1f}; limit {hmax}) - '{q['must_hold']}'")
        elif jump > hmax: warns.append(f"hold borderline: layout jump {jump:.1f} (limit {hmax}) - '{q['must_hold']}'")
    # --- sharpness collapse
    sh = np.array([np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean() for g in uniq])
    res["sharpness_min_over_median"] = round(float(sh.min() / np.median(sh)), 2)
    if sh.min() / np.median(sh) < 0.5: warns.append("a stretch of frames collapses in sharpness (smear?)")
    # --- audio
    a = audio_mono16k(video)
    if a is None: fails.append("no audio stream")
    else:
        fin = bool(np.isfinite(a).all()); rms = float(np.sqrt((a ** 2).mean())); pk = float(np.abs(a).max())
        sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2; fq = np.fft.rfftfreq(len(a), 1 / 16000); cent = float((fq * sp).sum() / (sp.sum() + 1e-12))
        w8 = a[:len(a) // 800 * 800].reshape(-1, 800); active = float((np.sqrt((w8 ** 2).mean(1)) > 0.003).mean())
        res["audio"] = dict(rms=round(rms, 4), peak=round(pk, 3), centroid_hz=round(cent), active_fraction=round(active, 2), finite=fin)
        if not fin or rms < 0.002: fails.append("audio is dead or invalid")
        elif rms < 0.02: warns.append(f"audio is very quiet (rms {rms:.3f}); fix in the sound pass")
        if pk > 0.99: warns.append("audio clips")
        if cent < 200 and rms >= 0.002: warns.append(f"audio is only a low rumble (centroid {cent:.0f} Hz)")
        # --- speech
        exp = q.get("expect_speech")
        if exp and do_asr:
            from faster_whisper import WhisperModel
            model = WhisperModel("small.en", device="cpu", compute_type="int8")
            segs, _ = model.transcribe(a.astype("float32"), word_timestamps=True, vad_filter=False, language="en")
            words = [w for sg in segs for w in sg.words]
            heard = " ".join(w.word for w in words).strip(); sim = difflib.SequenceMatcher(None, norm_words(exp), norm_words(heard)).ratio()
            onset = words[0].start if words else None; end = words[-1].end if words else None; dur = len(a) / 16000
            res["speech"] = dict(expected=exp, heard=heard, similarity=round(sim, 2), onset_s=None if onset is None else round(onset, 2), end_s=None if end is None else round(end, 2), clip_s=round(dur, 2))
            if sim < 0.6: fails.append(f"speech does not match the line (similarity {sim:.2f}): heard '{heard}'")
            elif sim < 0.8: warns.append(f"speech partly differs from the line (similarity {sim:.2f}): heard '{heard}'")
            if words and (onset > 2.5): warns.append(f"speech starts late ({onset:.1f} s)")
            if words and end >= dur - 0.1: warns.append("speech runs to the very end of the clip (may be cut off)")
    verdict = "FAIL" if fails else ("WARN" if warns else "PASS")
    out = dict(episode=episode, shot=shot, attempt=attempt, video=os.path.relpath(video, ROOT), frames=n_frames, verdict=verdict, fails=fails, warns=warns, metrics=res)
    os.makedirs(os.path.join(ep, "qc"), exist_ok=True)
    json.dump(out, open(os.path.join(ep, "qc", f"shot{shot}_a{attempt}.json"), "w", encoding="utf-8"), indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("shot"); ap.add_argument("video"); ap.add_argument("--attempt", type=int, default=0); ap.add_argument("--no-asr", action="store_true")
    a = ap.parse_args(); r = run_qc(a.episode, a.shot, a.video, a.attempt, not a.no_asr)
    print(json.dumps(r["metrics"], indent=1)); print("VERDICT:", r["verdict"])
    for f in r["fails"]: print("  FAIL:", f)
    for w in r["warns"]: print("  WARN:", w)
    sys.exit(1 if r["verdict"] == "FAIL" else 0)
