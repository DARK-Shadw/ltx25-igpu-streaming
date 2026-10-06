"""Extract audio from every video, compute simple diagnostics, save wav + spectrogram."""
import glob, os, sys, wave, numpy as np, av
from PIL import Image
rows = []
def spec_img(x, sr, path):
    n, hop = 2048, 512
    w = np.hanning(n); fr = [x[i:i + n] * w for i in range(0, len(x) - n, hop)]
    S = np.abs(np.fft.rfft(np.stack(fr), axis=1)).T
    S = 20 * np.log10(S + 1e-6); S = np.clip((S - (S.max() - 90)) / 90, 0, 1)[::-1]
    # log-ish frequency squeeze: keep up to ~12 kHz
    keep = int(S.shape[0] * min(1.0, 12000 / (sr / 2))); S = S[-keep:]
    img = Image.fromarray((S * 255).astype(np.uint8)).resize((900, 300)); img.save(path)
for p in sorted(glob.glob("videos/**/*.mp4", recursive=True)):
    try:
        c = av.open(p); st = [s for s in c.streams if s.type == "audio"]
        if not st: rows.append((p, "NO AUDIO STREAM")); continue
        sr = st[0].rate; ch = []
        for fr in c.decode(audio=0): ch.append(fr.to_ndarray())
        a = np.concatenate(ch, axis=1).astype(np.float32)      # (channels, samples) planar or (1, interleaved)
        if a.shape[0] == 1 and st[0].layout.nb_channels == 2: a = a.reshape(-1, 2).T
        x = a.mean(0)
        if np.abs(x).max() > 1.5: x = x / 32768.0
        name = os.path.relpath(p, "videos").replace(os.sep, "__")[:-4]
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        with wave.open(f"audio/extracted/{name}.wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
        spec_img(x, sr, f"audio/spectrograms/{name}.png")
        rms = float(np.sqrt((x ** 2).mean())); peak = float(np.abs(x).max())
        n = 2048; F = np.abs(np.fft.rfft(x[: len(x) // n * n].reshape(-1, n) * np.hanning(n), axis=1)); fr_ = np.fft.rfftfreq(n, 1 / sr)
        P = (F ** 2).sum(0) + 1e-12; tot = P.sum()
        cent = float((fr_ * P).sum() / tot); flat = float(np.exp(np.mean(np.log(P))) / np.mean(P))
        bass = float(P[fr_ < 150].sum() / tot); hi = float(P[fr_ > 8000].sum() / tot)
        fl = x[: len(x) // 1024 * 1024].reshape(-1, 1024); silent = float((np.sqrt((fl ** 2).mean(1)) < 10 ** (-60 / 20)).mean())
        rows.append((p, f"sr={sr} dur={len(x)/sr:.2f}s rms={20*np.log10(rms+1e-9):6.1f}dBFS peak={20*np.log10(peak+1e-9):6.1f}dBFS silent={silent*100:4.0f}% centroid={cent:6.0f}Hz flat={flat:.3f} bass<150Hz={bass*100:5.1f}% >8k={hi*100:5.1f}%"))
    except Exception as e:
        rows.append((p, f"ERROR {type(e).__name__}: {e}"))
for p, r in rows: print(f"{os.path.relpath(p, 'videos'):95s} {r}")
