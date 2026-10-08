"""Join shots into one mp4: hard cuts (optionally dropping the duplicated first frame of chained shots) + audio crossfades."""
import json, sys, numpy as np, av

def read_audio(path):
    c = av.open(path); st = c.streams.audio[0]; sr = st.rate
    rs = av.AudioResampler(format="fltp", layout="stereo", rate=sr); out = []
    for fr in c.decode(st):
        for r in rs.resample(fr): out.append(r.to_ndarray())
    for r in rs.resample(None): out.append(r.to_ndarray())
    c.close(); return np.concatenate(out, axis=1).astype(np.float32), sr

def stitch(shots, out_path, fps=24, xfade_ms=300, crf=17):
    """shots: list of dicts {path, drop_first_frames, frame_fn (optional), gain (optional linear)}"""
    W = H = 0                                                           # output size = the largest clip (smaller clips are upscaled, never the reverse)
    for sh in shots:
        c0 = av.open(sh["path"]); v0 = c0.streams.video[0]
        if v0.width * v0.height > W * H: W, H = v0.width, v0.height
        c0.close()
    out = av.open(out_path, "w")
    vs = out.add_stream("libx264", rate=fps); vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"; vs.options = {"crf": str(crf), "preset": "medium"}
    SR = 48000
    asr = out.add_stream("aac", rate=SR); asr.layout = "stereo"        # both streams must exist before the first packet is muxed
    sr = None; tracks = []; n_out = 0; kept = []
    for sh in shots:
        a, sr = read_audio(sh["path"]); drop = sh.get("drop_first_frames", 0)
        a = a * float(sh.get("gain", 1.0))                                                           # optional per-shot gain (loudness match)
        c = av.open(sh["path"]); k = 0
        for fr in c.decode(video=0):
            if k >= drop and (not sh.get("trim_end_s") or (k - drop) < int(round(sh["trim_end_s"] * fps))):
                arr = fr.to_ndarray(format="rgb24") if (fr.width, fr.height) == (W, H) else fr.reformat(width=W, height=H, format="rgb24", interpolation="LANCZOS").to_ndarray()   # clips of other sizes are resized to the output size
                if sh.get("frame_fn"): arr = sh["frame_fn"](arr)                                       # optional per-shot picture transform (colour match / grade)
                nf = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for pkt in vs.encode(nf): out.mux(pkt)
                n_out += 1
            k += 1
        c.close(); kept.append(k - drop)
        if sh.get("trim_end_s"):                                                                # cut the clip's end (e.g. cut on the last word)
            keep_n = min(kept[-1], int(round(sh["trim_end_s"] * fps))); kept[-1] = keep_n
        a = a[:, int(sr * drop / fps):]                                 # keep audio aligned with the dropped frames
        tracks.append(a)
    for pkt in vs.encode(None): out.mux(pkt)
    # audio: every clip's audio is placed EXACTLY under its own video (padded/trimmed to its frame count), so sync never drifts.
    # (An overlapping crossfade would shorten the track by xf at every join and push the sound ahead of the picture.)
    # At each join the clips just get a short fade-out/fade-in (xfade_ms/2 each side) so cuts do not click.
    f = int(sr * xfade_ms / 2000); parts = []
    for i, (t, nf) in enumerate(zip(tracks, kept)):
        want = int(round(nf / fps * sr)); t = t[:, :want] if t.shape[1] >= want else np.pad(t, ((0, 0), (0, want - t.shape[1])))
        t = t.copy(); n = min(f, want // 2); w = np.linspace(0, np.pi / 2, n)
        if i > 0: t[:, :n] *= np.sin(w)
        if i < len(tracks) - 1: t[:, -n:] *= np.cos(w)
        parts.append(t)
    mix = np.concatenate(parts, axis=1)
    mix = np.clip(mix, -1, 1).astype(np.float32)
    assert sr == SR, f"unexpected audio rate {sr}"
    pts = 0
    for i in range(0, mix.shape[1], 1024):
        ch = np.ascontiguousarray(mix[:, i:i + 1024]); fr = av.AudioFrame.from_ndarray(ch, format="fltp", layout="stereo"); fr.sample_rate = sr; fr.pts = pts; pts += ch.shape[1]
        for pkt in asr.encode(fr): out.mux(pkt)
    for pkt in asr.encode(None): out.mux(pkt)
    out.close()
    return dict(frames=n_out, seconds=round(n_out / fps, 2), size=f"{W}x{H}")

if __name__ == "__main__":
    spec = json.load(open(sys.argv[1])); print(stitch(spec["shots"], spec["out"], xfade_ms=spec.get("xfade_ms", 300)))
