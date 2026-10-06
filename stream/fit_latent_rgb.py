"""Fit a linear map latent(128 ch) -> mean RGB of the 32x32 pixel patch it covers, from clips we already rendered.
Used by the live denoising preview (stream/preview.py). Output: stream/latent_rgb.npy  (129 x 3: 128 weights + bias).
  python stream/fit_latent_rgb.py
"""
import glob, os, sys
import av, numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = []   # (stage-1 latent file, video file)
def add(lat, vid):
    if os.path.exists(lat) and vid and os.path.exists(vid): PAIRS.append((lat, vid))
for i in range(1, 10):
    add(f"{ROOT}/latents/emberfall_shot{i}_flash_seed{200 + i}.pt.stage1", next(iter(glob.glob(f"{ROOT}/videos/anime/emberfall/shots/shot{i}_flash_*_seed{200 + i}.mp4")), None))
for i, sd in zip(range(1, 12), (301, 302, 303, 304, 305, 306, 1307, 308, 309, 310, 311)):
    add(f"{ROOT}/latents/snowbridge_shot{i}_flash_seed{sd}.pt.stage1", next(iter(glob.glob(f"{ROOT}/videos/anime/snowbridge/shots/shot{i}_flash_*_seed{sd}.mp4")), None))
for i in range(1, 5):
    add(f"{ROOT}/latents/lantern-keeper_shot{i}_flash_seed{100 + i}.pt.stage1", next(iter(glob.glob(f"{ROOT}/videos/anime/lantern-keeper/shots/shot{i}_flash_*_seed{100 + i}.mp4")), None))
print(len(PAIRS), "clip pairs")


def samples(lat_path, vid_path):
    z = torch.load(lat_path)["video"][0].float().numpy()            # (128, F, h, w)  denormalized
    C, F, h, w = z.shape
    want = {2 * (0 if k == 0 else 8 * k - 4): k for k in range(F)}    # video frame (frames are doubled) for latent frame k
    got = {}
    for n, fr in enumerate(av.open(vid_path).decode(video=0)):
        if n in want: got[want[n]] = fr.to_ndarray(format="rgb24")
    X, Y = [], []
    for k, img in got.items():
        H, W = img.shape[:2]; ph, pw = H // h, W // w
        rgb = img[:ph * h, :pw * w].reshape(h, ph, w, pw, 3).mean((1, 3)) / 255.0   # (h, w, 3)
        X.append(z[:, k].transpose(1, 2, 0).reshape(-1, C)); Y.append(rgb.reshape(-1, 3))
    return np.concatenate(X), np.concatenate(Y)


data = [samples(*p) for p in PAIRS]
hold = max(2, len(data) // 6)
Xtr = np.concatenate([d[0] for d in data[:-hold]]); Ytr = np.concatenate([d[1] for d in data[:-hold]])
Xte = np.concatenate([d[0] for d in data[-hold:]]); Yte = np.concatenate([d[1] for d in data[-hold:]])
A = lambda X: np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1)
lam = 1e-1
W = np.linalg.solve(A(Xtr).T @ A(Xtr) + lam * np.eye(129), A(Xtr).T @ Ytr)
pred = np.clip(A(Xte) @ W, 0, 1)
r2 = 1 - ((pred - Yte) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
print(f"train samples {len(Xtr)}, held-out clips {hold}: held-out MAE {np.abs(pred - Yte).mean():.3f} (0-1 scale), R^2 {r2:.3f}")
Wall = np.linalg.solve(A(np.concatenate([d[0] for d in data])).T @ A(np.concatenate([d[0] for d in data])) + lam * np.eye(129),
                       A(np.concatenate([d[0] for d in data])).T @ np.concatenate([d[1] for d in data]))
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_rgb.npy"), Wall.astype(np.float32))
print("saved stream/latent_rgb.npy")
