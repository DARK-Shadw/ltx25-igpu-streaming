"""Layer-streaming executor prototype.

Weights live on NVMe. A loader thread reads each layer with unbuffered I/O into
pinned host memory, copies it to the iGPU on a side stream, and hands it to the
compute loop. Compares: compute-only, sequential load+compute, overlapped.
"""
import ctypes
import os
import queue
import sys
import threading
import time
from ctypes import wintypes

import torch
import torch.nn.functional as F

D, FF, T = 2048, 8192, 1024  # hidden, ffn, tokens
N_LAYERS = int(os.environ.get("N_LAYERS", 40))
PATH = os.path.join(os.path.dirname(__file__), "layers.bin")
dev = torch.device("xpu")
ALIGN = 4096

# (name, shape) of each tensor in a layer
SHAPES = [("w1", (D, FF)), ("w2", (FF, D))] + [(f"a{i}", (D, D)) for i in range(4)]
OFFS, off = {}, 0
for name, shp in SHAPES:
    n = shp[0] * shp[1] * 2
    OFFS[name] = (off, shp)
    off += n
LAYER_BYTES = (off + ALIGN - 1) // ALIGN * ALIGN


def make_file():
    if os.path.exists(PATH) and os.path.getsize(PATH) == LAYER_BYTES * N_LAYERS:
        return
    print(f"writing {LAYER_BYTES * N_LAYERS / 2**30:.2f} GB test model...")
    with open(PATH, "wb") as f:
        for _ in range(N_LAYERS):
            buf = torch.empty(LAYER_BYTES // 2, dtype=torch.bfloat16).normal_(0, 0.02)
            f.write(buf.view(torch.int16).numpy().tobytes())


# --- Windows unbuffered reads (bypass page cache -> real NVMe speed) ---
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
GENERIC_READ, SHARE_READ, OPEN_EXISTING = 0x80000000, 1, 3
NO_BUFFERING = 0x20000000


def open_raw():
    h = k32.CreateFileW(PATH, GENERIC_READ, SHARE_READ, None, OPEN_EXISTING, NO_BUFFERING, None)
    if h in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_last_error())
    return h


def read_layer(h, idx, host_buf):
    k32.SetFilePointerEx(h, idx * LAYER_BYTES, None, 0)
    ptr, left, got = host_buf.data_ptr(), LAYER_BYTES, wintypes.DWORD()
    while left:
        n = min(left, 64 << 20)
        if not k32.ReadFile(h, ptr, n, ctypes.byref(got), None) or got.value == 0:
            raise OSError(ctypes.get_last_error())
        ptr += got.value
        left -= got.value


def weights(gbuf):
    return {n: gbuf[o:o + s[0] * s[1] * 2].view(torch.bfloat16).view(s) for n, (o, s) in OFFS.items()}


def compute(x, w):
    x = x + F.gelu(x @ w["w1"]) @ w["w2"]
    for i in range(4):
        x = x + 0.1 * (x @ w[f"a{i}"])
    return F.rms_norm(x, (D,))


def sync():
    torch.xpu.synchronize()


def run_compute_only(x):
    g = torch.empty(LAYER_BYTES, dtype=torch.uint8, device=dev)
    g.view(torch.bfloat16).normal_(0, 0.02)
    w = weights(g)
    compute(x, w)
    sync()
    t = time.perf_counter()
    for _ in range(N_LAYERS):
        x = compute(x, w)
    sync()
    return time.perf_counter() - t


def run_sequential(x):
    h = open_raw()
    host = torch.empty(LAYER_BYTES, dtype=torch.uint8).pin_memory()
    g = torch.empty(LAYER_BYTES, dtype=torch.uint8, device=dev)
    t = time.perf_counter()
    for i in range(N_LAYERS):
        read_layer(h, i, host)
        g.copy_(host)
        x = compute(x, weights(g))
        sync()
    return time.perf_counter() - t


def run_overlapped(x, slots=3):
    h = open_raw()
    hosts = [torch.empty(LAYER_BYTES, dtype=torch.uint8).pin_memory() for _ in range(slots)]
    gbufs = [torch.empty(LAYER_BYTES, dtype=torch.uint8, device=dev) for _ in range(slots)]
    free, ready = queue.Queue(), queue.Queue()
    for s in range(slots):
        free.put(s)
    copy_stream = torch.xpu.Stream()

    def loader():
        for i in range(N_LAYERS):
            s = free.get()
            read_layer(h, i, hosts[s])
            with torch.xpu.stream(copy_stream):
                gbufs[s].copy_(hosts[s], non_blocking=True)
            copy_stream.synchronize()
            ready.put(s)

    th = threading.Thread(target=loader, daemon=True)
    t = time.perf_counter()
    th.start()
    for i in range(N_LAYERS):
        s = ready.get()
        x = compute(x, weights(gbufs[s]))
        sync()
        free.put(s)
    th.join()
    return time.perf_counter() - t


if __name__ == "__main__":
    make_file()
    gb = LAYER_BYTES * N_LAYERS / 2**30
    print(f"model: {N_LAYERS} layers x {LAYER_BYTES / 2**20:.0f} MB = {gb:.2f} GB, tokens={T}")
    x0 = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
    for name, fn in [("compute-only (weights resident)", run_compute_only),
                     ("sequential load+compute", run_sequential),
                     ("overlapped streaming", run_overlapped)]:
        dt = fn(x0.clone())
        print(f"{name:34s}: {dt:6.2f} s/pass  ({dt / N_LAYERS * 1000:5.1f} ms/layer, disk-equiv {gb / dt:.2f} GB/s)")
