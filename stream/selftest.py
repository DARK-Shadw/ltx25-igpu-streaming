"""Setup self-tests (each prints PASS/FAIL lines; exit code != 0 on failure).

  python stream/selftest.py device     # accelerator visible, matmul + SDPA + RAM->GPU copy work, speed numbers
  python stream/selftest.py loader     # streams 2 real transformer blocks from disk and compares them bit-for-bit with safetensors
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from dev import DEV, NAME, acc, MODELS

what = sys.argv[1] if len(sys.argv) > 1 else "device"
bad = 0


def check(name, ok, detail=""):
    global bad
    bad += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def timeit(fn, n=10):
    fn(); acc.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    acc.synchronize(); return (time.perf_counter() - t) / n


if what == "device":
    print(f"device: {NAME}  torch {torch.__version__}")
    check("accelerator available", (torch.cuda.is_available() if NAME == "cuda" else torch.xpu.is_available()))
    a = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16); b = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    c = a @ b; acc.synchronize()
    ref = (a.float().cpu() @ b.float().cpu())
    err = float((c.float().cpu() - ref).abs().max() / ref.abs().max())
    check("bf16 matmul correct", err < 0.02, f"(max rel err {err:.4f})")
    t = timeit(lambda: a @ b); check("bf16 matmul speed", True, f"{2 * 4096 ** 3 / t / 1e12:.1f} TFLOPS")
    q = torch.randn(1, 32, 2048, 128, device=DEV, dtype=torch.bfloat16)
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q); acc.synchronize()
    check("attention runs, finite", bool(torch.isfinite(o).all()))
    host = torch.empty(256 << 20, dtype=torch.uint8).pin_memory(); dst = torch.empty_like(host, device=DEV)
    t = timeit(lambda: dst.copy_(host, non_blocking=True), 5); check("pinned RAM->GPU copy", True, f"{(256 << 20) / t / 1e9:.1f} GB/s")
    try:
        print(f"  free GPU memory: {acc.mem_get_info()[0] / 2**30:.1f} GiB")
    except Exception:
        print("  free GPU memory: (not reported by this device)")
elif what == "loader":
    from safetensors import safe_open
    from blockstream import BlockStreamer, Plan, ShardIndex
    idx = ShardIndex(f"{MODELS}/transformer")
    units = ["transformer_blocks.0", "transformer_blocks.47"]
    plans = [Plan([t for k, t in idx.tensors.items() if k.startswith(u + ".")]) for u in units]
    check("found tensors for both blocks", all(len(p.views) > 0 for p in plans), f"({[len(p.views) for p in plans]} tensors)")
    t0 = time.perf_counter()
    st = BlockStreamer(plans, slots=2, passes=1)
    files, nbad, ntot = {}, 0, 0
    for i in range(len(plans)):
        slot, got = st.next()
        for name, g in got.items():
            t = idx.tensors[name]
            f = files.get(t.file) or files.setdefault(t.file, safe_open(t.file, "pt"))
            r = f.get_tensor(name); ntot += 1
            if not (g.shape == r.shape and g.dtype == r.dtype and torch.equal(g.cpu(), r)):
                nbad += 1
        st.release(slot)
    dt = time.perf_counter() - t0
    mb = sum(p.nbytes for p in plans) / 2**20
    check("streamed tensors identical to safetensors", nbad == 0, f"({ntot - nbad}/{ntot} tensors; {mb:.0f} MiB in {dt:.1f}s = {mb / 1024 / dt:.2f} GiB/s incl. verify)")
    st.close()
else:
    sys.exit(f"unknown test {what}")
sys.exit(1 if bad else 0)
