"""Find the fastest host->iGPU path for streaming layer weights."""
import time
import torch

dev = torch.device("xpu")
MB = 2**20


def timeit(fn, nbytes, iters=10):
    fn()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    dt = time.perf_counter() - t
    return nbytes * iters / dt / 2**30


for size_mb in (16, 64, 256):
    n = size_mb * MB // 2
    pageable = torch.empty(n, dtype=torch.bfloat16).normal_()
    dst = torch.empty(n, dtype=torch.bfloat16, device=dev)
    print(f"--- {size_mb} MB chunks ---")
    print(f"pageable .to(dev):          {timeit(lambda: pageable.to(dev), n * 2):.2f} GB/s")
    print(f"pageable copy_ into dst:    {timeit(lambda: dst.copy_(pageable), n * 2):.2f} GB/s")
    try:
        pinned = torch.empty(n, dtype=torch.bfloat16).pin_memory()
        pinned.normal_()
        print(f"pinned copy_ non_blocking:  {timeit(lambda: dst.copy_(pinned, non_blocking=True), n * 2):.2f} GB/s")
    except Exception as e:
        print("pinned failed:", e)
    del pageable, dst
