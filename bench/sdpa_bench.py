import time, torch
import torch.nn.functional as F
dev = torch.device("xpu")
for N in (2200, 4400, 8800, 11400, 17000):
    try:
        q = torch.randn(1, 32, N, 128, device=dev, dtype=torch.bfloat16)
        k = torch.randn_like(q); v = torch.randn_like(q)
        torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats()
        base = torch.xpu.memory_allocated()
        t = time.perf_counter()
        o = F.scaled_dot_product_attention(q, k, v)
        torch.xpu.synchronize(); dt = time.perf_counter() - t
        peak = (torch.xpu.max_memory_allocated() - base) / 2**30
        flops = 4 * N * N * 128 * 32
        print(f"N={N:6d}: {dt:6.2f} s  extra peak {peak:5.2f} GiB  ~{flops / dt / 1e12:.2f} TFLOPS", flush=True)
        del q, k, v, o
    except Exception as e:
        print(f"N={N:6d}: FAILED {type(e).__name__}: {str(e)[:90]}", flush=True)
        break
    torch.xpu.empty_cache()
