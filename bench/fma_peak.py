"""Raw FMA throughput of the iGPU: does fp16 run at 2x the fp32 rate?"""
import time, torch, triton, triton.language as tl
dev = torch.device("xpu")

@triton.jit
def fma_kernel(x_ptr, y_ptr, n, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    a = x * 0.5 + 0.25
    b = x * 0.25 + 0.5
    for _ in tl.static_range(ITERS):   # 4 independent chains of dependent FMAs
        x = x * a + b
        a = a * b + x
        b = b * x + a
    tl.store(y_ptr + offs, x, mask=m)

def bench(dt, ITERS=64, n=1 << 22, BLOCK=256):
    x = (torch.rand(n, device=dev) * 0.01).to(dt); y = torch.empty_like(x)
    grid = (triton.cdiv(n, BLOCK),)
    fma_kernel[grid](x, y, n, ITERS=ITERS, BLOCK=BLOCK); torch.xpu.synchronize()
    t = time.perf_counter(); reps = 5
    for _ in range(reps): fma_kernel[grid](x, y, n, ITERS=ITERS, BLOCK=BLOCK)
    torch.xpu.synchronize(); dtm = (time.perf_counter() - t) / reps
    return 3 * 2 * ITERS * n / dtm / 1e12     # 3 FMAs per iter, 2 flop each

for dt in (torch.float32, torch.float16, torch.float32, torch.float16):
    try: print(f"{str(dt):14s}: {bench(dt):5.2f} TFLOPS (pure FMA, registers only)", flush=True)
    except Exception as e: print(dt, "FAILED:", type(e).__name__, str(e)[:200]); break
