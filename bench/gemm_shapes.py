import time, torch
dev = torch.device("xpu")
# real shapes: tokens x 4096 @ 4096 x {4096 (attn proj), 16384 (ffn up)}, ffn down 16384->4096
shapes = [(11440, 4096, 4096), (11440, 4096, 16384), (11440, 16384, 4096)]
# theoretical: 112 EU * 16 fp32 flop/clk * clock
for clk in (2.1,):
    print(f"theoretical peak @ {clk} GHz: fp32 {112*16*clk/1000:.1f} TFLOPS, fp16 (2x packed) {112*32*clk/1000:.1f} TFLOPS")
for M, K, N in shapes:
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        a = torch.randn(M, K, device=dev, dtype=dt); b = torch.randn(K, N, device=dev, dtype=dt)
        for _ in range(2): a @ b
        torch.xpu.synchronize(); t = time.perf_counter(); n = 5
        for _ in range(n): a @ b
        torch.xpu.synchronize(); dtm = (time.perf_counter() - t) / n
        print(f"M={M} K={K} N={N} {str(dt):15s}: {2*M*K*N/dtm/1e12:5.2f} TFLOPS", flush=True)
        del a, b
    torch.xpu.empty_cache()
