import time, torch, torch.nn.functional as F
dev = torch.device("xpu")
M, K, N = 11440, 4096, 4096
def tf(fn, n=6):
    fn(); fn(); torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.xpu.synchronize(); return 2 * M * K * N / ((time.perf_counter() - t) / n) / 1e12
for dt in (torch.float16, torch.bfloat16, torch.float32):
    x = torch.randn(M, K, device=dev, dtype=dt); w = torch.randn(N, K, device=dev, dtype=dt); b = torch.randn(N, device=dev, dtype=dt)
    wt = w.t().contiguous(); xb = x.unsqueeze(0); wtb = wt.unsqueeze(0)
    print(f"{str(dt):15s} x@w.T {tf(lambda: x @ w.t()):.2f} | F.linear {tf(lambda: F.linear(x, w)):.2f} | linear+bias {tf(lambda: F.linear(x, w, b)):.2f} | x@wt(contig) {tf(lambda: x @ wt):.2f} | addmm {tf(lambda: torch.addmm(b, x, wt)):.2f} | bmm {tf(lambda: torch.bmm(xb, wtb)):.2f} | einsum {tf(lambda: torch.einsum('mk,kn->mn', x, wt)):.2f}", flush=True)
    del x, w, b, wt
    torch.xpu.empty_cache()
