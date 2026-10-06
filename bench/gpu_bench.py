"""iGPU microbenchmarks: matmul TFLOPS (fp32/fp16/bf16) and device memory bandwidth."""
import time
import torch

assert torch.xpu.is_available(), "XPU not available"
dev = torch.device("xpu")
print("device:", torch.xpu.get_device_name(0))
props = torch.xpu.get_device_properties(0)
print("total memory GB:", round(props.total_memory / 2**30, 1))


def sync():
    torch.xpu.synchronize()


def bench_matmul(dtype, n=2048, iters=20):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(3):
        a @ b
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync()
    dt = time.perf_counter() - t
    return 2 * n**3 * iters / dt / 1e12


for dt in (torch.float32, torch.float16, torch.bfloat16):
    try:
        print(f"matmul {str(dt):15s}: {bench_matmul(dt):.2f} TFLOPS")
    except Exception as e:
        print(f"matmul {dt}: failed ({e})")

# device memory bandwidth (copy within shared RAM)
n = 256 * 2**20 // 4
x = torch.empty(n, device=dev)
y = torch.empty(n, device=dev)
x.normal_()
sync()
t = time.perf_counter()
for _ in range(20):
    y.copy_(x)
sync()
dt = time.perf_counter() - t
print(f"device copy: {2 * n * 4 * 20 / dt / 2**30:.1f} GB/s (read+write)")

# host -> device transfer (pinned vs pageable)
h = torch.empty(n, dtype=torch.float32).normal_()
sync()
t = time.perf_counter()
for _ in range(5):
    h.to(dev)
sync()
dt = time.perf_counter() - t
print(f"host->device: {n * 4 * 5 / dt / 2**30:.1f} GB/s")
