"""OpenCL raw FMA throughput on the iGPU: fp32 vs fp16 (does fp16 run at 2x?)."""
import time, numpy as np, pyopencl as cl
ctx = cl.create_some_context(interactive=False); q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
SRC = """
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define KERNEL(NAME, T, VEC)                                                     \
__kernel void NAME(__global T##VEC *out, const T s) {                            \
    size_t i = get_global_id(0);                                                  \
    T##VEC a0 = (T##VEC)(s) + (T##VEC)(i), a1 = a0 * (T)0.5, a2 = a0 * (T)0.25, a3 = a0 * (T)0.125; \
    T##VEC b = (T##VEC)(s * (T)0.001), c = (T##VEC)((T)0.999);                    \
    for (int k = 0; k < ITERS; k++) {                                             \
        a0 = fma(a0, c, b); a1 = fma(a1, c, b); a2 = fma(a2, c, b); a3 = fma(a3, c, b); \
        a0 = fma(a0, c, b); a1 = fma(a1, c, b); a2 = fma(a2, c, b); a3 = fma(a3, c, b); \
    }                                                                             \
    out[i] = a0 + a1 + a2 + a3;                                                   \
}
KERNEL(fma_f32, float, 4)
KERNEL(fma_f16, half, 8)
KERNEL(fma_f16v16, half, 16)
"""
ITERS = 2048
prog = cl.Program(ctx, SRC).build(options=f"-DITERS={ITERS}")
GLOBAL = 112 * 8 * 64 * 16   # plenty of threads to saturate
def run(name, lanes, dtype, s):
    k = getattr(prog, name)
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, GLOBAL * lanes * np.dtype(dtype).itemsize)
    k(q, (GLOBAL,), None, out, dtype(s)).wait()
    best = 0
    for _ in range(4):
        ev = k(q, (GLOBAL,), None, out, dtype(s)); ev.wait()
        t = (ev.profile.end - ev.profile.start) * 1e-9
        best = max(best, GLOBAL * lanes * ITERS * 8 * 2 / t / 1e12)   # 8 fma per iter per lane, 2 flop each
    return best
print(f"theoretical @2.2GHz: fp32 {112*16*2.2/1000:.2f}  fp16(2x) {112*32*2.2/1000:.2f} TFLOPS")
print(f"fp32 float4  : {run('fma_f32', 4, np.float32, 1.0):.2f} TFLOPS")
print(f"fp16 half8   : {run('fma_f16', 8, np.float16, 1.0):.2f} TFLOPS")
print(f"fp16 half16  : {run('fma_f16v16', 16, np.float16, 1.0):.2f} TFLOPS")
