"""Custom fp16 GEMM for the iGPU (OpenCL, packed half2 FMAs). C[M,N] = A[M,K] @ B[K,N], row-major."""
import sys, time, numpy as np, pyopencl as cl

def gen_v0(TM=8, TN=8, flush=0):
    """Register-tiled GEMM. Each work-item computes TM x TN outputs. TN must be 8 (one half8 B load per k).
    flush>0: every `flush` k-steps fold the half2 partial sums into float accumulators (precision)."""
    assert TN == 8
    rows = range(TM)
    s = ["#pragma OPENCL EXTENSION cl_khr_fp16 : enable",
         "__attribute__((intel_reqd_sub_group_size(16)))",
         "__kernel void gemm(__global const half *A, __global const half *B, __global float *C, int M, int N, int K) {",
         "  int n0 = get_global_id(0) * 8; int m0 = get_global_id(1) * %d;" % TM]
    for r in rows:
        s.append("  half2 c%d_0=(half2)(0), c%d_1=(half2)(0), c%d_2=(half2)(0), c%d_3=(half2)(0);" % (r, r, r, r))
        if flush:
            s.append("  float2 f%d_0=(float2)(0), f%d_1=(float2)(0), f%d_2=(float2)(0), f%d_3=(float2)(0);" % (r, r, r, r))
    s.append("  for (int k = 0; k < K; k += 8) {")
    for r in rows:
        s.append("    half8 a%d = vload8(0, A + (m0+%d)*K + k);" % (r, r))
    for kk in range(8):
        s.append("    { half8 b = vload8(0, B + (size_t)(k+%d)*N + n0);" % kk)
        for r in rows:
            s.append("      half2 x%d = (half2)(a%d.s%d);" % (r, r, kk))
            for j in range(4):
                s.append("      c%d_%d = fma(x%d, b.s%d%d, c%d_%d);" % (r, j, r, 2*j, 2*j+1, r, j))
        s.append("    }")
    if flush:
        s.append("    if (((k >> 3) + 1) %% %d == 0) {" % flush)
        for r in rows:
            for j in range(4):
                s.append("      f%d_%d += convert_float2(c%d_%d); c%d_%d = (half2)(0);" % (r, j, r, j, r, j))
        s.append("    }")
    s.append("  }")
    for r in rows:
        src = "f" if flush else "c"
        conv = (lambda j: "f%d_%d" % (r, j)) if flush else (lambda j: "convert_float2(c%d_%d)" % (r, j))
        s.append("  vstore8((float8)(%s, %s, %s, %s), 0, C + (size_t)(m0+%d)*N + n0);" % (conv(0), conv(1), conv(2), conv(3), r))
    s.append("}")
    return "\n".join(s)

def bench(src, M, N, K, TM, local=None, reps=4, check=True, label=""):
    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16); B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    mf = cl.mem_flags
    dA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A); dB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    dC = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    t0 = time.perf_counter()
    prog = cl.Program(ctx, src).build()
    k = prog.gemm
    k.set_args(dA, dB, dC, np.int32(M), np.int32(N), np.int32(K))
    gsz = (N // 8, M // TM)
    cl.enqueue_nd_range_kernel(q, k, gsz, local).wait()
    best = 1e9
    for _ in range(reps):
        ev = cl.enqueue_nd_range_kernel(q, k, gsz, local); ev.wait()
        best = min(best, (ev.profile.end - ev.profile.start) * 1e-9)
    tf = 2 * M * N * K / best / 1e12
    err = ""
    if check:
        C = np.empty((M, N), np.float32); cl.enqueue_copy(q, C, dC)
        rows = rng.choice(M, 32, replace=False)
        ref = A[rows].astype(np.float32) @ B.astype(np.float32)
        rel = np.linalg.norm(C[rows] - ref) / np.linalg.norm(ref)
        err = f" rel.err vs fp32 ref {rel:.2e}"
    print(f"{label:34s} local={str(local):9s} {best*1000:7.1f} ms  {tf:5.2f} TFLOPS{err}", flush=True)
    return tf

if __name__ == "__main__":
    M, N, K = 11392, 4096, 4096
    print(f"shape M={M} N={N} K={K}  (torch/oneDNN fp16 on this shape: ~3.2-3.7 TFLOPS)")
    for TM in (4, 8):
        for flush in (0, 8):
            for local in (None, (16, 4), (16, 8)):
                try: bench(gen_v0(TM, 8, flush), M, N, K, TM, local, label=f"v0 tile {TM}x8 flush={flush}")
                except Exception as e: print(f"v0 tile {TM}x8 flush={flush} local={local}: FAILED {type(e).__name__}: {str(e)[:120]}")
