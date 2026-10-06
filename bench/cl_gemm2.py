"""GEMM v2 for the iGPU: packed-fp16 FMAs, A via sub-group block reads + shuffles (no per-lane A loads), SIMD8.

C[M,N] (fp32) = A[M,K] @ B[K,N]; A is stored *pre-doubled* as uint32 pairs (a,a) so a shuffled value is directly a half2
broadcast operand; B is [K,N] half row-major (the pre-transposed Linear weight). Partial sums are kept in half2 and folded
into float accumulators every FLUSH blocks of 8 k-steps (precision).
"""
import sys, time, numpy as np, pyopencl as cl

def gen_v2(TM=8, flush=4):
    s = ["#pragma OPENCL EXTENSION cl_khr_fp16 : enable",
         "#pragma OPENCL EXTENSION cl_intel_subgroups : enable",
         "__attribute__((intel_reqd_sub_group_size(8)))",
         "__kernel void gemm(__global const uint *A2, __global const half *B, __global float *C, int M, int N, int K) {",
         "  int n0 = get_global_id(0) * 8; int m0 = get_global_id(1) * %d;" % TM,
         "  __global const uint *Ap = A2 + (size_t)m0 * K;"]
    for r in range(TM):
        for j in range(4):
            s.append("  half2 c%d_%d = (half2)(0);" % (r, j))
            if flush:
                s.append("  float2 f%d_%d = (float2)(0);" % (r, j))
    s.append("  for (int kb = 0; kb < K; kb += 8) {")
    for r in range(TM):
        s.append("    uint a%d = intel_sub_group_block_read(Ap + (size_t)%d*K + kb);" % (r, r))
    for kk in range(8):
        s.append("    { half8 b = vload8(0, B + (size_t)(kb+%d)*N + n0);" % kk)
        for r in range(TM):
            s.append("      half2 x%d = as_half2(intel_sub_group_shuffle(a%d, %d));" % (r, r, kk))
            for j in range(4):
                s.append("      c%d_%d = fma(x%d, b.s%d%d, c%d_%d);" % (r, j, r, 2 * j, 2 * j + 1, r, j))
        s.append("    }")
    if flush:
        s.append("    if (((kb >> 3) + 1) %% %d == 0) {" % flush)
        for r in range(TM):
            for j in range(4):
                s.append("      f%d_%d += convert_float2(c%d_%d); c%d_%d = (half2)(0);" % (r, j, r, j, r, j))
        s.append("    }")
    s.append("  }")
    for r in range(TM):
        v = ["f%d_%d" % (r, j) if flush else "convert_float2(c%d_%d)" % (r, j) for j in range(4)]
        s.append("  vstore8((float8)(%s, %s, %s, %s), 0, C + (size_t)(m0+%d)*N + n0);" % (v[0], v[1], v[2], v[3], r))
    s.append("}")
    return "\n".join(s)

_ctx = None
def setup():
    global _ctx
    if _ctx is None:
        c = cl.create_some_context(interactive=False)
        _ctx = (c, cl.CommandQueue(c, properties=cl.command_queue_properties.PROFILING_ENABLE))
    return _ctx

def bench(src, M, N, K, TM, local=None, reps=4, label="", data=None):
    ctx, q = setup()
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16); B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    A2 = A.view(np.uint16).astype(np.uint32) * np.uint32(0x10001)
    mf = cl.mem_flags
    dA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A2); dB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    dC = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    t0 = time.perf_counter()
    prog = cl.Program(ctx, src).build(); k = prog.gemm
    k.set_args(dA, dB, dC, np.int32(M), np.int32(N), np.int32(K))
    spills = ""
    try:
        spills = f" regs/priv={k.get_work_group_info(cl.kernel_work_group_info.PRIVATE_MEM_SIZE, ctx.devices[0])}B"
    except Exception:
        pass
    gsz = (N // 8, M // TM)
    cl.enqueue_nd_range_kernel(q, k, gsz, local).wait()
    best = 1e9
    for _ in range(reps):
        ev = cl.enqueue_nd_range_kernel(q, k, gsz, local); ev.wait()
        best = min(best, (ev.profile.end - ev.profile.start) * 1e-9)
    tf = 2 * M * N * K / best / 1e12
    C = np.empty((M, N), np.float32); cl.enqueue_copy(q, C, dC)
    rows = rng.choice(M, 32, replace=False)
    ref = A[rows].astype(np.float32) @ B.astype(np.float32)
    rel = np.linalg.norm(C[rows] - ref) / np.linalg.norm(ref)
    print(f"{label:30s} local={str(local):9s} {best*1000:7.1f} ms {tf:5.2f} TFLOPS rel.err {rel:.1e}{spills} (build {time.perf_counter()-t0:.1f}s)", flush=True)
    return tf

if __name__ == "__main__":
    M, N, K = 11392, 4096, 4096
    print(f"M={M} N={N} K={K}; oneDNN fp16 reference ~3.4-3.7 TFLOPS; raw fp16 FMA peak 7.8")
    for TM in (4, 8, 6):
        for flush in (4, 0):
            for local in ((8, 8), (8, 4), (32, 4)):
                if M % (TM * local[1]) or N % (8 * local[0]): continue
                try: bench(gen_v2(TM, flush), M, N, K, TM, local, label=f"v2 TM={TM} flush={flush}")
                except Exception as e: print(f"v2 TM={TM} flush={flush} local={local}: FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
