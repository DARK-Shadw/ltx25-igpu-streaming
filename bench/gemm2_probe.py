import sys, time, numpy as np
sys.path.insert(0, "bench")
import cl_gemm2 as g
TM, flush, M = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
local = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] != "none" else None
N, K = 4096, 4096
t = time.perf_counter(); src = g.gen_v2(TM, flush); print(f"generated {len(src.splitlines())} lines", flush=True)
g.bench(src, M, N, K, TM, local, reps=2, label=f"v2 TM={TM} flush={flush} M={M}")
