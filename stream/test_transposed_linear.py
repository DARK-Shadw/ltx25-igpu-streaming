import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from safetensors import safe_open
from diffusers import LTX2VideoTransformer3DModel
from runtime import DEV, Streamed, build_empty
M = "models/ltx25/transformer"
dit = build_empty(lambda: LTX2VideoTransformer3DModel.from_config(LTX2VideoTransformer3DModel.load_config(M)))
unit = "transformer_blocks.0"
mode = sys.argv[1] if len(sys.argv) > 1 else "sync"
st = Streamed(dit, M, [unit], prefetch_passes=1 if mode == "prefetch" else 0, transpose_linear=True)
blk = dit.get_submodule(unit)
lin = [(n, m) for n, m in blk.named_modules() if isinstance(m, nn.Linear)]
print(f"mode={mode}: block has {len(lin)} Linear modules, {len(st.transposed_names)} weights flagged for transposition")
# trigger the streaming pre-hook by hand (same call the forward pre-hook would make)
st._pre(blk, ())
files = {}
def ref(name):
    t = st.idx.tensors[name]; f = files.get(t.file) or files.setdefault(t.file, safe_open(t.file, "pt"))
    return f.get_tensor(name).to(DEV)
worst, bad = 0.0, 0
for n, m in lin:
    full = f"{unit}.{n}.weight"
    if full not in st.transposed_names: continue
    w = ref(full); b = ref(full[:-6] + "bias") if m.bias is not None else None
    if b is not None and b.dtype != torch.bfloat16: b = b.to(torch.bfloat16)
    x = torch.randn(300, w.shape[1], device=DEV, dtype=torch.bfloat16)
    y = m(x); r = F.linear(x, w, b)
    rel = ((y.float() - r.float()).abs().max() / r.float().abs().max()).item()
    worst = max(worst, rel); bad += rel > 1e-2
print(f"patched vs reference F.linear: worst relative error {worst:.2e}, layers over 1e-2: {bad}")
# speed on the biggest layers at video-sized token counts
for n, m in lin:
    full = f"{unit}.{n}.weight"
    if full not in st.transposed_names or m.weight.numel() < 4096 * 8192: continue
    K = m.weight.shape[0]; Nn = m.weight.shape[1]
    w = ref(full); x = torch.randn(11440, K, device=DEV, dtype=torch.bfloat16)
    def tf(fn, reps=5):
        fn(); fn(); torch.xpu.synchronize(); t = time.perf_counter()
        for _ in range(reps): fn()
        torch.xpu.synchronize(); return 2 * 11440 * K * Nn / ((time.perf_counter() - t) / reps) / 1e12
    print(f"  {n:22s} [{K}x{Nn}]  F.linear {tf(lambda: F.linear(x, w)):.2f} TFLOPS | patched {tf(lambda: m(x)):.2f} TFLOPS")
    del w, x
