import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig
from safetensors import safe_open
from runtime import DEV, Streamed, build_empty
M = "models/ltx25/text_encoder"
tcfg = AutoConfig.from_pretrained(M)
m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16))
units = [f"model.language_model.layers.{i}" for i in range(48)]
te = Streamed(m, M, units)
files = {}
def ref(t):
    f = files.get(t.file) or files.setdefault(t.file, safe_open(t.file, "pt"))
    return f.get_tensor(t.name)
bad = tot = 0
for ui in (0, 5, 23, 47):
    got = te._load_unit_direct(te.plans[ui])
    for name, g in got.items():
        r = ref(te.idx.tensors[name]); tot += 1
        ok = g.shape == r.shape and g.dtype == r.dtype and torch.equal(g.cpu(), r)
        if not ok:
            bad += 1
            if bad <= 5: print("MISMATCH", name, tuple(g.shape), tuple(r.shape), g.dtype, r.dtype)
print(f"direct loader: {tot - bad}/{tot} tensors identical to safetensors")
# resident path
te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))
for name in ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight"):
    g = dict(m.named_parameters())[name]
    r = ref(te.idx.tensors[name])
    print(name, "identical:", torch.equal(g.detach().cpu(), r))
# which model params are still meta (never loaded) outside the layers?
print("still-meta non-layer params:", [n for n, p in m.named_parameters() if p.device.type == "meta" and ".layers." not in n][:12])
