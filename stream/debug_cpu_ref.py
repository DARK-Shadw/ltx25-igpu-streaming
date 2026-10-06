import copy, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
from runtime import DEV, Streamed, build_empty
M = "models/ltx25"
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer")
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16))
units = [f"model.language_model.layers.{i}" for i in range(48)]
te = Streamed(m, f"{M}/text_encoder", units)
te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))

def tocpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float() if x.is_floating_point() else x.detach().cpu()
    if isinstance(x, dict): return {k: tocpu(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return type(x)(tocpu(v) for v in x)
    return x

def replay(mod, args, kwargs, out):
    i = mod._unit_i
    if i not in (0, 1, 5, 23, 47): return
    memo = {}
    for p in mod.parameters():
        memo[id(p)] = torch.nn.Parameter(p.detach().cpu().float(), requires_grad=False)
    for b in mod.buffers():
        memo[id(b)] = b.detach().cpu().float()
    lay = copy.deepcopy(mod, memo)
    lay._forward_hooks.clear(); lay._forward_pre_hooks.clear()
    with torch.no_grad():
        ref = lay(*tocpu(args), **tocpu(kwargs))
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    got = (out[0] if isinstance(out, (tuple, list)) else out).detach().cpu().float()
    d = (ref - got).abs()
    print(f"layer {i}: gpu absmax={got.abs().max():.2f} cpu absmax={ref.abs().max():.2f}  "
          f"maxdiff={d.max():.3f} meandiff={d.mean():.4f} relmean={(d.mean()/ref.abs().mean()):.4f}", flush=True)

for i in range(48):
    m.get_submodule(units[i]).register_forward_hook(replay, with_kwargs=True, prepend=True)
ids = tok(["The capital of France is"], return_tensors="pt").input_ids.to(DEV)
with torch.no_grad():
    m.model(input_ids=ids, use_cache=False)
