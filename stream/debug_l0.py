import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
from runtime import DEV, Streamed, build_empty
M = "models/ltx25"
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer")
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16))
te = Streamed(m, f"{M}/text_encoder", ["model.language_model.layers.0"])
te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))
layer = m.get_submodule("model.language_model.layers.0")
print(type(layer).__name__, [n for n, _ in layer.named_children()])
def mk(name):
    def h(mod, args, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        w = getattr(mod, "weight", None)
        ws = f"w.finite={torch.isfinite(w).all().item()} w.absmax={w.float().abs().max().item():.3f} w.dtype={w.dtype}" if w is not None and w.device.type != "meta" else ""
        if torch.is_tensor(o):
            print(f"  {name:28s} out finite={torch.isfinite(o).all().item()} absmax={o.float().nan_to_num(0).abs().max().item():.2f} {ws}")
    return h
for n, sub in layer.named_modules():
    if n: sub.register_forward_hook(mk(n))
ids = tok(["A golden retriever running along a sunny beach"], return_tensors="pt").input_ids.to(DEV)
with torch.no_grad():
    out = m.model(input_ids=ids, output_hidden_states=True)
print("layer_scalar:", [(n, b.float().tolist()) for n, b in layer.named_buffers()])
