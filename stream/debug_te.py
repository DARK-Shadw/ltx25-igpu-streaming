import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
from runtime import DEV, Streamed, build_empty
M = "models/ltx25"
impl = os.environ.get("ATTN", "sdpa")
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer"); tok.padding_side = "left"
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
te_full = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16, attn_implementation=impl))
print("attn impl:", te_full.config._attn_implementation if hasattr(te_full.config, "_attn_implementation") else impl)
te = Streamed(te_full, f"{M}/text_encoder", [f"model.language_model.layers.{i}" for i in range(48)])
te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))
first = []
def chk(mod, args, out):
    o = out[0] if isinstance(out, (tuple, list)) else out
    ok = torch.isfinite(o).all().item()
    i = mod._unit_i
    if i in (0, 1, 2, 10, 47) or (not ok and not first):
        print(f"  layer {i}: finite={ok} absmax={o.float().nan_to_num(0).abs().max().item():.1f}")
    if not ok and not first: first.append(i)
for i in range(48): te_full.get_submodule(f"model.language_model.layers.{i}").register_forward_hook(chk)
for pad in (False, True):
    first.clear()
    print("padding to 256:" if pad else "no padding:")
    ti = tok(["A golden retriever running along a sunny beach"], padding="max_length" if pad else False, max_length=256,
             truncation=True, add_special_tokens=True, return_tensors="pt")
    ids, m = ti.input_ids.to(DEV), ti.attention_mask.to(DEV)
    print("  tokens:", ids.shape, "valid:", int(m.sum()))
    with torch.no_grad():
        out = te_full.model(input_ids=ids, attention_mask=m, output_hidden_states=True)
    hs = out.hidden_states
    print("  hidden states finite:", [torch.isfinite(h).all().item() for h in hs][:3], "...", all(torch.isfinite(h).all().item() for h in hs))
