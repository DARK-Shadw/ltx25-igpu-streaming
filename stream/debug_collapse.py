import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
from runtime import DEV, Streamed, build_empty
M = "models/ltx25"
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer"); tok.add_bos_token = True
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16))
te = Streamed(m, f"{M}/text_encoder", [f"model.language_model.layers.{i}" for i in range(48)])
te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))
ls = [b.float().item() for n, b in m.named_buffers() if n.endswith("layer_scalar")]
print("layer_scalar (resident-state, expect meta/ones):", ls[:3])
P = ["A golden retriever running along a sunny beach", "Stock prices fell sharply on Monday morning", "She baked a chocolate cake for the party"]
cos = torch.nn.functional.cosine_similarity
res = {}
for p in P:
    ids = tok([p], return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        res[p] = [h[0].float().cpu() for h in m.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states]
a, b = res[P[0]], res[P[1]]
print("layer | mean cos(last token, other prompt last token) | mean pairwise cos between positions within prompt 0")
for l in (0, 1, 2, 4, 8, 16, 24, 32, 40, 47, 48):
    x = a[l]; n = x.shape[0]
    xn = torch.nn.functional.normalize(x, dim=-1); within = ((xn @ xn.T).sum() - n) / (n * (n - 1))
    print(f"{l:5d} | {cos(a[l][-1], b[l][-1], dim=0):.3f} | {within:.3f}   (|h| last={a[l][-1].norm():.0f})")
