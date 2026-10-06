import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
from runtime import DEV, Streamed, build_empty
M = "models/ltx25"
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer"); tok.add_bos_token = True; tok.padding_side = "left"
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16))
te = Streamed(m, f"{M}/text_encoder", [f"model.language_model.layers.{i}" for i in range(48)])
te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))
topics = {
 "dog":   ["A golden retriever runs along the beach", "A happy puppy plays in the sand by the sea", "The dog chases waves on the shore"],
 "money": ["Stock prices fell sharply on Monday", "The bank raised interest rates again today", "Investors worry about inflation and the economy"],
 "food":  ["She baked a chocolate cake for the party", "The chef cooked pasta with tomato sauce", "Fresh bread and cheese were served for lunch"],
}
P = [p for v in topics.values() for p in v]; lab = [k for k, v in topics.items() for _ in v]
ti = tok(P, padding=True, return_tensors="pt")
ids, mk = ti.input_ids.to(DEV), ti.attention_mask.to(DEV)
with torch.no_grad():
    hs = m.model(input_ids=ids, attention_mask=mk, output_hidden_states=True, use_cache=False).hidden_states
same = torch.tensor([[a == b and i != j for j, b in enumerate(lab)] for i, a in enumerate(lab)])
diff = torch.tensor([[a != b for b in lab] for a in lab])
print("layer | within-topic cos | cross-topic cos | gap   (mean-pooled over real tokens, centered across prompts)")
for l in (2, 4, 8, 12, 16, 24, 32, 40, 47, 48):
    h = hs[l].float() * mk[..., None]
    v = (h.sum(1) / mk.sum(1, keepdim=True)).cpu()
    v = torch.nn.functional.normalize(v - v.mean(0, keepdim=True), dim=-1)
    c = v @ v.T
    print(f"{l:5d} | {c[same].mean():.3f} | {c[diff].mean():.3f} | {c[same].mean() - c[diff].mean():+.3f}")
