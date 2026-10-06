import gc, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
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
L = 128
ti = tok(P, padding="max_length", max_length=L, truncation=True, return_tensors="pt")
ids, mk = ti.input_ids.to(DEV), ti.attention_mask.to(DEV)
with torch.no_grad():
    hs = m.model(input_ids=ids, attention_mask=mk, output_hidden_states=True, use_cache=False).hidden_states
    emb = torch.stack(hs, dim=-1).flatten(2, 3).to(torch.bfloat16)
del hs; te.free_resident(); te.close(); del te, m; gc.collect(); torch.xpu.empty_cache()
print("text embeds", tuple(emb.shape))
conn = build_empty(lambda: LTX2TextConnectors.from_config(LTX2TextConnectors.load_config(f"{M}/connectors")))
cunits = ["video_text_proj_in", "audio_text_proj_in"] + [f"{s}_connector.transformer_blocks.{i}" for s in ("video", "audio") for i in range(8)]
cs = Streamed(conn, f"{M}/connectors", cunits); cs.load_resident()
with torch.no_grad():
    cv, ca, cm = conn(emb, mk, padding_side="left")
print("connector out", tuple(cv.shape), tuple(ca.shape), "mask", tuple(cm.shape), cm.dtype, "mask sum/row", cm.float().sum(-1)[:2].tolist() if cm.dim() == 2 else cm.float().flatten(1).sum(-1)[:2].tolist())
same = torch.tensor([[a == b and i != j for j, b in enumerate(lab)] for i, a in enumerate(lab)])
diff = torch.tensor([[a != b for b in lab] for a in lab])
def gap(name, x):
    v = x.float().mean(1).cpu()                      # mean-pool over all positions
    v = torch.nn.functional.normalize(v - v.mean(0, keepdim=True), dim=-1); c = v @ v.T
    print(f"{name:22s} within {c[same].mean():+.3f}  cross {c[diff].mean():+.3f}  gap {c[same].mean() - c[diff].mean():+.3f}")
w = mk.float()[..., None]
gap("gemma layer24 (pooled)", (emb.view(*emb.shape[:2], 3840, 49)[..., 24].float() * w))
gap("connector video out", cv); gap("connector audio out", ca)
