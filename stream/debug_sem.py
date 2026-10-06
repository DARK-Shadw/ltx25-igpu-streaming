import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
from runtime import DEV, Streamed, build_empty
M = "models/ltx25"
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer"); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16))
te = Streamed(m, f"{M}/text_encoder", [f"model.language_model.layers.{i}" for i in range(48)])
te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))
P = ["A golden retriever running along a sunny beach",
     "A puppy playing at the seashore on a bright day",
     "Stock market crash causes panic on the trading floor",
     "An old man with a long white beard"]
ti = tok(P, padding=True, return_tensors="pt", add_special_tokens=True)
print("input ids row0:", ti.input_ids[0].tolist(), "decoded:", tok.decode(ti.input_ids[0]))
ids, mk = ti.input_ids.to(DEV), ti.attention_mask.to(DEV)
with torch.no_grad():
    out = m.model(input_ids=ids, attention_mask=mk, output_hidden_states=True)
for layer in (0, 12, 24, 36, 48):
    h = out.hidden_states[layer].float()
    v = (h * mk[..., None]).sum(1) / mk.sum(1, keepdim=True)
    v = torch.nn.functional.normalize(v - v.mean(0, keepdim=True), dim=-1)   # center across prompts
    c = v @ v.T
    print(f"layer {layer:2d} centered cosine  dog~puppy={c[0,1]:.2f}  dog~stock={c[0,2]:.2f}  dog~oldman={c[0,3]:.2f}  puppy~stock={c[1,2]:.2f}")
