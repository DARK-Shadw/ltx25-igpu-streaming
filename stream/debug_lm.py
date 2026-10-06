import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
from runtime import DEV, Streamed, build_empty
M = "models/ltx25"
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer"); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16, attn_implementation=os.environ.get('ATTN', 'sdpa')))
te = Streamed(m, f"{M}/text_encoder", [f"model.language_model.layers.{i}" for i in range(48)])
te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))

for text in ["The capital of France is", "The dog chased the ball and then the dog"]:
    ids = tok([text], return_tensors="pt").input_ids.to(DEV)
    print("tokens:", ids[0].tolist()[:12])
    with torch.no_grad():
        out = m.model(input_ids=ids)
    h = out.last_hidden_state[0, -1]
    W = m.model.language_model.embed_tokens.weight
    logits = torch.cat([(W[i:i + 16384] @ h).float() for i in range(0, W.shape[0], 16384)])
    top = logits.topk(8).indices.tolist()
    print(repr(text), "->", [tok.decode([t]) for t in top])
