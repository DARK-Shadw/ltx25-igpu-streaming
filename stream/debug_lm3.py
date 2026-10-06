import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
import runtime
from runtime import DEV, Streamed, build_empty
M = "models/ltx25"
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer"); tok.add_bos_token = True
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
for default in (torch.bfloat16, torch.float32):
    m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=default), dtype=default)
    lm = m.model.language_model
    print(f"=== build default dtype {default}")
    print("  buffers:", [(n.split('.')[-1], str(b.dtype).replace('torch.', '')) for n, b in m.named_buffers() if 'layers.' not in n])
    te = Streamed(m, f"{M}/text_encoder", [f"model.language_model.layers.{i}" for i in range(48)])
    te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))
    W = lm.embed_tokens.weight
    ids = tok(["The capital of France is"], return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        out = m.model(input_ids=ids, use_cache=False)
    h = out.last_hidden_state[0, -1]
    logits = torch.cat([(W[i:i + 16384] @ h).float() for i in range(0, W.shape[0], 16384)])
    print("  top:", [ascii(tok.decode([t])) for t in logits.topk(6).indices.tolist()])
    te.close(); te.free_resident(); del te, m, lm, W; torch.xpu.empty_cache()
