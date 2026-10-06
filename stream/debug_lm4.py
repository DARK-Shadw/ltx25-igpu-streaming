import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
import runtime
from runtime import DEV, Streamed, build_empty
M = "models/ltx25"
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer"); tok.add_bos_token = True
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
for default in (torch.bfloat16,):
    m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=default), dtype=default)
    lm = m.model.language_model
    print(f"=== build default dtype {default}")
    print("  buffers:", [(n.split('.')[-1], str(b.dtype).replace('torch.', '')) for n, b in m.named_buffers() if 'layers.' not in n])
    te = Streamed(m, f"{M}/text_encoder", [f"model.language_model.layers.{i}" for i in range(48)])
    te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))
    W = lm.embed_tokens.weight
    for q in ["What is the capital of France?", "Name a color of the sky on a clear day.", "Count: 1, 2, 3, 4, 5,"]:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        ids = tok([prompt], return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        for step in range(5):  # greedy 5 tokens (no KV cache; tiny prompts)
            with torch.no_grad():
                out = m.model(input_ids=ids, use_cache=False)
            h = out.last_hidden_state[0, -1]
            logits = torch.cat([(W[i:i + 16384] @ h).float() for i in range(0, W.shape[0], 16384)])
            nxt = logits.argmax().view(1, 1)
            ids = torch.cat([ids, nxt], 1)
        print(ascii(q), "->", ascii(tok.decode(ids[0, -5:])))
    te.close(); te.free_resident(); del te, m, lm, W; torch.xpu.empty_cache()
