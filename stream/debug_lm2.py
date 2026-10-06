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
lm = m.model.language_model
W = lm.embed_tokens.weight
print("final norm weight finite/absmean:", torch.isfinite(lm.norm.weight).all().item(), lm.norm.weight.float().abs().mean().item())
print("text cfg softcap:", getattr(tcfg.text_config, "final_logit_softcapping", None), "hidden:", tcfg.text_config.hidden_size,
      "layer_types sample:", tcfg.text_config.layer_types[:6] if hasattr(tcfg.text_config, "layer_types") else None)
def top(h, k=6):
    logits = torch.cat([(W[i:i + 16384] @ h).float() for i in range(0, W.shape[0], 16384)])
    return [tok.decode([t]) for t in logits.topk(k).indices.tolist()]
for text in ["The capital of France is", "Roses are red, violets are"]:
    ids = tok([text], return_tensors="pt").input_ids.to(DEV)
    print(repr(text), "ids:", ids[0].tolist())
    with torch.no_grad():
        out = m.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    print("  last_hidden_state :", top(out.last_hidden_state[0, -1]))
    print("  hs[48]            :", top(hs[48][0, -1]))
    print("  norm(hs[47])      :", top(lm.norm(hs[47])[0, -1]))
    print("  hs[48] equals last_hidden_state:", torch.equal(hs[48], out.last_hidden_state))
