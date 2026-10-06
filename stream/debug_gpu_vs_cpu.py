import gc, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import transformers
from transformers import AutoConfig, AutoTokenizer
import runtime
from runtime import Streamed, build_empty
M = "models/ltx25"
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer")
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
units = [f"model.language_model.layers.{i}" for i in range(48)]
SKIP = ("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder")

def run(dev):
    runtime.DEV = torch.device(dev)
    m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16))
    te = Streamed(m, f"{M}/text_encoder", units)
    te.load_resident(skip_prefixes=SKIP)
    ids = tok(["The capital of France is"], return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = m.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = [h.float().cpu() for h in out.hidden_states]
    te.close(); del te, m, out; gc.collect()
    if dev == "xpu": torch.xpu.empty_cache()
    return hs

g = run("xpu")
c = run("cpu")
for i, (a, b) in enumerate(zip(g, c)):
    if i in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 47, 48):
        d = (a - b).abs()
        print(f"hidden[{i:2d}] xpu_absmax={a.abs().max():8.2f} cpu_absmax={b.abs().max():8.2f}  rel_mean_diff={(d.mean() / b.abs().mean()).item():.4f}")
