import sys, json, torch
sys.path.insert(0, "stream")
from runtime import build_empty
from blockstream import ShardIndex
import transformers, diffusers
from transformers import AutoConfig
from diffusers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

def cmp(name, model, comp):
    idx = ShardIndex(f"models/ltx25/{comp}")
    mk = {k for k, _ in model.named_parameters()}
    ck = set(idx.tensors)
    print(f"== {name}: model params {len(mk)}, ckpt tensors {len(ck)}, "
          f"in model not ckpt {len(mk-ck)}, in ckpt not model {len(ck-mk)}")
    print("   model-only:", sorted(mk - ck)[:4]); print("   ckpt-only :", sorted(ck - mk)[:4])

cfg = LTX2VideoTransformer3DModel.load_config("models/ltx25/transformer")
m = build_empty(lambda: LTX2VideoTransformer3DModel.from_config(cfg)); cmp("transformer", m, "transformer")
print("dtype", m.dtype)
cfg = LTX2TextConnectors.load_config("models/ltx25/connectors")
m = build_empty(lambda: LTX2TextConnectors.from_config(cfg)); cmp("connectors", m, "connectors")
tc = AutoConfig.from_pretrained("models/ltx25/text_encoder")
print("text cfg:", type(tc).__name__)
cls = transformers.Gemma4UnifiedForConditionalGeneration
m = build_empty(lambda: cls._from_config(tc, dtype=torch.bfloat16)); cmp("text_encoder", m, "text_encoder")
print("dtype", m.dtype)
