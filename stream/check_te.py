import sys, torch
sys.path.insert(0, "stream")
from runtime import build_empty
from blockstream import ShardIndex
import transformers
from transformers import AutoConfig
tc = AutoConfig.from_pretrained("models/ltx25/text_encoder")
m = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tc, dtype=torch.bfloat16))
idx = ShardIndex("models/ltx25/text_encoder")
mk = {k for k, _ in m.named_parameters()}; bk = {k for k, _ in m.named_buffers()}; ck = set(idx.tensors)
print("ckpt-only not layer_scalar:", sorted(k for k in ck - mk if "layer_scalar" not in k))
print("ckpt-only that are model buffers:", len([k for k in ck - mk if k in bk]))
print("model-only:", sorted(mk - ck))
print("buffers in layer0:", [(k, v.shape, v.dtype) for k, v in m.model.language_model.layers[0].named_buffers()])
print("model-level buffers:", [(k, tuple(v.shape)) for k, v in m.named_buffers() if "layers." not in k][:10])
print(type(m.model.language_model.layers[0]).__name__, [n for n, _ in m.named_children()], [n for n,_ in m.model.named_children()])
print(m.config.text_config.num_hidden_layers if hasattr(m.config,'text_config') else '')
