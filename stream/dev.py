"""Accelerator selection shared by every script: CUDA (NVIDIA) if available, else XPU (Intel Arc).
Override with LTX_DEVICE=cuda|xpu. `acc` is torch.cuda or torch.xpu (same API for synchronize/empty_cache/Stream/stream)."""
import os
import torch

NAME = os.environ.get("LTX_DEVICE") or ("cuda" if torch.cuda.is_available() else "xpu")
DEV = torch.device(NAME)
acc = getattr(torch, NAME)
MODELS = os.environ.get("LTX_MODELS", "models/ltx25")   # folder holding the downloaded checkpoint
