"""Glue: build a HF/diffusers model with NO weights in memory, then stream weights per unit
(block) from NVMe via forward hooks.

  build_empty(cls, config)          -> model with meta params, real (small) buffers
  Streamed(model, comp_dir, units)  -> installs hooks; units are submodule paths streamed per call
  Streamed.load_resident()          -> loads every param NOT inside a unit onto the GPU, permanently
"""
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights

from blockstream import BlockStreamer, DT, Plan, aligned_unpack, ShardIndex, _open_raw, _read_range, k32

from dev import DEV, acc


_STAGE = None


def _stage(size=64 << 20):
    """One shared pinned staging buffer for all chunked reads (pinned memory is never returned to the OS)."""
    global _STAGE
    if _STAGE is None:
        _STAGE = torch.empty(size, dtype=torch.uint8).pin_memory()
    return _STAGE


def build_empty(build_fn, dtype=torch.bfloat16):
    """build_fn() constructs the model; params land on meta, buffers stay real, default dtype bf16."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with init_empty_weights(include_buffers=False):
            m = build_fn()
    finally:
        torch.set_default_dtype(old)
    for mod in m.modules():  # move small real buffers to the GPU
        for k, b in list(mod._buffers.items()):
            if b is not None and b.device.type != "meta":
                mod._buffers[k] = b.to(DEV)
    return m


def _linear_fwd(self, x):
    """nn.Linear.forward that can use a pre-transposed weight [in, out]: x @ Wt is ~9% faster than x @ W.T on this iGPU."""
    if getattr(self, "_transposed", False):
        w = self.weight
        sh = x.shape
        x2 = x.reshape(-1, sh[-1])
        y = torch.addmm(self.bias, x2, w) if self.bias is not None else x2 @ w
        return y.reshape(*sh[:-1], w.shape[1])
    return F.linear(x, self.weight, self.bias)


def _set_param(mod_root, name, tensor):
    path, _, leaf = name.rpartition(".")
    sub = mod_root.get_submodule(path) if path else mod_root
    sub._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)


class Streamed:
    def __init__(self, model, comp_dir, units, prefetch_passes=0, slots=2, log=None, transpose_linear=False):
        self.model, self.units = model, units
        self.idx = ShardIndex(comp_dir)
        self.unit_names = list(units)
        self.plans = []
        for u in units:
            ts = [t for k, t in self.idx.tensors.items() if k.startswith(u + ".")]
            if not ts:
                raise KeyError(f"no tensors for unit {u}")
            self.plans.append(Plan(ts))
        self.log = log or (lambda *a: None)
        self.transposed_names = set()
        if transpose_linear:
            for plan in self.plans:
                for name, (o, n, dt, shp) in plan.views.items():
                    path, _, leaf = name.rpartition(".")
                    if leaf != "weight" or len(shp) != 2:
                        continue
                    sub = model.get_submodule(path)
                    if isinstance(sub, nn.Linear):
                        plan.transpose.add(name)
                        sub.forward = types.MethodType(_linear_fwd, sub)
                        sub._transposed = False
                        self.transposed_names.add(name)
        self.prefetch = prefetch_passes > 0
        self.bytes = 0
        self.wait_s = 0.0
        self._cursor = 0
        if self.prefetch:
            self.streamer = BlockStreamer(self.plans, slots=slots, passes=prefetch_passes)
        else:
            self.abuf = torch.empty(max(p.abytes for p in self.plans), dtype=torch.uint8, device=DEV)
            self.handles = {}
        self.hooks = []
        for i, u in enumerate(units):
            mod = model.get_submodule(u)
            mod._unit_i = i
            self.hooks.append(mod.register_forward_pre_hook(self._pre))
            self.hooks.append(mod.register_forward_hook(self._post))
        self._orig = {}

    # --- hooks ---
    def _pre(self, mod, args, kwargs=None):
        i = mod._unit_i
        u, plan = self.unit_names[i], self.plans[i]
        t0 = time.perf_counter()
        if self.prefetch:
            assert i == self._cursor % len(self.plans), f"out-of-order unit {u}"
            self._cursor += 1
            slot, tensors = self.streamer.next()
            mod._slot = slot
        else:
            tensors = self._load_unit_direct(plan)
        self.wait_s += time.perf_counter() - t0
        self.bytes += plan.nbytes
        for name, t in tensors.items():
            local = name[len(u) + 1:]
            path, _, leaf = local.rpartition(".")
            sub = mod.get_submodule(path) if path else mod
            if t.dtype == torch.float32:  # match from_pretrained(torch_dtype=bf16)
                t = t.to(torch.bfloat16)
            if leaf in sub._parameters:
                self._orig.setdefault((id(sub), leaf), sub._parameters[leaf])
                sub._parameters[leaf] = nn.Parameter(t, requires_grad=False)
                if name in self.transposed_names:
                    sub._transposed = True
            else:  # registered buffer stored in the checkpoint (e.g. layer_scalar)
                self._orig.setdefault((id(sub), leaf), sub._buffers[leaf])
                sub._buffers[leaf] = t.clone()
        mod._loaded = list(tensors)

    def _post(self, mod, args, output=None, *rest):
        acc.synchronize()  # compute on this unit's weights must finish before the slot is reused
        u = self.unit_names[mod._unit_i]
        for name in mod._loaded:
            local = name[len(u) + 1:]
            path, _, leaf = local.rpartition(".")
            sub = mod.get_submodule(path) if path else mod
            if leaf in sub._parameters:
                sub._parameters[leaf] = self._orig[(id(sub), leaf)]
                if name in self.transposed_names:
                    sub._transposed = False
            else:
                sub._buffers[leaf] = self._orig[(id(sub), leaf)]
        if self.prefetch:
            self.streamer.release(mod._slot)

    # --- resident (non-streamed) weights ---
    def load_resident(self, skip_prefixes=()):
        """Load every still-meta param not under a streamed unit directly to the GPU."""
        n = 0
        prefixes = tuple(u + "." for u in self.unit_names)
        host = {}
        for name, p in self.model.named_parameters():
            if p.device.type != "meta" or name.startswith(prefixes) or name.startswith(tuple(skip_prefixes)):
                continue
            t = self.idx.tensors.get(name)
            if t is None:
                self.log(f"  [warn] no checkpoint tensor for {name}")
                continue
            data = self._read_to_gpu(t)
            if data.dtype == torch.float32:  # match from_pretrained(torch_dtype=bf16)
                data = data.to(torch.bfloat16)
            _set_param(self.model, name, data)
            n += data.numel() * data.element_size()
        return n

    def _load_unit_direct(self, plan):
        """Read each tensor of a unit straight into its 256B-aligned slot (no raw staging copy on the GPU)."""
        stage = _stage()
        CH = stage.numel()
        out = {}
        for name, (o, n, dt, shp) in plan.views.items():
            t = self.idx.tensors[name]
            h = self.handles.get(t.file) or self.handles.setdefault(t.file, _open_raw(t.file))
            ao = plan.aoff[name]
            pos = t.start // 4096 * 4096
            end = (t.end + 4095) // 4096 * 4096
            while pos < end:
                m = min(CH, end - pos)
                _read_range(h, pos, m, stage.data_ptr())
                lo, hi = max(pos, t.start), min(pos + m, t.end)
                if hi > lo:
                    self.abuf[ao + lo - t.start:ao + hi - t.start].copy_(stage[lo - pos:hi - pos])
                pos += m
            if name in plan.transpose:
                tmp = self.abuf[ao:ao + n].clone()   # fresh aligned copy of the raw bytes
                dst = self.abuf[ao:ao + n].view(DT[dt]).view(shp[1], shp[0])
                dst.copy_(tmp.view(DT[dt]).view(shp).t())
                del tmp
                out[name] = dst
            else:
                out[name] = self.abuf[ao:ao + n].view(DT[dt]).view(shp)
        return out

    def _read_to_gpu(self, t):
        """Read one tensor from disk straight into a GPU tensor via a small reusable staging buffer."""
        stage = _stage()
        CH = stage.numel()
        plan = Plan([t])
        f, a0, ln, _ = plan.reads[0]
        o, nb, dt, shp = plan.views[t.name]
        g = torch.empty(ln, dtype=torch.uint8, device=DEV)
        h = _open_raw(f)
        try:
            for off in range(0, ln, CH):
                n = min(CH, ln - off)
                _read_range(h, a0 + off, n, stage.data_ptr())
                g[off:off + n].copy_(stage[:n])
        finally:
            k32.CloseHandle(h)
        dst = torch.empty(nb, dtype=torch.uint8, device=DEV)  # fresh allocation = aligned
        dst.copy_(g[o:o + nb])
        del g
        return dst.view(DT[dt]).view(shp)

    def free_resident(self):
        prefixes = tuple(u + "." for u in self.unit_names)
        for name, p in list(self.model.named_parameters()):
            if p.device.type != "meta" and not name.startswith(prefixes):
                path, _, leaf = name.rpartition(".")
                sub = self.model.get_submodule(path) if path else self.model
                sub._parameters[leaf] = nn.Parameter(torch.empty(p.shape, dtype=p.dtype, device="meta"),
                                                     requires_grad=False)
        acc.empty_cache()

    def close(self):
        """Remove hooks and release every buffer this streamer owns."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        if self.prefetch:
            self.streamer.close()
            self.streamer.hosts, self.streamer.gbufs, self.streamer.abuf = [], [], None
            self.streamer = None
        else:
            for h in self.handles.values():
                k32.CloseHandle(h)
            self.handles = {}
            self.abuf = None
        self._orig = {}
