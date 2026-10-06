"""Stream real safetensors blocks from NVMe -> pinned RAM -> iGPU, with prefetch.

Index every tensor in a component's shards, group them by block prefix, and read each
group's byte range with unbuffered I/O (page-cache bypass, real NVMe speed).
"""
import ctypes
import glob
import json
import os
import queue
import re
import struct
import threading
from ctypes import wintypes

import torch

from dev import NAME as _DEVNAME, acc

ALIGN = 4096
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
      "I64": torch.int64, "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn}

# ---- Windows unbuffered reads ----
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
k32.CloseHandle.argtypes = [wintypes.HANDLE]


def _open_raw(path):
    h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000, None)  # NO_BUFFERING
    if h in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_last_error(), path)
    return h


def _read_range(h, start, nbytes, ptr):
    """Read [start, start+nbytes) (both ALIGN-multiples) into raw pointer ptr."""
    k32.SetFilePointerEx(h, start, None, 0)
    got, left = wintypes.DWORD(), nbytes
    while left:
        n = min(left, 64 << 20)
        if not k32.ReadFile(h, ptr, n, ctypes.byref(got), None) or got.value == 0:
            raise OSError(ctypes.get_last_error())
        if got.value < n:  # EOF inside the final aligned chunk
            return
        ptr += got.value
        left -= got.value


class TensorInfo:
    __slots__ = ("name", "file", "start", "end", "dtype", "shape")

    def __init__(s, name, file, start, end, dtype, shape):
        s.name, s.file, s.start, s.end, s.dtype, s.shape = name, file, start, end, dtype, shape


class ShardIndex:
    def __init__(self, comp_dir):
        self.tensors = {}
        for p in sorted(glob.glob(os.path.join(comp_dir, "*.safetensors"))):
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.tensors[k] = TensorInfo(k, p, 8 + n + a, 8 + n + b, v["dtype"], tuple(v["shape"]))

    def group(self, pattern):
        """Return {group_key: [TensorInfo]} for tensor names matching regex with one capture group."""
        rx, out = re.compile(pattern), {}
        for k, t in self.tensors.items():
            m = rx.match(k)
            if m:
                out.setdefault(m.group(1), []).append(t)
        return out


class Plan:
    """Aligned read plan for one group of tensors: list of (file, aligned_start, aligned_len, buf_off)
    and the tensor views inside the destination buffer."""

    def __init__(self, tensors):
        by_file = {}
        for t in tensors:
            by_file.setdefault(t.file, []).append(t)
        self.reads, self.views, buf_off = [], {}, 0
        GAP = 1 << 20  # merge tensors closer than this into one read
        for f, ts in by_file.items():
            ts = sorted(ts, key=lambda t: t.start)
            runs, cur = [], [ts[0]]
            for t in ts[1:]:
                if t.start - cur[-1].end <= GAP:
                    cur.append(t)
                else:
                    runs.append(cur)
                    cur = [t]
            runs.append(cur)
            for run in runs:
                lo, hi = run[0].start, run[-1].end
                a0 = lo // ALIGN * ALIGN
                a1 = (hi + ALIGN - 1) // ALIGN * ALIGN
                self.reads.append((f, a0, a1 - a0, buf_off))
                for t in run:
                    self.views[t.name] = (buf_off + t.start - a0, t.end - t.start, t.dtype, t.shape)
                buf_off += a1 - a0
        self.nbytes = buf_off
        # GPU kernels need aligned weights; tensors sit at arbitrary byte offsets in the shard files,
        # so each is re-packed into a 256-byte-aligned slot after the raw copy.
        self.aoff, a = {}, 0
        for name, (o, n, dt, shp) in self.views.items():
            self.aoff[name] = a
            a += (n + 255) // 256 * 256
        self.abytes = a
        self.transpose = set()  # names of 2-D Linear weights to store transposed ([out,in] -> [in,out]) for faster GEMMs


def aligned_unpack(raw, plan, abuf):
    """Copy each tensor from the raw (unaligned) buffer into abuf at aligned offsets; return typed views.
    Names in plan.transpose are stored transposed (their shape in the result is [in, out])."""
    out = {}
    for name, (o, n, dt, shp) in plan.views.items():
        ao = plan.aoff[name]
        if name in plan.transpose:
            tmp = torch.empty(n, dtype=torch.uint8, device=abuf.device)   # fresh allocation => aligned
            tmp.copy_(raw[o:o + n])
            dst = abuf[ao:ao + n].view(DT[dt]).view(shp[1], shp[0])
            dst.copy_(tmp.view(DT[dt]).view(shp).t())
            del tmp
            out[name] = dst
        else:
            abuf[ao:ao + n].copy_(raw[o:o + n])
            out[name] = abuf[ao:ao + n].view(DT[dt]).view(shp)
    return out


class BlockStreamer:
    """Cycles through `plans` (list of Plan) forever (or `passes` times); yields GPU tensor dicts."""

    def __init__(self, plans, device=_DEVNAME, slots=2, passes=1):
        self.plans, self.dev, self.passes = plans, torch.device(device), passes
        size = max(p.nbytes for p in plans)
        self.hosts = [torch.empty(size, dtype=torch.uint8).pin_memory() for _ in range(slots)]
        self.gbufs = [torch.empty(size, dtype=torch.uint8, device=self.dev) for _ in range(slots)]
        self.abuf = torch.empty(max(p.abytes for p in plans), dtype=torch.uint8, device=self.dev)
        self.free, self.ready = queue.Queue(), queue.Queue()
        for s in range(slots):
            self.free.put(s)
        self.stream = acc.Stream()
        self.stop = False
        self.bytes_read = 0
        self.t = threading.Thread(target=self._loader, daemon=True)
        self.t.start()

    def _loader(self):
        handles = {}
        try:
            for _ in range(self.passes):
                for plan in self.plans:
                    s = self.free.get()
                    if self.stop:
                        return
                    host = self.hosts[s]
                    for f, a0, ln, boff in plan.reads:
                        h = handles.get(f) or handles.setdefault(f, _open_raw(f))
                        _read_range(h, a0, ln, host.data_ptr() + boff)
                    with acc.stream(self.stream):
                        self.gbufs[s][:plan.nbytes].copy_(host[:plan.nbytes], non_blocking=True)
                    self.stream.synchronize()
                    self.bytes_read += plan.nbytes
                    self.ready.put((s, plan))
        except Exception as e:  # surface loader errors to the consumer
            self.ready.put((None, e))
        finally:
            for h in handles.values():
                k32.CloseHandle(h)

    def next(self):
        """-> (slot, {name: gpu_tensor}). Call release(slot) when the block is no longer needed."""
        s, plan = self.ready.get()
        if s is None:
            raise plan
        return s, aligned_unpack(self.gbufs[s], plan, self.abuf)

    def release(self, slot):
        self.free.put(slot)

    def close(self):
        self.stop = True
        self.free.put(0)
