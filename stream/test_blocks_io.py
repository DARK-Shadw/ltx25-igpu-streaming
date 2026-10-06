"""Step 1: I/O-only throughput on the REAL transformer blocks."""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from blockstream import BlockStreamer, Plan, ShardIndex

idx = ShardIndex("models/ltx25/transformer")
groups = idx.group(r"(transformer_blocks\.\d+)\.")
keys = sorted(groups, key=lambda k: int(k.split(".")[1]))
plans = [Plan(groups[k]) for k in keys]
print(f"{len(plans)} blocks, max block read {max(p.nbytes for p in plans) / 2**20:.0f} MiB, "
      f"total {sum(p.nbytes for p in plans) / 2**30:.2f} GiB")

for slots in (2, 3):
    st = BlockStreamer(plans, slots=slots, passes=1)
    t = time.perf_counter()
    for _ in plans:
        s, w = st.next()
        torch.xpu.synchronize()
        st.release(s)
    dt = time.perf_counter() - t
    print(f"slots={slots}: {dt:.1f} s per full pass  -> {st.bytes_read / dt / 2**30:.2f} GiB/s  "
          f"({dt / len(plans) * 1000:.0f} ms/block)")
    # check a tensor survived the trip
    del st
