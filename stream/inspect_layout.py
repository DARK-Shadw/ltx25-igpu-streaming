import glob, json, struct, re, collections, sys
def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n
for comp in sys.argv[1:]:
    tot = collections.Counter(); groups = collections.defaultdict(lambda: [0, 0])
    for p in sorted(glob.glob(f"models/ltx25/{comp}/*.safetensors")):
        h, _ = header(p)
        for k, v in h.items():
            if k == "__metadata__": continue
            n = v["data_offsets"][1] - v["data_offsets"][0]
            m = re.match(r"(.*?\.\d+)\.", k)
            g = m.group(1) if m else "(other) " + k.split(".")[0]
            groups[g][0] += n; groups[g][1] += 1; tot[v["dtype"]] += n
    print(f"== {comp}: {sum(tot.values())/2**30:.2f} GiB, dtypes {dict((k, round(v/2**30,2)) for k,v in tot.items())}")
    items = list(groups.items())
    for g, (n, c) in items[:3] + items[-4:]:
        print(f"   {g:45s} {n/2**20:8.1f} MiB  {c} tensors")
    print(f"   groups: {len(items)}")
