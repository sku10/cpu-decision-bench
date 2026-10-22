#!/usr/bin/env python3
"""Per-layer byte footprint of a GGUF, from its tensor table."""
import collections, re, struct, sys

GT_SIZE = {  # ggml_type -> (block_elems, bytes_per_block) for the ones we need
    0:(1,4), 1:(1,2), 2:(32,18), 3:(32,20), 6:(32,22), 7:(32,24),
    8:(32,34), 9:(32,36), 10:(256,84), 11:(256,110), 12:(256,144),
    13:(256,176), 14:(256,210), 15:(256,292), 16:(256,66), 17:(256,74),
    18:(256,98), 19:(256,50), 20:(256,34), 24:(1,1), 25:(1,2), 26:(1,4),
    27:(1,8), 28:(1,8), 30:(1,2),
}

def read(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a gguf"
    ver, = struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))

    def rstr():
        n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
    def skip_val(t):
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8:   rstr()
        elif t == 9:
            et, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
            for _ in range(n): skip_val(et)
        else: f.read(sizes[t])

    meta = {}
    for _ in range(n_kv):
        k = rstr(); t, = struct.unpack("<I", f.read(4))
        if t in (4,5,10,11):           # ints we may want
            sizes = {4:("<I",4),5:("<i",4),10:("<Q",8),11:("<q",8)}
            fmt,sz = sizes[t]; meta[k] = struct.unpack(fmt, f.read(sz))[0]
        else: skip_val(t)

    per = collections.Counter(); other = 0; unknown = set()
    for _ in range(n_tensors):
        name = rstr()
        nd, = struct.unpack("<I", f.read(4))
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        gt, = struct.unpack("<I", f.read(4))
        struct.unpack("<Q", f.read(8))          # offset
        n_elem = 1
        for d in dims: n_elem *= d
        if gt in GT_SIZE:
            be, bb = GT_SIZE[gt]
            nbytes = (n_elem // be) * bb
        else:
            unknown.add(gt); nbytes = 0
        m = re.match(r"blk\.(\d+)\.", name)
        if m: per[int(m.group(1))] += nbytes
        else: other += nbytes
    return meta, per, other, unknown

if __name__ == "__main__":
    for path in sys.argv[1:]:
        meta, per, other, unknown = read(path)
        if not per:
            print(f"{path}: no blk.* tensors found"); continue
        sizes = sorted(per.values())
        tot = sum(per.values()) + other
        name = path.split("/")[-1]
        print(f"{name}")
        print(f"  layers            : {len(per)}")
        print(f"  per-layer (median): {sizes[len(sizes)//2]/2**20:8.1f} MiB")
        print(f"  per-layer (max)   : {sizes[-1]/2**20:8.1f} MiB")
        print(f"  non-layer (embed) : {other/2**20:8.1f} MiB")
        print(f"  total tensors     : {tot/2**30:8.2f} GiB")
        if unknown: print(f"  (unknown ggml types, counted 0: {sorted(unknown)})")
        print()
