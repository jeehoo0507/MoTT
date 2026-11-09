"""Check that the H1 val files (val_*10_1, _2) share no instance with any other 10x10 pool file (train, eval, val_*_0).

An instance is keyed by its valid mask, keep-out mask, probe/entry ports and target (bytes of each array).
Run: .venv/bin/python scripts/check_val_overlap.py
"""
import glob
import hashlib
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "runs/pool_v1")


def keys(path):
    d = pickle.load(open(path, "rb"))
    n = len(d["targets"]) if hasattr(d["targets"], "__len__") else 0
    out = set()
    for i in range(n):
        h = hashlib.sha1()
        for k in ("valid", "keepout", "probes", "entry", "targets"):
            h.update(np.ascontiguousarray(np.asarray(d[k][i])).tobytes())
        out.add(h.hexdigest())
    return out, n


new = sorted(glob.glob(os.path.join(POOL, "val_*10_[12].pkl")))
others = [p for p in sorted(glob.glob(os.path.join(POOL, "*.pkl"))) if p not in new]
ok = True
ks = {p: keys(p) for p in new + others}
for p in new:
    kp, n = ks[p]
    print(f"{os.path.basename(p)}: {n} instances, {len(kp)} distinct")
    for q in new + others:
        if q == p:
            continue
        both = kp & ks[q][0]
        if both:
            ok = False
            print(f"  OVERLAP with {os.path.basename(q)}: {len(both)}")
print("no overlap" if ok else "OVERLAP FOUND", f"({len(new)} new files vs {len(others)} others)")
