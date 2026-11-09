"""PROPOSALS 11/09 T: copy the first K problems of eval pool files into new files <set>_k<K>.pkl (originals untouched).
Every per-problem array / list (first dimension = number of problems B) is sliced; everything else is copied as is.
  .venv/bin/python scripts/speed_subset.py --pool runs/pool_v1 --sets eval_ip16_0,eval_pcb16_0,eval_pkg16_0 --k 10"""
import argparse
import os
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="runs/pool_v1")
ap.add_argument("--sets", required=True)
ap.add_argument("--k", type=int, default=10)
a = ap.parse_args()
for s in a.sets.split(","):
    src, dst = os.path.join(a.pool, s + ".pkl"), os.path.join(a.pool, f"{s}_k{a.k}.pkl")
    if os.path.exists(dst):
        print("exists", dst); continue
    d = pickle.load(open(src, "rb"))
    B = len(d["valid"])
    out, cut = {}, []
    for key, v in d.items():
        if not isinstance(v, (str, bytes, dict)) and hasattr(v, "__len__") and hasattr(v, "__getitem__") \
                and len(v) == B:          # numpy array, torch tensor or list with one entry per problem
            out[key] = v[:a.k]; cut.append(key)
        else:
            out[key] = v
    pickle.dump(out, open(dst, "wb"))
    print(f"{dst}: {B} -> {a.k} problems, sliced {cut}")
