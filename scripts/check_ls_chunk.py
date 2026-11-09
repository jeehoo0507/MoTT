"""11/05 LS16 memory fix check: local_search with --ls_chunk (memory only) must give the same solutions as the
recorded unchunked LS record of the same pool file (same ok / count / worst / occ per instance).
Run: .venv/bin/python scripts/check_ls_chunk.py <chunked record> <recorded record>"""
import json
import sys

import numpy as np

a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("ok", "count", "worst", "occ"):
    x, y = np.asarray(a[k], float), np.asarray(b[k], float)
    same = np.isclose(x, y, atol=1e-9).reshape(len(x), -1).all(1)
    print(f"{k:6s} same {int(same.sum())}/{len(same)}" + ("" if same.all() else f" differ at {np.flatnonzero(~same).tolist()}"))
print("rounds", a["rounds"], b["rounds"], "| F_final", a["F_final"], b["F_final"],
      "| peak_rss_mb", round(a["peak_rss_mb"]), round(b["peak_rss_mb"]), "| sec_total", round(a["sec_total"]), round(b["sec_total"]))
