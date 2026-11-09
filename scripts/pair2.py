"""Paired comparison of two methods on the same eval files (read only): success diff and count ratio (both ok),
each with the paired bootstrap 95% CI of collect_results.boot_ci.

Run: .venv/bin/python scripts/pair2.py 'runs/2026-10-29/lo_ip_r2pk_s1_rl/eval/eval_ip10_*.json' \
         'runs/2026-10-29/lo_ip_r0_s1_rl/eval/eval_ip10_*.json'
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import boot_ci  # noqa: E402


def load(pat):
    return {r["set"]: r for r in (json.load(open(f)) for f in sorted(glob.glob(pat)))}


a, b = load(sys.argv[1]), load(sys.argv[2])
sets = sorted(set(a) & set(b))
ok_a = np.concatenate([np.asarray(a[s]["ok"], bool) for s in sets])
ok_b = np.concatenate([np.asarray(b[s]["ok"], bool) for s in sets])
c_a = np.concatenate([np.asarray(a[s]["count"], float) for s in sets])
c_b = np.concatenate([np.asarray(b[s]["count"], float) for s in sets])
n = len(ok_a)
d = ok_a.astype(float) - ok_b
lo, hi = boot_ci(lambda i: d[i].mean(), n)
print(f"sets {sets} n={n}: success {ok_a.mean():.2f} vs {ok_b.mean():.2f}, diff {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
both = ok_a & ok_b
r = c_a[both] / np.maximum(c_b[both], 1)
if both.any():
    lo, hi = boot_ci(lambda i: r[i].mean(), len(r))
    print(f"count ratio (both ok, n={both.sum()}): {r.mean():.3f} [{lo:.3f}, {hi:.3f}]; "
          f"mean count {c_a[both].mean():.2f} vs {c_b[both].mean():.2f}")
print("only A ok:", int((ok_a & ~ok_b).sum()), " only B ok:", int((~ok_a & ok_b).sum()))
