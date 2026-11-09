"""Per-problem features of the 10×10 eval pools (QUEUE 2026-11-01 15:10 item 1: where does learning beat the rule?).

For every instance of runs/pool_v1/eval_*10_*.pkl, from the instance definition and the bare board only
(no method output): board shape, target kind, valid / keep-out / placeable cell counts, and on the pool's
working grid the bare-board max-over-probes log10(|Z_pp|/Zt): worst level, number of violating peaks
(local maxima above the target), log10 frequency of the worst peak, fraction of grid points violating.
Writes runs/analysis/problem_features.json {set: [row per instance]}; collect_results.py joins it with
the eval records. CPU, 1 thread, ~1 min.
Run: .venv/bin/python scripts/problem_features.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "runs/analysis/problem_features.json")


def features(path):
    from decap_loop.bank import load_bank
    from decap_loop.sim import EPS, probe_impedance
    b, _ = load_bank(path)
    inst = torch.arange(b.B)
    occ = torch.full((b.B, b.n), -1, dtype=torch.long)
    zpp = probe_impedance(b, inst, occ, b.zd("cpu"))                     # [B,F,P]
    lv = (torch.log10(zpp.abs() + EPS) - torch.log10(b.Zt)[..., None]).amax(-1).numpy()   # [B,F]
    placeable = b.placeable().cpu().numpy()
    rows = []
    for i in range(b.B):
        o = np.argsort(b.freqs[i])
        f, v = b.freqs[i][o], lv[i][o]
        pad = np.concatenate([[-np.inf], v, [-np.inf]])
        peak = (v >= pad[:-2]) & (v >= pad[2:])
        j = int(v.argmax())
        rows.append(dict(shape=b.shapes[i], target=b.targets[i], n_valid=int(b.valid[i].sum()),
                         n_keepout=int(b.keepout[i].sum()), n_placeable=int(placeable[i].sum()),
                         bare_worst=float(v[j]), log_f_worst=float(np.log10(f[j])),
                         n_viol_peaks=int((peak & (v > 0)).sum()), viol_frac=float((v > 0).mean())))
    return rows


def main():
    torch.set_num_threads(1)
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "runs/pool_v1/eval_*10_*.pkl"))):
        s = os.path.basename(p)[:-4]
        out[s] = features(p)
        print(s, len(out[s]), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=0)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
