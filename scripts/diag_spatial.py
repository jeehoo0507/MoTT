"""Spatial clustering of placed decaps per method (existing eval records only, no new runs).

Question (round 33, before looking): MoTT has no cell-cell attention (cells talk only through the modes, no
coordinates). If that is why r2 is the weakest rung on pcb in all 3 seeds and uses more decaps than r0 on pkg,
its placements should be spatially more clustered (decaps next to each other) than r0/r1, and clustering should
be higher on its failures. Metrics per instance with >= 2 decaps: fraction of decaps with another decap in the
8-neighbourhood (adj), mean nearest-neighbour Chebyshev distance (nn).
Usage: .venv/bin/python scripts/diag_spatial.py  -> runs/analysis/diag_spatial.json + printed table
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = {
    "r0_coord": ["runs/2026-10-28/r0_coord_rl", "runs/2026-10-29/r0_coord_s1_rl", "runs/2026-10-29/r0_coord_s2_rl"],
    "r1_modal": ["runs/2026-10-28/r1_modal_rl", "runs/2026-10-29/r1_modal_s1_rl", "runs/2026-10-29/r1_modal_s2_rl"],
    "r2_mott": ["runs/2026-10-28/r2_mott_rl", "runs/2026-10-29/r2_mott_s1_rl", "runs/2026-10-29/r2_mott_s2_rl"],
    "r3_loop": ["runs/2026-10-28/r3_loop_rl", "runs/2026-10-29/r3_loop_s1_rl", "runs/2026-10-29/r3_loop_s2_rl"],
}
BASE = ["greedy+prune", "modal_rule"]
SETS = ["eval_pcb10_0", "eval_pcb10_1", "eval_pkg10_0", "eval_pkg10_1", "eval_ip10_0", "eval_ip10_1"]


def inst_metrics(occ, W):
    idx = np.array([i for i, t in enumerate(occ) if t >= 0])
    if len(idx) < 2:
        return None
    r, c = idx // W, idx % W
    d = np.maximum(np.abs(r[:, None] - r[None]), np.abs(c[:, None] - c[None])).astype(float)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    return float((nn <= 1).mean()), float(nn.mean())


def load(path):
    d = json.load(open(path))
    W = d["size"]
    m = [inst_metrics(o, W) for o in d["occ"]]
    return m, list(map(bool, d["ok"]))


def boot(diff, B=2000, seed=0):
    diff = np.asarray(diff)
    if len(diff) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
    return [float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    out = {}
    for tech in ["pcb", "pkg", "ip"]:
        sets = [s for s in SETS if s.startswith(f"eval_{tech}")]
        rec = {}
        for name, runs in METHODS.items():
            for si, run in enumerate(runs):
                key = f"{name}_s{si}"
                ms, oks = [], []
                for s in sets:
                    p = os.path.join(ROOT, run, "eval", f"{s}__{os.path.basename(run)}.json")
                    if not os.path.exists(p):
                        ms, oks = None, None
                        break
                    m, ok = load(p)
                    ms += m
                    oks += ok
                if ms is not None:
                    rec[key] = (ms, oks)
        for b in BASE:
            ms, oks = [], []
            for s in sets:
                m, ok = load(os.path.join(ROOT, "runs/2026-10-28/baselines/eval", f"{s}__{b}.json"))
                ms += m
                oks += ok
            rec[b] = (ms, oks)
        res = {}
        for key, (ms, oks) in rec.items():
            adj = [m[0] for m in ms if m is not None]
            nn = [m[1] for m in ms if m is not None]
            adj_ok = [m[0] for m, o in zip(ms, oks) if m is not None and o]
            adj_f = [m[0] for m, o in zip(ms, oks) if m is not None and not o]
            res[key] = dict(n=len(adj), adj=float(np.mean(adj)), nn=float(np.mean(nn)),
                            adj_ok=float(np.mean(adj_ok)) if adj_ok else None,
                            adj_fail=float(np.mean(adj_f)) if adj_f else None, n_fail=len(adj_f))
        # paired r2 - r0 (same seed) on adj, instances where both have >= 2 decaps
        pair = {}
        for si in range(3):
            a, b = rec.get(f"r2_mott_s{si}"), rec.get(f"r0_coord_s{si}")
            if a and b:
                diff = [x[0] - y[0] for x, y in zip(a[0], b[0]) if x is not None and y is not None]
                pair[f"s{si}"] = boot(diff)
        out[tech] = dict(per_method=res, r2_minus_r0_adj=pair)
        print(f"== {tech}")
        for key, v in res.items():
            af = "-" if v["adj_fail"] is None else f"{v['adj_fail']:.3f}"
            ao = "-" if v["adj_ok"] is None else f"{v['adj_ok']:.3f}"
            print(f"  {key:16s} n {v['n']:3d} adj {v['adj']:.3f} nn {v['nn']:.2f} | adj ok {ao} fail {af} (n {v['n_fail']})")
        for s, ci in pair.items():
            print(f"  r2 - r0 adj {s}: {ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]")
    os.makedirs(os.path.join(ROOT, "runs/analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "runs/analysis/diag_spatial.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
