"""Time decap_loop.baselines.genetic per instance per generation on one 10-cell eval file per board type (11/02).

Decides the GA budget for the 10-cell GA baseline before queueing it: runs genetic with gens 0 and 3 (pop 64) on
the working grid (no cutting plane, no exact check; timing only) and prints sec/inst/gen and the 100-gen estimate.
One thread, run as a 4th-slot job (scripts/side_slot.py). Writes runs/analysis/smoke_ga.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

torch.set_num_threads(1)
from decap_loop.bank import load_bank  # noqa: E402
from decap_loop.baselines import genetic  # noqa: E402

out = {}
for s in ["eval_ip10_0", "eval_pcb10_0", "eval_pkg10_0"]:
    b, _ = load_bank(f"runs/pool_v1/{s}.pkl")
    t = {}
    for gens in (0, 3):
        t0 = time.time()
        r = genetic(b, b.cfg, pop=64, gens=gens, seed=0)
        t[gens] = time.time() - t0
        print(s, "gens", gens, f"{t[gens]:.1f}s", "grid-feasible", float(r["feasible"].float().mean()),
              "count", float(r["count"].float().mean()), flush=True)
    per = (t[3] - t[0]) / 3 / b.B
    out[s] = dict(B=int(b.B), sec_gens0=t[0], sec_gens3=t[3], sec_per_inst_gen=per,
                  est_sec_per_inst_100gens=t[0] / b.B + 100 * per)
    print(s, out[s], flush=True)
os.makedirs("runs/analysis", exist_ok=True)
json.dump(out, open("runs/analysis/smoke_ga.json", "w"), indent=1)
print("done")
