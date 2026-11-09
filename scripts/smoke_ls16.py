"""Time the LS baseline's local_search part on side-16 eval boards before queueing LS16 for Fig 2 (11/02 round 15).

eval_policy's local_search = greedy+prune then local_search (4 passes x 32 trials) inside the cutting plane; the
greedy+prune 16-cell records already give the first part, so this times only local_search on the working grid (no
cutting plane, no exact check; timing only) on the first instance of one 16-cell eval file per board type.
One thread, run as a 4th-slot filler (scripts/side_slot.py, "ls10 " prefix like the GA smoke).
Writes runs/analysis/smoke_ls16.json.
"""
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

torch.set_num_threads(1)
from decap_loop.bank import load_bank  # noqa: E402
from decap_loop.baselines import greedy, local_search  # noqa: E402

out = {}
for s in ["eval_ip16_0", "eval_pcb16_0", "eval_pkg16_0"]:
    t0 = time.time()
    b, _ = load_bank(f"runs/pool_v1/{s}.pkl")
    t_load = time.time() - t0
    t0 = time.time()
    g = greedy(b, b.cfg, inst=torch.tensor([0]))
    t_gp = time.time() - t0
    occ = torch.full((b.B, g["occ"].shape[1]), -1, dtype=g["occ"].dtype)
    occ[0] = g["occ"][0]
    feas = torch.zeros(b.B, dtype=torch.bool)
    feas[0] = g["feasible"][0]
    t0 = time.time()
    occ2 = local_search(b, b.cfg, occ, feas, passes=4, trials=32)
    t_ls = time.time() - t0
    out[s] = dict(B=int(b.B), sec_load=t_load, sec_gp_inst0=t_gp, sec_ls_inst0=t_ls,
                  gp_feasible=bool(feas[0]), gp_count=int((occ[0] >= 0).sum()), ls_count=int((occ2[0] >= 0).sum()),
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(s, out[s], flush=True)
    del b
os.makedirs("runs/analysis", exist_ok=True)
json.dump(out, open("runs/analysis/smoke_ls16.json", "w"), indent=1)
print("done")
