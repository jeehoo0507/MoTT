"""Smoke for the zero-shot kind `chip` before its eval jobs run (PROGRESS 11/03 round 25): a 2-problem 8-cell
chip bank (cutting plane, like build_pools.py) in runs/analysis/smoke_zs_chip/, then the caller runs
eval_policy.py on it with the four ladder checkpoints. Only checks that pool building, modal features and every
policy run on chip without errors; the numbers are not reported.
Run: .venv/bin/python scripts/smoke_zs_chip.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from decap_loop.bank import save_bank
    from decap_loop.baselines import greedy
    from decap_loop.pdn import sample_batch
    from decap_loop.techs import CHIP, HARD
    from decap_loop.verify import cutting_plane
    out = "runs/analysis/smoke_zs_chip"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(5)
    b = sample_batch(2, 8, 8, CHIP, np.random.default_rng(5), difficulty=HARD)
    r, info = cutting_plane(lambda bb: greedy(bb, CHIP), b, CHIP, max_rounds=10)
    ok = info["worst"] <= np.log10(1.01)
    cnt = (r["occ"] >= 0).sum(1).cpu().numpy()
    save_bank(os.path.join(out, "eval_chip8_0.pkl"), b,
              {"greedy+prune": dict(occ=r["occ"].cpu().numpy(), worst=info["worst"], ok=ok, count=cnt)})
    print("built", ok.tolist(), cnt.tolist(), "rounds", info["rounds"], flush=True)


if __name__ == "__main__":
    main()
