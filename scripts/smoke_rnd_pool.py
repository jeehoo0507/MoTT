"""Smoke for the randomized-physics kind "rnd" (PROPOSALS 11/07 R; decap_loop/techs.sample_rnd, build_pools 'rnd'), CPU.
1. sample_rnd is deterministic per seed, never equal to a preset, and spans the PCB-IP range (prints 5 draws).
2. build_pools.build() makes one small rnd file (8x8, 4 boards) in a temp dir; load_bank restores its own cfg;
   PkEnv + untrained MoTT+P1 greedy rollout runs end to end on it.
3. run_rung's "--rl_pool_glob" arm token is stripped from the arm and reaches only the rl command (string check).
Run: python scripts/smoke_rnd_pool.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
from decap_loop.bank import load_bank  # noqa: E402
from decap_loop.peak_env import PkEnv  # noqa: E402
from decap_loop.rollout import rollout  # noqa: E402
from decap_loop.techs import IP, PCB, PKG, sample_rnd  # noqa: E402
import build_pools  # noqa: E402
import train  # noqa: E402

# 1. draws
for i in range(5):
    cfg, diff = sample_rnd(np.random.default_rng(400000 + i))
    cfg2, _ = sample_rnd(np.random.default_rng(400000 + i))
    assert cfg == cfg2, "sample_rnd not deterministic"
    assert cfg not in (PCB, PKG, IP)
    srf = [1 / (2 * np.pi * np.sqrt(l * c)) / 1e9 for c, l, _ in cfg.decaps]
    print(f"1. draw {i}: cell {cfg.cell_size*1e6:5.0f}um diel {cfg.diel_thickness*1e6:5.2f}um metal {cfg.metal_thickness*1e6:4.2f}um "
          f"vrm_l {cfg.vrm_l*1e12:4.0f}pH C {[f'{c*1e12:.0f}p' for c, _, _ in cfg.decaps]} SRF {[f'{s:.1f}G' for s in srf]} "
          f"diff {diff}", flush=True)
# 2. one small file through build_pools.build
with tempfile.TemporaryDirectory() as td:
    name, status, dt, meta = build_pools.build((td, "train_rnd8_0", "rnd", 8, 4, build_pools.SEED["rnd"] + 800))
    assert status == "built", status
    print(f"2. built {name} in {dt:.0f}s: g+p ok {meta['gp_ok']:.2f} count {meta['gp_count']} F {meta['F']} diff {meta['difficulty']}", flush=True)
    batch, res = load_bank(os.path.join(td, name + ".pkl"))
    cfg_file = batch.cfg
    assert abs(cfg_file.cell_size - meta["cfg"]["cell_size"]) < 1e-12, "cfg not restored from the pool file"
    env = PkEnv(batch, torch.arange(batch.B), cfg_file, feat_mode="mott_pk", max_steps=40)
    pol = train.make_policy(cfg_file, dict(model="mott", d=32, layers=2, heads=4, peak_refine=True, feat="mott_pk"))
    out = rollout(pol, env, greedy=True, amp=False)
    print("   greedy rollout on the rnd file: count", out["count"].tolist(), "feasible", out["feasible"].tolist(), flush=True)
# 3. run_rung token handling (same code path, re-implemented check on the parsed strings)
arm = "--model mott --peak_refine --pool_glob train_pcb*.pkl,train_ip*.pkl --rl_pool_glob train_pcb*.pkl,train_ip*.pkl,train_rnd*.pkl"
toks = arm.split()
i = toks.index("--rl_pool_glob")
rl_extra, arm2 = f" --pool_glob {toks[i + 1]}", " ".join(toks[:i] + toks[i + 2:])
assert "--rl_pool_glob" not in arm2 and "train_rnd" not in arm2 and "train_rnd*.pkl" in rl_extra
src = open(os.path.join(ROOT, "scripts/run_rung.py")).read()
assert '--out {rl}{rl_extra}' in src and 'stage distill {a.arm}' in src and '{rl_extra}' not in src.split("stage distill")[1].split("\n")[0]
print("3. --rl_pool_glob reaches only the rl stage")
open("runs/analysis/smoke_rnd_pool.ok", "w").write("ok\n") if os.path.isdir("runs/analysis") else None
print("OK")
