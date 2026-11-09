"""Why is bm_manoharan distill val fail 1.00 on all three boards (2026-10-29 11:58)? Greedy vs sampled decoding of a
checkpoint on one val pool, CPU, 1 thread: decaps placed, skip share, exact-check success. Read-only diagnostic."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

torch.set_num_threads(1)
from decap_loop.bank import load_bank  # noqa: E402
from decap_loop.modal_env import make_env  # noqa: E402
from decap_loop.pdn import PDNConfig  # noqa: E402
from decap_loop.rollout import rollout  # noqa: E402
from train import make_policy  # noqa: E402

ck_path, pool = sys.argv[1], sys.argv[2]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
a = ck["args"]
vb, _ = load_bank(pool, torch.device("cpu"))
cfg = vb.cfg or PDNConfig(**ck["cfg"])
pol = make_policy(cfg, a)
pol.load_state_dict(ck["model"])
pol.eval()
print("ckpt iter", ck.get("iter"), "pool", os.path.basename(pool), "B", vb.B, "T", cfg.n_types)
with torch.no_grad():
    for mode in ("greedy", "sample"):
        env = make_env(vb, torch.arange(vb.B), cfg, a["feat"], allow_prune=False)
        g = torch.Generator().manual_seed(0)
        out = rollout(pol, env, greedy=(mode == "greedy"), amp=False, gen=g)
        acts = out["actions"]
        valid = acts >= 0
        stop = acts == env.stop_idx
        typ = torch.where(valid & ~stop, acts % (cfg.n_types + 1), torch.full_like(acts, -1))
        n_skip = int((typ == cfg.n_types).sum())
        n_place = int(((typ >= 0) & (typ < cfg.n_types)).sum())
        print(f"{mode:6s} steps {int(valid.sum())} placed {n_place} skip {n_skip} "
              f"count/prob {out['count'].float().mean():.2f} env-feasible {out['feasible'].float().mean():.2f}")
