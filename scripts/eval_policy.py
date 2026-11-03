"""Certified evaluation on the held-out pools: classical baselines and trained policies.

Every method runs inside verify.cutting_plane on a fresh copy of the pool (same rule for all: re-run
on a grown grid until its claimed solutions are certified or the round cap is hit); success is the
exact dense check (worst |Z| <= 1.01 Zt). One worker process per (pool file, method), torch
single-threaded (decap_loop/safe_verify.py), so timings are "one CPU core, same calculator".

Records -> <out>/eval/<set>__<method>.json:
  {set, tech, size, n, method, ok[], count[], worst[], sec_per_inst, sec_method_last_round, rounds,
   F_final, ckpt, git, time}
RESULTS.md is generated from these by scripts/collect_results.py.

Run:
  .venv/bin/python scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*' --methods gp,ls --out runs/2026-10-28/baselines
  .venv/bin/python scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*' --policy r1=runs/2026-10-28/r1_modal_rl/best.pt --out runs/2026-10-28/r1_modal_rl
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def parse_set(name):
    m = re.match(r"(eval|val|test)_([a-z]+)(\d+)_(\d+)", name)
    return (m.group(2), int(m.group(3))) if m else ("?", 0)


def job(args):
    path, method, spec, out, rounds, samples, ls_cfg, device, inst_chunk, ga_cfg, repair, start = args
    import resource
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from decap_loop.bank import load_bank
    from decap_loop.baselines import genetic, greedy, local_search, modal_heuristic
    from decap_loop.modal_env import make_env
    from decap_loop.rollout import rollout
    from decap_loop.sim import FEAS_TOL
    from decap_loop.verify import certify, cutting_plane, grid_worst
    from train import make_policy
    name = os.path.splitext(os.path.basename(path))[0]
    tech, size = parse_set(name)
    if start == "rule" and spec:                    # trained policies only; baselines in the same call run as usual
        method = f"modal_rule>{method}"
    rec_method = f"{method}+rep{repair}" if repair else method
    dst = os.path.join(out, "eval", f"{name}__{rec_method}.json")
    if os.path.exists(dst):
        return name, method, "exists", None
    b, _ = load_bank(path)                       # fresh copy of the grid for this method
    cfg = b.cfg
    box = {}

    def timed(fn):
        def run(bb):
            t = time.time()
            r = fn(bb)
            box["last"] = time.time() - t
            return r
        return run

    cont = None                                   # repair stage: continue the method from a placement
    if method == "greedy+prune":
        fn = lambda bb: greedy(bb, cfg)  # noqa: E731
        cont = lambda bb, inst, occ0: greedy(bb, cfg, inst=inst, occ0=occ0)  # noqa: E731
    elif method == "greedy":
        fn = lambda bb: greedy(bb, cfg, prune=False)  # noqa: E731
    elif method == "modal_rule":
        fn = lambda bb: modal_heuristic(bb, cfg)  # noqa: E731
        cont = lambda bb, inst, occ0: modal_heuristic(bb, cfg, inst=inst, occ0=occ0)  # noqa: E731
    elif method == "genetic":                     # 11/02: decap_loop.baselines.genetic, no seed solution
        pop, gens = ga_cfg
        fn = lambda bb: genetic(bb, cfg, pop=pop, gens=gens, seed=0)  # noqa: E731
    elif method == "local_search":
        passes, trials = ls_cfg

        def fn(bb):
            g = greedy(bb, cfg)
            occ = local_search(bb, cfg, g["occ"], g["feasible"], passes=passes, trials=trials)
            inst = torch.arange(bb.B)
            return dict(occ=occ, feasible=torch.from_numpy(grid_worst(bb, inst, occ, cfg) <= FEAS_TOL))
    else:                                         # trained policy: method = "<name>" or "<name>@S"
        ck = torch.load(spec, map_location="cpu", weights_only=False)
        a = ck["args"]
        dev = torch.device(device)
        pol = make_policy(cfg, a).to(dev)
        pol.load_state_dict(ck["model"])
        pol.eval()
        if hasattr(pol, "set_eval_R") and "@R" in method:
            pol.set_eval_R(int(method.split("@R")[1]))
        S = samples

        @torch.no_grad()
        def fn(bb):
            if S <= 1:
                # greedy decoding is per instance, so rolling out inst_chunk instances at a time gives the same
                # solutions with a smaller peak (side 16: ~22 GB per worker for 25 at once, 10/29 01:50)
                step = inst_chunk if inst_chunk > 0 else bb.B
                occ, feas = [], []
                for i in range(0, bb.B, step):
                    env = make_env(bb, torch.arange(i, min(i + step, bb.B)), cfg, a["feat"])
                    o = rollout(pol, env, greedy=True, amp=False)
                    occ.append(o["occ"])
                    feas.append(o["feasible"])
                return dict(occ=torch.cat(occ), feasible=torch.cat(feas))
            inst = torch.arange(bb.B).repeat_interleave(S)
            env = make_env(bb, inst, cfg, a["feat"])
            o = rollout(pol, env, amp=False, gen=torch.Generator().manual_seed(0))
            key = (o["count"] + (~o["feasible"]) * 10 ** 6).view(bb.B, S)
            j = key.argmin(1) + torch.arange(bb.B) * S
            return dict(occ=o["occ"][j], feasible=o["feasible"][j])

        if start == "rule":                       # (only reached for trained policies)
            # 11/03 T4 (PROPOSALS, QUEUE 11/03 obs. 3): modal_rule's solution (already pruned like g+p) is the
            # start; the policy continues from it with greedy decoding (on a met target the env only allows
            # remove / change type / stop). Inside the cutting-plane, so every round re-runs rule -> policy.
            assert S <= 1, "--start rule: greedy decoding only"

            @torch.no_grad()
            def fn(bb):
                occ0 = modal_heuristic(bb, cfg)["occ"]
                step = inst_chunk if inst_chunk > 0 else bb.B
                occ, feas = [], []
                for i in range(0, bb.B, step):
                    env = make_env(bb, torch.arange(i, min(i + step, bb.B)), cfg, a["feat"], occ0=occ0[i:i + step])
                    o = rollout(pol, env, greedy=True, amp=False)
                    occ.append(o["occ"])
                    feas.append(o["feasible"])
                return dict(occ=torch.cat(occ), feasible=torch.cat(feas))

        if S <= 1:
            @torch.no_grad()
            def cont(bb, inst, occ0):
                step = inst_chunk if inst_chunk > 0 else len(inst)
                occ = []
                for i in range(0, len(inst), step):
                    env = make_env(bb, inst[i:i + step], cfg, a["feat"], occ0=occ0[i:i + step])
                    occ.append(rollout(pol, env, greedy=True, amp=False)["occ"])
                return dict(occ=torch.cat(occ))
    t0 = time.time()
    r, info = cutting_plane(timed(fn), b, cfg, max_rounds=rounds)
    ok = info["worst"] <= np.log10(1.01)
    cnt = (r["occ"] >= 0).sum(1).cpu().numpy()
    extra = {}
    if repair:
        # 11/03 repair stage (QUEUE 11/03 obs. 1, PROPOSALS T2): after the cutting-plane, each instance that still
        # fails the exact check gets its exact-check failure frequencies added to the grid and the SAME method
        # continues from its own final placement (adds, and prunes once the grid is met); then the exact check
        # again, up to `repair` times. Instances that already pass are not touched. Same rule for every method.
        assert cont is not None, f"--repair not supported for {method}"
        occ, worst = r["occ"].clone(), info["worst"].copy()
        t1, used = time.time(), 0
        for _ in range(repair):
            fail = np.flatnonzero(worst > np.log10(1.01))
            if not len(fail):
                break
            used += 1
            fi = torch.as_tensor(fail, device=b.valid.device)
            _, _, add = certify(b, fi, occ[fi], cfg)
            b.add_freqs(add)
            occ[fi] = cont(b, fi, occ[fi])["occ"]
            worst[fail] = certify(b, fi, occ[fi], cfg)[0]
        extra = dict(ok_pre=ok.tolist(), count_pre=cnt.tolist(), worst_pre=[float(x) for x in info["worst"]],
                     repair=repair, repair_used=used, sec_repair_total=time.time() - t1, base_method=method)
        r = dict(occ=occ)
        info = dict(info, worst=worst)
        ok = worst <= np.log10(1.01)
        cnt = (occ >= 0).sum(1).cpu().numpy()
    rec = dict(set=name, tech=tech, size=size, n=int(b.B), method=rec_method, ok=ok.tolist(), count=cnt.tolist(),
               worst=[float(x) for x in info["worst"]], sec_per_inst=box.get("last", float("nan")) / b.B,
               sec_total=time.time() - t0, rounds=int(info["rounds"]), F_final=int(b.freqs.shape[1]),
               ckpt=spec, samples=samples if spec else None, inst_chunk=inst_chunk if spec else None,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               git=git_hash(), time=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
               occ=r["occ"].cpu().numpy().tolist(), **extra)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(rec, open(dst, "w"))
    return name, rec_method, f"ok {ok.mean():.2f} count {cnt[ok].mean() if ok.any() else float('nan'):.2f} {rec['sec_per_inst']:.3f}s/inst rounds {info['rounds']}", rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="runs/pool_v1")
    ap.add_argument("--sets", default="eval_*", help="glob of pool files (without .pkl), comma-separated for several")
    ap.add_argument("--methods", default="", help="comma list of gp,greedy,ls,modal_rule")
    ap.add_argument("--policy", nargs="*", default=[], help="name=path/to/ckpt.pt")
    ap.add_argument("--samples", type=int, default=1, help="policy: best-of-S sampling (1 = greedy decoding)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--ls", default="4x32")
    ap.add_argument("--ga", default="64x100", help="genetic: population x generations")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--inst_chunk", type=int, default=0, help="policy greedy rollout: instances per rollout (0 = all)")
    ap.add_argument("--repair", type=int, default=0, help="after the cutting-plane: up to K repair rounds (method "
                    "continues from its placement on the grid + its exact-check failures); records '<method>+repK'")
    ap.add_argument("--start", default="", choices=["", "rule"], help="policy: start from modal_rule's solution and "
                    "let the policy continue (remove / change type / stop); records 'modal_rule><name>'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted({f for s in a.sets.split(",") for f in glob.glob(os.path.join(a.pool, s + ".pkl"))})
    files.sort(key=lambda f: -os.path.getsize(f))
    names = {"gp": "greedy+prune", "greedy": "greedy", "ls": "local_search", "modal_rule": "modal_rule", "ga": "genetic"}
    methods = [(names[m], "") for m in a.methods.split(",") if m]
    for p in a.policy:
        n, path = p.split("=")
        methods.append((n if a.samples <= 1 else f"{n}@{a.samples}", path))
    ls_cfg = tuple(map(int, a.ls.split("x")))
    ga_cfg = tuple(map(int, a.ga.split("x")))
    jobs = [(f, m, spec, a.out, a.rounds, a.samples, ls_cfg, a.device, a.inst_chunk, ga_cfg, a.repair, a.start)
            for f in files for m, spec in methods]
    print(f"{len(files)} sets x {len(methods)} methods = {len(jobs)} jobs, {a.workers} workers -> {a.out}/eval", flush=True)
    t0 = time.time()
    # one fresh process per job: a side-16 job's peak stayed resident in the worker into the next job
    # (10/29 01:50, 22 GB), and peak_rss_mb in the record is then that job's own peak
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for name, m, status, rec in pool.imap_unordered(job, jobs):
            print(f"[{time.time()-t0:6.0f}s] {name:14s} {m:28s} {status}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
