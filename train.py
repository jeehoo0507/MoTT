"""Train the add / remove / change-type / stop policy (min decap count s.t. target impedance).

Fixed methodology (agent/AGENT_PROMPT.md, 2026-10-28), the same for every rung of the ladder:
  stage 1  teacher   : offline greedy+prune -> local search per training instance (scripts/make_teacher.py,
                       data/teacher/<pool>.npz). Not done here.
  stage 2  distill   : --stage distill  imitation only: replay the teacher solution (trajectory_to) and
                       maximise its log-likelihood. No reward, no lambda.
  stage 3  rl        : --stage rl --init <distill ckpt>  REINFORCE with a shared M-sample baseline
                       (POMO style), Lagrange penalty on certified failures, + imitation of the
                       best-known solution; a certified sample with fewer decaps than the stored
                       teacher replaces it (expert iteration).
  --stage joint      : legacy behaviour (rl from scratch with the teacher as imitation target).
Reward = -count / N_ref - lambda * (failed + violation), N_ref = greedy+prune count of the instance.

Pools (--pool) hold certified instances of several board kinds (decap_loop/techs.py); each
iteration draws one pool file, so the policy sees a mixture of kinds. Exact checks (verify.py) run
through decap_loop/safe_verify.certify_parallel (single-threaded workers; MKL hang otherwise).

Example (A5000):
  python train.py --stage distill --feat coord --out runs/2026-10-28/r0_coord_distill --hours 0.75
  python train.py --stage rl --init runs/2026-10-28/r0_coord_distill/best.pt --feat coord --out runs/2026-10-28/r0_coord_rl --hours 1.25
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np

# rl rollouts allocate [E,F,n,k] complex128 blocks whose k changes every step; with the default allocator
# 5.7-6.5 GB stayed reserved but unusable and the rl stage hit CUDA OOM (10/28 17:15, 17:22)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from decap_loop.bank import load_bank
from decap_loop.baselines import greedy, local_search, trajectory_to
from decap_loop.modal_env import feat_dim_any, make_env
from decap_loop.model import MoTT, GridPolicy
from decap_loop.pdn import PDNConfig, sample_batch
from decap_loop.rollout import backward_logp, rollout
from decap_loop.safe_verify import certify_parallel, pin_threads
from decap_loop.sim import K_MODES, mott_dims

TOL1 = np.log10(1.01)


def rng_range(s, typ=int):
    a, b = s.split("-") if "-" in s else (s, s)
    return typ(a), typ(b)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/gnn_phys")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", choices=["distill", "rl", "joint"], default="joint")
    p.add_argument("--init", default="", help="checkpoint to start from (stage rl: the distilled policy)")
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--B", type=int, default=16, help="instances per batch")
    p.add_argument("--M", type=int, default=8, help="samples per instance (shared baseline, G in the method text)")
    p.add_argument("--chunk", type=int, default=2, help="rl: instances per rollout/backward pass; gradients of the "
                   "B/chunk passes accumulate into one step (memory only: B=16 x M=8 in one pass and chunk 4 "
                   "both hit CUDA OOM on 24 GB); halved for the rest of the run after a CUDA OOM")
    p.add_argument("--grid_growth", type=float, default=0.5, help="rl: a pool's working grid grows by at most this "
                   "fraction of its initial F (exact-check failures beyond it still count as failures); the dense "
                   "pool and every rollout block scale with F, so unbounded growth ends in CUDA OOM")
    p.add_argument("--sizes", default="8-12", help="(no pool) train board side range")
    p.add_argument("--shapes", default="rect,L,holes")
    p.add_argument("--targets", default="flat,step")
    p.add_argument("--n_probe", default="1-3")
    p.add_argument("--difficulty", default="0.02-0.2", help="(no pool) target level u (0 = hardest)")
    p.add_argument("--max_steps", type=int, default=64)
    p.add_argument("--feat", choices=["phys", "coord", "mott", "modal", "modal_only", "manoharan"], default="phys")
    p.add_argument("--model", choices=["grid", "mott", "loop", "park_am", "manoharan", "devformer"], default="grid",
                   help="mott/loop imply --feat mott; park_am (prior-work baseline B-Park, baselines_ml/park_am.py) --feat coord; "
                   "manoharan (B-Manoharan, baselines_ml/manoharan.py: fixed port order, type-or-skip) --feat manoharan; "
                   "devformer (B-DevFormer, baselines_ml/devformer.py, distill stage only) --feat coord")
    p.add_argument("--df_perm", type=int, default=0, help="distill (B-DevFormer expert exploitation, PROPOSALS_DRAFT_devformer): "
                   "also imitate P random permutations of the teacher's add order (paper P = 4); 0 = off")
    p.add_argument("--df_self", type=float, default=0.0, help="distill (B-DevFormer self-exploitation): weight lambda of "
                   "the order-bias term on the policy's own samples (log space, baselines_ml/devformer.py); 0 = off")
    p.add_argument("--df_layers", type=int, default=3, help="devformer: encoder layers (paper L = 3; --layers is left "
                   "to the other models, run_rung's COMMON sets it to 4)")
    p.add_argument("--baseline", choices=["group", "rollout"], default="group", help="rl advantage: group = normalised "
                   "within the M samples of an instance (POMO, ours); rollout = R - R(greedy rollout of a frozen copy, "
                   "replaced at every new best validation) (Kool 2019 / Park 2022, B-Park)")
    p.add_argument("--pool", default="", help="directory of certified pools (scripts/build_pools.py)")
    p.add_argument("--pool_glob", default="train_*.pkl", help="comma-separated globs of training pool files")
    p.add_argument("--val_pools", default="val_pcb10_0,val_pkg10_0,val_ip10_0")
    p.add_argument("--teacher_dir", default="data/teacher", help="stage-1 teacher solutions (<pool name>.npz)")
    p.add_argument("--hours", type=float, default=0, help="stop after this wall time (0 = off)")
    p.add_argument("--certify", type=int, default=1, help="exact-check the best sample of each instance every iteration; 2: every distinct grid-feasible sample")
    p.add_argument("--feas_margin", type=float, default=0.0, help="rl reward: a sample counts as feasible only if its "
                   "worst working-grid log10(|Z|/Zt) <= -this (0 = the env's feasibility); certify / val unchanged")
    p.add_argument("--hard_weight", type=float, default=0.0, help="rl: weight each instance's loss terms by "
                   "1 + this * (percentile of its g+p count among the loaded training instances of its board kind), "
                   "normalised to mean 1 over the pool (0 = off; sampling unchanged)")
    p.add_argument("--kind_weight", default="", help="pool-file draw weight per board kind, e.g. 'pkg=2' (other kinds 1; "
                   "'' = the uniform draw, unchanged RNG stream) (QUEUE 11/07 10:45 KST candidate C2)")
    p.add_argument("--self_repair", type=int, default=0, help="rl: verified self-imitation (QUEUE 11/07 10:45 KST C1): "
                   "samples with fewer decaps than the stored solution are exact-checked, failures repaired up to this "
                   "many times as in eval_policy --repair (failure frequencies onto the grid, the policy continues "
                   "greedily from its placement); the fewest-decap certified one replaces the stored solution (0 = off)")
    p.add_argument("--prior", action="store_true", help="grid model: add the physics prior (as in MoTT)")
    p.add_argument("--no_prune", action="store_true", help="ablation: no remove / change-type actions")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--glob", action="store_true", help="add global self-attention to every block")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--loop_train", default="2-8", help="loop model: R ~ U{a..b} during training")
    p.add_argument("--loop_eval", type=int, default=8, help="loop model: R at validation")
    p.add_argument("--fc_bias", action="store_true", help="mott/loop (day-2 arch a, QUEUE 10/29 03:15): add the "
                   "frequency-match bias -|log f_k - log SRF_t| / tau_h (tau learned per head) to dec_from_mode")
    p.add_argument("--mm_bias", action="store_true", help="mott (PROPOSALS 11/07 M1, human): add the log-frequency "
                   "distance bias -|log f_k - log f_k'| / tau_h (tau learned per head) to the mode self-attention")
    p.add_argument("--mode_head", action="store_true", help="mott/loop (day-2 arch c', PROPOSALS 10/29 08:00): add "
                   "a mode-routed add logit sum_k alpha_ik phi(m_k, c_t), zero-init")
    p.add_argument("--peak_gate", action="store_true", help="mott (11/07 C1', PROPOSALS): P1's refined peak level also "
                   "enters the env's target-met test (decap_loop/pkgate_env.py); implies --peak_refine")
    p.add_argument("--no_prior", action="store_true", help="mott (11/07 A2 ablation, PROPOSALS A1/A2): drop the "
                   "physics prior term (w_dv*dV - w_mn*margin) from the logits; the policy learns the scores alone")
    p.add_argument("--peak_refine", action="store_true", help="mott/loop (PROPOSALS 11/03 D7 -> P1): append to each "
                   "mode token the parabolic-vertex excess of its peak level over the working grid and its violating "
                   "flag (--feat mott_pk, decap_loop/peak_env.py)")
    p.add_argument("--dedup_modes", action="store_true", help="mott/loop (PROPOSALS 11/04 D8b -> P2): pick the mode "
                   "tokens on distinct working-grid frequencies (the padded grid repeats frequencies, which made "
                   "copies of one peak fill the token slots) (--feat mott_dd, decap_loop/dedup_env.py)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--imit", type=float, default=1.0, help="weight of imitating the best-known solution (0=off)")
    p.add_argument("--teacher", choices=["gp", "ls"], default="gp", help="(no pool) greedy+prune, or + local search")
    p.add_argument("--lam", type=float, default=1.0, help="initial Lagrange multiplier")
    p.add_argument("--lam_lr", type=float, default=0.05)
    p.add_argument("--fail_target", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_sizes", default="10,20")
    p.add_argument("--val_n", type=int, default=16)
    p.add_argument("--ckpt_min", type=float, default=20, help="write <out>/resume.pt every this many minutes; "
                   "a restarted run with the same --out continues from it")
    return p.parse_args()


def make_policy(cfg, a):
    if a.get("peak_refine"):
        from decap_loop.peak_env import mott_pk_dims as mott_dims  # noqa: F811 (mode descriptor 4 -> 6)
    else:
        from decap_loop.sim import mott_dims
    if a.get("model") == "mott":
        pol = MoTT(cfg.n_types, K_MODES, mott_dims(cfg.n_types), d=a["d"], layers=a["layers"], heads=a["heads"],
                   fc=a.get("fc_bias", False), mode_head=a.get("mode_head", False), mm=a.get("mm_bias", False))
        pol.use_prior = not a.get("no_prior", False)   # 11/07 A2: learned scores only, no physics prior
        return pol
    if a.get("model") == "loop":
        from decap_loop.loop_model import LoopMoTT
        lo, hi = rng_range(a.get("loop_train", "2-8"))
        return LoopMoTT(cfg.n_types, K_MODES, mott_dims(cfg.n_types), d=a["d"], heads=a["heads"],
                        r_train=(lo, hi), r_eval=a.get("loop_eval", 8), fc=a.get("fc_bias", False),
                        mode_head=a.get("mode_head", False))
    if a.get("model") == "manoharan":
        from baselines_ml.manoharan import ManoharanNet
        return ManoharanNet(cfg.n_types, cfg.n_feat_freq, d=a["d"], heads=a["heads"])
    if a.get("model") == "devformer":
        from baselines_ml.devformer import DevFormer
        return DevFormer(feat_dim_any(cfg, a["feat"]), cfg.n_types, d=a["d"], layers=a.get("df_layers", 3),
                         heads=a["heads"], ff=4 * a["d"])
    if a.get("model") == "park_am":
        from baselines_ml.park_am import ParkAM
        return ParkAM(feat_dim_any(cfg, a["feat"]), cfg.n_types, d=a["d"], layers=a["layers"], heads=a["heads"])
    return GridPolicy(feat_dim_any(cfg, a["feat"]), cfg.n_types, d=a["d"], layers=a["layers"],
                      heads=a["heads"], glob=a["glob"], prior=a.get("prior", False))


def batch_kw(args):
    return dict(shapes=tuple(args.shapes.split(",")), targets=tuple(args.targets.split(",")),
                n_probe=rng_range(args.n_probe), difficulty=rng_range(args.difficulty, float))


def gpu_peak():
    return f" | gpu peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if torch.cuda.is_available() else ""


class Tee:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, *a, **k):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


@torch.no_grad()
def validate(policy, val_sets, args, cfg, dev):
    """Greedy decoding on each validation pool; success by the exact check. Returns
    {name: (count / greedy+prune on jointly solved, certified fail rate)}."""
    policy.eval()
    if hasattr(policy, "set_eval_R"):
        policy.set_eval_R(args.loop_eval)
    res = {}
    for name, (vb, ref) in val_sets.items():
        env = make_env(vb, torch.arange(vb.B, device=dev), vb.cfg or cfg, args.feat, allow_prune=not args.no_prune)
        out = rollout(policy, env, greedy=True, amp=not args.no_amp)
        feas = out["feasible"].clone()
        if args.pool:   # certified validation: a solution that violates the exact check is a failure
            wv, _, _ = certify_parallel(vb, torch.arange(vb.B, device=dev), out["occ"], vb.cfg)
            feas &= torch.from_numpy(wv <= TOL1).to(dev)
        ok = feas & ref["feasible"]
        ratio = (out["count"][ok].float() / ref["count"][ok].clamp(min=1).float()).mean().item() if ok.any() else float("nan")
        res[name] = (ratio, 1 - feas.float().mean().item())
    policy.train()
    return res


def load_teacher(path, batch, dev, log):
    """Teacher store for one pool file: occ [B,n], count [B], ok [B], gp_count [B] (all on dev)."""
    if os.path.exists(path):
        d = np.load(path)
        occ, cnt, ok, gp = d["occ"], d["count"], d["ok"], d["gp_count"]
        src = "teacher file"
    else:
        r = greedy(batch, batch.cfg)
        occ, cnt, ok, gp = r["occ"].cpu().numpy(), r["count"].cpu().numpy(), r["feasible"].cpu().numpy(), r["count"].cpu().numpy()
        src = "greedy+prune (no teacher file)"
    log(f"  teacher {os.path.basename(path)}: {src}, ok {np.mean(ok):.2f}, count {np.mean(cnt[ok]):.2f}, g+p {np.mean(gp):.2f}")
    return dict(occ=torch.from_numpy(np.asarray(occ)).long().to(dev), count=torch.from_numpy(np.asarray(cnt)).long().to(dev),
                ok=torch.from_numpy(np.asarray(ok)).bool().to(dev), gp=torch.from_numpy(np.asarray(gp)).long().to(dev),
                replaced=0)


SR_PER = 2   # --self_repair: distinct candidates exact-checked per instance and step (time bound, PROPOSALS 11/07 M)


@torch.no_grad()
def self_repair(policy, batch, cfg, tch, inst, occ, args, amp, room):
    """--self_repair (QUEUE 11/07 10:45 KST C1, verified self-imitation): exact-check candidate solutions occ [N, n] of
    pool instances inst [N] (samples with fewer decaps than the stored solution); each failure gets its exact-check
    failure frequencies on the working grid (within the --grid_growth room) and the policy continues greedily from its
    own placement, as eval_policy --repair, up to args.self_repair times; per instance the fewest-decap certified
    candidate replaces the stored solution if it has fewer decaps. Returns counters."""
    was_training = policy.training
    policy.eval()
    if hasattr(policy, "set_eval_R"):
        policy.set_eval_R(args.loop_eval)
    wv, _, add = certify_parallel(batch, inst, occ, cfg)
    ok = wv <= 1e-3                                   # the training tolerance (as the expert-iteration check)
    st = dict(checked=len(ok), pass0=int(ok.sum()), repaired=0, replaced=0, no_room=0)
    for _ in range(args.self_repair):
        bad = np.flatnonzero(~ok)
        if not len(bad):
            break
        r = room()
        if r <= 0:                                    # the grid cannot take the failure points: no repair
            st["no_room"] += len(bad)
            break
        batch.add_freqs([a[:r] for a in add])
        bi = torch.from_numpy(bad).to(inst.device)
        env = make_env(batch, inst[bi], cfg, args.feat, max_steps=args.max_steps, allow_prune=not args.no_prune,
                       occ0=occ[bi])
        occ[bi] = rollout(policy, env, greedy=True, amp=amp)["occ"]
        env = None
        wb, _, add = certify_parallel(batch, inst[bi], occ[bi], cfg)
        ok[bad] = wb <= 1e-3
        st["repaired"] += int((wb <= 1e-3).sum())
    if was_training:
        policy.train()
    okt = torch.from_numpy(ok).to(inst.device)
    cnt = (occ >= 0).sum(1)
    key = cnt.float() + (~okt).float() * 1e6
    for b in inst.unique():
        idx = torch.nonzero(inst == b).squeeze(-1)
        j = idx[key[idx].argmin()]
        if okt[j] and cnt[j] < tch["count"][b]:
            tch["occ"][b] = occ[j]
            tch["count"][b] = cnt[j]
            tch["ok"][b] = True
            tch["replaced"] += 1
            st["replaced"] += 1
    return st


def main():
    args = parse()
    if args.model in ("mott", "loop"):
        # both flags = PROPOSALS 11/04 P3 (P2 tokens on top of P1, compared with the P1 run: still one change)
        if args.peak_gate:
            args.peak_refine = True
        args.feat = ("mott_pkgate" if args.peak_gate else
                     "mott_pkdd" if args.peak_refine and args.dedup_modes else "mott_pk" if args.peak_refine
                     else "mott_dd" if args.dedup_modes else "mott")
    elif args.model in ("park_am", "devformer"):
        args.feat = "coord"
        if args.model == "devformer" and args.stage != "distill":
            # the paper trains offline by imitation only; an old pipeline_day2 driver (no "distill_only") would
            # otherwise follow the distill stage with rl
            raise SystemExit("B-DevFormer is distill-only (offline imitation, no rl): use --stage distill")
    elif args.model == "manoharan":
        args.feat = "manoharan"
    pin_threads()
    os.makedirs(args.out, exist_ok=True)
    log = Tee(os.path.join(args.out, "log.txt"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    cfg = PDNConfig()
    smin, smax = rng_range(args.sizes)
    kw = batch_kw(args)
    amp = not args.no_amp
    t_start = time.time()

    policy = make_policy(cfg, vars(args)).to(dev)
    if args.init:
        ck0 = torch.load(args.init, map_location=dev, weights_only=False)
        policy.load_state_dict(ck0["model"])
        log(f"init from {args.init} (iter {ck0.get('iter')})")
    policy.grad_ckpt = args.grad_ckpt
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    base_policy = None
    if args.stage != "distill" and args.baseline == "rollout":
        import copy
        base_policy = copy.deepcopy(policy).eval()
        for q in base_policy.parameters():
            q.requires_grad_(False)
    log(f"params: {sum(p.numel() for p in policy.parameters())/1e6:.2f}M  device: {dev}  stage: {args.stage}  feat: {args.feat}  model: {args.model}")
    meta = dict(args=vars(args), cfg=asdict(cfg), git=git_hash(), cmd=" ".join(sys.argv))
    json.dump(meta, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    vrng = np.random.default_rng(12345)  # same val sets for every run
    val_sets, pool, teachers, pool_names = {}, [], [], []
    if args.pool:
        files = sorted({f for g in args.pool_glob.split(",") for f in glob.glob(os.path.join(args.pool, g))})
        for f in files:
            pb, _ = load_bank(f, dev, dense=True, z_dtype=torch.complex64)
            pool.append(pb)
            name = os.path.splitext(os.path.basename(f))[0]
            pool_names.append(name)
            teachers.append(load_teacher(os.path.join(args.teacher_dir, name + ".npz"), pb, dev, log))
        for name in args.val_pools.split(","):
            vb, vres = load_bank(os.path.join(args.pool, name + ".pkl"), dev)
            if "greedy+prune" in vres:
                g = vres["greedy+prune"]
                ref = dict(count=torch.from_numpy(np.asarray(g["count"])).long().to(dev),
                           feasible=torch.from_numpy(np.asarray(g["ok"])).bool().to(dev))
            else:
                ref = greedy(vb, vb.cfg)
            val_sets[name] = (vb, ref)
        log(f"pool: {len(pool)} train batches, {sum(p.B for p in pool)} instances; "
            f"GPU mem {torch.cuda.memory_allocated() / 2**30:.2f} GB" if dev.type == "cuda" else
            f"pool: {len(pool)} train batches, {sum(p.B for p in pool)} instances")
        F0 = [pb.freqs.shape[1] for pb in pool]   # before a resume restores grown grids
    else:
        for s in map(int, args.val_sizes.split(",")):
            vb = sample_batch(args.val_n, s, s, cfg, vrng, device=dev, **kw)
            val_sets[s] = (vb, greedy(vb, cfg))
    hw = None
    if pool and args.hard_weight > 0:
        # 11/03 round 29 (QUEUE 11/03 observation 2, scripts/diag_difficulty.py): the eval failures sit in the top
        # quartile of the training pool's g+p count, inside its range, where the teacher covers least
        kind = [n.split("_")[1].rstrip("0123456789") for n in pool_names]
        hw = []
        for k, t in zip(kind, teachers):
            g = np.concatenate([tt["gp"].cpu().numpy() for kk, tt in zip(kind, teachers) if kk == k])
            x = t["gp"].cpu().numpy()
            q = ((g[None, :] < x[:, None]).sum(1) + 0.5 * (g[None, :] == x[:, None]).sum(1)) / len(g)   # mid-rank
            hw.append(1.0 + args.hard_weight * q)
        mean = np.concatenate(hw).mean()
        hw = [torch.from_numpy(w / mean).float().to(dev) for w in hw]
        log(f"hard_weight {args.hard_weight}: instance weights {min(w.min().item() for w in hw):.2f}.."
            f"{max(w.max().item() for w in hw):.2f}")

    pw = None
    if pool and args.kind_weight:
        kwt = {k: float(v) for k, v in (s.split("=") for s in args.kind_weight.split(","))}
        kind = [n.split("_")[1].rstrip("0123456789") for n in pool_names]
        assert set(kwt) <= set(kind), (kwt, set(kind))
        pw = np.array([kwt.get(k, 1.0) for k in kind])
        pw = pw / pw.sum()
        log("kind_weight " + ", ".join(f"{k} {pw[[i for i, kk in enumerate(kind) if kk == k]].sum():.3f}"
                                       for k in sorted(set(kind))))

    best_val, best_iter, lam, t0 = 1e9, 0, args.lam, time.time()
    last_val, it_done, n_replaced, n_oom, n_capped = {}, 0, 0, 0, 0
    sr_on = bool(pool) and args.self_repair > 0 and args.stage == "rl"
    sr_tot = dict(checked=0, pass0=0, repaired=0, replaced=0, no_room=0)
    resume_path = os.path.join(args.out, "resume.pt")

    def save_resume(it):
        """Everything needed to continue after a pod restart (QUEUE 2026-10-28: every 20 min)."""
        st = dict(model=policy.state_dict(), opt=opt.state_dict(), iter=it, lam=lam, best_val=best_val,
                  best_iter=best_iter, last_val=last_val, n_replaced=n_replaced, elapsed=time.time() - t0,
                  n_oom=n_oom, n_capped=n_capped, chunk=args.chunk, sr_tot=sr_tot,
                  rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
                  teachers=[{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in t.items()} for t in teachers],
                  pool_freqs=[pb.freqs for pb in pool],
                  base=base_policy.state_dict() if base_policy is not None else None)
        torch.save(st, resume_path + ".tmp")
        os.replace(resume_path + ".tmp", resume_path)

    start_it = 0
    if os.path.exists(resume_path):
        st = torch.load(resume_path, map_location=dev, weights_only=False)
        policy.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_it, lam, best_val, best_iter = st["iter"], st["lam"], st["best_val"], st["best_iter"]
        last_val, n_replaced, it_done = st["last_val"], st["n_replaced"], st["iter"]
        n_oom, n_capped, args.chunk = st.get("n_oom", 0), st.get("n_capped", 0), st.get("chunk", args.chunk)
        sr_tot = st.get("sr_tot", sr_tot)
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"].cpu())
        if st["cuda_rng"] is not None and dev.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        for t, ts in zip(teachers, st["teachers"]):
            t.update({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ts.items()})
        for pb, fr in zip(pool, st["pool_freqs"]):   # grid points the exact checks added before the restart
            if fr.shape[1] > pb.freqs.shape[1]:
                pb.add_freqs([list(fr[b, pb.freqs.shape[1]:]) for b in range(pb.B)])
        if base_policy is not None and st.get("base") is not None:
            base_policy.load_state_dict(st["base"])
        t0 = time.time() - st["elapsed"]
        log(f"resumed from {resume_path}: iter {start_it}, {st['elapsed']/3600:.2f} h used, lam {lam:.2f}, "
            f"best iter {best_iter}, replaced {n_replaced}")
    logf = open(os.path.join(args.out, "log.csv"), "a" if start_it else "w", newline="")
    w = csv.writer(logf)
    if not start_it:
        w.writerow(["iter", "ratio", "best_ratio", "fail", "lam", "loss", "sec", "replaced"] +
                   [f"val{s}_{m}" for s in val_sets for m in ("ratio", "fail")])
    t_ck = time.time()
    df_st = {}   # B-DevFormer distill stats of the last step (empty when --df_perm / --df_self are off)
    for it in range(start_it + 1, args.iters + 1):
        if args.hours and time.time() - t0 > args.hours * 3600:
            log(f"time limit reached at iter {it}")
            break
        it_done = it
        if hasattr(policy, "sample_R"):     # loop model: one R per iteration (rollout and backward must agree)
            policy.sample_R(rng)
        if pool:
            pi = int(rng.choice(len(pool), p=pw)) if pw is not None else int(rng.integers(len(pool)))
            batch, tch = pool[pi], teachers[pi]
            H, W = batch.H, batch.W
            cfg = batch.cfg
            ib = torch.from_numpy(rng.permutation(batch.B)[:args.B]).to(dev)
            n_ref = tch["gp"][ib].clamp(min=1).float()
            # the imitation target: replay of the stored best-known solution (greedy order + prune)
            if args.model == "manoharan":   # the teacher solution replayed in the fixed port order
                from baselines_ml.manoharan import manoharan_teacher
                teach = manoharan_teacher(batch, cfg, ib, tch["occ"][ib], max_steps=args.max_steps)
            else:
                teach = trajectory_to(batch, cfg, ib, tch["occ"][ib], max_steps=args.max_steps)
        else:
            H, W = (int(x) for x in rng.integers(smin, smax + 1, size=2))
            batch = sample_batch(args.B, H, W, cfg, rng, device=dev, **kw)
            ib = torch.arange(args.B, device=dev)
            teach = greedy(batch, cfg, inst=ib, max_steps=args.max_steps)
            n_ref = teach["count"].clamp(min=1).float()
            if args.no_prune:  # the teacher may only use the actions the policy has
                teach = greedy(batch, cfg, prune=False, inst=ib, max_steps=args.max_steps)
            elif args.teacher == "ls":
                occ = local_search(batch, cfg, teach["occ"], teach["feasible"], passes=2, trials=16, seed=it)
                teach = trajectory_to(batch, cfg, ib, occ, max_steps=args.max_steps)
        t_key = teach["count"].float() + (~teach["feasible"]) * 1e6

        opt.zero_grad(set_to_none=True)
        if args.stage == "distill":
            tb = torch.nonzero(teach["feasible"]).squeeze(-1)
            loss, ratio, best_ratio, fail_rate, n_bad, n_margin = 0.0, float("nan"), float("nan"), float("nan"), 0, 0
            if tb.numel() and (args.df_perm > 0 or args.df_self > 0):   # B-DevFormer CSE (baselines_ml/devformer.py)
                from baselines_ml.devformer import distill_step
                loss, df_st = distill_step(policy, batch, cfg, ib[tb], teach["actions"][tb], args, amp, rng, H, W)
                ratio = (teach["count"][tb].float() / n_ref[tb]).mean().item()
            elif tb.numel():
                tenv = make_env(batch, ib[tb], cfg, args.feat, max_steps=args.max_steps, allow_prune=not args.no_prune)
                rep = rollout(policy, tenv, forced=teach["actions"][tb], store=True, amp=amp)
                tsteps = (rep["actions"] >= 0).sum(1).clamp(min=1).float()
                loss = backward_logp(policy, rep["traj"], 1.0 / tsteps, H, W, amp, scale=1.0 / tb.numel())
                ratio = (teach["count"][tb].float() / n_ref[tb]).mean().item()
        else:
            # The B instances go through rollout + backward `chunk` at a time and the gradients accumulate into
            # one step. Advantages are normalised within each instance's M samples and the loss scales are the
            # full-batch ones (1/(B*M), 1/B), so this is the same update as one B*M pass with less GPU memory.
            E, loss, n_bad, cnts, fails = args.B * args.M, 0.0, 0, [], []
            n_margin = 0   # grid-feasible samples turned into failures by --feas_margin
            adds = [[] for _ in range(batch.B)]   # new grid points of this step, added once after all chunks
            sr_inst, sr_occ = [], []              # --self_repair candidates of this step (checked after it)
            try:
                for c0 in range(0, args.B, args.chunk or args.B):
                    cb = torch.arange(c0, min(c0 + (args.chunk or args.B), args.B), device=dev)   # positions in ib
                    nb, ibc = cb.numel(), ib[cb]
                    env = make_env(batch, ibc.repeat_interleave(args.M), cfg, args.feat, max_steps=args.max_steps,
                                   allow_prune=not args.no_prune)
                    out = rollout(policy, env, store=True, amp=amp)
                    cnt, fail = out["count"].float().view(nb, args.M), (~out["feasible"]).view(nb, args.M).clone()
                    if args.feas_margin > 0:
                        # 11/02 round 7, diagnosis H2: exact-check failures are near misses through grid gaps, so
                        # a grid-feasible sample must also clear the target by this margin to earn the success reward
                        n_margin += int((~fail & (out["m"].view(nb, args.M) > -args.feas_margin)).sum())
                        fail |= out["m"].view(nb, args.M) > -args.feas_margin
                    j = (cnt + fail * 1e6).argmin(1)                      # best sample per instance
                    cert_ok = torch.zeros(nb, dtype=torch.bool, device=dev)
                    if pool and args.certify >= 2:
                        # every distinct working-grid-feasible sample (11/02, diagnosis H2): with the best one only,
                        # the other samples of the group still earn the success reward through grid gaps
                        flat = torch.nonzero(~fail.view(-1)).squeeze(-1)
                        if flat.numel():
                            rows = torch.cat([(flat // args.M)[:, None], out["occ"][flat].view(flat.numel(), -1).long()], 1)
                            uniq, inv = torch.unique(rows, dim=0, return_inverse=True)
                            rep_e = torch.zeros(uniq.shape[0], dtype=torch.long, device=dev).scatter_(0, inv, flat)
                            wv, _, add = certify_parallel(batch, ibc[rep_e // args.M], out["occ"][rep_e], cfg)
                            bad_u = torch.from_numpy(wv > 1e-3).to(dev)
                            fail.view(-1)[flat[bad_u[inv]]] = True
                            n_bad += int(bad_u.sum())
                            for b, fs in enumerate(add):
                                adds[b] += fs
                        cert_ok = ~fail.all(1)     # every sample left feasible is certified
                    elif pool and args.certify:
                        # the reward is computed on the working grid; without this the policy learns to prune into
                        # resonance peaks that fall between grid points
                        sel = torch.nonzero(~fail[torch.arange(nb, device=dev), j]).squeeze(-1)
                        if sel.numel():
                            e_idx = sel * args.M + j[sel]
                            wv, _, add = certify_parallel(batch, ibc[sel], out["occ"][e_idx], cfg)
                            bad = torch.from_numpy(wv > 1e-3).to(dev)
                            fail[sel[bad], j[sel[bad]]] = True
                            cert_ok[sel[~bad]] = True
                            n_bad += int(bad.sum())
                            if bad.any():
                                for b, fs in enumerate(add):
                                    adds[b] += fs
                    R = -cnt / n_ref[cb][:, None] - lam * (fail.float() + out["V"].float().view(nb, args.M))
                    if base_policy is None:
                        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)).view(-1)
                    else:   # greedy rollout of the frozen copy; its failure is the working-grid one (not certified)
                        with torch.no_grad():
                            if hasattr(base_policy, "R_eval"):
                                base_policy.set_eval_R(args.loop_eval)
                            benv = make_env(batch, ibc, cfg, args.feat, max_steps=args.max_steps, allow_prune=not args.no_prune)
                            bo = rollout(base_policy, benv, greedy=True, amp=amp)
                            bfail = ~bo["feasible"] | (bo["m"] > -args.feas_margin) if args.feas_margin > 0 else ~bo["feasible"]
                            Rb = -bo["count"].float() / n_ref[cb] - lam * (bfail.float() + bo["V"].float())
                            benv = bo = None
                        adv = (R - Rb[:, None]).view(-1)
                    wc = hw[pi][ibc] if hw is not None else torch.ones(nb, device=dev)   # --hard_weight
                    if hw is not None:
                        adv = adv * wc.repeat_interleave(args.M)
                    loss += backward_logp(policy, out["traj"], adv, H, W, amp, scale=1.0 / E)
                    if args.imit > 0:  # imitate the best-known solution of each instance
                        key = cnt + fail * 1e6
                        best = key.min(1)
                        use_sample = best.values <= t_key[cb]
                        ep = torch.arange(nb, device=dev) * args.M + best.indices
                        steps = (out["actions"] >= 0).sum(1).clamp(min=1).float()
                        wt = torch.zeros(nb * args.M, device=dev)
                        wt[ep[use_sample]] = args.imit * wc[use_sample] / steps[ep[use_sample]]
                        loss += backward_logp(policy, out["traj"], wt, H, W, amp, scale=1.0 / args.B)
                        tb = torch.nonzero(~use_sample & teach["feasible"][cb]).squeeze(-1)
                        if tb.numel():
                            tenv = make_env(batch, ibc[tb], cfg, args.feat, max_steps=args.max_steps, allow_prune=not args.no_prune)
                            rep = rollout(policy, tenv, forced=teach["actions"][cb][tb], store=True, amp=amp)
                            tsteps = (rep["actions"] >= 0).sum(1).clamp(min=1).float()
                            loss += backward_logp(policy, rep["traj"], args.imit * wc[tb] / tsteps, H, W, amp,
                                                  scale=1.0 / args.B)
                        if pool and args.stage == "rl":
                            # expert iteration: a certified sample with fewer decaps than the stored teacher replaces it
                            better = cert_ok & (best.values < tch["count"][ibc].float())
                            if better.any():
                                bi = torch.nonzero(better).squeeze(-1)
                                tch["occ"][ibc[bi]] = out["occ"][ep[bi]]
                                tch["count"][ibc[bi]] = cnt[bi, best.indices[bi]].long()
                                tch["ok"][ibc[bi]] = True
                                tch["replaced"] += int(bi.numel())
                                n_replaced += int(bi.numel())
                    if sr_on:
                        # samples with fewer decaps than the stored solution (after this chunk's expert iteration),
                        # distinct per instance, SR_PER per instance: grid-feasible first, then the lowest counts
                        cand = torch.nonzero((cnt < tch["count"][ibc].float()[:, None]).view(-1)).squeeze(-1)
                        if cand.numel():
                            rows = torch.unique(torch.cat([(cand // args.M)[:, None], fail.view(-1)[cand].long()[:, None],
                                                           cnt.view(-1)[cand].long()[:, None], out["occ"][cand].long()], 1),
                                                dim=0)   # sorted (inst, grid fail, count, occ)
                            for b in rows[:, 0].unique():
                                rb = rows[rows[:, 0] == b][:SR_PER]
                                sr_inst.append(ibc[b].repeat(rb.shape[0]))
                                sr_occ.append(rb[:, 3:])
                    cnts.append(cnt)
                    fails.append(fail)
                    env = out = tenv = rep = None     # free this chunk's trajectories before the next one
            except torch.OutOfMemoryError:
                # memory only: drop this step (no optimizer update) and use smaller chunks from here on
                opt.zero_grad(set_to_none=True)
                env = out = tenv = rep = cnts = fails = None
                torch.cuda.empty_cache()
                n_oom += 1
                args.chunk = max(1, (args.chunk or args.B) // 2)
                log(f"it {it}: CUDA OOM ({batch.P=} F={batch.freqs.shape[1]}), step skipped, chunk -> {args.chunk} "
                    f"(OOM {n_oom}){gpu_peak()}")
                if n_oom > 20:
                    raise
                continue
            if any(adds):
                # one call per step, as with a single B*M pass (each call pads every instance to its longest list)
                room = int(F0[pi] * (1 + args.grid_growth)) - batch.freqs.shape[1]
                if room < max(len(a) for a in adds):
                    n_capped += 1
                if room > 0:
                    batch.add_freqs([a[:room] for a in adds])
            if sr_inst:
                sr = self_repair(policy, batch, cfg, tch, torch.cat(sr_inst), torch.cat(sr_occ), args, amp,
                                 room=lambda: int(F0[pi] * (1 + args.grid_growth)) - batch.freqs.shape[1])
                for k in sr_tot:
                    sr_tot[k] += sr[k]
                n_replaced += sr["replaced"]
            cnt, fail = torch.cat(cnts), torch.cat(fails)
            fail_rate = fail.float().mean().item()
            lam = max(0.0, lam + args.lam_lr * (fail_rate - args.fail_target))
            ok = ~fail
            ratio = (cnt / n_ref[:, None])[ok].mean().item() if ok.any() else float("nan")
            solved = ok.any(1)
            best_ratio = ((cnt + fail * 1e6).min(1).values / n_ref)[solved].mean().item() if solved.any() else float("nan")
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip)
        opt.step()

        if it % 25 == 0 or it == args.iters:
            pname = pool_names[pi] if pool else f"{H}x{W}"
            log(f"it {it:6d} | {pname} P={batch.P} F={batch.freqs.shape[1]} cert-bad {n_bad}"
                f"{f' margin-fail {n_margin}' if args.feas_margin > 0 else ''} | count/ref {ratio:.3f} "
                f"best-of-M {best_ratio:.3f} fail {fail_rate:.2f} lam {lam:.2f} | teach ok {teach['feasible'].float().mean():.2f} "
                f"| loss {loss:.3f} | replaced {n_replaced} | grid capped {n_capped} oom {n_oom} | {time.time()-t0:.0f}s{gpu_peak()}"
                + (" | self-repair " + " ".join(f"{k} {v}" for k, v in sr_tot.items()) if sr_on else "")
                + (f" | df aug_ok {df_st['aug_ok']:.2f} self {df_st['self_loss']:.3f} self_ok {df_st['self_ok']:.2f}"
                   if df_st else ""))
        if it % args.val_every == 0 or it == args.iters:
            v = validate(policy, val_sets, args, cfg, dev)
            last_val = v
            log("  val " + "  ".join(f"{s}: count/greedy+prune={r:.3f} fail={f:.2f}" for s, (r, f) in v.items()))
            w.writerow([it, ratio, best_ratio, fail_rate, lam, loss, time.time() - t0, n_replaced] +
                       [x for r, f in v.values() for x in (r, f)])
            logf.flush()
            ck = dict(model=policy.state_dict(), **meta, iter=it)
            torch.save(ck, os.path.join(args.out, "last.pt"))
            score = float(np.mean([(r if r == r else 9.0) + 2 * f for r, f in v.values()]))
            if score < best_val:
                best_val, best_iter = score, it
                torch.save(ck, os.path.join(args.out, "best.pt"))
                if base_policy is not None:      # rollout baseline: replaced when the policy improves
                    base_policy.load_state_dict(policy.state_dict())
                log(f"  new best (score {score:.3f}) at iter {it}")
        if args.ckpt_min and time.time() - t_ck > args.ckpt_min * 60:
            save_resume(it)
            t_ck = time.time()
            log(f"  resume checkpoint at iter {it}")
    if not os.path.exists(os.path.join(args.out, "best.pt")) or not last_val:
        v = validate(policy, val_sets, args, cfg, dev)
        last_val = v
        ck = dict(model=policy.state_dict(), **meta, iter=it_done)
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if not os.path.exists(os.path.join(args.out, "best.pt")):
            torch.save(ck, os.path.join(args.out, "best.pt"))
            best_val, best_iter = float(np.mean([(r if r == r else 9.0) + 2 * f for r, f in v.values()])), it_done
    if pool and args.stage == "rl":
        os.makedirs(os.path.join(args.out, "teacher_final"), exist_ok=True)
        for name, tch in zip(pool_names, teachers):
            np.savez(os.path.join(args.out, "teacher_final", name + ".npz"), occ=tch["occ"].cpu().numpy(),
                     count=tch["count"].cpu().numpy(), ok=tch["ok"].cpu().numpy(), gp_count=tch["gp"].cpu().numpy(),
                     replaced=tch["replaced"])
    metrics = dict(args=vars(args), git=meta["git"], iters=it_done, hours=(time.time() - t_start) / 3600,
                   final_val={k: list(v) for k, v in last_val.items()}, best_val_score=best_val, best_iter=best_iter,
                   teacher_replaced=n_replaced, lam=lam, cuda_oom_skipped=n_oom, grid_capped_steps=n_capped,
                   chunk_final=args.chunk, **({"self_repair": sr_tot} if sr_on else {}), pool_F={n: [f0, pb.freqs.shape[1]] for n, f0, pb in zip(pool_names, F0, pool)}
                   if pool else {})
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    log(f"done: {it_done} iters, {metrics['hours']:.2f} h, best iter {best_iter} score {best_val:.3f}, replaced {n_replaced}"
        f"{gpu_peak()}")


if __name__ == "__main__":
    main()
