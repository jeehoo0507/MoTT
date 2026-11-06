"""Paired causal probe of P1 warnings versus action masks; CPU, fixed cases, no training."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from decap_loop.pdn import PDNBatch, PDNConfig, ZBank, build_Y0, target_at
from decap_loop.modal_env import make_env
from decap_loop.rollout import _logits
from decap_loop.sim import FEAS_TOL, K_MODES, mott_dims
from decap_loop.safe_verify import certify_parallel
from train import make_policy

ARMS = ("observe", "mask_open", "warning_refine", "generic_refine")
TOL = float(np.log10(1.01))


def single(d, i):
    """Rebuild one exact saved instance, without cross-instance frequency padding effects."""
    cfg = PDNConfig(**d["cfg"])
    h, w = d["H"], d["W"]
    mask, ent = d["valid"][i].reshape(h, w), np.flatnonzero(d["entry"][i])
    freqs, par = d["freqs"][i:i + 1].copy(), d["zt_par"][i:i + 1].copy()
    probes = torch.from_numpy(d["probes"][i:i + 1].copy())
    bank = ZBank([build_Y0(mask, ent, cfg, freqs[0])], probes, "cpu", True)
    return PDNBatch(h, w, torch.from_numpy(d["valid"][i:i + 1].copy()),
        torch.from_numpy(d["keepout"][i:i + 1].copy()), probes,
        torch.from_numpy(d["entry"][i:i + 1].copy()), torch.from_numpy(target_at(par, freqs[0])),
        bank, [d["shapes"][i]], [d["targets"][i]], freqs=freqs, zt_par=par, cfg=cfg)


def warning(obs, types):
    dn, _, de, _ = mott_dims(types)
    start = dn + K_MODES * de
    modes = obs["x"][0, 0, start:start + K_MODES * 6].reshape(K_MODES, 6)
    return bool(((modes[:, 3] > .5) & (modes[:, 5] > .5)).any())


@torch.no_grad()
def run_case(policy, batch, feat, arm, checks=4):
    env = make_env(batch, torch.tensor([0]), batch.cfg, feat, max_steps=64)
    events, cache, checked, mismatch_states = [], {}, 0, 0
    started = time.monotonic()
    while True:
        obs = env.observe()
        if bool(env.done.all()):
            break
        warn, feasible = warning(obs, env.T), bool(obs["feas"][0])
        if warn and feasible:
            mismatch_states += 1
        trigger = feasible and (warn or arm == "generic_refine")
        mask = obs["mask"].clone()
        key = tuple(env.occ[0].tolist())
        event = None
        if trigger and (key in cache or checked < checks):
            if key not in cache:
                worst, freq, add = certify_parallel(batch, torch.tensor([0]), env.occ, batch.cfg, workers=1)
                cache[key] = (float(worst[0]), float(freq[0]), add)
                checked += 1
            worst, freq, add = cache[key]
            dn, _, de, _ = mott_dims(env.T)
            start_mode = dn + K_MODES * de
            modes = obs["x"][0, 0, start_mode:start_mode + K_MODES * 6].reshape(K_MODES, 6)
            flagged = (modes[:, 3] > .5) & (modes[:, 5] > .5)
            token_freqs = (10 ** modes[flagged, 0].double() *
                           np.sqrt(batch.cfg.f_min * batch.cfg.f_max)).tolist()
            event = dict(step=int(env.steps[0]), warning=warn, grid_margin=float(obs["m"][0]),
                         exact_margin=worst, exact_failed=worst > TOL,
                         exact_peak_hz=freq, warned_token_hz=token_freqs, occupancy=list(key),
                         adds_allowed_before=int(mask[0, :-1].reshape(env.n, env.T + 1)[:, :env.T].sum()))
            if worst > TOL and arm in ("warning_refine", "generic_refine"):
                added = batch.add_freqs(add)
                event["frequencies_added"] = added
                if added:
                    env.refresh()
                    obs = env.observe()
                    mask = obs["mask"].clone()
            elif worst > TOL and arm == "mask_open":
                legal = mask[:, :-1].reshape(1, env.n, env.T + 1)
                legal[:, :, :env.T] |= ((env.occ < 0) & env.placeable)[:, :, None]
            event["adds_allowed_after"] = int(mask[0, :-1].reshape(env.n, env.T + 1)[:, :env.T].sum())
        if bool(env.done.all()):
            if event:
                events.append(event)
            break
        action = _logits(policy, obs["x"], env.valid, batch.H, batch.W, mask, False).argmax(-1)
        if event:
            event["action"] = int(action[0])
            event["selected_add"] = bool(action[0] != env.stop_idx and
                action[0] % (env.T + 1) < env.T and env.occ[0, action[0] // (env.T + 1)] < 0)
            events.append(event)
        env.step(action)
    worst, _, _ = certify_parallel(batch, torch.tensor([0]), env.occ, batch.cfg, workers=1)
    return dict(arm=arm, success=bool(worst[0] <= TOL), worst=float(worst[0]),
        count=int(env.count()[0]), steps=int(env.steps[0]), checks=checked,
        mismatch_states=mismatch_states, events=events, seconds=time.monotonic() - started)


def save(path, obj):
    grouped = {}
    for row in obj["rows"]:
        grouped.setdefault(row["instance"], {})[row["arm"]] = row
    obj["summary"] = {}
    for arm in ARMS:
        rows = [g[arm] for g in grouped.values() if arm in g]
        pairs = [(g["observe"], g[arm]) for g in grouped.values() if "observe" in g and arm in g]
        obj["summary"][arm] = dict(n=len(rows), successes=sum(r["success"] for r in rows),
            confirmed_warning_fail_events=sum(e["warning"] and e["exact_failed"] for r in rows for e in r["events"]),
            paired_n=len(pairs), rescued=sum(not b["success"] and t["success"] for b, t in pairs),
            harmed=sum(b["success"] and not t["success"] for b, t in pairs))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_seconds", type=float, default=3600)
    args = ap.parse_args()
    torch.set_num_threads(1)
    with open(args.pool, "rb") as f:
        d = pickle.load(f)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ck["args"]["feat"] != "mott_pk":
        raise ValueError("This diagnostic requires the existing P1 model")
    policy = make_policy(PDNConfig(**d["cfg"]), ck["args"]).eval()
    policy.load_state_dict(ck["model"])
    path, start = Path(args.out), time.monotonic()
    result = dict(pool=args.pool, ckpt=args.ckpt, arms=list(ARMS), rows=[], complete=False,
        note="Exploratory mechanism probe; single-instance base-grid rollouts, not main cutting-plane metrics")
    for i in range(min(args.n, len(d["valid"]))):
        rows = {}
        for arm in ARMS:
            if time.monotonic() - start > args.max_seconds:
                save(path, result)
                print("Time budget reached; partial results retained", flush=True)
                return
            rows[arm] = run_case(policy, single(d, i), ck["args"]["feat"], arm)
            result["rows"].append(dict(instance=i, **rows[arm]))
            save(path, result)
            print(json.dumps(dict(instance=i, arm=arm, success=rows[arm]["success"],
                mismatch_states=rows[arm]["mismatch_states"], checks=rows[arm]["checks"])), flush=True)
    result["complete"] = True
    save(path, result)
    path.with_suffix(".done").write_text("complete\n")


if __name__ == "__main__":
    main()
