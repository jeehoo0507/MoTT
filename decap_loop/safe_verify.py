"""Process-parallel wrapper around verify.certify (no change to verify.py).

Why: on this server torch's batched complex128 CPU solve (used by verify._probe_z_exact for
n <= 900 cells) hangs inside MKL ("Intel oneMKL ERROR: Parameter 6 was incorrect on entry to
ZLASWP") whenever n >= 200 and torch uses >= 2 threads (scripts/mkl_repro.py, 2026-10-28).
Single-threaded it is always correct, so exact checks run one instance per worker process with
torch.set_num_threads(1), and the main process also pins its CPU thread count to 1 before any
exact check.

certify_parallel() has the same signature and return as verify.certify.
"""
import multiprocessing as mp
import os

import numpy as np
import torch

from .pdn import PDNConfig
from .verify import _inst_data, exact_worst

_POOL = None
_WORKERS = int(os.environ.get("CERTIFY_WORKERS", max(1, min(14, (os.cpu_count() or 2) - 2))))


def pin_threads():
    """Call early in every process that runs exact checks on boards with >= 200 cells."""
    torch.set_num_threads(1)


def _init():
    torch.set_num_threads(1)


def _job(args):
    mask, ent, P, occ, cfg_d, par, tol, k_peaks, rel, span = args
    cfg = PDNConfig(**cfg_d)
    w, fw, (f, r) = exact_worst(mask, ent, P, occ, cfg, par)
    worst = float(np.log10(w))
    add = []
    if worst > tol:
        i = np.flatnonzero((r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:]) & (r[1:-1] > 10 ** tol)) + 1
        for fp in [fw] + list(f[i[np.argsort(-r[i])[:k_peaks]]]):
            add += list(fp * np.exp(rel * np.arange(-span, span + 1)))
    return worst, float(fw), add


def _pool():
    global _POOL
    if _POOL is None:
        ctx = mp.get_context("fork")
        _POOL = ctx.Pool(_WORKERS, initializer=_init)
    return _POOL


def _cfg_dict(cfg):
    from dataclasses import asdict
    d = asdict(cfg)
    d["decaps"] = tuple(map(tuple, d["decaps"]))
    return d


@torch.no_grad()
def certify_parallel(batch, inst, occ, cfg, tol=1e-3, k_peaks=3, rel=3e-3, span=3, workers=None):
    """Same contract as verify.certify: (worst log10 ratio [E], f at worst [E], add[b] per instance)."""
    inst_np, occ_np = inst.cpu().numpy(), occ.cpu().numpy()
    cfg_d = _cfg_dict(cfg)
    jobs = []
    for e, b in enumerate(inst_np):
        mask, ent, P = _inst_data(batch, b)
        jobs.append((mask, ent, P, occ_np[e], cfg_d, batch.zt_par[b:b + 1], tol, k_peaks, rel, span))
    if len(jobs) <= 1 or (workers is not None and workers <= 1):
        _init()
        out = [_job(j) for j in jobs]
    else:
        out = _pool().map(_job, jobs, chunksize=1)
    worst = np.array([o[0] for o in out])
    fworst = np.array([o[1] for o in out])
    add = [[] for _ in range(batch.B)]
    for e, b in enumerate(inst_np):
        add[b] += out[e][2]
    return worst, fworst, add
