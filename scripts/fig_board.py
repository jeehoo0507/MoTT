"""Figure 1: one trained policy, three board kinds. For each kind, one evaluation instance:
left  the board (valid cells, keep-out, probes, power entries) with the policy's decaps (marker per type),
right |Z_probe(f)| / Zt(f) before (bare board) and after the decaps on the exact 700-point check grid,
      with the target (= 1) as a line.

Inputs: a pool file per kind and the eval record of one method (scripts/eval_policy.py JSON, which
stores occ per instance). Instance choice: --pick best (largest certified count that succeeded) or index.

Run: .venv/bin/python scripts/fig_board.py --pool runs/pool_v1 --records runs/<day>/<run>/eval/eval_pcb20_0__<m>.json,... --out reports/fig1_board.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decap_loop.bank import load_bank  # noqa: E402
from decap_loop.pdn import target_at  # noqa: E402
from decap_loop.verify import _inst_data, _probe_z_exact  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd7"
KIND = {"pcb": "PCB-like", "pkg": "package-like", "ip": "interposer-like"}   # English: no Korean font on the server


def draw(ax_b, ax_z, batch, rec, b):
    import torch
    torch.set_num_threads(1)
    cfg = batch.cfg
    H, W = batch.H, batch.W
    mask, ent, P = _inst_data(batch, b)
    occ = np.asarray(rec["occ"][b])
    f = cfg.check_freqs()
    zt = target_at(batch.zt_par[b:b + 1], f)[0]
    bare = (_probe_z_exact(mask, ent, P, None, cfg, f) / zt[:, None]).max(1)
    after = (_probe_z_exact(mask, ent, P, occ, cfg, f) / zt[:, None]).max(1)
    # board
    img = np.full((H, W, 3), 1.0)
    img[~mask] = np.array([0.93, 0.93, 0.92])
    ko = batch.keepout[b].view(H, W).numpy()
    img[ko & mask] = np.array([0.85, 0.85, 0.83])
    ax_b.imshow(img, origin="upper", interpolation="nearest")
    for y in range(H + 1):
        ax_b.axhline(y - 0.5, color=GRID, lw=0.4)
    for x in range(W + 1):
        ax_b.axvline(x - 0.5, color=GRID, lw=0.4)
    for t in range(cfg.n_types):
        cells = np.flatnonzero(occ == t)
        c, esl, _ = cfg.decaps[t]
        ax_b.scatter(cells % W, cells // W, s=55, color=SERIES[t], edgecolor="white", lw=0.8, zorder=3,
                     label=f"decap {t}: {c*1e12:.0f} pF" if c < 1e-9 else f"decap {t}: {c*1e9:.0f} nF")
    ax_b.scatter(P % W, P // W, marker="*", s=140, color=INK, zorder=4, label="probe")
    ax_b.scatter(ent % W, ent // W, marker="s", s=40, facecolor="none", edgecolor=INK2, lw=1.2, zorder=4, label="power entry")
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for s in ax_b.spines.values():
        s.set_visible(False)
    n_dec = int((occ >= 0).sum())
    ax_b.set_title(f"{KIND.get(rec['tech'], rec['tech'])} {H}x{W} #{b}, {rec['method']}: {n_dec} decaps", fontsize=10, color=INK, loc="left")
    # impedance
    ax_z.plot(f, bare, color=INK2, lw=1.2, ls="--", label="bare board")
    ax_z.plot(f, after, color=SERIES[0], lw=2, label="with decaps")
    ax_z.axhline(1.0, color=INK, lw=1, label="target Zt")
    ax_z.set_xscale("log"); ax_z.set_yscale("log")
    ax_z.set_xlabel("frequency (Hz)", color=INK2, fontsize=9)
    ax_z.set_ylabel("max |Z_probe| / Zt(f)", color=INK2, fontsize=9)   # Zt(f) is piecewise -> steps in the ratio
    ax_z.grid(True, which="major", color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax_z.spines[s].set_visible(False)
    ax_z.tick_params(colors=INK2, labelsize=8)
    ax_z.text(0.02, 0.95, f"exact-check worst |Z|/Zt = {after.max():.3f}" + ("" if rec["ok"][b] else "  (fails)"), transform=ax_z.transAxes, va="top", fontsize=8, color=INK)
    return n_dec, after.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="runs/pool_v1")
    ap.add_argument("--records", required=True, help="comma list of eval record JSONs (one per kind)")
    ap.add_argument("--pick", default="best", help="best | median | <index>")
    ap.add_argument("--out", default="reports/fig1_board.png")
    ap.add_argument("--title", default="", help="override the suptitle (e.g. two methods on one instance)")
    a = ap.parse_args()
    recs = [json.load(open(r)) for r in a.records.split(",")]
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(len(recs), 2, figsize=(9, 3.2 * len(recs)), gridspec_kw=dict(width_ratios=[1, 1.5]))
    axes = np.atleast_2d(axes)
    for i, rec in enumerate(recs):
        batch, _ = load_bank(os.path.join(a.pool, rec["set"] + ".pkl"))
        ok = np.asarray(rec["ok"], bool)
        cnt = np.asarray(rec["count"])
        if a.pick == "best":
            b = int(np.argmax(np.where(ok, cnt, -1)))
        elif a.pick == "median":
            idx = np.flatnonzero(ok)
            b = int(idx[np.argsort(cnt[idx])[len(idx) // 2]])
        else:
            b = int(a.pick)
        n_dec, worst = draw(axes[i, 0], axes[i, 1], batch, rec, b)
        print(f"{rec['set']} instance {b}: {n_dec} decaps, certified worst {worst:.3f}, method {rec['method']}")
        # decap library differs per kind -> type legend on every row (type index != same part across kinds)
        axes[i, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        if i == 0:
            axes[i, 1].legend(fontsize=7, frameon=False, loc="lower left")
    title = a.title or f"One policy ({recs[0]['method']}) on three board kinds: placement and impedance"
    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
