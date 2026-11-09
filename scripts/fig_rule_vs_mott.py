"""QUEUE 11/05 08:05 KST: report figure "same board, modal_rule vs MoTT + P1 (r2pk_rl)" — placement and |Z|/Zt of the
two methods on one instance, stacked (same look as reports/fig_ip_gp_fail_ip10_0_13.png, drawing = fig_board.draw).
No new evaluation: the eval records' stored occ only.

Instance rule (fixed in the QUEUE before looking at any result): among problems both methods pass on the exact check,
the one whose count difference (modal_rule − r2pk_rl) is closest to that file's median difference; ties -> smallest
index. Usage: fig_rule_vs_mott.py <rule record.json> <mott record.json> <out.png>"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_board import INK, KIND, draw  # noqa: E402
from decap_loop.bank import load_bank  # noqa: E402

LABEL = {"modal_rule": "modal_rule", "r2pk_rl": "MoTT+P1"}


def pick(rule, mott):
    ok = np.asarray(rule["ok"], bool) & np.asarray(mott["ok"], bool)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None, None, idx
    d = np.asarray(rule["count"])[idx] - np.asarray(mott["count"])[idx]
    med = float(np.median(d))
    b = int(idx[np.argmin(np.abs(d - med))])   # argmin returns the first (smallest index) on ties
    return b, med, idx


def main():
    rule, mott = (json.load(open(p)) for p in sys.argv[1:3])
    out = sys.argv[3]
    assert rule["set"] == mott["set"] and rule["method"] == "modal_rule" and mott["method"] == "r2pk_rl"
    b, med, idx = pick(rule, mott)
    if b is None:
        print(f"{rule['set']}: no problem passed by both methods -> skipped")
        return
    nr, nm = int(rule["count"][b]), int(mott["count"][b])
    print(f"{rule['set']}: both pass {len(idx)}/{len(rule['ok'])}, median count diff (rule - MoTT+P1) {med:+.1f}, "
          f"pick #{b}: modal_rule {nr} vs MoTT+P1 {nm}")
    batch, _ = load_bank(os.path.join("runs/pool_v1", rule["set"] + ".pkl"))
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), gridspec_kw=dict(width_ratios=[1, 1.5]))
    for i, rec in enumerate((rule, mott)):
        n_dec, worst = draw(axes[i, 0], axes[i, 1], batch, rec, b)
        assert n_dec == int(rec["count"][b]), (n_dec, rec["count"][b])
        print(f"  {rec['method']}: {n_dec} decaps, exact-check worst {worst:.3f}")
        axes[i, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        if i == 0:
            axes[i, 1].legend(fontsize=7, frameon=False, loc="lower left")
    H, W = batch.H, batch.W
    fig.suptitle(f"{KIND.get(rule['tech'], rule['tech'])} {H}x{W} #{b}, same instance: modal_rule {nr} decaps vs "
                 f"MoTT+P1 {nm} decaps\n(instance = median count difference among {len(idx)} problems both pass)",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
