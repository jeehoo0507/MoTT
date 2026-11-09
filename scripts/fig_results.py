"""Figures 2 and 3 from the eval records (scripts/eval_policy.py JSON) — numbers come from the logs only.

fig 2 (big boards, 16x16 = 256 and 20x20 = 400 cells): per kind, two panels — seconds per instance
      (log axis) and decap count / greedy+prune (paired, jointly solved) — one bar per method.
fig 3 (ladder): per kind x size, success rate (exact check) and count / greedy+prune for the rungs
      in fixed order (rung 0, 1, 2, 3) plus the two classical baselines as reference lines.

Run: .venv/bin/python scripts/fig_results.py --fig 2 --records 'runs/2026-10-28/*/eval/*.json' \
         --methods greedy+prune,local_search,r0_coord_rl,r1_modal_rl --sizes 16,20 --out reports/fig2_bigboard.png
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd7"
KIND = {"pcb": "PCB-like", "pkg": "package-like", "ip": "interposer-like"}


def load(pattern):
    """{(tech, size, method): merged record over the _0/_1 files of a set}.
    Keyed by pool file too (a re-run of the same file replaces, not doubles), so a method evaluated on
    fewer files than greedy+prune is paired only on the files both have."""
    per = defaultdict(dict)
    for f in sorted(glob.glob(pattern, recursive=True)):
        r = json.load(open(f))
        per[(r["tech"], int(r["size"]), r["method"])][r["set"]] = r
    out = {}
    for k, sets in per.items():
        rs = [sets[s] for s in sorted(sets)]
        out[k] = dict(ok=sum((r["ok"] for r in rs), []), count=sum((r["count"] for r in rs), []),
                      sec=[r["sec_per_inst"] for r in rs], n=sum(r["n"] for r in rs), sets=dict(sets))
    return out


def paired_ratio(rec, ref):
    common = sorted(set(rec["sets"]) & set(ref["sets"]))
    ok = np.concatenate([np.asarray(rec["sets"][s]["ok"], bool) for s in common]) if common else np.zeros(0, bool)
    cnt = np.concatenate([np.asarray(rec["sets"][s]["count"], float) for s in common]) if common else np.zeros(0)
    rok = np.concatenate([np.asarray(ref["sets"][s]["ok"], bool) for s in common]) if common else np.zeros(0, bool)
    rcnt = np.concatenate([np.asarray(ref["sets"][s]["count"], float) for s in common]) if common else np.zeros(0)
    both = ok & rok
    if not both.any():
        return np.nan, np.nan
    r = cnt[both] / np.maximum(rcnt[both], 1)
    # paired bootstrap 95% CI of the mean ratio
    rng = np.random.default_rng(0)
    bs = [r[rng.integers(len(r), size=len(r))].mean() for _ in range(2000)]
    return float(r.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def fig2(recs, methods, sizes, out):
    techs = [t for t in ("pcb", "pkg", "ip") if any(k[0] == t for k in recs)]
    # rows: time, success (so the count ratio, taken on problems both solved, never hides failures), count ratio
    fig, axes = plt.subplots(3, len(techs), figsize=(3.4 * len(techs), 8.0), squeeze=False)
    width = 0.8 / len(methods)
    for j, t in enumerate(techs):
        ax_t, ax_s, ax_c = axes[0, j], axes[1, j], axes[2, j]
        for i, m in enumerate(methods):
            xs, secs, succ, ns, rels, los, his = [], [], [], [], [], [], []
            for s_i, s in enumerate(sizes):
                rec = recs.get((t, s, m))
                ref = recs.get((t, s, "greedy+prune"))
                if rec is None or ref is None:
                    continue
                xs.append(s_i + (i - (len(methods) - 1) / 2) * width)
                secs.append(np.mean(rec["sec"]))
                succ.append(np.mean(rec["ok"]))
                ns.append(len(rec["ok"]))
                mean, ci = paired_ratio(rec, ref)
                rels.append(mean)
                los.append(mean - ci[0] if ci == ci else 0)
                his.append(ci[1] - mean if ci == ci else 0)
            if not xs:
                continue
            ax_t.bar(xs, secs, width * 0.92, color=SERIES[i], label=m)
            ax_s.bar(xs, succ, width * 0.92, color=SERIES[i], label=m)
            for x, v, n in zip(xs, succ, ns):
                ax_s.text(x, v, f"{v:.2f}\nn={n}", ha="center", va="bottom", fontsize=5, color=INK2)
            ax_c.bar(xs, rels, width * 0.92, color=SERIES[i], yerr=[los, his], error_kw=dict(lw=0.8, color=INK2), label=m)
            for x, v in zip(xs, secs):
                ax_t.text(x, v, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=6, color=INK2)
        ax_t.set_yscale("log")
        ax_t.set_title(KIND[t], fontsize=10, color=INK, loc="left")
        ax_c.axhline(1.0, color=INK, lw=0.8)
        ax_s.set_ylim(0, 1.2)
        for ax in (ax_t, ax_s, ax_c):
            style(ax)
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels([f"{s}x{s} ({s*s} cells)" for s in sizes])
        if j == 0:
            ax_t.set_ylabel("seconds per board (1 CPU core)", color=INK2, fontsize=9)
            ax_s.set_ylabel("success rate (exact check)", color=INK2, fontsize=9)
            ax_c.set_ylabel("count / greedy+prune (both solved, 95% CI)", color=INK2, fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Big boards: time and decap count, same calculator for every method", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def fig3(recs, methods, sizes, out):
    techs = [t for t in ("pcb", "pkg", "ip") if any(k[0] == t for k in recs)]
    base = ["greedy+prune", "local_search"]
    rungs = [m for m in methods if m not in base]
    fig, axes = plt.subplots(2, len(techs), figsize=(3.4 * len(techs), 5.6), squeeze=False)
    width = 0.8 / max(1, len(rungs))
    for j, t in enumerate(techs):
        ax_s, ax_c = axes[0, j], axes[1, j]
        for i, m in enumerate(rungs):
            xs, succ, rels, los, his = [], [], [], [], []
            for s_i, s in enumerate(sizes):
                rec, ref = recs.get((t, s, m)), recs.get((t, s, "greedy+prune"))
                if rec is None or ref is None:
                    continue
                xs.append(s_i + (i - (len(rungs) - 1) / 2) * width)
                succ.append(np.mean(rec["ok"]))
                mean, ci = paired_ratio(rec, ref)
                rels.append(mean)
                los.append(mean - ci[0] if ci == ci else 0)
                his.append(ci[1] - mean if ci == ci else 0)
            if xs:
                ax_s.bar(xs, succ, width * 0.92, color=SERIES[i], label=m)
                ax_c.bar(xs, rels, width * 0.92, color=SERIES[i], yerr=[los, his], error_kw=dict(lw=0.8, color=INK2), label=m)
        for bi, m in enumerate(base):
            ys, yc = [], []
            for s in sizes:
                rec, ref = recs.get((t, s, m)), recs.get((t, s, "greedy+prune"))
                ys.append(np.mean(rec["ok"]) if rec else np.nan)
                yc.append(paired_ratio(rec, ref)[0] if rec and ref else np.nan)
            ax_s.plot(range(len(sizes)), ys, ls="--" if bi else "-", color=INK, lw=1, marker="_", ms=14, label=m)
            ax_c.plot(range(len(sizes)), yc, ls="--" if bi else "-", color=INK, lw=1, marker="_", ms=14, label=m)
        ax_s.set_ylim(0, 1.05)
        ax_s.set_title(KIND[t], fontsize=10, color=INK, loc="left")
        for ax in (ax_s, ax_c):
            style(ax)
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels([f"{s}x{s}" for s in sizes])
        if j == 0:
            ax_s.set_ylabel("success rate (exact 700-pt check)", color=INK2, fontsize=9)
            ax_c.set_ylabel("decap count / greedy+prune (paired, 95% CI)", color=INK2, fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Ladder ablation: same data, same training, one component added per rung", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def seed_names(m):
    """r2_mott_rl -> [r2_mott_rl, r2_mott_s1_rl, r2_mott_s2_rl] (seed 0 has no suffix, as in agent/day2_queue.json)."""
    stem = m[:-3] if m.endswith("_rl") else m
    return [m, f"{stem}_s1_rl", f"{stem}_s2_rl"]


def fig3_seeds(recs, methods, size, out, ref="gp_type_aware", lines=(("gp_type_aware", "E0+ type-aware greedy", "-"),
                                                                     ("modal_rule", "modal rule (no learning)", "--"))):
    """Fig 3 with seeds (QUEUE 10/29 item 3): one board size, x = rung. Bar = mean over the seeds that have records,
    dots = each seed. Top: success rate; bottom: decap count / `ref` on problems both solved (per seed, paired).
    Rungs are named (with their value) on the x axis, so identity is never colour alone; reference methods are
    horizontal lines in the legend."""
    techs = [t for t in ("pcb", "pkg", "ip") if any(k[0] == t and k[1] == size for k in recs)]
    fig, axes = plt.subplots(2, len(techs), figsize=(3.6 * len(techs), 5.8), squeeze=False)
    rng = np.random.default_rng(0)
    for j, t in enumerate(techs):
        ax_s, ax_c = axes[0, j], axes[1, j]
        rf = recs.get((t, size, ref))
        # only the four ladder rungs are "rung k"; other r2* runs (e.g. the adopted r2pk_rl) keep their name
        names = [f"rung {m[1]}" if m in ("r0_coord_rl", "r1_modal_rl", "r2_mott_rl", "r3_loop_rl") else m for m in methods]
        ticks = {ax_s: list(names), ax_c: list(names)}
        for i, m in enumerate(methods):
            succ, rel = [], []
            for name in seed_names(m):
                rec = recs.get((t, size, name))
                if rec is None:
                    continue
                succ.append(np.mean(rec["ok"]))
                rel.append(paired_ratio(rec, rf)[0] if rf else np.nan)
            for ax, v in ((ax_s, succ), (ax_c, rel)):
                v = [x for x in v if x == x]
                if not v:
                    continue
                ax.bar(i, np.mean(v), 0.62, color=SERIES[i], alpha=0.85)
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=INK, zorder=3, lw=0)
                ticks[ax][i] += f"\n{np.mean(v):.2f} (n={len(v)})"
        for m, label, ls in lines:
            rec = recs.get((t, size, m))
            if rec is None:
                continue
            for ax, y in ((ax_s, np.mean(rec["ok"])), (ax_c, paired_ratio(rec, rf)[0] if rf and m != ref else (1.0 if m == ref else np.nan))):
                if y != y:
                    continue
                ax.axhline(y, color=INK, lw=0.9, ls=ls, label=label)
        ax_s.set_ylim(0, 1.12)
        ax_s.set_title(f"{KIND[t]} {size}x{size}", fontsize=10, color=INK, loc="left")
        for ax in (ax_s, ax_c):
            style(ax)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(ticks[ax], fontsize=7)
        if j == 0:
            ax_s.set_ylabel("success rate (exact check), seed mean", color=INK2, fontsize=9)
            ax_c.set_ylabel("decap count / E0+ (both solved)", color=INK2, fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor=GRID, loc="lower left")
    fig.text(0.99, 0.005, "bar = seed mean (value under each rung), dots = seeds; lines = reference methods (legend)",
             fontsize=7, color=INK2, ha="right")
    fig.suptitle("Ladder ablation, 3 seeds: same data and training budget, one component added per rung",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, choices=[2, 3], required=True)
    ap.add_argument("--seeds", action="store_true", help="fig 3: seed mean + per-seed dots, one size (first of --sizes)")
    ap.add_argument("--records", required=True, help="glob of eval JSON records")
    ap.add_argument("--methods", required=True, help="comma list, in display order")
    ap.add_argument("--sizes", default="16,20")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = load(a.records)
    methods = a.methods.split(",")
    sizes = [int(s) for s in a.sizes.split(",")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    if a.fig == 3 and a.seeds:
        fig3_seeds(recs, methods, sizes[0], a.out)
    else:
        (fig2 if a.fig == 2 else fig3)(recs, methods, sizes, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
