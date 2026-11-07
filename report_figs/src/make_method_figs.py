"""Method figures for the R2E report (structure read from the code, not from memory):
fig_arch_v2  : MoTT block (decap_loop/model.py MoTTBlock / MoTT.forward, loop_model.py LoopMoTT)
fig_train    : distill -> RL with expert iteration (train.py docstring and rl loop)
fig_infer    : one episode + exact check + cutting plane + repair (decap_loop/sim.py, scripts/eval_policy.py)
Usage: python make_method_figs.py <outdir>"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 9, "savefig.dpi": 300,
                     "mathtext.fontset": "custom", "mathtext.rm": "Helvetica Neue",
                     "mathtext.it": "Helvetica Neue:italic", "mathtext.bf": "Helvetica Neue:bold"})
INK, MUTED, LINE = "#24292f", "#57606a", "#6e7781"
C = dict(attn="#dcefe0", ffn="#dbe6f3", emb="#fde8e8", head="#f3f4f6", edge="#ece4fb", phys="#fcf0d6", chk="#eef6ff")
EC = dict(attn="#4d8b5c", ffn="#4f6d93", emb="#c25b5b", head="#57606a", edge="#7c3aed", phys="#b7791f", chk="#2563a8")
MODE, CELL, DEC = "#c2410c", "#2563a8", "#15803d"


def box(ax, x, y, w, h, txt, kind="head", fs=9, bold=False, tc=INK, lw=1.0, ls="-", r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=C[kind], ec=EC[kind],
                                lw=lw, ls=ls, zorder=2))
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc, zorder=3,
                fontweight="bold" if bold else "normal", linespacing=1.3)


def arr(ax, p, q, col=LINE, lw=1.0, ls="-", rad=0.0, head=True):
    ax.annotate("", xy=q, xytext=p, zorder=1, arrowprops=dict(
        arrowstyle="-|>" if head else "-", color=col, lw=lw, ls=ls, mutation_scale=9, shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}"))


def plus(ax, x, y, r=0.012):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=0.9, zorder=3))
    ax.text(x, y - 0.001, "+", ha="center", va="center", fontsize=9, zorder=4)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ===================================================================== architecture
def arch():
    fig = plt.figure(figsize=(8.6, 9.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, H = 0.25, 0.058
    xm, xc, xd = 0.05, 0.40, 0.72                     # left edges of the three streams
    cm, cc, cd = xm + W / 2, xc + W / 2, xd + W / 2   # centres
    # inputs + embeddings
    for x, c, lab, col in [(xm, cm, "Mode tokens  $m_k$ (K = 8)", MODE), (xc, cc, "Cell tokens  $u_i$ (n)", CELL),
                           (xd, cd, "Decap-type tokens  $c_t$ (T)", DEC)]:
        ax.text(c, 0.025, lab, ha="center", fontsize=9.5, color=col, fontweight="bold")
        box(ax, x, 0.06, W, 0.045, "Embedding (MLP)", "emb")
        arr(ax, (c, 0.045), (c, 0.06))
    # edge branch
    box(ax, 0.30, 0.135, 0.20, 0.07, "Cell–mode edges $e_{ik}$\nMLP: bias $b^{h}_{ik}$ + message $v_{ik}$", "edge", fs=8.2)
    ax.text(0.40, 0.115, "coupling $g_{ik}$ · $|Z_{ii}|/Z_t$ · peak after each type", ha="center", fontsize=7.2,
            color=EC["edge"])
    # block frame
    ax.add_patch(FancyBboxPatch((0.02, 0.225), 0.96, 0.6, boxstyle="round,pad=0,rounding_size=0.015",
                                fc="none", ec="#9aa1a9", lw=1.0, ls=(0, (4, 3)), zorder=0))
    ax.text(0.975, 0.835, "× L", ha="right", fontsize=11, style="italic", color=INK)
    # mode stream
    y_sa, y_ca, y_ff = 0.27, 0.40, 0.53
    box(ax, xm, y_sa, W, H, "Norm → Self-Attention\nmodes ↔ modes", "attn", fs=8.4)
    box(ax, xm, y_ca, W, H, "Norm → Cross-Attention\nQ: modes   K,V: cells", "attn", fs=8.4)
    box(ax, xm, y_ff, W, H, "Norm → Feed-Forward", "ffn", fs=8.4)
    for y0, y1 in [(0.105, y_sa), (y_sa + H, y_ca), (y_ca + H, y_ff)]:
        arr(ax, (cm, y0), (cm, y1 - 0.024 if y1 != y_sa else y1))
    for yp in (y_ca - 0.012, y_ff - 0.012, y_ff + H + 0.03):
        plus(ax, cm, yp)
    # residual skips (left side)
    for y0, y1 in [(y_sa - 0.02, y_ca - 0.012), (y_ca - 0.012, y_ff - 0.012), (y_ff - 0.012, y_ff + H + 0.03)]:
        ax.plot([cm, xm - 0.015, xm - 0.015, cm - 0.012], [y0, y0, y1, y1], color=LINE, lw=0.8)
    arr(ax, (cm, y_ff + H), (cm, y_ff + H + 0.018))
    m_out = y_ff + H + 0.03
    # cell stream
    y_cca, y_cff = 0.62, 0.73
    box(ax, xc, y_cca, W, H, "Norm → Cross-Attention\nQ: cells   K,V: modes", "attn", fs=8.4)
    box(ax, xc, y_cff, W, H, "Norm → Feed-Forward", "ffn", fs=8.4)
    arr(ax, (cc, 0.105), (cc, y_cca))
    plus(ax, cc, y_cff - 0.014); plus(ax, cc, y_cff + H + 0.025)
    arr(ax, (cc, y_cca + H), (cc, y_cff - 0.026))
    arr(ax, (cc, y_cff + H), (cc, y_cff + H + 0.013))
    # decap stream
    box(ax, xd, y_cca, W, H, "Norm → Cross-Attention\nQ: decaps   K,V: modes", "attn", fs=8.4)
    arr(ax, (cd, 0.105), (cd, y_cca))
    plus(ax, cd, y_cca + H + 0.025)
    arr(ax, (cd, y_cca + H), (cd, y_cca + H + 0.013))
    ax.text(cd + 0.012, y_cca - 0.012, "reads modes,\nnever writes back", ha="left", va="top", fontsize=7.4, color=DEC)
    # cells -> mode cross-attn (K,V), before the cell update
    ax.plot([cc, cc - 0.06], [0.33, 0.33], color=CELL, lw=0.9)
    ax.plot([cc - 0.06, cc - 0.06], [0.33, y_ca + H / 2], color=CELL, lw=0.9)
    arr(ax, (cc - 0.06, y_ca + H / 2), (xm + W, y_ca + H / 2), col=CELL, lw=0.9)
    ax.text(xm + W + 0.01, y_ca + H / 2 + 0.008, "K, V (cells)", fontsize=7.6, color=CELL, ha="left", va="bottom")
    # updated modes -> cell / decap cross-attn
    ax.plot([cm, cm], [m_out + 0.012, 0.66 + 0.03], color=MODE, lw=0.9)
    ax.plot([cm, xd - 0.03], [0.69, 0.69], color=MODE, lw=0.9)
    arr(ax, (xc - 0.03, 0.69), (xc - 0.03, y_cca + H / 2), col=MODE, lw=0.9)
    arr(ax, (xc - 0.03, y_cca + H / 2), (xc, y_cca + H / 2), col=MODE, lw=0.9)
    arr(ax, (xd - 0.03, 0.69), (xd - 0.03, y_cca + H / 2), col=MODE, lw=0.9)
    arr(ax, (xd - 0.03, y_cca + H / 2), (xd, y_cca + H / 2), col=MODE, lw=0.9)
    ax.text(xc - 0.035, 0.70, "updated modes (K, V)", fontsize=7.6, color=MODE, ha="left", va="bottom")
    # edge -> both mode<-cell and cell<-mode attention
    arr(ax, (0.33, 0.205), (xm + W - 0.01, y_ca), col=EC["edge"], ls=(0, (3, 2)))
    arr(ax, (0.47, 0.205), (xc + 0.03, y_cca), col=EC["edge"], ls=(0, (3, 2)))
    ax.text(0.325, 0.31, "+ $b_{ik}$", fontsize=7.6, color=EC["edge"])
    ax.text(0.462, 0.52, "+ $b_{ik}$, + $v_{ik}$", fontsize=7.6, color=EC["edge"])
    # loop note
    ax.text(0.975, 0.215, "MoTT: L = 4 blocks (d 128, 8 heads)\nLoop variant: one shared block × R, "
            "input re-injected: $h_{r+1}$ = Block($h_r + h_0$)", fontsize=7.6, color=MUTED, ha="right", va="top",
            linespacing=1.35, zorder=5, bbox=dict(fc="white", ec="none", pad=1.5))
    # heads
    box(ax, 0.05, 0.875, 0.62, 0.075, "add $s_{it}$ = MLP[$u_i, c_t, u_i \\odot c_t$]   ·   remove $r_i$ = MLP($u_i$)   ·"
        "   stop = MLP[mean $u$, mean $m$]", "head", fs=8.4)
    box(ax, 0.72, 0.875, 0.25, 0.075, "physics prior\n$w_{\\Delta}\\Delta V_{it} - w_{m}\\,$margin$_{it}$", "phys", fs=8.2)
    ax.text(0.695, 0.912, "+", fontsize=12, ha="center", va="center")
    ax.text(0.5, 0.975, "Action: place (cell, type) / remove / stop   (masked softmax)", ha="center", fontsize=9.5,
            fontweight="bold")
    for x in (cm, cc):
        arr(ax, (x, 0.80 if x == cc else m_out + 0.012), (x, 0.875))
    ax.plot([cd, cd, 0.60], [y_cca + H + 0.037, 0.855, 0.855], color=LINE, lw=1.0); arr(ax, (0.60, 0.855), (0.60, 0.875))
    # prior comes straight from the cell features (exact one-step effect)
    ax.plot([xc + W + 0.01, xc + W + 0.01], [0.083, 0.86], color=EC["phys"], lw=0.9, ls=(0, (2, 2)))
    arr(ax, (xc + W + 0.01, 0.86), (0.80, 0.875), col=EC["phys"], ls=(0, (2, 2)))
    ax.text(xc + W + 0.016, 0.47, "exact 1-step\neffect (skip)", fontsize=7.2, color=EC["phys"], rotation=90,
            va="center")
    save(fig, "fig_arch_v2")


# ===================================================================== training
def train():
    fig = plt.figure(figsize=(10.5, 4.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.95, "Training: teacher → distillation (0.75 h) → RL with expert iteration (1.25 h)",
            fontsize=11, fontweight="bold")
    box(ax, 0.01, 0.55, 0.17, 0.25, "27 training boards\npcb · pkg · ip\nsides 8 / 10 / 12\n(one file per iteration,\nkinds mixed over time)",
        "emb", fs=8.6)
    box(ax, 0.21, 0.55, 0.17, 0.25, "Teacher (offline)\ngreedy+prune\n→ local search\ncertified solutions", "phys", fs=8.6)
    box(ax, 0.41, 0.55, 0.17, 0.25, "Distill (0.75 h)\nreplay teacher trajectory\nmax log-likelihood\n(no reward)",
        "ffn", fs=8.6)
    arr(ax, (0.18, 0.675), (0.21, 0.675)); arr(ax, (0.38, 0.675), (0.41, 0.675)); arr(ax, (0.58, 0.675), (0.61, 0.675))
    # RL loop
    ax.add_patch(FancyBboxPatch((0.61, 0.08), 0.38, 0.8, boxstyle="round,pad=0,rounding_size=0.015", fc="#fafbfc",
                                ec="#8c959f", lw=1.0, zorder=0))
    ax.text(0.80, 0.84, "RL (1.25 h): one iteration", ha="center", fontsize=9.5, fontweight="bold")
    steps = [("sample B = 16 boards × M = 8 rollouts", "attn"),
             ("reward  R = −count / N$_{ref}$ − λ·(fail + violation)", "head"),
             ("exact check (700 pts) of the best sample\n→ failing freqs added to working grid", "chk"),
             ("loss = REINFORCE (group baseline)\n+ imitation of best-known solution", "ffn"),
             ("expert iteration: certified sample with\nfewer decaps replaces the teacher", "phys")]
    y = 0.73
    for i, (t, k) in enumerate(steps):
        box(ax, 0.63, y, 0.34, 0.09, t, k, fs=8.2)
        if i < len(steps) - 1:
            arr(ax, (0.80, y), (0.80, y - 0.06))
        y -= 0.15
    arr(ax, (0.97, 0.155), (0.985, 0.155), head=False)
    ax.plot([0.985, 0.985], [0.155, 0.775], color=LINE, lw=1.0)
    arr(ax, (0.985, 0.775), (0.97, 0.775))
    ax.text(0.993, 0.46, "repeat", rotation=90, fontsize=7.5, color=MUTED, va="center")
    box(ax, 0.41, 0.12, 0.17, 0.2, "validation every 100 it\n(val boards, exact check)\n→ best.pt", "head", fs=8.4)
    arr(ax, (0.63, 0.22), (0.58, 0.22))
    ax.text(0.01, 0.42, "λ: Lagrange multiplier, raised while samples fail.\nN$_{ref}$: greedy+prune count of the board.\n"
            "Same budget for every model (ladder, loop, prior-work reimplementations).", fontsize=8, color=MUTED,
            va="top", linespacing=1.4)
    save(fig, "fig_train")


# ===================================================================== inference
def infer():
    fig = plt.figure(figsize=(10.5, 3.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.94, "Inference: one decision per step, exact physics between steps, exact check at the end",
            fontsize=11, fontweight="bold")
    xs = [0.01, 0.2, 0.39, 0.58]
    txt = [("board state\n(placed decaps)", "emb"), ("features\nmodes · cells · edges\n(rank-1 Z update)", "edge"),
           ("MoTT\nmasked action", "attn"), ("apply action\nplace / remove", "ffn")]
    for x, (t, k) in zip(xs, txt):
        box(ax, x, 0.5, 0.16, 0.24, t, k, fs=8.6)
    for a, b in zip(xs[:-1], xs[1:]):
        arr(ax, (a + 0.16, 0.62), (b, 0.62))
    ax.annotate("", xy=(0.09, 0.5), xytext=(0.66, 0.5), arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.0,
                connectionstyle="arc3,rad=-0.35", mutation_scale=9))
    ax.text(0.375, 0.27, "repeat until the policy chooses stop", ha="center", fontsize=8, color=MUTED)
    box(ax, 0.79, 0.62, 0.2, 0.15, "exact check\n700 freqs + peak refine\n|Z|/Zt ≤ 1.01 ?", "chk", fs=8.4)
    arr(ax, (0.74, 0.66), (0.79, 0.69))
    ax.text(0.765, 0.705, "stop", fontsize=7.6, color=MUTED, ha="center")
    box(ax, 0.79, 0.86, 0.2, 0.06, "pass → solution", "head", fs=8.4)
    arr(ax, (0.89, 0.77), (0.89, 0.86))
    box(ax, 0.79, 0.25, 0.2, 0.25, "fail → cutting plane:\nadd failing freqs to the\nworking grid and re-solve\n(≤ 4 rounds, every method)",
        "phys", fs=8)
    arr(ax, (0.89, 0.62), (0.89, 0.5))
    box(ax, 0.79, 0.03, 0.2, 0.17, "optional repair (≤ 4):\nsame method continues\nfrom its own placement", "phys",
        fs=8, ls=(0, (3, 2)))
    arr(ax, (0.89, 0.25), (0.89, 0.2))
    ax.text(0.01, 0.12, "Every method (MoTT, rule, greedy+prune, prior-work policies) uses the same exact check, "
            "cutting plane and repair rule.\nRepair is reported separately: policy-alone numbers compare models; "
            "with-repair numbers compare deployable pipelines.", fontsize=8, color=MUTED, linespacing=1.4)
    save(fig, "fig_infer")


OUT.mkdir(parents=True, exist_ok=True)
arch(); train(); infer()
print("ok")
