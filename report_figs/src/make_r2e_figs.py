"""R2E 보고서 그림 (한국어). 숫자는 R2E RESULTS.md (2026-11-05 07:44 KST) 기준, 손으로 옮김.
fig_tokens: MoTT 토큰 구성도 · fig_loop: 루프 vs 기본 트랜스포머 · fig_baselines: 10포트 기준선 비교 · fig_scale: 판 크기 곡선
Usage: python make_r2e_figs.py <outdir>"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "font.size": 8, "axes.linewidth": 0.6, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
INK, MUTED, RULE = "#1f2328", "#6a737d", "#d0d7de"
C = {"mode": "#c2410c", "cell": "#2563a8", "edge": "#7a9e7e", "dec": "#8b5cf6", "ours": "#c2410c", "loop": "#f59e0b",
     "rule": "#4b5563", "search": "#9ca3af", "prior": "#7c3aed", "r0": "#93c5fd", "r1": "#60a5fa"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1. 토큰 구성
def fig_tokens():
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.0); ax.axis("off")

    def box(x, y, w, h, title, lines, color, fill="white", lw=1.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=color, lw=lw))
        ax.text(x + 0.12, y + h - 0.12, title, ha="left", va="top", fontsize=9, fontweight="bold", color=color)
        for i, s in enumerate(lines):
            ax.text(x + 0.12, y + h - 0.45 - i * 0.26, s, ha="left", va="top", fontsize=7.2, color=INK)

    def arrow(p, q, text="", color=MUTED, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>", mutation_scale=10, color=color, lw=1.0,
                                     connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.08, text, ha="center", va="bottom", fontsize=6.8, color=color, style="italic")

    box(3.6, 3.05, 2.8, 1.75, "모드 토큰  (K = 8)", ["판의 상위 공진 피크", "· 피크 주파수 log(f / f_c)", "· 피크 높이 |Z_pp| / Zt",
                                                  "· 목표 위반 여부 · 유효 표시", "P1: + 보정 피크 높이 2채널"], C["mode"], "#fdf1ec", 1.8)
    box(0.2, 0.35, 3.0, 2.3, "칸 토큰  (n = 칸 수)", ["· 칸 플래그 (keepout · probe · 진입)", "· 이미 놓인 decap 종류",
                                                  "· 전역 상태 (만족 · 남은 수 · 채운 비율)", "· 정확한 1수 효과", "  = 종류별로 놓았을 때 위반 감소량",
                                                  "  (물리 사전 점수)"], C["cell"], "#eef4fb")
    box(6.8, 0.35, 3.0, 2.3, "decap 토큰  (T = 3 종류)", ["· log(SRF / f_c)", "· ESL · ESR · 1/(ωC)", "  (Zt로 정규화)", "",
                                                     "→ 피크 주파수와 SRF의", "   궁합을 보게 함"], C["dec"], "#f4f0fe")
    box(3.6, 0.35, 2.8, 1.55, "칸–모드 엣지  (n × K)", ["· 결합도 g = |Z_ip| / √(|Z_pp||Z_ii|)", "· |Z_ii| / Zt",
                                                     "· 종류별로 놓은 뒤 피크 높이", "→ 이 칸이 이 피크를 얼마나 누르나"], C["edge"], "#f1f6f1")
    arrow((3.2, 2.2), (3.6, 3.6), "", C["cell"], rad=-0.2)
    ax.text(2.55, 2.95, "cross-attn\n칸 ↔ 모드", ha="center", fontsize=6.8, color=C["cell"], style="italic")
    arrow((6.8, 2.2), (6.4, 3.6), "", C["dec"], rad=0.2)
    ax.text(7.45, 2.95, "cross-attn\ndecap ↔ 모드", ha="center", fontsize=6.8, color=C["dec"], style="italic")
    ax.add_patch(FancyArrowPatch((5.0, 1.9), (5.0, 3.05), arrowstyle="-|>", mutation_scale=10, color=C["edge"], lw=1.0))
    ax.text(5.08, 2.45, "attention 편향", fontsize=6.8, color=C["edge"], style="italic")
    ax.text(5.0, 4.93, "MoTT 입력: 좌표 · 고정 주파수 없음, 모두 현재 배치의 물리량", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(7.0, 4.55, "모드끼리 self-attention\n칸끼리는 attention 없음\n출력 = 학습 가중치 × 물리 사전 점수 + 학습 보정",
            ha="left", va="top", fontsize=7, color=MUTED, linespacing=1.4)
    save(fig, "fig_tokens")


# ------------------------------------------------------------------ 2. 루프 vs 기본
def fig_loop():
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.1), gridspec_kw={"wspace": 0.38, "width_ratios": [1.15, 1, 1]})
    # (a) seed별 10포트
    ax = axs[0]
    seeds = {"pcb": ([0.88, 0.78, 0.84], [0.84, 0.94, 0.96]), "pkg": ([0.72, 0.84, 0.62], [0.88, 0.78, 0.82]),
             "ip": ([1.00, 1.00, 1.00], [0.98, 1.00, 1.00])}
    for i, (k, (b, l)) in enumerate(seeds.items()):
        for j, (vals, c, dx, lab) in enumerate([(b, C["ours"], -0.17, "기본 (블록 4개)"), (l, C["loop"], 0.17, "루프 (블록 1개 × R)")]):
            ax.bar(i + dx, np.mean(vals) - 0.5, bottom=0.5, width=0.3, color=c, alpha=0.25, ec=c, lw=0.8, label=lab if i == 0 else None)
            ax.scatter(np.full(3, i + dx) + np.array([-0.06, 0, 0.06]), vals, s=10, color=c, zorder=3, lw=0)
            ax.text(i + dx, max(vals) + 0.015, f"{np.mean(vals):.2f}", ha="center", fontsize=6.6, color=c, fontweight="bold")
    ax.set_xticks(range(3), ["pcb", "pkg", "ip"]); ax.set_ylim(0.5, 1.08); ax.set_ylabel("정확 검사 성공률")
    ax.legend(fontsize=6.2, frameon=False, loc="lower left")
    ax.set_title("a  10포트, seed 3개", loc="left", fontsize=8.6, fontweight="bold", pad=10)
    ax.text(0, 1.01, "pkg seed 0: +0.16 [+0.06, +0.28] · 블록 파라미터 1/4", transform=ax.transAxes, fontsize=6.2, color=MUTED)
    # (b) 크기별 성공률 차 (루프 − 기본)
    ax = axs[1]
    sizes = [10, 16, 20]
    diff = {"pcb": [-0.04, -0.04, -0.25], "pkg": [0.16, 0.06, 0.12], "ip": [-0.02, 0.00, 0.12]}
    ci = {"pcb": [(-0.16, 0.08), (-0.18, 0.10), (-0.50, -0.06)], "pkg": [(0.06, 0.28), (-0.12, 0.24), (-0.19, 0.44)],
          "ip": [(-0.06, 0.00), (0.0, 0.0), (0.0, 0.38)]}
    cols = {"pcb": "#2563a8", "pkg": "#c2410c", "ip": "#7a9e7e"}
    for j, (k, v) in enumerate(diff.items()):
        x = np.arange(3) + (j - 1) * 0.18
        lo = [v[i] - ci[k][i][0] for i in range(3)]; hi = [ci[k][i][1] - v[i] for i in range(3)]
        ax.errorbar(x, v, yerr=[lo, hi], fmt="o", color=cols[k], ms=4, capsize=2, lw=1, label=k)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(3), ["10", "16", "20*"]); ax.set_xlabel("판 크기 (포트)  *20은 8~16문제")
    ax.set_ylabel("성공률 차 (루프 - 기본)"); ax.set_ylim(-0.6, 0.5)
    ax.legend(fontsize=6.2, frameon=False, loc="upper right", ncol=3)
    ax.set_title("b  크기별 차이 (seed 0, 95% CI)", loc="left", fontsize=8.6, fontweight="bold", pad=10)
    # (c) R 스윕 16포트
    ax = axs[2]
    R = [2, 4, 8, 16]
    for k, v, c in [("pcb 16", [0.92, 0.90, 0.86, 0.84], "#2563a8"), ("pkg 16", [0.58, 0.64, 0.58, 0.50], "#c2410c"),
                    ("ip 16", [1.00, 1.00, 1.00, 1.00], "#7a9e7e"), ("pkg 10", [0.74, 0.88, 0.88, 0.78], "#f59e0b")]:
        ax.plot(R, v, marker="o", ms=3.5, lw=1.1, color=c, label=k, ls="--" if "10" in k else "-")
    ax.set_xscale("log", base=2); ax.set_xticks(R, [str(r) for r in R]); ax.minorticks_off()
    ax.set_xlabel("평가 때 반복 수 R (학습 R 2~8)"); ax.set_ylabel("성공률"); ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=6.2, frameon=False, loc="lower left", ncol=2)
    ax.set_title("c  반복 수를 늘리면?", loc="left", fontsize=8.6, fontweight="bold", pad=10)
    ax.text(0, 1.01, "늘려도 오르지 않음 · R 2가 가장 빠름 (16포트 pcb 2.1 s vs R16 4.3 s)", transform=ax.transAxes, fontsize=6.2, color=MUTED)
    for a in axs:
        a.grid(axis="y", color=RULE, lw=0.4); a.set_axisbelow(True); a.tick_params(labelsize=7, width=0.6)
    save(fig, "fig_loop")


# ------------------------------------------------------------------ 3. 기준선 비교 (10포트)
def fig_baselines():
    # 성공률, 개수÷g+p, 문제당 초 (없으면 None)
    D = {
        "greedy+prune":   {"ip": (0.40, 1.00, 2.9), "pcb": (0.90, 1.00, 4.8), "pkg": (0.90, 1.00, 3.8), "g": "search"},
        "종류 인식 greedy": {"ip": (1.00, 0.74, 11.7), "pcb": (0.66, 4.83, 24.8), "pkg": (0.82, 1.59, 9.2), "g": "search"},
        "local search":   {"ip": (0.40, 0.75, 10.8), "pcb": (0.78, 0.75, 25.0), "pkg": (0.42, 0.80, 16.0), "g": "search"},
        "GA":             {"ip": (0.40, 1.02, 9.2), "pcb": (0.78, 0.96, 16.4), "pkg": (0.42, 0.88, 31.8), "g": "search"},
        "공진 규칙":        {"ip": (0.94, 1.10, 0.6), "pcb": (1.00, 0.84, 0.5), "pkg": (0.88, 0.86, 2.2), "g": "rule"},
        "Park식 (RL만)*":   {"ip": (0.12, 6.37, None), "pcb": (0.86, 8.14, None), "pkg": (0.86, 7.05, None), "g": "prior"},
        "Manoharan식 (RL만)*": {"ip": (0.96, 1.23, None), "pcb": (0.00, None, None), "pkg": (0.88, 6.59, None), "g": "prior"},
        "r0 좌표":          {"ip": (1.00, 1.05, 1.1), "pcb": (0.96, 0.80, 2.4), "pkg": (0.76, 0.77, 2.6), "g": "r0"},
        "r2 MoTT":         {"ip": (1.00, 0.74, 0.5), "pcb": (0.88, 0.88, 0.7), "pkg": (0.72, 0.86, 1.6), "g": "r2"},
        "r2 + P1 피크 보정": {"ip": (1.00, 0.74, 0.3), "pcb": (0.92, 0.85, 0.4), "pkg": (0.88, 0.87, 0.7), "g": "ours"},
    }
    gc = {"search": C["search"], "rule": C["rule"], "prior": C["prior"], "r0": C["r0"], "r2": "#2563a8", "ours": C["ours"]}
    names = list(D)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9), gridspec_kw={"wspace": 0.08}, sharey=True)
    for ax, k in zip(axs, ["ip", "pcb", "pkg"]):
        for i, n in enumerate(names):
            p, c, t = D[n][k]
            col = gc[D[n]["g"]]
            ax.barh(i, p, color=col, alpha=0.85 if D[n]["g"] == "ours" else 0.55, height=0.62, ec=col, lw=0.6)
            lab = f"{p:.2f}" + (f"  · {c:.2f}×" if c is not None else "") + (f"  · {t:.1f}s" if t is not None else "")
            ax.text(p + 0.02, i, lab, va="center", fontsize=6.2, color=INK)
        ax.set_xlim(0, 1.75); ax.set_xticks([0, 0.5, 1.0]); ax.axvline(1.0, color=RULE, lw=0.6)
        ax.set_title(f"{k} 10포트", loc="left", fontsize=9, fontweight="bold", pad=6)
        ax.set_xlabel("정확 검사 성공률")
        ax.tick_params(labelsize=7, width=0.6); ax.grid(axis="x", color=RULE, lw=0.4); ax.set_axisbelow(True)
    axs[0].set_yticks(range(len(names)), names, fontsize=7.4); axs[0].invert_yaxis()
    fig.suptitle("기준선 비교 — 막대 = 성공률, 글자 = 성공률 · 개수÷greedy+prune (둘 다 성공) · 문제당 초 (CPU 한 코어)",
                 x=0.12, ha="left", y=1.02, fontsize=9, fontweight="bold")
    fig.text(0.12, -0.04, "*선행 재구현: 정책 구조·입력만 우리 문제(목표 만족 + 개수 최소)로 옮김, 같은 2시간 예산. 원 논문 재현 아님 "
             "(Manoharan 원 예산 약 24 h, critic + replay).", fontsize=6.8, color=MUTED)
    save(fig, "fig_baselines")


# ------------------------------------------------------------------ 4. 판 크기 곡선
def fig_scale():
    S = [10, 16, 20]
    # 개수비 vs 공진 규칙 (둘 다 성공), 성공률, 문제당 초
    data = {
        "ip": {"공진 규칙": ([0.94, 1.00, 1.00], [1, 1, 1], [0.62, 4.19, 8.30]),
               "greedy+prune": ([0.40, 0.56, 0.62], [1.46, 5.37, 3.97], [2.94, 82.5, 72.1]),
               "r2 MoTT": ([1.00, 1.00, 0.94], [0.71, 0.73, 0.84], [0.52, 1.84, 23.2]),
               "r2 + P1 (seed 1)": ([1.00, 1.00, 1.00], [0.71, 0.70, 0.72], [0.32, 1.46, 3.92])},
        "pcb": {"공진 규칙": ([1.00, 1.00, 1.00], [1, 1, 1], [0.50, 4.86, 1.71]),
                "greedy+prune": ([0.90, 0.92, 1.00], [1.65, 1.37, 0.81], [4.84, 61.9, 1.32]),
                "r2 MoTT": ([0.88, 0.90, 1.00], [1.02, 1.00, 0.82], [0.68, 2.36, 1.11]),
                "r2 + P1 (seed 1)": ([0.92, 0.96, 0.88], [0.99, 0.99, 0.81], [0.36, 1.85, 1.27])},
        "pkg": {"공진 규칙": ([0.88, 0.60, 0.56], [1, 1, 1], [2.20, 8.51, 7.36]),
                "greedy+prune": ([0.90, 0.82, 0.75], [1.21, 1.73, 1.28], [3.79, 130.9, 564.8]),
                "r2 MoTT": ([0.72, 0.52, 0.31], [1.02, 0.85, 1.00], [1.59, 4.15, 5.90]),
                "r2 + P1 (seed 0)": ([0.78, 0.58, 0.69], [1.00, 0.98, 1.14], [0.70, 3.28, 4.45])},
    }
    col = {"공진 규칙": C["rule"], "greedy+prune": C["search"], "r2 MoTT": "#2563a8"}
    fig, axs = plt.subplots(3, 3, figsize=(10, 7.4), gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    for r, k in enumerate(["ip", "pcb", "pkg"]):
        for c, (idx, ylab, logy) in enumerate([(0, "성공률", False), (1, "개수 ÷ 공진 규칙", False), (2, "문제당 초 (log)", True)]):
            ax = axs[r, c]
            for m, v in data[k].items():
                cc = col.get(m, C["ours"])
                ax.plot(S, v[idx], marker="o", ms=3.8, lw=1.3 if "P1" in m else 1.0, color=cc, label=m)
            if logy: ax.set_yscale("log")
            if idx == 0: ax.set_ylim(0, 1.05)
            if idx == 1: ax.axhline(1, color=C["rule"], lw=0.6, ls=":")
            ax.set_xticks(S, ["10", "16", "20*"]); ax.set_ylabel(ylab, fontsize=7.5)
            ax.grid(axis="y", color=RULE, lw=0.4); ax.set_axisbelow(True); ax.tick_params(labelsize=7, width=0.6)
            if c == 0: ax.set_title(f"{k}", loc="left", fontsize=9.5, fontweight="bold", pad=6)
    axs[0, 0].legend(fontsize=6.4, frameon=False, loc="lower right")
    axs[2, 1].set_xlabel("판 크기 (포트)   *20은 종류당 8~16문제, CI 넓음", fontsize=7.5)
    fig.suptitle("판 크기가 커질 때: ip는 모든 크기에서 규칙보다 약 30% 적게, pcb는 20포트에서 18% 적게, pkg는 성공률이 약점",
                 x=0.07, ha="left", y=0.995, fontsize=9.2, fontweight="bold")
    save(fig, "fig_scale")


OUT.mkdir(parents=True, exist_ok=True)
fig_tokens(); fig_loop(); fig_baselines(); fig_scale()
print("ok")
