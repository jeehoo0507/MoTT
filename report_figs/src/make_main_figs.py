"""R2E 보고서 본문 그림 (강점 위주). 숫자는 R2E RESULTS.md (2026-11-05 07:44 KST) 기준, 손으로 옮김.
main1_speed_quality: 16포트 문제당 시간 vs decap 개수 (ip, pcb)
main2_size: ip·pcb 판 크기별 개수 ÷ 공진 규칙
main3_unseen: 처음 보는 pcb에서 입력 표현별 성공률 (seed 3개)
main4_p1: P1 피크 보정 seed 3개 묶음 효과
Usage: python make_main_figs.py <outdir>"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "font.size": 8, "axes.linewidth": 0.6, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
INK, MUTED, RULE = "#1f2328", "#6a737d", "#d0d7de"
OURS, OURS2, RULEC, SEARCH, LADDER = "#c2410c", "#2563a8", "#4b5563", "#9ca3af", "#93c5fd"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.tick_params(labelsize=7, width=0.6)
    ax.grid(color=RULE, lw=0.4); ax.set_axisbelow(True)


# ------------------------------------------------------------------ 1. 속도 vs 품질 (16포트)
def main1():
    # (방법, 문제당 초, 성공 문제의 평균 개수, 성공률, 색, 마커)
    D = {"ip 16포트": [("greedy+prune", 82.5, 73.9, 0.56, SEARCH, "s"), ("종류 인식 greedy", 188.6, 13.5, 1.00, SEARCH, "D"),
                     ("공진 규칙", 4.19, 20.3, 1.00, RULEC, "^"), ("MoTT", 1.84, 14.6, 1.00, OURS2, "o"),
                     ("MoTT + P1", 1.46, 14.2, 1.00, OURS, "o")],
         "pcb 16포트": [("greedy+prune", 61.9, 12.6, 0.92, SEARCH, "s"), ("종류 인식 greedy", 547.4, 65.1, 0.46, SEARCH, "D"),
                      ("공진 규칙", 4.86, 9.7, 1.00, RULEC, "^"), ("MoTT", 2.36, 9.2, 0.90, OURS2, "o"),
                      ("MoTT + P1", 1.85, 9.6, 0.96, OURS, "o")]}
    LAB = {"ip 16포트": {"MoTT": (1.3, 0.84, "left"), "MoTT + P1": (1.3, 1.17, "left"), "공진 규칙": (1.25, 1.12, "left")},
           "pcb 16포트": {"MoTT": (1.25, 0.78, "left"), "MoTT + P1": (1.0, 1.32, "center"), "공진 규칙": (1.3, 1.22, "left")}}
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"wspace": 0.28})
    for ax, (title, rows) in zip(axs, D.items()):
        for n, t, c, p, col, mk in rows:
            ax.scatter(t, c, s=40 + 80 * p, color=col, marker=mk, zorder=3, ec="white", lw=0.6)
            ox, oy, ha = LAB[title].get(n, (1.18, 1.0, "left"))
            ax.text(t * ox, c * oy, f"{n}\n{c:.1f}개 · {t:.1f} s · 성공 {p:.2f}", fontsize=6.4, color=col, ha=ha, va="center",
                    fontweight="bold" if "MoTT" in n else "normal", linespacing=1.15)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_yticks([7, 10, 20, 40, 70], ["7", "10", "20", "40", "70"]); ax.set_ylim(6, 95); ax.minorticks_off()
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.set_xlabel("문제당 시간 (초, log) — 왼쪽일수록 빠름"); ax.set_ylabel("놓은 decap 수 (log) — 아래일수록 적음")
        ax.set_xlim(0.8, 3000); ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=8)
        style(ax)
    for a in axs:
        a.text(0.98, 0.03, "왼쪽 아래일수록 빠르고 적게", ha="right", transform=a.transAxes, fontsize=6.8, color=OURS, fontweight="bold", alpha=0.8)
    fig.suptitle("MoTT는 탐색 기준선보다 30~300배 빠르면서 같거나 더 적은 decap으로 목표를 만족한다",
                 x=0.08, ha="left", y=1.03, fontsize=9.5, fontweight="bold")
    fig.text(0.08, -0.05, "같은 문제 50개, 같은 정확 검사, CPU 한 코어. 점 크기 = 성공률. 개수는 각 방법이 성공한 문제의 평균.",
             fontsize=6.8, color=MUTED)
    save(fig, "main1_speed_quality")


# ------------------------------------------------------------------ 2. 판 크기 (ip, pcb)
def main2():
    S = [10, 16, 20]
    rows = {"ip": {"MoTT + P1": ([0.71, 0.70, 0.72], [None, None, (0.67, 0.78)]), "MoTT": ([0.71, 0.73, 0.84], [None, None, (0.68, 1.10)]),
                   "greedy+prune": ([1.46, 5.37, 3.97], [None, None, None])},
            "pcb": {"MoTT": ([1.02, 1.00, 0.82], [None, None, (0.74, 0.92)]), "MoTT + P1": ([0.99, 0.99, 0.81], [None, None, (0.68, 0.95)]),
                    "greedy+prune": ([1.65, 1.37, 0.81], [None, None, (0.68, 0.94)])}}
    col = {"MoTT + P1": OURS, "MoTT": OURS2, "greedy+prune": SEARCH}
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"wspace": 0.28})
    for ax, (k, d) in zip(axs, rows.items()):
        ax.axhline(1.0, color=RULEC, lw=1.0, ls="--"); ax.text(10, 1.03, "공진 규칙 = 1", fontsize=6.6, color=RULEC)
        for n, (v, ci) in d.items():
            ax.plot(S, v, marker="o", ms=4, lw=1.4 if "MoTT" in n else 0.9, color=col[n], label=n, zorder=3)
            for x, y, c in zip(S, v, ci):
                if c: ax.errorbar(x, y, yerr=[[y - c[0]], [c[1] - y]], color=col[n], capsize=2.5, lw=0.9, zorder=2)
        ax.set_xticks(S, ["10", "16", "20*"]); ax.set_xlabel("판 크기 (포트)")
        ax.set_ylabel("놓은 decap 수 ÷ 공진 규칙 (둘 다 성공)")
        ax.set_yscale("log"); ax.set_yticks([0.6, 0.8, 1, 1.5, 2, 4], ["0.6", "0.8", "1", "1.5", "2", "4"]); ax.minorticks_off()
        style(ax)
    axs[0].set_title("ip: 모든 크기에서 규칙보다 약 30% 적게 (성공률 1.00)", loc="left", fontsize=8.6, fontweight="bold", pad=8)
    axs[1].set_title("pcb: 판이 커지면 규칙보다 18% 적게", loc="left", fontsize=8.6, fontweight="bold", pad=8)
    axs[0].legend(fontsize=6.6, frameon=False, loc="upper left")
    fig.text(0.08, -0.05, "*20포트는 종류당 16문제, 오차 막대 = 95% bootstrap CI. 10·16포트는 50문제.", fontsize=6.8, color=MUTED)
    save(fig, "main2_size")


# ------------------------------------------------------------------ 3. 처음 보는 pcb
def main3():
    held = {"좌표 입력\n(Park식, r0)": [0.60, 0.64, 0.68], "모달 특징\n(r1)": [0.80, 0.84, 0.88], "모드 토큰\n(MoTT, r2)": [0.92, 0.84, 0.94]}
    full = {"좌표 입력\n(Park식, r0)": [0.96, 0.94, 0.98], "모달 특징\n(r1)": [0.98, 0.84, 0.96], "모드 토큰\n(MoTT, r2)": [0.88, 0.78, 0.84]}
    cols = [LADDER, "#60a5fa", OURS]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for i, (k, v) in enumerate(held.items()):
        ax.bar(i, np.mean(v), width=0.55, color=cols[i], alpha=0.85, ec=cols[i])
        ax.scatter(np.full(3, i) + np.array([-0.1, 0, 0.1]), v, s=12, color=INK, zorder=3, lw=0)
        ax.text(i, np.mean(v) + 0.04, f"{np.mean(v):.2f}", ha="center", fontsize=8, fontweight="bold", color=cols[i] if i else "#3b82f6")
        f = np.mean(full[k]); ax.plot([i - 0.3, i + 0.3], [f, f], color=MUTED, lw=1.0, ls=":")
    ax.text(2.33, np.mean(full["모드 토큰\n(MoTT, r2)"]), "세 종류 다\n학습했을 때", fontsize=6.2, color=MUTED, va="center")
    ax.set_xticks(range(3), list(held)); ax.set_ylim(0.4, 1.05); ax.set_ylabel("처음 보는 pcb 성공률 (10포트)")
    ax.set_title("pcb를 빼고 학습해도 MoTT는 유지, 좌표 입력은 무너짐", loc="left", fontsize=8.8, fontweight="bold", pad=8)
    ax.text(0, 1.005, "seed 3개 (점), 사전 등록한 기준 통과: 2단 - 0단 > 0 seed 3개 모두", transform=ax.transAxes, fontsize=6.3, color=MUTED)
    style(ax); ax.grid(axis="x", visible=False)
    save(fig, "main3_unseen")


# ------------------------------------------------------------------ 4. P1
def main4():
    rows = [("같은 종류 pcb", [0.02, 0.14, 0.10], 0.087, (0.020, 0.153)), ("같은 종류 pkg", [0.06, 0.04, 0.18], 0.093, (0.007, 0.180)),
            ("같은 종류 ip", [-0.02, 0.00, -0.02], -0.013, (-0.033, 0.000)), ("처음 보는 pcb", [-0.04, 0.04, -0.04], -0.013, (-0.080, 0.053))]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for i, (n, s, m, ci) in enumerate(rows):
        c = OURS if ci[0] > 0 else MUTED
        ax.errorbar(m, i, xerr=[[m - ci[0]], [ci[1] - m]], fmt="o", color=c, ms=6, capsize=3, lw=1.4)
        ax.scatter(s, [i + 0.22] * 3, s=9, color=c, alpha=0.6, lw=0)
        ax.text(ci[1] + 0.01, i, f"{m:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", va="center", fontsize=6.8, color=c,
                fontweight="bold" if ci[0] > 0 else "normal")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(4), [r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(-0.12, 0.33)
    ax.set_xlabel("성공률 차 (피크 보정 - MoTT), seed 3개 묶음 95% CI")
    ax.set_title("P1 피크 보정: pcb·pkg 성공률 +9%p, 해로운 판 없음", loc="left", fontsize=8.8, fontweight="bold", pad=8)
    ax.text(0, 1.005, "작은 점 = seed별 차이 · 사전 등록 판정 통과 (seed 3개)", transform=ax.transAxes, fontsize=6.3, color=MUTED)
    style(ax); ax.grid(axis="y", visible=False)
    save(fig, "main4_p1")


OUT.mkdir(parents=True, exist_ok=True)
main1(); main2(); main3(); main4()
print("ok")
