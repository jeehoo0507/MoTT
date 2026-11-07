"""R2E 탐구 결과 한 장 (MaskedKD fig_explorations와 같은 꼴). 숫자는 RESULTS.md / 토큰 분석 결과.md에서 손으로 옮김.
A 입력 특징 AUC 뒤집힘 · B 입력 사다리 (처음 보는 pcb) · C 시도한 수정 13개 (seed 0 짝 차이) · D P1 3 seed 묶음 ·
E 판 크기별 개수 ÷ 공진 규칙 · F 루프 반복 수 R
Usage: python make_r2e_explorations.py <outdir>"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "font.size": 7.6, "axes.linewidth": 0.6, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
INK, MUTED, RULE = "#1f2328", "#6a737d", "#d0d7de"
OURS, BLUE, GREEN, GRAY, RED = "#c2410c", "#2563a8", "#2f855a", "#8c959f", "#b42318"
PCB, PKG, IP = "#c97b2a", "#2f855a", "#3d6e8f"


def head(ax, tag, title, sub):
    ax.set_title(f"{tag}  {title}", loc="left", fontsize=9, fontweight="bold", color=INK, pad=16)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=6.6, color=MUTED, va="bottom")


def style(ax, grid="y"):
    ax.tick_params(labelsize=6.8, width=0.6)
    ax.grid(axis=grid, color=RULE, lw=0.4)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(11, 12.2))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 1.25, 1], hspace=0.62, wspace=1.3)

# ---------------------------------------------------------------- A token AUC flip
ax = fig.add_subplot(gs[0, 0:3])
rows = [("좌표", 0.77, 0.46), ("loop L", 0.70, 0.43), ("고정 8점 |Z|", 0.58, 0.06), ("모달 결합도 g", 0.78, 0.57),
        ("모달 참여비", 0.56, 0.87), ("정확한 1수 효과", 0.59, 0.98)]
ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
for n, a, b in rows:
    c = RED if b < 0.5 else (OURS if b > a else BLUE)
    ax.plot([0, 1], [a, b], color=c, lw=1.6, marker="o", ms=4)
    yb = b + {"좌표": 0.022, "loop L": -0.022}.get(n, 0)
    ax.text(1.05, yb, f"{n}  {b:.2f}", va="center", fontsize=6.6, color=c, fontweight="bold" if c != BLUE else "normal")
ax.set_xticks([0, 1], ["PCB형", "인터포저형"]); ax.set_xlim(-0.15, 1.75); ax.set_ylim(0, 1.05)
ax.set_ylabel("선생님 자리 예측 AUC")
ax.text(-0.13, 0.52, "무작위", fontsize=6.2, color=MUTED)
head(ax, "A", "좌표 입력은 판 종류가 바뀌면 뒤집힌다", "학습 없는 분석, 20×20, 종류당 판 10개 → 모드 토큰 + 물리 1수 효과로 설계")
style(ax)

# ---------------------------------------------------------------- B ladder
ax = fig.add_subplot(gs[0, 3:6])
held = [("좌표\n(r0, Park식)", [0.60, 0.64, 0.68], "#93c5fd"), ("모달 특징\n(r1)", [0.80, 0.84, 0.88], "#60a5fa"),
        ("모드 토큰\n(MoTT, r2)", [0.92, 0.84, 0.94], OURS)]
for i, (n, v, c) in enumerate(held):
    ax.bar(i, np.mean(v), width=0.55, color=c, alpha=0.9)
    ax.scatter(np.full(3, i) + np.array([-0.1, 0, 0.1]), v, s=11, color=INK, zorder=3, lw=0)
    ax.text(i, np.mean(v) + 0.035, f"{np.mean(v):.2f}", ha="center", fontsize=8, fontweight="bold", color=c if i else BLUE)
ax.set_xticks(range(3), [h[0] for h in held]); ax.set_ylim(0.4, 1.05); ax.set_ylabel("처음 보는 pcb 성공률")
head(ax, "B", "pcb를 빼고 학습해도 모드 토큰은 버틴다", "seed 3개 (점), 사전 등록 기준 통과 · ip는 둘 다 ~1.00이라 구분 불가, pkg는 실패")
style(ax)

# ---------------------------------------------------------------- C attempts forest
ax = fig.add_subplot(gs[1, :])
att = [  # name, kind(group), pcb (m, lo, hi), pkg (m, lo, hi), verdict
    ("주파수 궁합 편향", "구조", (0.08, 0.02, 0.16), (0.12, 0.02, 0.24), "seed 0만, 처음 보는 판 효과 없음"),
    ("모드 경유 점수", "구조", (-0.02, -0.08, 0.04), (0.02, -0.08, 0.12), "효과 없음"),
    ("루프 (블록 1개 × R)", "구조", (-0.04, -0.16, 0.08), (0.16, 0.06, 0.28), "큰 판에서 섞임 → 버림"),
    ("모든 샘플 정확 검사", "학습", (0.02, -0.10, 0.14), (0.08, -0.06, 0.22), "처음 보는 pcb 해로움"),
    ("검증 문제 3배", "학습", (0.02, -0.06, 0.10), (0.00, -0.18, 0.16), "효과 없음"),
    ("보상 5% 여유", "학습", (0.08, 0.00, 0.18), (0.02, -0.12, 0.16), "효과 없음"),
    ("작업 격자 상한 완화", "학습", (0.00, -0.10, 0.10), (0.04, -0.12, 0.22), "효과 없음"),
    ("규칙 보강 선생님 (RL만)", "학습", (-0.04, -0.16, 0.08), (-0.06, -0.20, 0.08), "효과 없음"),
    ("규칙 보강 선생님 (전체)", "학습", (0.00, -0.10, 0.10), (0.06, -0.08, 0.20), "효과 없음"),
    ("어려운 문제 가중", "학습", (0.02, -0.06, 0.10), (0.00, -0.16, 0.16), "처음 보는 pcb 해로움"),
    ("P1 피크 보정", "입력", (0.02, -0.08, 0.12), (0.06, -0.08, 0.22), "채택 (seed 3개 묶음 → D)"),
    ("P2 중복 없는 모드 토큰", "입력", (0.04, -0.08, 0.14), (0.14, 0.00, 0.28), "seed 2개 묶음 효과 없음"),
    ("P3 P1 + P2", "입력", (0.06, -0.02, 0.14), (0.04, -0.10, 0.18), "효과 없음"),
]
GC = {"구조": "#7c3aed", "학습": GRAY, "입력": OURS}
ax.axvspan(-0.16, 0.16, color="#f6f8fa", zorder=0)
ax.axvline(0, color=INK, lw=0.8)
for i, (n, g, pcb, pkg, v) in enumerate(att):
    y = len(att) - 1 - i
    for (m, lo, hi), dy, c in ((pcb, 0.17, PCB), (pkg, -0.17, PKG)):
        ax.errorbar(m, y + dy, xerr=[[m - lo], [hi - m]], fmt="o", ms=3.6, color=c, capsize=2, lw=1.0,
                    markeredgecolor="white" if lo <= 0 <= hi else c, markeredgewidth=0.6)
    ax.text(-0.335, y, n, ha="right", va="center", fontsize=7, color=GC[g], fontweight="bold" if g == "입력" and "P1" in n else "normal")
    ax.text(0.345, y, v, ha="left", va="center", fontsize=6.6, color=OURS if "채택" in v else MUTED,
            fontweight="bold" if "채택" in v else "normal")
ax.set_yticks([]); ax.set_xlim(-0.33, 0.34); ax.set_ylim(-0.7, len(att) - 0.3)
ax.set_xlabel("같은 종류 성공률 차 (수정 - 기준), seed 0 짝지은 bootstrap 95% CI")
ax.text(0.0, -0.62, "seed 잡음 범위 ±0.16", ha="center", va="bottom", fontsize=6.4, color=MUTED)
ax.plot([], [], "o", color=PCB, label="pcb"); ax.plot([], [], "o", color=PKG, label="pkg")
for g, c in GC.items():
    ax.plot([], [], "s", color=c, label=f"{g} 변경")
ax.legend(loc="upper right", fontsize=6.6, frameon=False, ncol=5, bbox_to_anchor=(1.0, -0.13))
head(ax, "C", "시도한 수정 13개: 대부분 seed 잡음 범위 안 → 하나씩 사전 등록 판정, 채택은 P1 하나",
     "회색 띠 = 수정 없이 seed만 바꿨을 때 차이 (d* = 0.16). 결과 전에 정한 규칙으로 판정, 실패도 그대로 기록")
style(ax, "x")

# ---------------------------------------------------------------- D P1 pooled
ax = fig.add_subplot(gs[2, 0:2])
p1 = [("같은 종류 pcb", 0.087, 0.020, 0.153), ("같은 종류 pkg", 0.093, 0.007, 0.180), ("같은 종류 ip", -0.013, -0.033, 0.000),
      ("처음 보는 pcb", -0.013, -0.080, 0.053)]
for i, (n, m, lo, hi) in enumerate(p1):
    c = OURS if lo > 0 else MUTED
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=5, capsize=3, lw=1.3)
    ax.text(hi + 0.01, i, f"{m:+.3f}", va="center", fontsize=6.8, color=c, fontweight="bold" if lo > 0 else "normal")
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(4), [r[0] for r in p1]); ax.invert_yaxis(); ax.set_xlim(-0.12, 0.26)
ax.set_xlabel("성공률 차 (P1 - MoTT)")
head(ax, "D", "P1: pcb·pkg +9%p", "seed 3개 묶음 95% CI, 해로운 판 없음")
style(ax, "x")

# ---------------------------------------------------------------- E size
ax = fig.add_subplot(gs[2, 2:4])
S = [10, 16, 20]
ax.axhline(1.0, color=GRAY, lw=1.0, ls="--")
ax.plot(S, [0.71, 0.70, 0.72], "o-", color=IP, lw=1.6, ms=4, label="ip (MoTT+P1)")
ax.plot(S, [0.99, 0.99, 0.81], "o-", color=PCB, lw=1.6, ms=4, label="pcb (MoTT+P1)")
ax.plot(S, [1.46, 5.37, 3.97], "s--", color=GRAY, lw=0.9, ms=3, label="ip greedy+prune")
ax.set_yscale("log"); ax.set_yticks([0.6, 0.8, 1, 2, 4], ["0.6", "0.8", "1", "2", "4"]); ax.minorticks_off()
ax.set_xticks(S, ["10", "16", "20*"]); ax.set_xlabel("판 크기 (포트)"); ax.set_ylabel("decap 수 ÷ 공진 규칙")
ax.text(10.2, 1.06, "공진 규칙 = 1", fontsize=6.2, color=GRAY)
ax.legend(fontsize=6.2, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
head(ax, "E", "ip는 모든 크기에서 30% 적게", "학습은 8~12칸, 16·20칸은 처음 보는 크기 · *20칸 16문제")
style(ax)

# ---------------------------------------------------------------- F loop R
ax = fig.add_subplot(gs[2, 4:6])
R = [2, 4, 8, 16]
for lab, v, c in (("pcb 16", [0.92, 0.90, 0.86, 0.84], PCB), ("pkg 16", [0.58, 0.64, 0.58, 0.50], PKG),
                  ("ip 16", [1.00, 1.00, 1.00, 1.00], IP), ("pkg 10", [0.74, 0.88, 0.88, 0.78], "#86c79b")):
    ax.plot(R, v, "o-", color=c, lw=1.3, ms=3.5, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks(R, [str(r) for r in R]); ax.minorticks_off()
ax.set_ylim(0.4, 1.05); ax.set_xlabel("평가 때 반복 수 R (학습 R 2~8)"); ax.set_ylabel("성공률")
ax.legend(fontsize=6.2, frameon=False, loc="lower left", ncol=2)
head(ax, "F", "루프: 반복을 늘려도 좋아지지 않음", "블록 1개 가중치 공유, seed 0 → 채택 안 함")
style(ax)

fig.suptitle("R2E 탐구 결과: 입력 설계 → 사다리 → 수정 13개 → P1 채택 → 크기 확인", x=0.08, y=0.995, ha="left",
             fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.08, 0.972, "모든 학습 모델은 같은 예산(모방 0.75 h + 강화 1.25 h), 같은 문제·같은 정확 검사(700점). 판정은 결과 전에 정한 규칙으로.",
         fontsize=7.6, color=MUTED)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_r2e_explorations.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig_r2e_explorations.pdf", bbox_inches="tight", facecolor="white")
print("ok")
