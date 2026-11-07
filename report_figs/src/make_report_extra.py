"""R2E 보고서 보조 그림. 숫자는 손으로 옮김.
fig_auc_flip: research_notes/토큰 데이터 분석/결과.md v2 (20x20, LS 선생님, 종류당 판 10개) 단일 특징 AUC, 방향은 PCB에서 정함
tab_setup: 비교 조건 표 (RESULTS.md, QUEUE, research_notes/선행 기준선 재구현 차이 표.md)
Usage: python make_report_extra.py <outdir>"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "font.size": 8, "axes.linewidth": 0.6, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
INK, MUTED, RULE = "#1f2328", "#6a737d", "#d0d7de"
BAD, GOOD, MID = "#b42318", "#c2410c", "#2563a8"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ AUC 뒤집힘
def auc_flip():
    # (특징, 누가 쓰나, PCB AUC, 인터포저형 AUC)
    rows = [("좌표 (probe와 거리)", "Park식 입력", 0.77, 0.46),
            ("loop L (한 주파수)", "Manoharan식 입력", 0.70, 0.43),
            ("고정 8점 |Z|", "Manoharan식 입력", 0.58, 0.06),
            ("모달 결합도 g", "MoTT 셀-모드 간선", 0.78, 0.57),
            ("모달 참여비", "MoTT 셀-모드 간선", 0.56, 0.87),
            ("정확한 1수 효과", "MoTT 물리 prior", 0.59, 0.98)]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.text(-0.23, 0.515, "무작위 = 0.5", fontsize=6.5, color=MUTED, va="bottom")
    for n, who, a, b in rows:
        c = BAD if b < 0.5 else (GOOD if b > a else MID)
        ax.plot([0, 1], [a, b], color=c, lw=1.6, marker="o", ms=4.5, zorder=3)
        yb = b + {"좌표 (probe와 거리)": 0.012, "loop L (한 주파수)": -0.012}.get(n, 0)
        ax.text(1.04, yb, f"{a:.2f} → {b:.2f}  {n}  ({who})", ha="left", va="center", fontsize=6.6, color=c,
                fontweight="bold" if c != MID else "normal")
    ax.set_xticks([0, 1], ["PCB형\n(방향을 정한 판)", "인터포저형\n(같은 방향 그대로)"])
    ax.set_xlim(-0.25, 1.0); ax.set_ylim(0, 1.05)
    ax.set_ylabel("선생님이 고른 자리 예측 AUC")
    ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
    ax.set_title("좌표와 한 주파수 L은 판 종류가 바뀌면 신호가 뒤집힌다", loc="left", fontsize=9, fontweight="bold", pad=10)
    fig.text(0.02, -0.06, "학습 없는 분석, 20x20칸, 종류당 판 10개, LS 선생님, CI 없음. 인터포저형 라벨은 사실상 greedy 해라서\n"
             "1수 효과 0.98은 부풀려져 있음. 모달 특징도 결합도 g는 약해짐 → 여러 종류를 섞어 학습하는 근거.",
             fontsize=6.4, color=MUTED, linespacing=1.4)
    save(fig, "fig_auc_flip")


# ------------------------------------------------------------------ 비교 조건 표
def setup_table():
    rows = [("평가 문제", "종류(pcb·pkg·ip)마다 같은 문제 50개 (10·16포트), 20포트는 16개"),
            ("성공 판정", "모든 방법에 같은 정확 검사 (700점 주파수, 최악 |Z|/Zt ≤ 1)"),
            ("시간 측정", "CPU 한 코어, 문제당 벽시계 초"),
            ("학습 예산", "모든 학습 방법 같음: 증류 0.75 h + RL 1.25 h, GPU 1장"),
            ("seed", "주요 비교는 seed 3개, 판정은 사전 등록 규칙"),
            ("통계", "같은 문제를 짝지은 bootstrap 95% CI (2000회)"),
            ("잡음 기준", "수정 없이 seed만 바꾼 차이 d* = 0.16 → 이보다 작은 차이는 주장 안 함"),
            ("선행 재구현", "재현 아님. Park: 원래 목표 '개수 고정 + Z 감소'를 우리 목표로 바꿈\n"
                          "Manoharan: 원 예산 ~24 h vs 우리 2 h, 일부 설정 원문 미확인")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0)); ax.axis("off")
    y = 1.0
    for i, (k, v) in enumerate(rows):
        h = 0.155 if "\n" in v else 0.11
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - h), 1, h, color="#f6f8fa", transform=ax.transAxes, lw=0))
        ax.text(0.01, y - h / 2, k, fontsize=7.6, fontweight="bold", va="center", color=INK, transform=ax.transAxes)
        ax.text(0.17, y - h / 2, v, fontsize=7.2, va="center", color=BAD if k == "선행 재구현" else INK,
                transform=ax.transAxes, linespacing=1.35)
        y -= h
    ax.set_title("비교 조건: 모든 방법이 같은 문제·같은 검사·같은 예산", loc="left", fontsize=9, fontweight="bold")
    save(fig, "tab_setup")


OUT.mkdir(parents=True, exist_ok=True)
auc_flip(); setup_table()
print("ok")
