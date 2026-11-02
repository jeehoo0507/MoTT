"""Generate RESULTS.md from metrics.json files (never type numbers by hand).

Sources
  runs/**/eval/*.json          evaluation records written by scripts/eval_policy.py:
                               {set, method, tech, size, n, ok[], count[], sec_per_inst, ...}
  runs/**/metrics.json         training records written by train.py (final / best validation)
  data/teacher/*.json          teacher statistics (LS vs greedy+prune on the training pools)

Run: .venv/bin/python scripts/collect_results.py [--out RESULTS.md]
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_evals():
    recs = []
    for f in sorted(glob.glob("runs/**/eval/*.json", recursive=True)):
        try:
            d = json.load(open(f))
            d["_file"] = f
            recs.append(d)
        except Exception as e:  # noqa: BLE001
            print("skip", f, e, file=sys.stderr)
    return recs


TA = "gp_type_aware"   # E0+ type-aware greedy (scripts/make_type_aware.py), the day-2 reference


def order(k):
    return (k != "greedy+prune", k != TA, k != "local_search", k)


def fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def merged(recs):
    """{(tech, size): {method: merged record}}. A (tech, size) has several pool files (eval_pcb10_0, _1, ...);
    each method's files are concatenated, and the count ratio to greedy+prune is paired only within the same
    pool file (same problems), never across files. The latest record per (file, method) wins."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        by[(r["tech"], int(r["size"]))][r["method"]][r["set"]] = r
    out = defaultdict(dict)
    for key, ms in by.items():
        gp = ms.get("greedy+prune", {})
        ta = ms.get(TA, {})
        for m, sets in ms.items():
            ok, cnt, ratio, sec, files, rta, ta_ok = [], [], [], [], [], [], []
            for s in sorted(sets):
                r = sets[s]
                o = np.asarray(r["ok"], bool)
                c = np.asarray(r["count"], float)
                ok.append(o)
                cnt.append(c)
                sec.append((r.get("sec_per_inst", float("nan")), len(o)))
                files.append(r["_file"])
                if s in gp:
                    g = gp[s]
                    both = o & np.asarray(g["ok"], bool)
                    ratio.append(c[both] / np.maximum(np.asarray(g["count"], float)[both], 1))
                if s in ta:
                    g = ta[s]
                    both = o & np.asarray(g["ok"], bool)
                    rta.append(c[both] / np.maximum(np.asarray(g["count"], float)[both], 1))
                    ta_ok.append(np.asarray(g["ok"], bool))
            ok, cnt = np.concatenate(ok), np.concatenate(cnt)
            ratio = np.concatenate(ratio) if ratio else np.zeros(0)
            rta = np.concatenate(rta) if rta else np.zeros(0)
            out[key][m] = dict(ok=ok, cnt=cnt, rel=float(ratio.mean()) if ratio.size else None,
                               rel_ta=float(rta.mean()) if rta.size else None,
                               ta_ok=float(np.concatenate(ta_ok).mean()) if ta_ok else None,
                               sec=sum(t * n for t, n in sec) / sum(n for _, n in sec), sets=sorted(sets), files=files)
    return out


def eval_tables(recs):
    """One table per (tech, size): rows = methods. Columns: problems evaluated (all pool files of that size),
    success (exact), count on solved, count / greedy+prune (paired within a file, on instances both solve),
    sec per instance."""
    by = merged(recs)
    lines = []
    for (tech, size) in sorted(by):
        ms = by[(tech, size)]
        lines += [f"### {tech} {size}×{size}", "",
                  "| 방법 | 문제 수 (풀 파일) | 정확 검사 성공률 | 개수 (성공 평균) | 개수 / greedy+prune (둘 다 성공, 같은 파일끼리 짝지음) | "
                  "개수 / 종류 인식 greedy (둘 다 성공) | 문제당 시간 (s) | 출처 |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in sorted(ms, key=order):
            r = ms[m]
            ok, cnt = r["ok"], r["cnt"]
            sets = ", ".join(s.rsplit("_", 1)[1] for s in r["sets"])
            lines.append(f"| {m} | {len(ok)} ({sets}) | {ok.mean():.2f} | {fmt(cnt[ok].mean() if ok.any() else None)} | "
                         f"{fmt(r['rel'], 3)} | {fmt(r['rel_ta'], 3)} | {fmt(r['sec'], 3)} | " + ", ".join(f"`{f}`" for f in r["files"]) + " |")
        lines.append("")
    return lines


def ladder_table(recs):
    """Success / count ratio per method across kinds and sizes (one row per method; all pool files merged)."""
    by = merged(recs)
    keys = sorted(by)
    if not keys:
        return []
    methods = sorted({m for k in keys for m in by[k]}, key=order)
    head = "| 방법 | " + " | ".join(f"{t} {s}" for t, s in keys) + " |"
    lines = ["### 요약: 정확 검사 성공률 / 개수÷greedy+prune / 개수÷종류 인식 greedy (각각 둘 다 성공한 문제) [문제 수]", "",
             f"종류 인식 greedy (`{TA}`) = 종류 하나만 쓰는 greedy+prune을 종류마다 돌려 문제마다 성공한 해 중 가장 적은 것 "
             "(`scripts/make_type_aware.py`). 10/29부터 모든 비교의 기준선.", "", head, "|---|" + "---|" * len(keys)]
    for m in methods:
        cells = []
        for k in keys:
            r = by[k].get(m)
            cells.append("–" if r is None else f"{r['ok'].mean():.2f} / {fmt(r['rel'])} / {fmt(r['rel_ta'])} [{len(r['ok'])}]")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines + [""]


def type_table(recs):
    """Share of each decap type among the placed decaps (record occ: cell -> type, -1 empty), split by
    whether the exact check passed. Library per kind in decap_loop/techs.py (type 0 = largest C)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if "occ" not in r:
            continue
        occ, ok = np.asarray(r["occ"]), np.asarray(r["ok"], bool)
        for tag, sel in (("성공", ok), ("실패", ~ok)):
            acc[(r["tech"], int(r["size"]))][(r["method"], tag)].append(occ[sel])
    if not acc:
        return []
    lines = ["## decap 종류 비율 (놓인 decap 중, 정확 검사 성공/실패 문제별; 종류 0 = C가 가장 큰 것, `decap_loop/techs.py`)", "",
             "| 판 | 방법 | 문제 | 문제 수 | 평균 개수 | 종류 0 | 종류 1 | 종류 2 |", "|---|---|---|---|---|---|---|---|"]
    for key in sorted(acc):
        for (m, tag) in sorted(acc[key], key=lambda k: (k[0] != "greedy+prune", k[0], k[1])):
            o = np.concatenate(acc[key][(m, tag)])
            if len(o) == 0:
                continue
            placed = o[o >= 0]
            share = np.bincount(placed, minlength=3) / max(len(placed), 1)
            lines.append(f"| {key[0]} {key[1]} | {m} | {tag} | {len(o)} | {(o >= 0).sum(1).mean():.1f} | "
                         + " | ".join(f"{s:.2f}" for s in share[:3]) + " |")
    return lines + [""]


def gp_fail_table():
    """scripts/analyze_ip_gp_fail.py output: where greedy+prune fails (exact-check worst frequency vs the bare
    board's worst) and total placed C vs a policy that passes the same instance."""
    lines = []
    for f in sorted(glob.glob("runs/analysis/*gp_fail*.json")):
        d = json.load(open(f))
        rows = d["rows"]
        if not rows:
            continue
        pm = d["args"]["policy"].split(":")[1]
        rat = np.array([r["gp_f"] / r["bare_f"] for r in rows])
        pol = [r for r in rows if r["pol_ok"]]
        if not lines:
            lines = ["## greedy+prune 실패 문제의 최악 주파수 (`scripts/analyze_ip_gp_fail.py`, 기록된 occ로 정확 검사 격자 재계산)", "",
                     "| 세트 | g+p 실패 문제 | g+p 최악 Z/Zt (중앙값) | g+p 최악 주파수 / 맨판 최악 주파수 (중앙값, 범위) | "
                     "맨판 최악보다 낮은 비율 | 같은 문제 정책 성공 | 총 C g+p / 정책 (중앙값) | 출처 |", "|---|---|---|---|---|---|---|---|"]
        cr = np.median([r["gp_cap_pF"] / r["pol_cap_pF"] for r in pol]) if pol else None
        lines.append(f"| {d['args']['sets']} | {len(rows)} | {np.median([r['gp_worst'] for r in rows]):.2f} | "
                     f"{np.median(rat):.2f} ({rat.min():.2f}~{rat.max():.2f}) | {np.mean(rat < 1):.2f} | {pm} {len(pol)} | {fmt(cr)} | `{f}` |")
    return lines + [""] if lines else []


def swap_table():
    """scripts/analyze_ip_swap.py output: greedy+prune failures re-checked with the recorded positions and
    every placed decap set to one library type."""
    lines = []
    for f in sorted(glob.glob("runs/analysis/*swap*.json")):
        d = json.load(open(f))
        rows = d["rows"]
        if not rows:
            continue
        pm = d["args"]["policy"].split(":")[1]
        T = len(rows[0]["gp_at"])
        if not lines:
            lines = ["## g+p 실패 문제에서 위치는 그대로, 모든 decap을 한 종류로 바꾸면 (`scripts/analyze_ip_swap.py`, 정확 검사 격자 재계산)", "",
                     "| 세트 | 위치 출처 | 문제 수 | 기록 그대로: 성공률 / 최악 Z/Zt 중앙값 | "
                     + " | ".join(f"모두 종류 {t}: 성공률 / 최악 중앙값" for t in range(T)) + " | 출처 |",
                     "|---|---|---|---|" + "---|" * T + "---|"]
        for src, key, sel in (("greedy+prune", "gp", rows), (pm, "pol", [r for r in rows if r["pol_ok"]])):
            if not sel:
                continue
            own = np.array([r[key] for r in sel])
            at = np.array([r[key + "_at"] for r in sel])
            lines.append(f"| {d['args']['sets']} | {src} | {len(sel)} | {np.mean(own <= 1.01):.2f} / {np.median(own):.2f} | "
                         + " | ".join(f"{np.mean(at[:, t] <= 1.01):.2f} / {np.median(at[:, t]):.2f}" for t in range(T))
                         + f" | `{f}` |")
    return lines + [""] if lines else []


def one_type_table(recs, ref="r3_loop_rl"):
    """scripts/analyze_gp_one_type.py output: greedy+prune allowed only one decap type (analysis, not a
    reported baseline), against the full greedy+prune and a policy record on the same pool file."""
    files = sorted(glob.glob("runs/analysis/gp_one_type/*.json"))
    if not files:
        return []
    ev = {(r["set"], r["method"]): r for r in recs}
    lines = ["## greedy+prune을 한 종류만 쓰게 하면 (`scripts/analyze_gp_one_type.py`, 같은 규칙·cutting-plane·정확 검사; 분석용, 보고 기준선 아님)", "",
             f"| 세트 | 허용 종류 (C, ESL, ESR) | 성공률 | 개수 (성공 평균) | 개수 / greedy+prune (둘 다 성공) | 개수 / {ref} (둘 다 성공) | "
             "greedy+prune 성공률 | " + ref + " 성공률 | 문제당 시간 (s) | 출처 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        d = json.load(open(f))
        o, c = np.asarray(d["ok"], bool), np.asarray(d["count"], float)
        cells = []
        for m in ("greedy+prune", ref):
            r = ev.get((d["set"], m))
            if r is None:
                cells.append(("–", "–"))
                continue
            ro, rc = np.asarray(r["ok"], bool), np.asarray(r["count"], float)
            both = o & ro
            cells.append((fmt((c[both] / np.maximum(rc[both], 1)).mean() if both.any() else None, 3), f"{ro.mean():.2f}"))
        C, L, R = d["decap"]
        lines.append(f"| {d['set']} | {d['method'].split('@type')[1]} ({C * 1e12:g} pF, {L * 1e12:g} pH, {R:g} Ω) | {o.mean():.2f} | "
                     f"{fmt(c[o].mean() if o.any() else None)} | {cells[0][0]} | {cells[1][0]} | {cells[0][1]} | {cells[1][1]} | "
                     f"{fmt(d['sec_per_inst'], 3)} | `{f}` |")
    return lines + [""]


def type_aware_table():
    """Single-type greedy+prune success per board kind and which type the type-aware greedy picked
    (QUEUE 10/29 03:15 item 4: a rule has to re-pick the type per board kind). The closing sentence is
    generated from the same numbers."""
    by_set = defaultdict(dict)
    for f in glob.glob("runs/analysis/gp_one_type/*.json"):
        d = json.load(open(f))
        by_set[d["set"]][int(d["method"].split("@type")[1])] = np.asarray(d["ok"], bool)
    one = defaultdict(lambda: defaultdict(list))   # only pool files with all three types: same problems per type
    for s, ts in sorted(by_set.items()):
        if len(ts) == 3:
            tech, size = s.split("_")[1][:-2], int(s.split("_")[1][-2:])
            for t, o in ts.items():
                one[(tech, size)][t].append(o)
    picks = defaultdict(list)
    for f in glob.glob(f"runs/**/eval/*__{TA}.json", recursive=True):
        d = json.load(open(f))
        picks[(d["tech"], int(d["size"]))].append(np.asarray(d["type_picked"])[np.asarray(d["ok"], bool)])
    keys = sorted(k for k in one if len(one[k]) == 3)
    if not keys:
        return []
    lines = ["## 종류 인식 greedy: 판 종류마다 맞는 decap 종류 (`scripts/analyze_gp_one_type.py`, `scripts/make_type_aware.py`)", "",
             "| 판 | 문제 수 | 종류 0만: 성공률 | 종류 1만: 성공률 | 종류 2만: 성공률 | 종류 인식 greedy가 고른 종류 비율 (성공 문제, 0/1/2) |",
             "|---|---|---|---|---|---|"]
    best = {}
    for k in keys:
        rates = [np.concatenate(one[k][t]).mean() for t in range(3)]
        best[k] = int(np.argmax(rates))
        p = np.concatenate(picks[k]) if picks.get(k) else np.zeros(0, int)
        share = " / ".join(f"{s:.2f}" for s in np.bincount(p, minlength=3) / max(len(p), 1)) if len(p) else "–"
        n = len(np.concatenate(one[k][0]))
        lines.append(f"| {k[0]} {k[1]} | {n} | " + " | ".join(f"{r:.2f}" for r in rates) + f" | {share} |")
    kinds = ", ".join(f"{k[0]} {k[1]} → 종류 {best[k]}" for k in keys)
    same = len(set(best.values())) == 1
    lines += ["", f"단일 종류 g+p로 가장 잘 되는 종류: {kinds}. " +
              ("(판 종류 하나만 끝남 — 비교 문장은 둘 이상 끝난 뒤.)" if len(keys) < 2 else
               "판 종류마다 같다." if same else "**판 종류마다 다르다** — 규칙 기준선은 판 종류마다 종류를 다시 골라야 하고 "
               "(종류 인식 greedy는 그래서 종류 수만큼 다시 돈다), 정책은 모델 하나가 판마다 고른다 (위 \"decap 종류 비율\" 표)."), ""]
    return lines


def paired(recs, ref):
    """{(tech, size): {method: (ok_m, ok_ref, cnt_m, cnt_ref)}} over the instances of the pool files both have."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        by[(r["tech"], int(r["size"]))][r["method"]][r["set"]] = r
    out = defaultdict(dict)
    for key, ms in by.items():
        if ref not in ms:
            continue
        for m, sets in ms.items():
            common = sorted(set(sets) & set(ms[ref]))
            if m == ref or not common:
                continue
            def cat(src, k):
                return np.concatenate([np.asarray(src[s][k], float) for s in common])
            out[key][m] = (cat(sets, "ok").astype(bool), cat(ms[ref], "ok").astype(bool),
                           cat(sets, "count"), cat(ms[ref], "count"))
    return out


def boot_ci(fn, n, B=2000, seed=0):
    """Percentile 95% CI of fn(idx) over B paired resamples of the n instances (fixed seed: reproducible tables)."""
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    v = np.asarray([x for x in v if np.isfinite(x)])
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def ci_table(recs, ref=TA, title="종류 인식 greedy", only=None):
    """QUEUE 10/29 item 3: paired bootstrap 95% CI (instances resampled jointly for method and reference) of
    the success-rate difference and of the mean count ratio on instances both solve. only: restrict methods."""
    pr = paired(recs, ref)
    if only is not None:
        pr = {k: {m: v for m, v in ms.items() if m in only} for k, ms in pr.items()}
        pr = {k: ms for k, ms in pr.items() if ms}
    if not pr:
        return []
    lines = [f"## {title} 대비 짝지은 bootstrap 95% CI (같은 문제를 함께 다시 뽑음, 2000회, seed 0)", "",
             "| 판 | 방법 | 문제 수 | 성공률 차 (방법 − 기준) [95% CI] | 개수 비 방법/기준 (둘 다 성공) [95% CI] |", "|---|---|---|---|---|"]
    for key in sorted(pr):
        for m in sorted(pr[key], key=order):
            lines.append(f"| {key[0]} {key[1]} | {m} | " + ci_cells(*pr[key][m]) + " |")
    return lines + [""]


def ci_cells(om, orf, cm, cr):
    """'n | success diff [CI] | count ratio [CI]' for one paired (method, reference) comparison."""
    n = len(om)
    d = om.mean() - orf.mean()
    lo, hi = boot_ci(lambda i: om[i].mean() - orf[i].mean(), n)

    def ratio(i):
        b = om[i] & orf[i]
        return (cm[i][b] / np.maximum(cr[i][b], 1)).mean() if b.any() else np.nan
    rt = ratio(np.arange(n))
    rlo, rhi = boot_ci(ratio, n)
    return f"{n} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | {fmt(rt, 3)} [{fmt(rlo, 3)}, {fmt(rhi, 3)}]"


DISTILL_RL = [(f"{r}_rl", f"{r}_distill") for r in ("r0_coord", "r1_modal", "r2_mott", "r3_loop")]


def distill_rl_table(recs):
    """Method question (AGENT_PROMPT): does RL beat distillation alone? Same rung, same eval instances, seed 0."""
    rows = []
    for var, base in DISTILL_RL:
        pr = paired(recs, base)
        rows += [f"| {key[0]} {key[1]} | {var} | {base} | " + ci_cells(*pr[key][var]) + " |"
                 for key in sorted(pr) if var in pr[key]]
    if not rows:
        return []
    return ["## 증류 → RL: 같은 단 `*_rl` vs `*_distill` 짝지은 bootstrap 95% CI (같은 문제를 함께 다시 뽑음, 2000회, seed 0)", "",
            "| 판 | RL | 증류만 | 문제 수 | 성공률 차 (RL − 증류) [95% CI] | 개수 비 RL/증류 (둘 다 성공) [95% CI] |",
            "|---|---|---|---|---|---|"] + rows + [""]


ABLATIONS = [  # QUEUE 10/29 03:15 item 3: one change vs the same rung without it (same budget, seed 0)
    ("r3a_rl", "r3_loop_rl", "3단 + 주파수 궁합 편향 (dec_from_mode)", "같은 단 편향 없는"),
    ("r2a_rl", "r2_mott_rl", "2단 + 주파수 궁합 편향 (dec_from_mode)", "같은 단 편향 없는"),
    ("r2am_rl", "r2a_rl", "r2a + 모드 경유 add 점수 (--mode_head)", "모드 경유 점수 없는 부모"),
    ("r3_loop_rl", "r2_mott_rl", "루프 트랜스포머 (MoTT 블록 1개를 가중치 공유로 R회 반복)", "기본 트랜스포머 (블록 6개)"),
    # 11/02 diagnosis H2: rl reward exact-checks every distinct grid-feasible sample; same distilled init, rl 1.25 h
    ("r2c_rl", "r2_mott_rl", "2단 + 보상 정확 검사를 모든 샘플에 (`--certify 2`, 같은 distill에서 RL만)", "최선 샘플만 검사"),
    ("lo_pcb_r2c_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + `--certify 2` (같은 distill에서 RL만)", "최선 샘플만 검사"),
    # 11/02 diagnosis H1: best.pt chosen on 3 val files per kind (48 problems) instead of 1 (16); same distilled init
    ("r2v_rl", "r2_mott_rl", "2단 + best.pt 선택 val 종류당 48문제 (같은 distill에서 RL만)", "val 종류당 16문제"),
    ("lo_pcb_r2v_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + val 종류당 48문제 (같은 distill에서 RL만)", "val 종류당 16문제"),
]


PRIOR = {  # prior-work re-implementations (QUEUE 10/29 03:30 C): how each differs from the paper
    "bp_park": ("B-Park: Park 2022 / DevFormer식 좌표 입력 attention-model 포인터 정책 (`baselines_ml/park_am.py`)", [
        "학습 예산: 원 논문은 RL만 — 여기서는 우리와 같은 distill 0.75 h + RL 1.25 h (같은 선생님 해, 같은 학습 풀).",
        "행동 공간: 원 논문은 포트 선택(종류 고정) — 여기서는 우리 행동 공간 (칸 × 종류 추가/제거 + 멈춤), 채널마다 포인터 query.",
        "보상: 원 논문은 임피던스 점수 — 여기서는 우리 보상 (−개수/N_ref − λ·(실패+위반), λ 라그랑주), 정확 검사 실패 반영.",
        "구조·학습: Kool식 encode-once 대신 매 수 다시 인코딩 (상태가 특징에 있음), BatchNorm 대신 LayerNorm, 기준선 교체는 "
        "paired t-test 대신 val 최고 갱신 때, 기준선 rollout의 실패는 작업 격자 기준 (정확 검사 안 함), 모방 항·선생님 교체 없음 (`--imit 0`).",
        "원 설정 (DevFormer 공개 코드의 AM-PG 기준선에서 확인, 원문 PDF는 미확인): 10×10 판 한 장, 한 종류 decap **K = 20개 고정**, "
        "보상 = 201점 임피던스 감소 합 (목표 Zt 없음), 입력 (x, y, 탐침/keep-out), 5만 문제 × 10 epoch → 원 논문은 개수 최소화·목표 만족 문제가 아님. "
        "표: `research_notes/선행 기준선 설정 확인 (DevFormer 공개 코드).md`."]),
    "bm_manoharan": ("B-Manoharan: Manoharan 2026식 고정 포트 순서 + decap 토큰으로 종류/빈칸 선택 (`baselines_ml/manoharan.py`)", [
        "학습: 원 논문의 RL 세부 대신 우리 파이프라인 그대로 (distill 0.75 h: 선생님 해를 고정 순서로 다시 걸은 궤적, RL 1.25 h: 그룹 기준선 + 모방 + 선생님 교체).",
        "포트 순서: 최악 탐침에서 본 loop L (포트 i 단락, 한 주파수 f0) 오름차순 — f0는 맨판 최악 위반 주파수 (원 논문은 고정 설계 주파수).",
        "상태: 고정 특징 주파수 8점의 |Z|/Zt + 전역 블록 MLP와 현재 포트 L 임베딩; 원 논문의 Z 행렬 입력 전체는 아님. 갱신은 우리 env의 rank-1 (Schur/Woodbury).",
        "원 설정은 `reports/최종 모델 설계.md`·코드 docstring의 설명 기준 — TSIPI 2026 원문 대조는 확인 필요 (무인 라운드에서 원문 접근 불가).",
        "행동: 현재 포트에 종류 또는 빈칸, 제거·종류 변경 없음; 에피소드 길이 = 놓을 수 있는 칸 수 + 1 (학습의 max_steps 64 대신).",
        "`*_rlonly_l20` (bp_park_rlonly 포함): 원 논문처럼 RL만 2.0 h (= distill + RL 예산), 무작위 초기화, 모방 없음, 초기 λ 20 "
        "(우리 run은 1). 초기 λ 1로 한 bm_manoharan_rlonly는 it 75부터 모든 샘플이 0개 놓고 실패 → advantage 0으로 붕괴해 34분에 멈춤 "
        "(평가 없음, `runs/2026-10-29/bm_manoharan_rlonly_rl/log.txt`)."]),
}


def prior_table(recs):
    by = merged(recs)
    lines = []
    for key, (title, notes) in PRIOR.items():
        rows = [(k, m, r) for k in sorted(by) for m, r in by[k].items() if m.startswith(key)]
        if not rows:
            continue
        if not lines:
            lines = ["## 선행 연구 재구현 (재구현 기준, 같은 평가 풀·정확 검사)", ""]
        lines += [f"### {title}", "", "| 판 | 방법 | 문제 수 | 성공률 | 개수÷greedy+prune | 개수÷종류 인식 greedy | 문제당 시간 (s) |",
                  "|---|---|---|---|---|---|---|"]
        for k, m, r in rows:
            lines.append(f"| {k[0]} {k[1]} | {m} | {len(r['ok'])} | {r['ok'].mean():.2f} | {fmt(r['rel'])} | {fmt(r['rel_ta'])} | {fmt(r['sec'], 3)} |")
        lines += [""] + [f"- {n}" for n in notes] + [""]
    return lines


LO_TECHS = ("ip", "pcb", "pkg")
LO_RUNGS = (("r0", "r0_coord_rl", "0단 좌표"), ("r1", "r1_modal_rl", "1단 모달"), ("r2", "r2_mott_rl", "2단 MoTT"))


def leave_out_table(recs):
    """QUEUE 10/29 14:20: train on two board types (pools and val), evaluate on the held-out type's 10×10 eval
    pools. Per held-out type: lo_<t>_r{0,1,2}_rl, the same rung trained on all three types, E0+ and modal_rule;
    paired CIs (a) r1 vs r0, (b) r2 vs r0, (c) each lo run vs the same rung trained on all types."""
    by = merged(recs)
    if not any(m.startswith("lo_") for ms in by.values() for m in ms):
        return []
    lines = ["## 한 종류 빼고 학습 (처음 보는 기술 종류)", "",
             "`lo_<t>_r*`: 종류 t의 학습 풀·val을 빼고 나머지 두 종류로 학습 (같은 예산 distill 0.75 h + RL 1.25 h, 아래 두 표는 seed 0, "
             "arm은 사다리 0·1·2단 그대로 + `--pool_glob`·`--val_pools`). 아래는 **뺀 종류 t의 10×10 eval 풀**에서만 잰 값. 없는 결과는 \"–\".", "",
             "| 뺀 종류 (평가 판) | 방법 | 문제 수 | 정확 검사 성공률 | 개수 (성공 평균) | 개수÷종류 인식 greedy (둘 다 성공) |",
             "|---|---|---|---|---|---|"]
    for t in LO_TECHS:
        ms = by.get((t, 10), {})
        for m in [f"lo_{t}_{r}_rl" for r, _, _ in LO_RUNGS] + [full for _, full, _ in LO_RUNGS] + [TA, "modal_rule"]:
            r = ms.get(m)
            if r is None:
                lines.append(f"| {t} 10 | {m} | – | – | – | – |")
                continue
            ok, cnt = r["ok"], r["cnt"]
            lines.append(f"| {t} 10 | {m} | {len(ok)} | {ok.mean():.2f} | {fmt(cnt[ok].mean() if ok.any() else None)} | {fmt(r['rel_ta'], 3)} |")
    lines += ["", "짝지은 bootstrap 95% CI (seed 0 run끼리, 뺀 종류의 eval 문제, 같은 문제를 함께 다시 뽑음, 2000회, seed 0):", "",
              "| 뺀 종류 | 비교 | 방법 | 기준 | 문제 수 | 성공률 차 (방법 − 기준) [95% CI] | 개수 비 방법/기준 (둘 다 성공) [95% CI] |",
              "|---|---|---|---|---|---|---|"]
    for t in LO_TECHS:
        lo = {r: f"lo_{t}_{r}_rl" for r, _, _ in LO_RUNGS}
        pairs = [("(a) 1단 vs 0단", lo["r1"], lo["r0"]), ("(b) 2단 vs 0단", lo["r2"], lo["r0"])]
        pairs += [(f"(c) {desc}: 뺀 학습 vs 전체 학습", lo[r], full) for r, full, desc in LO_RUNGS]
        for what, m, ref in pairs:
            pm = paired(recs, ref).get((t, 10), {}).get(m)
            cells = ci_cells(*pm) if pm is not None else "– | – | –"
            lines.append(f"| {t} 10 | {what} | {m} | {ref} | {cells} |")
    return lines + [""] + leave_out_seed_table(by) + loop_vs_plain_table(recs, by)


LO_SEEDS = (0, 1, 2)


def seed_run(base, s):
    """Run name of rung `base` (e.g. lo_ip_r0, r0_coord) at seed s: seed 0 has no suffix."""
    return f"{base}_rl" if s == 0 else f"{base}_s{s}_rl"


def loop_vs_plain_table(recs, by):
    """QUEUE 10/31: looped transformer (r3_loop, one shared MoTT block run R times) vs the plain one (r2_mott, six
    blocks), same budget. Trained on all types at seeds 0-2 (success and count÷E0+ per seed on each board's 10×10
    eval), and on the leave-out pcb run lo_pcb_r3 (seed 0): vs lo_pcb_r2, and vs r3_loop_rl trained on all types."""
    if not any("r3_loop" in m for ms in by.values() for m in ms):
        return []

    def cell(t, m):
        r = by.get((t, 10), {}).get(m)
        return "–" if r is None else f"{r['ok'].mean():.2f} / {fmt(r['rel_ta'], 2)}"
    lines = ["### 루프 트랜스포머 vs 기본 트랜스포머 (같은 예산; 성공률 / 개수÷종류 인식 greedy)", "",
             "| 판 | seed | 기본 `r2_mott` | 루프 `r3_loop` |", "|---|---|---|---|"]
    for t in LO_TECHS:
        for s in LO_SEEDS:
            lines.append(f"| {t} 10 | {s} | {cell(t, seed_run('r2_mott', s))} | {cell(t, seed_run('r3_loop', s))} |")
    lines += ["", "뺀 종류 pcb (seed 0, 짝지은 bootstrap 95% CI):", "",
              "| 비교 | 방법 | 기준 | 문제 수 | 성공률 차 (방법 − 기준) [95% CI] | 개수 비 방법/기준 (둘 다 성공) [95% CI] |",
              "|---|---|---|---|---|---|"]
    for what, m, ref in (("루프 vs 기본 (둘 다 pcb 빼고 학습)", "lo_pcb_r3_rl", "lo_pcb_r2_rl"),
                         ("루프: 뺀 학습 vs 전체 학습", "lo_pcb_r3_rl", "r3_loop_rl")):
        pm = paired(recs, ref).get(("pcb", 10), {}).get(m)
        lines.append(f"| {what} | {m} | {ref} | {ci_cells(*pm) if pm is not None else '– | – | –'} |")
    return lines + [""]


def leave_out_seed_table(by):
    """QUEUE 10/31: the leave-one-type-out rungs at seeds 0-2. Success on the held-out type's 10×10 eval pools per
    seed, so a seed-0 result can be told apart from luck: (b) 2단 − 0단 within the leave-out runs, and (c) each
    leave-out run minus the same rung trained on all types at the same seed."""
    def rate(t, m):
        r = by.get((t, 10), {}).get(m)
        return r["ok"].mean() if r is not None else None

    def cells(vals):
        have = [v for v in vals if v is not None]
        mean = f"{np.mean(have):+.2f}" if len(have) == len(vals) else "–"
        return " / ".join("–" if v is None else f"{v:+.2f}" for v in vals), mean, len(have)
    lines = ["### seed별 (뺀 종류의 10×10 eval, 정확 검사 성공률; 평균은 seed가 다 있을 때만)", "",
             "| 뺀 종류 | 단 | 뺀 학습 seed 0 / 1 / 2 | 평균 | 전체 학습 seed 0 / 1 / 2 | 평균 | 뺀 − 전체 (같은 seed) | 평균 |",
             "|---|---|---|---|---|---|---|---|"]
    for t in LO_TECHS:
        for r, full, desc in LO_RUNGS:
            lo = [rate(t, seed_run(f"lo_{t}_{r}", s)) for s in LO_SEEDS]
            fu = [rate(t, seed_run(full[:-3], s)) for s in LO_SEEDS]
            d = [a - b if a is not None and b is not None else None for a, b in zip(lo, fu)]
            (lc, lm, _), (fc, fm, _), (dc, dm, _) = cells(lo), cells(fu), cells(d)
            lines.append(f"| {t} 10 | {desc} | {lc.replace('+', '')} | {lm.replace('+', '')} | {fc.replace('+', '')} | "
                         f"{fm.replace('+', '')} | {dc} | {dm} |")
    lines += ["", "| 뺀 종류 | 비교 (뺀 학습끼리) | seed 0 / 1 / 2 | 평균 | 0보다 큰 seed 수 |", "|---|---|---|---|---|"]
    for t in LO_TECHS:
        for what, a, b in (("(a) 1단 − 0단", "r1", "r0"), ("(b) 2단 − 0단", "r2", "r0")):
            d = [None if x is None or y is None else x - y
                 for x, y in ((rate(t, seed_run(f"lo_{t}_{a}", s)), rate(t, seed_run(f"lo_{t}_{b}", s))) for s in LO_SEEDS)]
            dc, dm, n = cells(d)
            lines.append(f"| {t} 10 | {what} | {dc} | {dm} | {sum(v > 0 for v in d if v is not None)}/{n} |")
    return lines + [""]


FEAT = "runs/analysis/problem_features.json"   # scripts/problem_features.py (bare-board features, no method output)


def learn_vs_rule_table(recs):
    """QUEUE 2026-11-01 15:10 item 1: on the held-out type's 10×10 eval problems, the leave-out MoTT runs
    (lo_<t>_r2, seeds 0-2) vs the untrained modal_rule and E0+: success, count, time, paired CIs; then where
    learning wins. Per problem and seed, vs modal_rule: +1 if the policy solves and the rule fails or both solve
    with fewer decaps, -1 for the reverse, 0 otherwise; averaged over the seeds present. Grouped by board shape,
    target kind and terciles (within the type) of the bare-board worst level and of the greedy+prune count."""
    if not os.path.exists(FEAT):
        return []
    feat = json.load(open(FEAT))
    by = defaultdict(dict)
    for r in recs:
        if int(r["size"]) == 10:
            by[r["method"]][r["set"]] = r
    lines = ["## 학습 vs 규칙: 처음 보는 종류에서 MoTT (lo_*_r2) vs `modal_rule` vs 종류 인식 greedy", "",
             "뺀 종류 t의 10×10 eval 50문제. lo_t_r2 = t를 학습·val에서 뺀 2단 MoTT (seed 0·1·2). 시간은 문제당 초 (방법 시간, 같은 CPU 1 thread). "
             "CI는 짝지은 bootstrap 95% (같은 문제를 함께 다시 뽑음, 2000회, seed 0).", "",
             "| 뺀 종류 | 방법 | 문제 수 | 성공률 | 개수 (성공 평균) | 문제당 시간 (s) | vs modal_rule: 성공률 차 [CI] | 개수 비 [CI] | vs 종류 인식 greedy: 성공률 차 [CI] | 개수 비 [CI] |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    grp_lines = []
    for t in LO_TECHS:
        sets = sorted(s for s in feat if s.startswith(f"eval_{t}10_"))

        def get(m):
            if m not in by or not all(s in by[m] for s in sets):
                return None
            cat = lambda k: np.concatenate([np.asarray(by[m][s][k], float) for s in sets])  # noqa: E731
            sec = sum(by[m][s]["sec_per_inst"] * by[m][s]["n"] for s in sets) / sum(by[m][s]["n"] for s in sets)
            return cat("ok").astype(bool), cat("count"), sec
        rule, ta, gp = get("modal_rule"), get(TA), get("greedy+prune")
        runs = [(seed_run(f"lo_{t}_r2", s), get(seed_run(f"lo_{t}_r2", s))) for s in LO_SEEDS]
        runs = [(m, v) for m, v in runs if v is not None]
        if rule is None or ta is None or not runs:
            continue
        for m, v in runs + [("modal_rule", rule), (TA, ta)]:
            ok, cnt, sec = v
            cells = []
            for ref in (rule, ta):
                if v is ref:
                    cells.append("– | –")
                else:
                    c = ci_cells(ok, ref[0], cnt, ref[1]).split(" | ", 1)[1]
                    cells.append(c)
            lines.append(f"| {t} 10 | {m} | {len(ok)} | {ok.mean():.2f} | {fmt(cnt[ok].mean() if ok.any() else None)} | "
                         f"{fmt(sec, 3)} | {cells[0]} | {cells[1]} |")
        F = [row for s in sets for row in feat[s]]
        score = np.zeros(len(F))
        for _, (ok, cnt, _) in runs:
            both = ok & rule[0]
            win = (ok & ~rule[0]) | (both & (cnt < rule[1]))
            loss = (~ok & rule[0]) | (both & (cnt > rule[1]))
            score += win.astype(float) - loss.astype(float)
        score /= len(runs)
        groups = [(f"모양 {v}", np.array([r["shape"] == v for r in F])) for v in sorted({r["shape"] for r in F})]
        groups += [(f"목표 {v}", np.array([r["target"] == v for r in F])) for v in sorted({r["target"] for r in F})]
        for what, x in (("맨판 최악 log10(Z/Zt)",np.array([r["bare_worst"] for r in F])),
                        ("greedy+prune 개수", gp[1] if gp is not None else None)):
            if x is None:
                continue
            q = np.quantile(x, [1 / 3, 2 / 3])
            for name, m in (("하", x <= q[0]), ("중", (x > q[0]) & (x <= q[1])), ("상", x > q[1])):
                groups.append((f"{what} {name} ({x[m].min():.2f}~{x[m].max():.2f})", m))
        for what, m in groups:
            grp_lines.append(f"| {t} 10 | {what} | {int(m.sum())} | {score[m].mean():+.2f} | {rule[0][m].mean():.2f} | "
                             f"{np.mean([v[0][m].mean() for _, v in runs]):.2f} | {len(runs)} |")
        grp_lines.append(f"| {t} 10 | **전체** | {len(F)} | {score.mean():+.2f} | {rule[0].mean():.2f} | "
                         f"{np.mean([v[0].mean() for _, v in runs]):.2f} | {len(runs)} |")
    if not grp_lines:
        return []
    lines += ["", "어디서 학습이 이기나 (문제마다 vs modal_rule: 정책만 성공 또는 둘 다 성공·개수 적음 = +1, 반대 = −1, 그 외 0; seed 평균):", "",
              "| 뺀 종류 | 문제 묶음 | 문제 수 | 평균 점수 (+ = 학습 우세) | modal_rule 성공률 | lo_r2 성공률 (seed 평균) | seed 수 |",
              "|---|---|---|---|---|---|---|"] + grp_lines + [""]
    return lines


def ckpt_table(recs):
    """Diagnosis H1 (PROGRESS 11/02): the same run's last.pt (`<run>@last`, runs/analysis/ckpt_last) vs the best.pt
    that every table reports (chosen on 3 RL validations of 16 problems per type)."""
    names = {r["method"] for r in recs}
    lasts = sorted(m for m in names if m.endswith("@last"))
    rows = []
    for m in lasts:
        # best.pt re-evaluated with the same eval code as last.pt (`<run>@best`) when it exists, else the reported record
        pb, pr = paired(recs, m[:-5] + "@best"), paired(recs, m[:-5])
        for key in sorted(set(pb) | set(pr)):
            ref, src = (m[:-5] + "@best", pb) if m in pb.get(key, {}) else (m[:-5], pr)
            if m in src.get(key, {}):
                rows.append(f"| {key[0]} {key[1]} | {m[:-5]} | {ref} | " + ci_cells(*src[key][m]) + " |")
    same = []
    for m in sorted(n for n in names if n.endswith("@best")):
        pr = paired(recs, m[:-5])
        for key in sorted(pr):
            if m in pr[key]:
                om, orf = pr[key][m][0], pr[key][m][1]
                same.append(f"| {key[0]} {key[1]} | {m[:-5]} | {int((om != orf).sum())} / {len(om)} | {om.mean() - orf.mean():+.2f} |")
    if not rows:
        return []
    return ["## 진단: 같은 run의 last.pt vs best.pt (`<run>@last`, 짝지은 bootstrap 95% CI, 2000회, seed 0)", "",
            "best.pt = RL 중 val 3번 (판마다 16문제, score = 평균(개수비 + 2·fail))에서 가장 좋은 것 — 모든 표가 쓰는 체크포인트. 진단용, 보고 숫자 아님.", "",
            "| 판 | run | 기준 (best.pt 기록) | 문제 수 | 성공률 차 (last − best) [95% CI] | 개수 비 last/best (둘 다 성공) [95% CI] |",
            "|---|---|---|---|---|---|"] + rows + [""] + ([
            "같은 best.pt를 지금 평가 코드로 다시 평가 (`<run>@best`) vs 보고된 기록: 성공 여부가 다른 문제 수 (평가 재현성; "
            "day 1 기록은 25문제를 한 번에 rollout, 이후는 `--inst_chunk 5` — 배치 크기에 따른 부동소수 차이로 동점 근처 argmax가 바뀔 수 있음):", "",
            "| 판 | run | 성공 여부가 다른 문제 | 성공률 차 (다시 − 기록) |", "|---|---|---|---|"] + same + [
            "", "r2_mott_rl (seed 0)은 best.pt = last.pt (같은 가중치, iter 300)인데 기록은 day 1 (한 번에 rollout), last는 `--inst_chunk 5`라 "
            "위 표의 seed 0 행이 곧 배치 크기에 따른 평가 차이다 (pcb 3/50 문제 해가 다름). day 1 run (seed 0 사다리)과 나중 run의 차이 ±0.04~0.06은 이 잡음 안일 수 있다.", ""]
            if same else [])


def grid_diag_table(recs):
    """Diagnosis H2 (PROGRESS 11/02): RL rewards use the working grid; only each instance's best sample is exact-checked,
    and a pool's grid stops growing at 1.5x its first size. From each RL log.txt (every 25 iterations): the number of
    iterations at the grid cap by the end, mean exact-check failures of the best samples (of B=16) per board type, and
    the run's 10×10 eval success per type."""
    import re
    pat = re.compile(r"it\s+\d+ \| train_([a-z]+)\d+_\d .*?cert-bad (\d+) .*?grid capped (\d+)")
    by = merged(recs)
    rows = []
    for f in sorted(glob.glob("runs/2026-10-2*/*_rl/log.txt")):
        name = os.path.basename(os.path.dirname(f))
        cb, cap, n_it = defaultdict(list), 0, None
        for line in open(f):
            m = pat.search(line)
            if m:
                cb[m.group(1)].append(int(m.group(2)))
                cap = int(m.group(3))
            if line.startswith("done:"):
                n_it = int(line.split()[1])
        if not cb:
            continue
        cells = []
        for t in LO_TECHS:
            r = by.get((t, 10), {}).get(name)
            cells.append(f"{np.mean(cb[t]):.1f}" if cb[t] else "–")
            cells.append(f"{r['ok'].mean():.2f}" if r is not None else "–")
        rows.append(f"| {name} | {cap} / {n_it if n_it else '–'} | " + " | ".join(cells) + " |")
    if not rows:
        return []
    return ["## 진단: 학습 격자 (RL log.txt, 25 iter마다 기록)", "",
            "cert-bad = 그 iter에서 작업 격자로는 성공인 최선 샘플 중 정확 검사 실패 수 (B=16 중). 격자 상한 = 풀마다 처음 F의 1.5배.", "",
            "| run | 격자 상한에 닿은 iter / 전체 iter | ip cert-bad | ip 10 성공률 | pcb cert-bad | pcb 10 성공률 | pkg cert-bad | pkg 10 성공률 |",
            "|---|---|---|---|---|---|---|---|"] + rows + [""]


def train_table():
    lines = []
    files = sorted(glob.glob("runs/**/metrics.json", recursive=True))
    if not files:
        return lines
    lines += ["## 학습 기록 (train.py metrics.json)", "",
              "| run | feat / model | stage | iters | 시간 (h) | 최종 val (count/g+p, fail) | 출처 |", "|---|---|---|---|---|---|---|"]
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        a = d.get("args", {})
        val = d.get("final_val", {})
        vs = ", ".join(f"{k}: {fmt(v[0])}/{fmt(v[1])}" for k, v in val.items())
        lines.append(f"| {os.path.dirname(f)} | {a.get('feat')} / {a.get('model')} | {a.get('stage')} | {d.get('iters')} | "
                     f"{fmt(d.get('hours'), 2)} | {vs} | `{f}` |")
    return lines + [""]


def teacher_table():
    files = sorted(glob.glob("data/teacher/*.json"))
    if not files:
        return []
    lines = ["## 선생님 해 (학습 풀, LS vs greedy+prune)", "",
             "| 풀 | g+p 성공 | LS 정확 성공 | LS/g+p 개수 | LS가 줄인 비율 | 선생님 평균 개수 | LS 시간 (s) |", "|---|---|---|---|---|---|---|"]
    for f in files:
        d = json.load(open(f))
        lines.append(f"| {d['name']} | {fmt(d['gp_ok'])} | {fmt(d['ls_ok'])} | {fmt(d['ls_over_gp'], 3)} | "
                     f"{fmt(d['improved_frac'])} | {fmt(d['teacher_count_mean'])} | {fmt(d['t_ls'], 0)} |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RESULTS.md")
    a = ap.parse_args()
    import make_type_aware
    make_type_aware.main()
    recs = load_evals()
    lines = [f"# RESULTS (자동 생성: `scripts/collect_results.py`, {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})", "",
             "성공 = `verify.py` 700점 + 피크 정밀화 정확 검사에서 최악 |Z| ≤ 1.01·Zt. 모든 방법은 cutting-plane 안에서 실행. "
             "시간은 문제당 초(판 준비 제외, 같은 CPU 계산기). 손으로 적은 숫자는 없다.", "",
             "## 평가 (정확 검사)", ""]
    lines += ladder_table(recs)
    lines += ci_table(recs)
    lines += ci_table(recs, ref="greedy+prune", title="greedy+prune")
    lines += ci_table(recs, ref="modal_rule", title="비학습 모달 규칙 `modal_rule` (US 6789241식, 공진 피크에 SRF 맞는 decap)",
                      only={f"{r}_rl" for r in ("r0_coord", "r1_modal", "r2_mott", "r3_loop", "r2a", "r3a", "r2am", "bp_park", "bm_manoharan")})
    for var, base, what, base_desc in ABLATIONS:
        lines += ci_table(recs, ref=base, title=f"{what}: `{var}` vs {base_desc} `{base}`", only={var})
    lines += distill_rl_table(recs)
    lines += eval_tables(recs)
    lines += type_table(recs)
    lines += gp_fail_table()
    lines += swap_table()
    lines += one_type_table(recs)
    lines += type_aware_table()
    lines += prior_table(recs)
    lines += leave_out_table(recs)
    lines += learn_vs_rule_table(recs)
    lines += ckpt_table(recs)
    lines += grid_diag_table(recs)
    lines += train_table()
    lines += teacher_table()
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {a.out}: {len(recs)} eval records")


if __name__ == "__main__":
    main()
