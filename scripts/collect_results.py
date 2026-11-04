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
import re
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
    if any("local_search" in by[k] for k in keys):
        lines += ["", "local_search = greedy+prune 해 (작업 격자에서 성공한 문제만)를 destroy-and-repair로 줄임 "
                  "(`decap_loop/baselines.py` `local_search`, 판정은 작업 격자). 줄인 해가 정확 검사에서 떨어지면 "
                  "성공률이 greedy+prune보다 낮아진다 — 짝지은 차이는 아래 판별 표."]
    if any("genetic" in by[k] for k in keys):
        lines += ["", "genetic = 문제마다 GA (`decap_loop/baselines.py` `genetic`, 개체 64 × 100세대, 무작위 초기화, "
                  "g+p 시드 없음; 적합도 = 개수 + 목표 위반 벌점, 작업 격자 판정). 마지막 최선 해를 greedy로 "
                  "(목표 미달이면) 채우고 가지치기 — 실패 해의 개수는 아래 종류 비율 표."]
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
    # r2_mott_rl's own record is from day 1 (no --inst_chunk); @last = the same weights (best.pt == last.pt, max diff 0)
    # re-evaluated with --inst_chunk 5 like r2c/r2v, so the pair is free of eval-batching noise (PROGRESS 11/02 round 5)
    ("r2c_rl", "r2_mott_rl@last", "2단 + `--certify 2` (같은 평가 설정 짝)", "최선 샘플만 검사, 같은 가중치를 `--inst_chunk 5`로 다시 평가한"),
    ("lo_pcb_r2c_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + `--certify 2` (같은 distill에서 RL만)", "최선 샘플만 검사"),
    # 11/02 diagnosis H1: best.pt chosen on 3 val files per kind (48 problems) instead of 1 (16); same distilled init
    ("r2v_rl", "r2_mott_rl", "2단 + best.pt 선택 val 종류당 48문제 (같은 distill에서 RL만)", "val 종류당 16문제"),
    ("r2v_rl", "r2_mott_rl@last", "2단 + val 종류당 48문제 (같은 평가 설정 짝)", "val 종류당 16문제, 같은 가중치를 `--inst_chunk 5`로 다시 평가한"),
    # 11/02 round 5: r2a (beats r2_mott on 3 seeds, same kind) on a held-out kind; vs lo_<t>_r2 and vs r2a trained on all
    *[(f"lo_{t}_r2a_rl", f"lo_{t}_r2_rl", f"{t} 빼고 학습 2단 + 주파수 궁합 편향 (`--fc_bias`)", "편향 없는 같은 단 (같은 종류 빼고)")
      for t in ("pcb", "ip", "pkg")],
    *[(f"lo_{t}_r2a_rl", "r2a_rl", f"{t} 빼고 학습 r2a (처음 보는 종류에서 얼마나 떨어지나)", "세 종류 모두 학습한 r2a")
      for t in ("pcb", "ip", "pkg")],
    ("lo_pcb_r2v_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + val 종류당 48문제 (같은 distill에서 RL만)", "val 종류당 16문제"),
    # 11/02 round 7, diagnosis H2 cheap fix: rl success needs worst working-grid log10(|Z|/Zt) <= -log10(1.05)
    ("r2m_rl", "r2_mott_rl", "2단 + 보상 성공에 5% 여유 (`--feas_margin`, 같은 distill에서 RL만)", "여유 없음"),
    ("r2m_rl", "r2_mott_rl@last", "2단 + `--feas_margin` (같은 평가 설정 짝)", "여유 없음, 같은 가중치를 `--inst_chunk 5`로 다시 평가한"),
    ("lo_pcb_r2m_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + `--feas_margin` (같은 distill에서 RL만)", "여유 없음"),
    # 11/02 round 14, diagnosis H2 other fix: working-grid growth cap 0.5 -> 1.0 of the initial F
    ("r2g_rl", "r2_mott_rl", "2단 + 작업 격자 상한 완화 (`--grid_growth 1.0`, 같은 distill에서 RL만)", "상한 0.5"),
    ("r2g_rl", "r2_mott_rl@last", "2단 + `--grid_growth 1.0` (같은 평가 설정 짝)", "상한 0.5, 같은 가중치를 `--inst_chunk 5`로 다시 평가한"),
    ("lo_pcb_r2g_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + `--grid_growth 1.0` (같은 distill에서 RL만)", "상한 0.5"),
    # 11/03 round 27 (QUEUE 11/03 11:40 KST obs. 3, PROPOSALS T1): teacher store = per instance the certified solution with
    # fewer decaps among {stored LS/g+p teacher, modal_rule in 4-round cutting-plane}; same distilled init, rl 1.25 h
    ("r2t_rl", "r2_mott_rl", "2단 + 규칙으로 보강한 선생님 (`--teacher_dir data/teacher_rcp`, 같은 distill에서 RL만)", "지금 선생님"),
    ("r2t_rl", "r2_mott_rl@last", "2단 + 규칙 보강 선생님 (같은 평가 설정 짝)", "지금 선생님, 같은 가중치를 `--inst_chunk 5`로 다시 평가한"),
    ("lo_pcb_r2t_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + 규칙 보강 선생님 (같은 distill에서 RL만)", "지금 선생님"),
    ("r2t_rl", "modal_rule", "2단 + 규칙 보강 선생님 vs 선생님에 쓴 규칙 자체 (판정 아님, QUEUE 11/03 관찰 3)", "비학습 규칙"),
    ("lo_pcb_r2t_rl", "modal_rule", "pcb 빼고 학습 2단 + 규칙 보강 선생님 vs 규칙 자체 (판정 아님)", "비학습 규칙"),
    # PROPOSALS T1': the same teacher in distill + rl (full 0.75 h + 1.25 h run), vs the full r2_mott runs
    ("r2tf_rl", "r2_mott_rl", "2단 + 규칙 보강 선생님, 증류+RL 전체 (`--teacher_dir data/teacher_rcp`)", "지금 선생님"),
    ("r2tf_rl", "r2_mott_rl@last", "2단 + 규칙 보강 선생님 증류+RL 전체 (같은 평가 설정 짝)", "지금 선생님, 같은 가중치를 `--inst_chunk 5`로 다시 평가한"),
    ("lo_pcb_r2tf_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + 규칙 보강 선생님, 증류+RL 전체", "지금 선생님"),
    ("r2tf_rl", "modal_rule", "2단 + 규칙 보강 선생님 증류+RL 전체 vs 규칙 자체 (판정 아님)", "비학습 규칙"),
    ("lo_pcb_r2tf_rl", "modal_rule", "pcb 빼고 학습 2단 + 규칙 보강 선생님 증류+RL 전체 vs 규칙 자체 (판정 아님)", "비학습 규칙"),
    # 11/03 round 29 (PROPOSALS T3): rl loss weighted by the instance's g+p-count percentile in its board kind
    ("r2h_rl", "r2_mott_rl", "2단 + 어려운 문제 가중 RL (`--hard_weight 2`, 같은 distill에서 RL만)", "가중 없음"),
    ("r2h_rl", "r2_mott_rl@last", "2단 + `--hard_weight 2` (같은 평가 설정 짝)", "가중 없음, 같은 가중치를 `--inst_chunk 5`로 다시 평가한"),
    ("lo_pcb_r2h_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + `--hard_weight 2` (같은 distill에서 RL만)", "가중 없음"),
    # 11/03 round 32 (PROPOSALS T5): no change, rl only again from the same distill with seed 1 -> rl training noise
    ("r2s1_rl", "r2_mott_rl", "재현 잡음: 2단 같은 distill에서 RL만 seed 1 (수정 없음)", "RL seed 0"),
    ("lo_pcb_r2s1_rl", "lo_pcb_r2_rl", "재현 잡음: pcb 빼고 학습 2단 같은 distill에서 RL만 seed 1 (수정 없음)", "RL seed 0"),
    # 11/03 round 33 (PROPOSALS T5b): the same with rl seed 2
    ("r2s2_rl", "r2_mott_rl", "재현 잡음: 2단 같은 distill에서 RL만 seed 2 (수정 없음)", "RL seed 0"),
    ("lo_pcb_r2s2_rl", "lo_pcb_r2_rl", "재현 잡음: pcb 빼고 학습 2단 같은 distill에서 RL만 seed 2 (수정 없음)", "RL seed 0"),
    # 11/03 round 36 (PROPOSALS P1, D7 H-underres): mode-token parabolic peak refinement input, full distill + rl, seeds 0/1
    ("r2pk_rl", "r2_mott_rl", "2단 + 모드 토큰 피크 보정 입력 (`--peak_refine`, 증류+RL 전체, seed 0)", "보정 없음"),
    ("r2pk_s1_rl", "r2_mott_s1_rl", "2단 + `--peak_refine` (seed 1)", "보정 없음"),
    ("lo_pcb_r2pk_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + `--peak_refine` (seed 0)", "보정 없음"),
    ("lo_pcb_r2pk_s1_rl", "lo_pcb_r2_s1_rl", "pcb 빼고 학습 2단 + `--peak_refine` (seed 1)", "보정 없음"),
    ("r2pk_distill", "r2_mott_distill", "증류만: 2단 + `--peak_refine` (seed 0, D1 표적)", "보정 없음"),
    ("r2pk_s1_distill", "r2_mott_s1_distill", "증류만: 2단 + `--peak_refine` (seed 1, D1 표적)", "보정 없음"),
    # 11/04 round 37 (PROPOSALS P2, D8b H-dup): mode tokens on distinct working-grid frequencies, full pipeline, seeds 0/1
    ("r2dd_rl", "r2_mott_rl", "2단 + 중복 없는 모드 토큰 (`--dedup_modes`, 증류+RL 전체, seed 0)", "중복 그대로"),
    ("r2dd_s1_rl", "r2_mott_s1_rl", "2단 + `--dedup_modes` (seed 1)", "중복 그대로"),
    ("lo_pcb_r2dd_rl", "lo_pcb_r2_rl", "pcb 빼고 학습 2단 + `--dedup_modes` (seed 0)", "중복 그대로"),
    ("lo_pcb_r2dd_s1_rl", "lo_pcb_r2_s1_rl", "pcb 빼고 학습 2단 + `--dedup_modes` (seed 1)", "중복 그대로"),
    ("r2dd_distill", "r2_mott_distill", "증류만: 2단 + `--dedup_modes` (seed 0, D1 표적)", "중복 그대로"),
    ("r2dd_s1_distill", "r2_mott_s1_distill", "증류만: 2단 + `--dedup_modes` (seed 1, D1 표적)", "중복 그대로"),
    # 11/04 round 39 (PROPOSALS P1 확장, queued after the seed 0+1 "win"): seed 2 + held-out ip / pkg
    ("r2pk_s2_rl", "r2_mott_s2_rl", "2단 + `--peak_refine` (seed 2, P1 확장)", "보정 없음"),
    ("lo_pcb_r2pk_s2_rl", "lo_pcb_r2_s2_rl", "pcb 빼고 학습 2단 + `--peak_refine` (seed 2, P1 확장)", "보정 없음"),
    ("lo_ip_r2pk_rl", "lo_ip_r2_rl", "ip 빼고 학습 2단 + `--peak_refine` (seed 0, P1 확장)", "보정 없음"),
    ("lo_pkg_r2pk_rl", "lo_pkg_r2_rl", "pkg 빼고 학습 2단 + `--peak_refine` (seed 0, P1 확장)", "보정 없음"),
    # 11/04 round 40 (PROPOSALS P3, written before the P2 rl results): P2 tokens + distinct-neighbour refine on top of P1
    ("r2pkdd_rl", "r2pk_rl", "2단 P1 + 고유 주파수 토큰·이웃 (`--peak_refine --dedup_modes`, seed 0, P3)", "P1"),
    ("r2pkdd_s1_rl", "r2pk_s1_rl", "2단 P1 + 고유 주파수 (seed 1, P3)", "P1"),
    ("lo_pcb_r2pkdd_rl", "lo_pcb_r2pk_rl", "pcb 빼고 학습 2단 P1 + 고유 주파수 (seed 0, P3)", "P1"),
    ("lo_pcb_r2pkdd_s1_rl", "lo_pcb_r2pk_s1_rl", "pcb 빼고 학습 2단 P1 + 고유 주파수 (seed 1, P3)", "P1"),
    ("r2pkdd_distill", "r2pk_distill", "증류만: 2단 P1 + 고유 주파수 (seed 0, P3)", "P1"),
    ("r2pkdd_s1_distill", "r2pk_s1_distill", "증류만: 2단 P1 + 고유 주파수 (seed 1, P3)", "P1"),
]


PRIOR = {  # prior-work re-implementations (QUEUE 10/29 03:30 C): how each differs from the paper
    "bp_park": ("B-Park: Park 2022 / DevFormer식 좌표 입력 attention-model 포인터 정책 (`baselines_ml/park_am.py`)", [
        "**원 문제가 다름 (Park 2022 arXiv v2 원문 확인):** 원 논문은 HBM PDN 한 장에서 **decap 개수 m 고정 + 임피던스 감소 최대화** "
        "(목표 임피던스·성공 판정 없음, 166점 |Z11|·|Z21| 감소 끝 보상) — 우리는 목표 만족 + 개수 최소화. 그래서 이 행들은 "
        "\"Park의 정책 구조·입력을 우리 문제로 옮긴 재구현\"이지 원 논문 재현이 아님. 전체 차이 표: `research_notes/선행 기준선 재구현 차이 표.md`.",
        "학습 예산: 원 논문은 RL만 (rollout + paired t-test 기준선, 2만 문제) — `bp_park_rl`은 우리와 같은 distill 0.75 h + RL 1.25 h (같은 선생님 해, 같은 학습 풀), "
        "`bp_park_rlonly_rl`은 RL만 2.0 h.",
        "입력: 원 논문은 노드마다 (x, y, z, probe 여부)뿐 — 우리 재구현 (`--feat coord`)은 여기에 놓인 종류·전역 위반 (고정 특징 주파수 8점)·첫 probe까지 거리가 더 들어감.",
        "행동 공간: 원 논문은 포트 선택(종류 고정) — 여기서는 우리 행동 공간 (칸 × 종류 추가/제거 + 멈춤), 채널마다 포인터 query.",
        "보상: 원 논문은 임피던스 점수 — 여기서는 우리 보상 (−개수/N_ref − λ·(실패+위반), λ 라그랑주), 정확 검사 실패 반영.",
        "구조·학습: Kool식 encode-once 대신 매 수 다시 인코딩 (상태가 특징에 있음), BatchNorm 대신 LayerNorm, 기준선 교체는 "
        "paired t-test 대신 val 최고 갱신 때, 기준선 rollout의 실패는 작업 격자 기준 (정확 검사 안 함), 모방 항·선생님 교체 없음 (`--imit 0`).",
        "DevFormer 공개 코드의 AM-PG 기준선 설정: 10×10 판 한 장, 한 종류 decap **K = 20개 고정**, "
        "보상 = 201점 임피던스 감소 합 (목표 Zt 없음), 입력 (x, y, 탐침/keep-out), 5만 문제 × 10 epoch → 원 논문은 개수 최소화·목표 만족 문제가 아님. "
        "표: `research_notes/선행 기준선 설정 확인 (DevFormer 공개 코드).md`.",
        "`bp_park_rlonly_rl` (원 논문처럼 RL만 2.0 h, 무작위 초기화, 초기 λ 20 — 아래 B-Manoharan 줄과 같은 규칙): 해 모양 (위 \"decap 종류 비율\" 표)은 "
        "**세 판 종류 모두 종류 1 하나만** 쓰고 많이 놓음 → pcb·pkg는 성공률이 distill 판보다 높지만 개수가 E0+의 몇 배, ip는 대부분 실패. "
        "좌표 입력 + 포인터 정책은 RL만으로 판 종류에 맞는 종류 선택을 배우지 못함. best.pt는 it 100 (2 h 중 앞쪽)."]),
    "bm_manoharan": ("B-Manoharan: Manoharan 2026식 고정 포트 순서 + decap 토큰으로 종류/빈칸 선택 (`baselines_ml/manoharan.py`)", [
        "**학습 예산이 가장 크게 다름 (박사논문 Paper I 메모):** 원 논문은 RTX 4090 ~24 h, 에피소드 5만, REINFORCE + critic + replay buffer, PCB만 학습 — "
        "우리는 같은 총 예산 **2.0 h**, critic·replay 없음, 세 판 종류. 아래 pcb 결과 (0.00 포함)는 **같은 2 h 예산에서**의 숫자로만 쓴다. "
        "전체 차이 표: `research_notes/선행 기준선 재구현 차이 표.md`.",
        "`bm_manoharan_rl` 학습: 원 논문의 RL 세부 대신 우리 파이프라인 그대로 (distill 0.75 h: 선생님 해를 고정 순서로 다시 걸은 궤적, RL 1.25 h: 그룹 기준선 + 모방 + 선생님 교체).",
        "포트 순서: 최악 탐침에서 본 loop L (포트 i 단락, 한 주파수 f0) 오름차순 — f0는 맨판 최악 위반 주파수 (원 논문은 고정 설계 주파수).",
        "상태: 고정 특징 주파수 8점의 |Z|/Zt + 전역 블록 MLP와 현재 포트 L 임베딩; 원 논문의 Z 행렬 입력 전체는 아님. 갱신은 우리 env의 rank-1 (Schur/Woodbury).",
        "원문 미확인 항목 (사람 메모 기준, 우리 재구현의 선택): 보상 식 (끝 보상 −개수/N_ref − λ·실패), \"빈칸\" 행동, 종료 조건 (목표 만족 또는 순서 끝), "
        "n_f = 8과 주파수 범위, 네트워크 크기·batch·lr (사다리와 같게). 패드 크기 one-hot은 우리 판에 없어 뺌.",
        "행동: 현재 포트에 종류 또는 빈칸, 제거·종류 변경 없음; 에피소드 길이 = 놓을 수 있는 칸 수 + 1 (학습의 max_steps 64 대신).",
        "`*_rlonly_l20` (bp_park_rlonly 포함): 원 논문처럼 RL만 2.0 h (= distill + RL 예산), 무작위 초기화, 모방 없음, 초기 λ 20 "
        "(우리 run은 1). 초기 λ 1로 한 bm_manoharan_rlonly는 it 75부터 모든 샘플이 0개 놓고 실패 → advantage 0으로 붕괴해 34분에 멈춤 "
        "(평가 없음, `runs/2026-10-29/bm_manoharan_rlonly_rl/log.txt`).",
        "`bm_manoharan_rlonly_l20_rl` 해 모양 (위 \"decap 종류 비율\" 표): 판 종류마다 **한 종류만** 씀 (ip 종류 0, pkg 종류 1, pcb 종류 2). "
        "pkg에서는 고정 순서의 거의 모든 칸에 놓아 성공하지만 개수가 g+p의 몇 배, pcb에서는 몇 개만 놓고 순서 끝에 닿아 전부 실패. "
        "best.pt는 it 100 (val 16문제 × 3, 2 h 중 앞쪽) — RL만으로는 고정 순서 + 종류/빈칸 행동에서 '어디서 멈출지·어느 종류'를 판마다 배우지 못함."]),
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
    return lines + [""] + leave_out_seed_table(by) + final_leave_out_table(by) + loop_vs_plain_table(recs, by)


LO_SEEDS = (0, 1, 2)


def final_leave_out_table(by):
    """PROPOSALS 11/04 F1 (written round 43, before lo_{ip,pkg}_r2pk_s{1,2} ran): the QUEUE 10/31 11:30 leave-out rule
    applied unchanged to the adopted final structure r2pk (2단 + `--peak_refine`): per held-out type t, (b') lo_t_r2pk
    − lo_t_r0 > 0 at all three seeds and (c') mean over seeds of lo_t_r2pk − r2pk (same seed) ≥ −0.05 -> "통과".
    The r2_mott verdict is recomputed next to it with the same rule (not changed)."""
    if not any(m.startswith("lo_") and "_r2pk" in m for ms in by.values() for m in ms):
        return []

    def rate(t, m):
        r = by.get((t, 10), {}).get(m)
        return r["ok"].mean() if r is not None else None

    def diffs(t, a, b):
        return [None if x is None or y is None else x - y
                for x, y in ((rate(t, a(s)), rate(t, b(s))) for s in LO_SEEDS)]

    def fmt3(v, sign=True):
        return " / ".join("–" if x is None else (f"{x:+.2f}" if sign else f"{x:.2f}") for x in v)

    def verdict(b, c):
        if any(x is None for x in b + c):
            return "seed 3개가 다 나온 뒤"
        cm = float(np.mean(c))
        return ("**통과**" if all(x > 0 for x in b) and cm >= -0.05 else
                f"실패 ((b') seed별 {fmt3(b)}, (c') 평균 {cm:+.2f})")
    lines = ["### 최종 후보 r2pk (2단 + `--peak_refine`)의 한 종류 빼고 학습, seed별 (PROPOSALS 11/04 F1, QUEUE 10/31 11:30 기준 그대로)", "",
             "| 뺀 종류 | 단 | 뺀 학습 seed 0 / 1 / 2 | 전체 학습 seed 0 / 1 / 2 | (b') 뺀 학습 − 0단 뺀 학습 | (c') 뺀 − 전체 (같은 seed) | 평균 (c') | 판정 |",
             "|---|---|---|---|---|---|---|---|"]
    for t in LO_TECHS:
        for r, full, desc in (("r2pk", "r2pk", "최종 후보 r2pk"), ("r2", "r2_mott", "2단 MoTT (기존, 같은 규칙)")):
            lo = [rate(t, seed_run(f"lo_{t}_{r}", s)) for s in LO_SEEDS]
            fu = [rate(t, seed_run(full, s)) for s in LO_SEEDS]
            b = diffs(t, lambda s: seed_run(f"lo_{t}_{r}", s), lambda s: seed_run(f"lo_{t}_r0", s))
            c = diffs(t, lambda s: seed_run(f"lo_{t}_{r}", s), lambda s: seed_run(full, s))
            cm = "–" if any(x is None for x in c) else f"{np.mean(c):+.2f}"
            lines.append(f"| {t} 10 | {desc} | {fmt3(lo, False)} | {fmt3(fu, False)} | {fmt3(b)} | {fmt3(c)} | {cm} | "
                         f"{verdict(b, c)} |")
    return lines + [""]


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


ZS_RUNGS = (("r0_coord", "0단 좌표"), ("r1_modal", "1단 modal"), ("r2_mott", "2단 MoTT"), ("r3_loop", "3단 루프"),
            ("r2a", "최종 후보 r2a (2단 + fc_bias)"))


def zs_chip_table(recs):
    """PROPOSALS 11/03 (fixed before any result): zero-shot to the unseen fourth kind `chip` (AGENT_PROMPT bonus).
    The ladder runs trained on pcb+pkg+ip at seeds 0-2, no training on chip; 50 problems (eval_chip10_0/1).
    Verdict rule: (b) 2단 − 0단 success > 0 at all three seeds -> "모달 표현이 chip으로 0단보다 잘 옮겨 감"."""
    ms = merged(recs).get(("chip", 10), {})
    if not ms:
        return []

    def cell(m):
        r = ms.get(m)
        return "–" if r is None else f"{r['ok'].mean():.2f} / {fmt(r['rel'], 2)}"

    def rate(m):
        r = ms.get(m)
        return None if r is None else r["ok"].mean()
    lines = ["## 보너스: 학습에 없던 네 번째 종류 `chip` zero-shot (PROPOSALS 11/03, 판정 규칙은 결과 전에 고정)", "",
             "세 종류 (pcb·pkg·ip) 전체 학습 run을 학습 없이 chip 10×10 eval 50문제 (`runs/pool_zs/eval_chip10_{0,1}`)에 평가. "
             "칸 = 정확 검사 성공률 / 개수÷g+p (둘 다 성공, 같은 파일끼리).", "",
             "| 방법 | seed 0 | seed 1 | seed 2 | 성공률 평균 (seed 3개) |", "|---|---|---|---|---|"]
    for m in ("greedy+prune", TA, "modal_rule"):
        if m in ms:
            lines.append(f"| {m} | {cell(m)} | | | |")
    for r, desc in ZS_RUNGS:
        v = [rate(seed_run(r, s)) for s in LO_SEEDS]
        mean = f"{np.mean(v):.2f}" if all(x is not None for x in v) else "–"
        lines.append(f"| {desc} `{r}` | " + " | ".join(cell(seed_run(r, s)) for s in LO_SEEDS) + f" | {mean} |")
    lines += ["", "| 비교 | seed 0 / 1 / 2 | 평균 | 0보다 큰 seed 수 |", "|---|---|---|---|"]
    verdict = None
    for what, a, b in (("(a) 1단 − 0단", "r1_modal", "r0_coord"), ("(b) 2단 − 0단", "r2_mott", "r0_coord"),
                       ("3단 − 0단", "r3_loop", "r0_coord"), ("r2a − 0단", "r2a", "r0_coord")):
        d = [None if x is None or y is None else x - y
             for x, y in ((rate(seed_run(a, s)), rate(seed_run(b, s))) for s in LO_SEEDS)]
        have = [x for x in d if x is not None]
        mean = f"{np.mean(have):+.2f}" if len(have) == len(d) else "–"
        lines.append(f"| {what} | " + " / ".join("–" if x is None else f"{x:+.2f}" for x in d) +
                     f" | {mean} | {sum(x > 0 for x in have)}/{len(have)} |")
        if a == "r2_mott" and len(have) == len(d):
            verdict = all(x > 0 for x in have)
    lines += ["", "판정 (미리 적은 규칙: (b) 2단 − 0단이 seed 3개 모두 > 0): " +
              ("seed 3개가 다 나온 뒤." if verdict is None else
               "**충족** — 모달 표현이 처음 보는 종류 chip으로 0단보다 잘 옮겨 감 (고찰용, 보너스 — 주장 아님)." if verdict else
               "**불충족** — \"모달 표현이 chip으로 0단보다 잘 옮겨 감\"이라고 쓰지 않는다."), "",
              "seed 0 짝지은 bootstrap 95% CI (같은 50문제):", "",
              "| 방법 | 기준 | 문제 수 | 성공률 차 (방법 − 기준) [95% CI] | 개수 비 방법/기준 (둘 다 성공) [95% CI] |",
              "|---|---|---|---|---|"]
    for ref in ("greedy+prune", "modal_rule", TA, "r0_coord_rl"):   # vs 0단: descriptive (count), not the verdict
        pr = paired(recs, ref).get(("chip", 10), {})
        for r, _ in ZS_RUNGS:
            if f"{r}_rl" in pr:
                lines.append(f"| {r}_rl | {ref} | " + ci_cells(*pr[f"{r}_rl"]) + " |")
    if ms.get("r0_coord_rl") is not None and ms["r0_coord_rl"]["ok"].all():
        lines += ["", "참고: seed 0에서 0단의 성공률이 이미 1.00이라 성공률로 정한 판정 (b)는 > 0이 될 수 없다 (천장). "
                  "단 차이는 위 \"vs r0_coord_rl\" 개수 비에만 보인다 — 미리 적은 판정 대상이 아니므로 기술용으로만 쓴다."]
    return lines + [""]


TOKEN_NOTE = "research_notes/토큰 데이터 분석/결과.md"   # training-free token analysis (token_analysis2.py, run2_10x20.log)


def token_transfer_rows():
    """Rows of the v2 single-feature table in TOKEN_NOTE (20×20, LS teacher): feature, KS, teacher-label AUC PCB → IP."""
    if not os.path.exists(TOKEN_NOTE):
        return []
    rows, on = [], False
    for line in open(TOKEN_NOTE):
        if line.startswith("## 단일 특징"):
            on = True
        elif on and line.startswith("|") and not line.startswith("|---") and "KS" not in line:
            rows.append([c.replace("*", "").strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))])
        elif on and rows and not line.startswith("|"):
            break
    return rows


def design_rationale(recs):
    """QUEUE 2026-11-01 15:10 item 6: which observation led to which design choice, and what the leave-one-type-out
    ladder then showed. Only the parts whose pre-registered claim holds are written as claims; the verdict per held-out
    type is recomputed here from the 10/31 11:30 rule ((b) 2단 − 0단 > 0 at all three seeds, (c) mean of 2단 뺀 −
    전체 ≥ −0.05)."""
    by = merged(recs)

    def rate(t, m):
        r = by.get((t, 10), {}).get(m)
        return r["ok"].mean() if r is not None else None

    def seeds(t, f):
        v = [f(s) for s in LO_SEEDS]
        return None if any(x is None for x in v) else v
    verdict, drop = {}, {}
    for t in LO_TECHS:
        b = seeds(t, lambda s: None if rate(t, seed_run(f"lo_{t}_r2", s)) is None or rate(t, seed_run(f"lo_{t}_r0", s)) is None
                  else rate(t, seed_run(f"lo_{t}_r2", s)) - rate(t, seed_run(f"lo_{t}_r0", s)))
        for r, full, _ in LO_RUNGS:
            d = seeds(t, lambda s: None if rate(t, seed_run(f"lo_{t}_{r}", s)) is None or rate(t, seed_run(full[:-3], s)) is None
                      else rate(t, seed_run(f"lo_{t}_{r}", s)) - rate(t, seed_run(full[:-3], s)))
            drop[t, r] = None if d is None else float(np.mean(d))
        c = drop[t, "r2"]
        verdict[t] = ("–" if b is None or c is None else
                      "통과" if all(x > 0 for x in b) and c >= -0.05 else
                      f"실패 ((b) seed별 {' / '.join(f'{x:+.2f}' for x in b)}, (c) 평균 {c:+.2f})")
    if not any(v != "–" for v in verdict.values()):
        return []
    tok = token_transfer_rows()
    lines = ["## 설계 근거: 관찰 → 설계 → 결과 (QUEUE 11/01 할 일 6; 미리 적은 기준이 선 부분만 주장)", "",
             "1. **선행 표현 (좌표, 주파수 한 점의 loop L)은 판 종류를 넘으면 뒤집힌다; 모달 참여비와 정확한 1수 효과는 옮겨간다** — 학습 없는 분석 "
             f"(`{TOKEN_NOTE}` v2, 20×20, PCB형·인터포저형만 (pkg 없음), 종류마다 판 10개, LS 선생님 해 예측 AUC, 방향은 PCB에서 정함, "
             "신뢰구간 없음, 선형 모델 기준):", ""]
    if tok:
        lines += ["| 특징 | KS (PCB vs IP 분포) | AUC PCB → IP |", "|---|---|---|"] + [f"| {' | '.join(r)} |" for r in tok] + [""]
    lines += ["   → 설계: 포트 특징에 **모달 참여비 + 정확한 1수 효과** (1단 `--feat modal`), 모드 토큰과 포트↔모드 cross-attention (2단), "
              "세 종류를 **섞어** 학습. 매 수 정확 Z 갱신과 decap 토큰은 Manoharan 2026에 이미 있음 (기여 아님); Park 2022는 Z를 단계마다 갱신하지 않고 "
              "좌표만 입력 (`research_notes/선행 기준선 재구현 차이 표.md`).", "",
              "2. **학습으로 확인 (한 종류 빼고 학습, 10×10, seed 0·1·2 평균, 뺀 종류의 eval 50문제 성공률):** 뺀 학습 − 전체 학습 (같은 seed)", "",
              "| 뺀 종류 | 0단 좌표 | 1단 모달 | 2단 MoTT | 10/31 기준 판정 (2단) |", "|---|---|---|---|---|"]
    for t in LO_TECHS:
        lines.append(f"| {t} | " + " | ".join("–" if drop[t, r] is None else f"{drop[t, r]:+.2f}" for r, _, _ in LO_RUNGS)
                     + f" | {verdict[t]} |")
    ok_t = [t for t in LO_TECHS if verdict[t] == "통과"]
    lines += ["", f"   → 주장할 수 있는 범위: **{', '.join(ok_t) if ok_t else '없음'}** 에서만 \"처음 보는 종류로의 범용화\" (seed 3개). "
              "그 밖의 종류는 위 판정 그대로 주장에서 뺀다. 좌표 (0단)가 무너지는 것은 1번 관찰 (좌표가 종류를 넘으면 뒤집힘)과 같은 방향; "
              "pkg는 학습한 두 종류 사이 (decap 라이브러리·맨판 위반이 pcb와 ip 사이, `scripts/type_distance.py`)라 내삽에 가깝다.", "",
              "3. 주장하지 않는 것 (미리 적은 기준 불충족): 루프 > 기본 (\"루프 트랜스포머 vs 기본\" 표), MoTT 개선 시도 `--certify 2`·val 48문제·best-of-8 샘플링·"
              "보상 성공 5% 여유 `--feas_margin` (r2m·lo_pcb_r2m: 성공률 CI 하한이 0 위인 판정 갈래 없음, 개수는 늘어남)·"
              "작업 격자 상한 완화 `--grid_growth 1.0` (r2g·lo_pcb_r2g: 격자 상한에 닿은 step은 줄었지만 성공률 CI 하한이 0 위인 갈래 없음)·"
              "규칙으로 보강한 선생님 RL만 (r2t·lo_pcb_r2t)·증류+RL 전체 (r2tf·lo_pcb_r2tf) (둘 다 성공률 CI 하한이 0 위인 갈래 없음, "
              "뺀 pcb run의 pkg는 내려감)·규칙 해에서 출발해 정책이 다듬기 (평가만: r2는 규칙 배치를 거의 안 바꾸고, 0단은 pkg를 해침) "
              "(위 ABLATIONS·\"추론 샘플링\" 표), 주파수 궁합 편향 (`--fc_bias`)이 처음 보는 pcb·ip로의 범용화를 돕는다는 것 "
              "(`lo_<t>_r2a_rl` vs `lo_<t>_r2_rl`, seed 0, ABLATIONS), 학습이 처음 보는 종류에서 `modal_rule`을 이긴다는 것 (\"학습 vs 규칙\" 표에서 판마다 확인).", ""]
    return lines


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


def sampling_table(recs):
    """PROGRESS 11/02 round 6: best-of-S policy sampling at inference (`<run>@S`, runs/analysis/sampling) vs the same
    checkpoint's greedy decoding, the untrained modal_rule and E0+; success, count, time per problem, paired CIs."""
    samp = sorted({r["method"] for r in recs if re.fullmatch(r".+_rl@\d+", r["method"])})
    rows = []
    for m in samp:
        base = m.rsplit("@", 1)[0]
        for ref in (base, "modal_rule", TA):
            pr = paired(recs, ref)
            for key in sorted(pr):
                if m in pr[key]:
                    om, orf, cm, cr = pr[key][m]
                    sec = lambda meth: np.mean([r["sec_per_inst"] for r in recs if r["method"] == meth and  # noqa: E731
                                                r["tech"] == key[0] and int(r["size"]) == key[1]])
                    rows.append(f"| {key[0]} {key[1]} | {m} | {ref} | {om.mean():.2f} / {orf.mean():.2f} | "
                                f"{fmt(sec(m), 2)} / {fmt(sec(ref), 2)} | " + ci_cells(om, orf, cm, cr) + " |")
    if not rows:
        return []
    return ["## 추론 샘플링 (best-of-S): 같은 체크포인트를 S번 샘플링, 작업 격자 성공 중 개수 최소 → cutting-plane 정확 검사", "",
            "학습·평가 문제는 그대로, 추론 계산만 S배. 판정 규칙은 PROGRESS 11/02 라운드 6에 결과 전에 적음. "
            "CI는 짝지은 bootstrap 95% (2000회, seed 0). 시간은 파일별 문제당 초의 평균 (같은 CPU 1 thread).", "",
            "| 판 | 샘플링 | 기준 | 성공률 (샘플링 / 기준) | 문제당 시간 s (샘플링 / 기준) | 문제 수 | 성공률 차 [95% CI] | 개수 비 (둘 다 성공) [95% CI] |",
            "|---|---|---|---|---|---|---|---|"] + rows + [""]


def repair_table(recs):
    """PROPOSALS 11/03 T2: the repair stage (`<method>+repK`, eval_policy.py --repair). (R1/R3) after vs before the
    repair inside the same record (ok_pre/count_pre: same run, no eval-batch noise); (R2) the policy with repair vs
    modal_rule / greedy+prune with the same repair, paired on the same problems."""
    reps = [r for r in recs if "ok_pre" in r]
    if not reps:
        return []
    by = defaultdict(lambda: defaultdict(list))
    for r in reps:
        by[(r["tech"], int(r["size"]))][r["method"]].append(r)
    rows = []
    for key in sorted(by):
        for m in sorted(by[key], key=order):
            rs = sorted(by[key][m], key=lambda r: r["set"])
            om, orf = (np.concatenate([np.asarray(r[k], bool) for r in rs]) for k in ("ok", "ok_pre"))
            cm, cr = (np.concatenate([np.asarray(r[k], float) for r in rs]) for k in ("count", "count_pre"))
            fixed = om & ~orf
            add = fmt((cm[fixed] - cr[fixed]).mean() if fixed.any() else None)
            sec = sum(r["sec_repair_total"] for r in rs) / max(1, sum(r["n"] for r in rs))
            rows.append(f"| {key[0]} {key[1]} | {m} | {orf.mean():.2f} → {om.mean():.2f} | {int(fixed.sum())} | {add} | "
                        f"{fmt(sec, 2)} | " + ci_cells(om, orf, cm, cr) + " |")
    lines = ["## 끝에서 고치기 (repair, PROPOSALS 11/03 T2): cutting-plane 뒤 정확 검사 실패 문제만, 실패 주파수를 격자에 더하고 "
             "같은 방법이 자기 배치에서 이어서 놓음 (최대 K번)", "",
             "판정 규칙은 결과 전에 PROPOSALS 11/03 T2에 적음. 본 표 (위)의 숫자는 고치기 없는 평가 그대로. "
             "전후는 같은 기록 안 (`ok_pre`)이라 평가 배치 잡음 없음. CI는 짝지은 bootstrap 95% (2000회, seed 0).", "",
             "| 판 | 방법 | 성공률 (전 → 후) | 고친 문제 수 | 고친 문제의 개수 변화 (평균) | 고치기 시간 s/문제 | 문제 수 | "
             "성공률 차 후 − 전 [95% CI] | 개수 비 후/전 (둘 다 성공) [95% CI] |", "|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    pol = sorted({r["method"] for r in reps if r.get("base_method") not in ("modal_rule", "greedy+prune")})
    for ref in sorted({r["method"] for r in reps if r.get("base_method") in ("modal_rule", "greedy+prune")}):
        lines += ci_table(reps, ref=ref, title=f"같은 고치기를 준 `{ref}`", only=set(pol))
    return lines


def rulestart_table(recs):
    """PROPOSALS 11/03 T4: modal_rule's solution as the start, the policy continues (`modal_rule><policy>`,
    eval_policy.py --start rule). (S1/S2) vs modal_rule, paired on the same problems; descriptive: how often the
    policy changed the rule's placement, and vs the same policy from an empty board."""
    rs = [r for r in recs if r["method"].startswith("modal_rule>")]
    if not rs:
        return []
    rule = {r["set"]: r for r in recs if r["method"] == "modal_rule" and "ok_pre" not in r}
    rows = []
    by = defaultdict(lambda: defaultdict(list))
    for r in rs:
        by[(r["tech"], int(r["size"]))][r["method"]].append(r)
    for key in sorted(by):
        for m in sorted(by[key]):
            ch = n = 0
            for r in by[key][m]:
                if r["set"] in rule:
                    ch += sum(a != b for a, b in zip(r["occ"], rule[r["set"]]["occ"]))
                    n += r["n"]
            rows.append(f"| {key[0]} {key[1]} | {m} | {ch} / {n} |")
    pol = {r["method"] for r in rs}
    lines = ["## 규칙 해에서 출발해 정책이 다듬기 (PROPOSALS 11/03 T4, 평가만): cutting-plane 각 round에서 modal_rule 해를 "
             "시작 배치로, 정책이 greedy decoding으로 이어 감 (목표 만족 상태라 remove·종류 바꿈·stop만)", "",
             "판정 규칙은 결과 전에 PROPOSALS 11/03 T4에 적음. 고치기 없음. 정책이 규칙 배치를 바꾼 문제 수는 같은 판의 "
             "`modal_rule` 기록 (cutting-plane 마지막 해)과 배치가 다른 문제 (기술용).", "",
             "| 판 | 방법 | 배치가 규칙과 다른 문제 |", "|---|---|---|"] + rows + [""]
    lines += ci_table(recs, ref="modal_rule", title="`modal_rule` (규칙 단독)", only=pol)
    for m in sorted(pol):
        lines += ci_table(recs, ref=m.split(">", 1)[1], title=f"같은 정책 빈 판에서 시작 (`{m.split('>', 1)[1]}`, 판정 아님)",
                          only={m})
    return lines


def count_gap_table():
    """PROGRESS 11/03 round 32 diagnosis: decaps per kind per method under the same repair
    (scripts/diag_count_gap.py), per ladder rung and seed without repair (scripts/diag_kind_mix.py), and how many
    decaps of a certified solution are still removable on the exact check (scripts/diag_prune_headroom.py)."""
    lines = []
    f = "runs/analysis/diag_count_gap.json"
    if os.path.exists(f):
        d = json.load(open(f))
        lines += ["## 진단: 같은 고치기에서 개수 차이는 어느 종류에서 오나 (`scripts/diag_count_gap.py`, 네 방법 모두 "
                  "정확 검사 성공한 문제, seed 0; 종류 0 = C가 가장 큰 것)", "",
                  "| 판 | 방법 | 문제 수 | 평균 개수 | 종류 0 / 1 / 2 평균 개수 | 개수 비 vs r0_coord_rl+rep4 [95% CI] | "
                  "규칙 배치와 Jaccard |", "|---|---|---|---|---|---|---|"]
        for tech, row in d.items():
            for m, v in row.items():
                if not isinstance(v, dict):
                    continue
                ratio = v.get("ratio_vs_r0")
                rs = f"{ratio[0]:.3f} [{ratio[1]:.3f}, {ratio[2]:.3f}]" if ratio else "–"
                jac = f"{v['jaccard_vs_rule']:.3f}" if "jaccard_vs_rule" in v else "–"
                lines.append(f"| {tech} 10 | {m} | {row['n_all_ok']} | {v['mean_count']:.2f} | "
                             f"{' / '.join(f'{x:.2f}' for x in v['mean_per_kind'])} | {rs} | {jac} |")
        lines.append("")
    f = "runs/analysis/diag_kind_mix.json"
    if os.path.exists(f):
        d = json.load(open(f))
        lines += ["## 진단: 사다리 단·seed별 종류 사용 (고치기 없음, 같은 seed의 네 단 모두 성공한 문제; "
                  "`scripts/diag_kind_mix.py`) — 평균 개수 (종류 0 / 1 / 2), 성공률은 50문제 전체", "",
                  "| 판 | seed | 문제 수 | r0_coord | r1_modal | r2_mott | r3_loop | modal_rule | 성공률 r0 / r1 / r2 / r3 / 규칙 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key, row in d.items():
            tech, s = key.rsplit("_s", 1)
            cells = [f"{row[r]['mean_count']:.2f} ({' / '.join(f'{x:.2f}' for x in row[r]['per_kind'])})"
                     for r in ("r0", "r1", "r2", "r3", "rule")]
            sc = " / ".join(f"{row['success'][r]:.2f}" for r in ("r0", "r1", "r2", "r3", "rule"))
            lines.append(f"| {tech} 10 | {s} | {row['n_all_ok']} | " + " | ".join(cells) + f" | {sc} |")
        lines.append("")
    fs = sorted(glob.glob("runs/analysis/prune_headroom_*.json"))
    if fs:
        lines += ["## 진단: 성공한 해에서 정확 검사를 지키며 더 뺄 수 있는 decap (`scripts/diag_prune_headroom.py`, "
                  "하나씩 빼 보고 통과하면 뺌, seed 0 기록)", "",
                  "| 방법 | 판 | 성공 문제 | 평균 개수 | 뺄 수 있는 평균 개수 | 뺄 것이 있는 문제 비율 |", "|---|---|---|---|---|---|"]
        for f in fs:
            m = os.path.basename(f)[len("prune_headroom_"):-5]
            for s, v in json.load(open(f)).items():
                lines.append(f"| {m} | {s} | {v['n_solved']} | {v['mean_count']:.2f} | {v['mean_removable']:.2f} | "
                             f"{v['frac_with_removable']:.2f} |")
        lines.append("")
    f = "runs/analysis/diag_spatial.json"
    if os.path.exists(f):
        d = json.load(open(f))
        lines += ["## 진단: 배치의 공간 군집 (`scripts/diag_spatial.py`, 고치기 없음, decap 2개 이상인 문제; "
                  "adj = 8-이웃에 다른 decap이 있는 decap 비율, nn = 가장 가까운 decap까지 Chebyshev 거리 평균)", "",
                  "| 판 | 방법 | 문제 수 | adj | nn | adj (성공) | adj (실패, 문제 수) |", "|---|---|---|---|---|---|---|"]
        for tech, row in d.items():
            for m, v in row["per_method"].items():
                af = "–" if v["adj_fail"] is None else f"{v['adj_fail']:.3f} ({v['n_fail']})"
                ao = "–" if v["adj_ok"] is None else f"{v['adj_ok']:.3f}"
                lines.append(f"| {tech} 10 | {m} | {v['n']} | {v['adj']:.3f} | {v['nn']:.2f} | {ao} | {af} |")
        lines += ["", "| 판 | seed | r2_mott − r0_coord adj (같은 문제) [95% CI] |", "|---|---|---|"]
        for tech, row in d.items():
            for s, ci in row["r2_minus_r0_adj"].items():
                lines.append(f"| {tech} 10 | {s} | {ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}] |")
        lines.append("")
    return lines


def distill_seed_table():
    """PROPOSALS 11/03 D1: 10-cell exact success of the distill-only checkpoints (best.pt of `<rung>[_sK]_distill`)
    per seed, r0_coord / r1_modal / r2_mott; '–' while a seed's eval is missing."""
    rows, pkg = [], {}
    for tech in ("pcb", "pkg", "ip"):
        for s in (0, 1, 2):
            cells = []
            for r in ("r0_coord", "r1_modal", "r2_mott"):
                n = f"{r}_distill" if s == 0 else f"{r}_s{s}_distill"
                d = f"runs/2026-10-28/{n}" if s == 0 else f"runs/2026-10-29/{n}"
                fs = [f"{d}/eval/eval_{tech}10_{k}__{n}.json" for k in (0, 1)]
                if all(os.path.exists(f) for f in fs):
                    ok = np.concatenate([np.asarray(json.load(open(f))["ok"], bool) for f in fs])
                    cells.append(f"{ok.mean():.2f}")
                    if tech == "pkg":
                        pkg[(r, s)] = ok.mean()
                else:
                    cells.append("–")
            rows.append(f"| {tech} 10 | {s} | " + " | ".join(cells) + " |")
    verdict = []
    if len(pkg) == 9:   # the D1 rule, as written in PROPOSALS before the seed 1/2 results
        held = all(pkg[("r2_mott", s)] < min(pkg[("r0_coord", s)], pkg[("r1_modal", s)]) for s in (0, 1, 2))
        verdict = ["", "D1 규칙 (미리 적음: r2 pkg가 seed 3개 모두 r0·r1 둘 다보다 낮은가): "
                   + ("**성립** → \"MoTT는 pkg에서 선생님을 덜 따라 한다 (증류 단계)\"; 다음 MoTT 후보의 표적은 증류 단계."
                      if held else "불성립 → seed 0 차이는 seed 하나로만 적음.")]
    return ["## 진단 D1: 증류만 한 체크포인트의 seed별 정확 검사 성공률 (PROPOSALS 11/03 D1, 10칸 50문제)", "",
            "| 판 | seed | r0_coord | r1_modal | r2_mott |", "|---|---|---|---|---|"] + rows + verdict + [""]


def _log_rows(path, key):
    """{name: {tech: dict}} from the '<name> <tech> {...}' lines a diag script prints (for runs stopped before json)."""
    import ast
    out = {}
    if os.path.exists(path):
        for line in open(path):
            p = line.split(" ", 2)
            if len(p) == 3 and p[1] in key and p[2].startswith("{"):
                out.setdefault(p[0], {})[p[1]] = ast.literal_eval(p[2])
    return out


def distill_mech_table():
    """PROPOSALS 11/03 D3-D6 (round 35): mechanism of the D1 gap, rules written before each run."""
    L = ["## 진단 D3~D6: D1 (MoTT 증류 pkg 약함)의 기제 (PROPOSALS 11/03 D3~D6, 규칙은 실행 전에 적음)", ""]
    d3 = _log_rows("runs/analysis/diag_imit_heldout.log", ("pkg", "pcb", "ip"))
    if d3:
        L += ["D3 학습 밖 모방 (eval 10칸 판, 선생님 규칙과 같은 해: LS 통과면 LS, 아니면 g+p): nll 학습 안 → 학습 밖 (격차)", "",
              "| 체크포인트 | pkg | pcb | ip |", "|---|---|---|---|"]
        for n, r in d3.items():
            L.append(f"| {n} | " + " | ".join(f"{r[t]['train_nll']:.2f} → {r[t]['nll']:.2f} ({r[t]['gap']:+.2f})" if t in r else "–"
                                         for t in ("pkg", "pcb", "ip")) + " |")
        s0 = [d3.get(n, {}).get("pkg") for n in ("r0_coord", "r1_modal", "r2_mott")]
        if all(s0):
            ok0 = s0[2]["nll"] > max(s0[0]["nll"], s0[1]["nll"]) and s0[2]["gap"] > max(s0[0]["gap"], s0[1]["gap"])
            L += ["", "D3 규칙 (seed 3개 모두 r2의 학습 밖 pkg nll과 격차가 r0·r1보다 큼): "
                  + ("seed 0 성립, seed 1·2 필요" if ok0 else "**seed 0에서 불성립 → 기각** (규제 후보 버림; seed 1·2는 멈춤)")]
        L.append("")
    if os.path.exists("runs/analysis/diag_prune_feasible.json"):
        d4 = json.load(open("runs/analysis/diag_prune_feasible.json"))
    else:
        d4 = _log_rows("runs/analysis/diag_prune_feasible.log", ("pkg", "pcb"))
    if d4:
        L += ["D4 기본 격자 greedy 한 번 (cutting-plane 없음, eval 10칸 50문제): 정확 통과 빼기 허용 / 금지, 개수 허용 / 금지, cost = 금지 − 허용", "",
              "| 판 | seed | r0_coord | r1_modal | r2_mott |", "|---|---|---|---|---|"]
        for tech in ("pkg", "pcb"):
            for s, sfx in enumerate(("", "_s1", "_s2")):
                cells = []
                for m in ("r0_coord", "r1_modal", "r2_mott"):
                    r = d4.get(m + sfx, {}).get(tech)
                    cells.append(f"{r['ok_p']:.2f} / {r['ok_n']:.2f}, {r['cnt_p']:.1f} / {r['cnt_n']:.1f}, cost {r['cost']:+.2f}" if r else "–")
                L.append(f"| {tech} 10 | {s} | " + " | ".join(cells) + " |")
        hp = []
        for sfx in ("", "_s1", "_s2"):
            c = [d4.get(m + sfx, {}).get("pkg") for m in ("r0_coord", "r1_modal", "r2_mott")]
            if all(c):
                hp.append(c[2]["cost"] > 0.04 and c[2]["cost"] > max(c[0]["cost"], c[1]["cost"]))
        if hp:
            L += ["", f"D4 규칙 (seed 3개 모두 r2 pkg cost > 0.04 이고 r0·r1보다 큼): seed별 {hp} → "
                  + ("성립" if len(hp) == 3 and all(hp) else "**기각**" if not all(hp) else "대기")]
        L.append("")
    if os.path.exists("runs/analysis/diag_rm_prior.json"):
        d5 = json.load(open("runs/analysis/diag_rm_prior.json"))
        L += ["D5 r2 증류의 제거 채널 prior를 추론 때만 0 (진단용 수술): drop (D4 → 수술), 정확 통과 차", "",
              "| seed | drop | 정확 통과 (D4 허용 → 수술) |", "|---|---|---|"]
        for s, n in enumerate(("r2_mott", "r2_mott_s1", "r2_mott_s2")):
            r = d5[n]
            L.append(f"| {s} | {r['drop_d4']:.2f} → {r['drop_s']:.2f} | {r['ok_p']:.2f} → {r['ok_s']:.2f} ({r['d_ok']:+.2f}) |")
        L += ["", "D5 규칙 (seed 3개 모두 drop ≤ D4의 절반 이고 정확 통과 ≥ +0.04): "
              + ("성립" if d5["H_rmprior"] else "**기각** (빼기는 prior가 아니라 학습된 잔차에서 옴)"), ""]
    if os.path.exists("runs/analysis/diag_replay_exact.json"):
        d6 = json.load(open("runs/analysis/diag_replay_exact.json"))
        L += ["D6 증류 목표 (기본 격자에서 다시 걸은 선생님 궤적의 끝 배치)의 정확 통과 (학습 풀: pkg 8~12 전부, pcb·ip 10칸)", "",
              "| 판 | 문제 | 재생 끝 정확 통과 | 저장된 선생님 정확 통과 | 개수 재생 / 선생님 |", "|---|---|---|---|---|"]
        for t in ("pkg", "pcb", "ip"):
            r = d6[t]
            L.append(f"| {t} | {r['n']} | {r['rep_ok']:.2f} | {r['tch_ok']:.2f} | {r['rep_cnt']:.2f} / {r['tch_cnt']:.2f} |")
        L += ["", "D6 규칙 (pkg 재생 끝 정확 통과 ≤ 0.80이면 증류 단계 cutting-plane 후보): "
              + ("성립" if d6["C_cp"] else "**기각** (증류 목표는 정확 검사를 통과함)"), ""]
    if os.path.exists("runs/analysis/diag_fail_freq.json"):
        d7 = json.load(open("runs/analysis/diag_fail_freq.json"))
        L += ["D7 격자 만족·정확 실패 해의 최악 주파수 (기본 격자 greedy 한 번, eval 10칸 50판; 덮임 = 모드 토큰 8개 중 가장 "
              "가까운 것이 격자 간격 안; 덜 풀림 = 정확 worst − 가장 가까운 격자점의 격자 값, decade)", "",
              "| 판 | 단 | seed | 격자 만족 | 정확 통과 | 실패 수 | 덮임 | 토큰 거리 중앙값 | 격자 간격 중앙값 | 덜 풀림 중앙값 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t in ("pkg", "pcb"):
            for r in ("r1_modal", "r2_mott"):
                for s, sfx in ((0, ""), (1, "_s1"), (2, "_s2")):
                    x = d7.get(f"{r}{sfx}", {}).get(t, {}).get("summary")
                    if not x:
                        continue
                    if x["n_fail"] == 0:
                        L.append(f"| {t} | {r} | {s} | {x['grid_feasible']:.2f} | {x['exact_pass']:.2f} | 0 | – | – | – | – |")
                        continue
                    L.append(f"| {t} | {r} | {s} | {x['grid_feasible']:.2f} | {x['exact_pass']:.2f} | {x['n_fail']} | "
                             f"{x['covered']:.2f} | {x['d_tok_med']:.3f} | {x['s_med']:.3f} | {x['underres_med']:.2f} |")
        L += ["", "D7 규칙 (pkg, seed 3개 모두): H-uncov (r2 안 덮임 ≥ 0.5 이고 r1보다 큼) "
              + ("성립" if d7["H_uncov"] else "**기각**") + "; H-underres (r2 덮임 ≥ 0.5 이고 덜 풀림 중앙값 ≥ 0.02) "
              + ("**성립** → 후보 P1 `--peak_refine` (PROPOSALS 11/03 P1)" if d7["H_underres"] else "**기각**"), ""]
        if os.path.exists("runs/analysis/diag_fail_freq_refine.json"):
            c = json.load(open("runs/analysis/diag_fail_freq_refine.json"))["P1_check"]
            L += [f"P1 큐 전 점검 (r2 seed 0~2 pkg 덮임 실패 {c['n_covered']}개에서 덮는 토큰의 세 점 포물선 보정 초과량): "
                  f"> 0.02 비율 {c['frac_gt_002']:.2f}, 중앙값 {c['median']:.3f} (규칙: ≥ 0.20이면 큐)", ""]
    if os.path.exists("runs/analysis/diag_fail_rank.json"):
        d8 = json.load(open("runs/analysis/diag_fail_rank.json"))
        L += ["D8 / D8b 안 덮인 실패 옆의 격자 극대 (D7과 같은 해, pkg). D8 순위는 작업 격자 그대로 (패딩으로 같은 주파수가 "
              "반복됨), D8b 순위는 중복을 지운 곡선. 토큰 고유 = 최종 상태의 모드 토큰 8칸 중 서로 다른 주파수 수", "",
              "| 단 | seed | 실패 수 | 안 덮임 | 극대 수 (그대로 / 고유) | 토큰 고유 중앙값 | D8 A (순위 > 8) | D8b D_dup (고유 순위 ≤ 8) |",
              "|---|---|---|---|---|---|---|---|"]
        for r in ("r1_modal", "r2_mott"):
            for s, sfx in ((0, ""), (1, "_s1"), (2, "_s2")):
                x = d8.get(f"{r}{sfx}", {}).get("pkg", {}).get("summary")
                if not x or not x.get("n_uncov"):
                    continue
                L.append(f"| {r} | {s} | {x['n_fail']} | {x['n_uncov']} | {x['n_max_med']:.0f} / {x['n_max_u_med']:.0f} | "
                         f"{x['n_tok_distinct_med']:.0f} | {x['A_budget']:.2f} | {x['D_dup']:.2f} |")
        L += ["", "D8 규칙 (r2, seed 3개 모두 A ≥ 0.5): " + ("성립" if d8["H_K"] else "기각")
              + " — 그러나 극대 수가 패딩 중복으로 부풀어 거의 자동으로 성립하는 측정이라 K 8 → 16 후보는 큐에 넣지 않음 "
              "(PROPOSALS 11/04 D8). D8b 규칙 (r2, seed 3개 모두 D_dup ≥ 0.5): "
              + ("**성립** → 후보 P2 `--dedup_modes` (PROPOSALS 11/04 P2)" if d8.get("H_dup") else "**기각**"), ""]
        # P1/P2 "같이 보고" (report only, no rule): D7 + D8b of extra distill ckpts vs the same-seed r2_mott distill
        extra = sorted(glob.glob("runs/analysis/diag_fail_rank_*_distill.json"))
        if extra:
            L += ["P1/P2 같이 보고 (판정 아님): 같은 경로 (eval pkg10 50판, 기본 격자 greedy 한 번)의 증류 ckpt 비교. "
                  "mott_dd ckpt는 DedupEnv와 같은 고유 주파수 토큰으로 덮임을 셈", "",
                  "| ckpt | 비교 (같은 seed) | 정확 통과 | 실패 수 | 덮임 | 안 덮임 수 | 덜 풀림 중앙값 | 토큰 고유 중앙값 |",
                  "|---|---|---|---|---|---|---|---|"]
            for f in extra:
                n = os.path.basename(f)[len("diag_fail_rank_"):-len(".json")]
                ref = "r2_mott" + ("_s1" if "_s1_" in n else "_s2" if "_s2_" in n else "")
                for name, x in ((n, json.load(open(f)).get(n, {}).get("pkg", {}).get("summary")),
                                (ref, d8.get(ref, {}).get("pkg", {}).get("summary"))):
                    if x:
                        L.append(f"| {name} | {ref if name == n else '–'} | {x['exact_pass']:.2f} | {x['n_fail']} | "
                                 f"{x['covered']:.2f} | {x['n_uncov']} | {x['underres_med']:.2f} | "
                                 f"{x['n_tok_distinct_med']:.0f} |")
            L.append("")
    return L


RL_REPLICATES = [("r2s1_rl", "r2_mott_rl", False), ("r2s2_rl", "r2_mott_rl", False),
                 ("lo_pcb_r2s1_rl", "lo_pcb_r2_rl", True), ("lo_pcb_r2s2_rl", "lo_pcb_r2_rl", True)]
RL_ONLY_MODS = [(f"{p}{m}_rl", f"{p}r2{'_mott' if not p else ''}_rl") for m in ("r2c", "r2v", "r2m", "r2g", "r2t", "r2h")
                for p in ("", "lo_pcb_")]


def rl_noise_table(recs):
    """PROPOSALS 11/03 T5/T5b usage, written before the replicate results: d* = max |success-diff point estimate| of the
    unmodified rl-seed replicates (same distilled init) over same-kind pcb/pkg and held-out pcb; (N1) every rl-only
    modification's board estimate with |.| <= d* is labelled 'within rl noise' (its verdict stays as ruled); (N2) any
    replicate CI excluding 0 -> a seed-0 'win' needs an rl seed-1 replicate before it is extended."""
    def est(var, base, tech):
        pr = paired(recs, base).get((tech, 10), {}).get(var)
        if pr is None:
            return None
        om, orf = pr[0], pr[1]
        lo, hi = boot_ci(lambda i: om[i].mean() - orf[i].mean(), len(om))
        return om.mean() - orf.mean(), lo, hi
    reps, n2 = [], False
    for var, base, lo_run in RL_REPLICATES:
        for tech in ("pcb",) if lo_run else ("pcb", "pkg"):
            e = est(var, base, tech)
            if e is not None:
                reps.append(abs(e[0]))
        for tech in ("pcb", "pkg", "ip"):
            e = est(var, base, tech)
            n2 |= e is not None and (e[1] > 0 or e[2] < 0)
    if not reps:
        return []
    dstar = max(reps)
    rows = []
    for var, base in RL_ONLY_MODS:
        cells = []
        for tech in ("pcb", "pkg", "ip"):
            e = est(var, base, tech)
            cells.append("–" if e is None else f"{e[0]:+.2f}" + (" (잡음 안)" if abs(e[0]) <= dstar + 1e-9 else ""))
        if any(c != "–" for c in cells):
            rows.append(f"| `{var}` | `{base}` | " + " | ".join(cells) + " |")
    return [f"## RL 재현 잡음 (PROPOSALS 11/03 T5·T5b, 사용법 결과 전에 적음): d* = {dstar:.2f} (재현 쌍 {len(reps)}개 값), "
            f"N2 {'발동' if n2 else '미발동'}", "",
            "수정 없이 같은 distill에서 RL seed만 바꾼 run (`r2s1`·`r2s2`·`lo_pcb_r2s1`·`lo_pcb_r2s2`) 대비 seed 0 run의 성공률 차 "
            "|점추정| 최댓값이 d*. 아래는 RL만 다시 돈 수정의 판별 점추정 (vs 같은 distill의 seed 0 RL); |·| ≤ d*이면 \"잡음 안\" "
            "(판정은 원래 규칙 그대로). N2 발동: 재현 쌍의 판 CI 중 0을 뺀 것이 있음 → 이후 seed 0 단독 \"이김\"은 RL seed 1 재현 뒤에만 확장.", "",
            "| 수정 run | 기준 | pcb 10 | pkg 10 | ip 10 |", "|---|---|---|---|---|"] + rows + [""]


# (tag, run infix, label, same-kind reference run, held-out reference infix): runs r2{x}*, lo_{t}_r2{x}* vs ref*, lo_{t}_{lref}*
POOLED_MODS = [("P1", "pk", "`--peak_refine`", "r2_mott", "r2"), ("P2", "dd", "`--dedup_modes`", "r2_mott", "r2"),
               ("P3", "pkdd", "`--peak_refine --dedup_modes` (기준 P1 run)", "r2pk", "r2pk")]


def pooled_mod_table(recs):
    """PROPOSALS 11/03 P1 / 11/04 P2 (rules written before training): seed 0+1 pooled paired bootstrap per board
    (problems resampled within each seed, 2000x). Judged boards: same-kind pcb / pkg / ip and held-out pcb of the
    lo_pcb runs. Win: (i) same-kind pcb or pkg or held-out pcb has pooled CI low > 0 and both seed estimates > 0,
    (ii) no judged board has a pooled estimate below -0.05. Harm: any judged board's pooled CI high < 0.
    Distill stage (D1 target): r2X_distill vs r2_mott_distill pkg, same (i) condition."""
    cache = {}

    def arr(m, ref, tech):
        if ref not in cache:
            cache[ref] = paired(recs, ref)
        v = cache[ref].get((tech, 10), {}).get(m)
        return None if v is None else (v[0].astype(float), v[1].astype(float))

    def pool(pairs, B=2000, seed=0):
        d = [a - b for a, b in pairs]
        rng = np.random.default_rng(seed)
        bs = [np.mean(np.concatenate([x[rng.integers(0, len(x), len(x))] for x in d])) for _ in range(B)]
        return [float(x.mean()) for x in d], float(np.concatenate(d).mean()), np.percentile(bs, [2.5, 97.5])

    L = []
    for tag, x, label, rs, rl in POOLED_MODS:
        boards = [("같은 종류 pcb", "pcb", f"r2{x}", rs, True), ("같은 종류 pkg", "pkg", f"r2{x}", rs, True),
                  ("같은 종류 ip", "ip", f"r2{x}", rs, False), ("뺀 pcb", "pcb", f"lo_pcb_r2{x}", f"lo_pcb_{rl}", True)]
        rows, res, missing = [], {}, False
        for name, tech, m, ref, in_i in boards:
            pairs = [arr(f"{m}{s}_rl", f"{ref}{s}_rl", tech) for s in ("", "_s1")]
            if pairs[0] is None:
                missing = True
                continue
            if pairs[1] is None:
                missing = True
                rows.append(f"| {name} | {np.mean(pairs[0][0] - pairs[0][1]):+.2f} | – | – | {len(pairs[0][0])} |")
                continue
            ps, pe, ci = pool(pairs)
            res[name] = (ps, pe, ci, in_i)
            rows.append(f"| {name} | {ps[0]:+.2f} | {ps[1]:+.2f} | {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                        f"{len(pairs[0][0]) + len(pairs[1][0])} |")
        if not rows:
            continue
        if missing:
            verdict = "seed 0·1 네 run이 다 끝나야 판정"
        else:
            win_i = [n for n, (ps, pe, ci, in_i) in res.items() if in_i and ci[0] > 0 and min(ps) > 0]
            low = [n for n, (ps, pe, ci, in_i) in res.items() if pe < -0.05]
            harm = [n for n, (ps, pe, ci, in_i) in res.items() if ci[1] < 0]
            verdict = ("**해로움** (" + ", ".join(harm) + ")" if harm else
                       "**이김** (" + ", ".join(win_i) + ")" if win_i and not low else
                       "효과 없음" + (f" ((i) {', '.join(win_i)} 성립, (ii) {', '.join(low)} < −0.05)" if win_i else ""))
        dl = [arr(f"r2{x}{s}_distill", f"{rs}{s}_distill", "pkg") for s in ("", "_s1")]
        if all(p is not None for p in dl):
            ps, pe, ci = pool(dl)
            dv = (f"증류만 pkg: seed {ps[0]:+.2f} / {ps[1]:+.2f}, 묶음 {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] → "
                  + ("**성립** (D1 차이를 일부 설명)" if ci[0] > 0 and min(ps) > 0 else "D1은 이것으로 설명 안 됨"))
        else:
            dv = "증류만 pkg: seed 0·1 평가가 다 있어야 판정"
        L += [f"## {tag} {label} seed 0·1 묶음 판정 (PROPOSALS {tag} 규칙, 학습 전에 적음; 판마다 seed 안에서 문제를 다시 뽑는 "
              "bootstrap 2000회)", "",
              "| 판 | seed 0 차 | seed 1 차 | 묶음 차 [95% CI] | 짝 수 |", "|---|---|---|---|---|"] + rows + [
              "", f"판정: {verdict}. {dv}", ""]
        # extension (PROPOSALS P1 "확장 계획", written before lo_pcb_r2pk_s1): seed 0+1+2 pooled + held-out ip/pkg seed 0
        if missing or not verdict.startswith("**이김**"):
            continue
        rows3, res3, rows_lo = [], {}, []
        for name, tech, m, ref, in_i in boards:
            pairs = [arr(f"{m}{s}_rl", f"{ref}{s}_rl", tech) for s in ("", "_s1", "_s2")]
            if pairs[2] is None:
                rows3.append(f"| {name} | – | – | {len(pairs[0][0]) + len(pairs[1][0])} |")
                continue
            ps, pe, ci = pool(pairs)
            res3[name] = (ps, pe, ci)
            rows3.append(f"| {name} | {ps[2]:+.2f} | {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                         f"{sum(len(p[0]) for p in pairs)} |")
        for t in ("ip", "pkg"):
            p = arr(f"lo_{t}_r2{x}_rl", f"lo_{t}_{rl}_rl", t)
            if p is None:
                rows_lo.append(f"| 뺀 {t} | – | – | – |")
                continue
            _, pe, ci = pool([p])
            rows_lo.append(f"| 뺀 {t} | {pe:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {len(p[0])} | "
                           + ("−0.16 위" if pe > -0.16 else "**−0.16 이하**") + " |")
        judged = [n for n in win_i]
        if not all(n in res3 for n in judged) or any(r.startswith("| 뺀") and "| – |" in r for r in rows_lo):
            v3 = "seed 2·뺀 ip·pkg run이 다 끝나야 판정"
        else:
            harm3 = [n for n, (ps, pe, ci) in res3.items() if ci[1] < 0]
            ok3 = [n for n in judged if res3[n][2][0] > 0 and res3[n][0][2] >= 0]
            lo_ok = all("−0.16 위" in r for r in rows_lo)
            v3 = ("**해로움** (" + ", ".join(harm3) + ")" if harm3 else
                  ("**이김 (seed 3개 확인)** (" + ", ".join(ok3) + ")" if ok3 else "seed 2에서 확인 안 됨")
                  + ("; 뺀 ip·pkg 모두 −0.16 위 → 최종 구조 후보" if ok3 and lo_ok else
                     "; 뺀 ip·pkg 손해 가능 → 범용화 주장은 r2_mott" if ok3 else ""))
        L += [f"### {tag} 확장: seed 0·1·2 묶음 + 뺀 ip·pkg (PROPOSALS {tag} 확장 계획, 결과 전에 적음)", "",
              "| 판 | seed 2 차 | seed 0·1·2 묶음 차 [95% CI] | 짝 수 |", "|---|---|---|---|"] + rows3 + [
              "", "| 뺀 종류 (seed 0) | 차 [95% CI] | 짝 수 | d* 기준 |", "|---|---|---|---|"] + rows_lo + [
              "", f"판정: {v3}.", ""]
    return L


SCALE_METHODS = ["greedy+prune", TA, "modal_rule", "r0_coord_rl", "r1_modal_rl", "r2_mott_rl", "r2pk_rl", "r2pk_s1_rl"]


def scale_curve_table(recs):
    """QUEUE 11/04 18:20 KST (human): per kind, rows = methods, columns = 10 / 16 / 20 cells (exact success, count ratio
    vs modal_rule on problems both solve paired within the same pool file, sec per problem, problems). 20 cells has
    8-16 problems: success and ratio get a 95% bootstrap CI and the "작은 표본" mark. A `*.json.memskip` marker
    (scripts/mem_guard.py, 45 GiB rule) shows as "메모리로 생략". Item 4 (record only): r2_mott_rl - r1_modal_rl at 20."""
    by = merged(recs)
    pr = paired(recs, "modal_rule")
    skips = defaultdict(list)
    for f in glob.glob("runs/**/eval/*.json.memskip", recursive=True):
        s, m = os.path.basename(f)[:-len(".json.memskip")].split("__", 1)
        t, n = re.match(r"eval_([a-z]+)(\d+)_\d+", s).groups()
        skips[(t, int(n), m)].append(s)
    if not any(k[1] == 20 for k in by):
        return []
    L = ["## 규모 곡선: 판 크기 10 → 16 → 20 (QUEUE 11/04 18:20 KST; 새 학습 없음, 평가만)", "",
         "칸마다 성공률 / 개수비 vs `modal_rule` (둘 다 성공, 같은 풀 파일끼리 짝) / 문제당 초 / 문제 수. 20칸은 문제 수가 적어 "
         "성공률·개수비에 95% bootstrap CI를 붙이고 \"작은 표본\"으로 표시. `gp_type_aware`는 종류별 g+p 기록이 있는 크기만.", ""]
    for tech in ("pcb", "pkg", "ip"):
        L += [f"### {tech}", "", "| 방법 | 10칸 | 16칸 | 20칸 (작은 표본) |", "|---|---|---|---|"]
        for m in SCALE_METHODS:
            cells = []
            for n in (10, 16, 20):
                r = by.get((tech, n), {}).get(m)
                if r is None:
                    sk = skips.get((tech, n, m))
                    cells.append(f"메모리로 생략 ({', '.join(sorted(sk))})" if sk else "–")
                    continue
                ok = r["ok"].astype(float)
                if m == "modal_rule":
                    rt = "1 (기준)"
                else:
                    p = pr.get((tech, n), {}).get(m)
                    if p is None:
                        rt = "–"
                    else:
                        om, orf, cm, cr = p
                        def ratio(i, om=om, orf=orf, cm=cm, cr=cr):
                            b = om[i] & orf[i]
                            return (cm[i][b] / np.maximum(cr[i][b], 1)).mean() if b.any() else np.nan
                        rt = fmt(ratio(np.arange(len(om))), 3)
                        if n == 20:
                            lo, hi = boot_ci(ratio, len(om))
                            rt += f" [{fmt(lo, 2)}, {fmt(hi, 2)}]"
                sr = f"{ok.mean():.2f}"
                if n == 20:
                    lo, hi = boot_ci(lambda i, ok=ok: ok[i].mean(), len(ok))
                    sr += f" [{lo:.2f}, {hi:.2f}]"
                sk = skips.get((tech, n, m))
                extra = f"; 메모리로 생략 {', '.join(sorted(sk))}" if sk else ""
                cells.append(f"{sr} / {rt} / {fmt(r['sec'], 2)} s / {len(ok)}{extra}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    p20 = paired(recs, "r1_modal_rl")
    rows = [f"| {t} 20 | " + ci_cells(*p20[(t, 20)]["r2_mott_rl"]) + " |" for t in ("pcb", "pkg", "ip")
            if "r2_mott_rl" in p20.get((t, 20), {})]
    if rows:
        L += ["#### 20칸 모드 토큰 효과 (기록만, 판정 없음): `r2_mott_rl` − `r1_modal_rl` (seed 0, 짝지은 bootstrap)", "",
              "| 판 | 문제 수 | 성공률 차 [95% CI] | 개수비 r2/r1 (둘 다 성공) [95% CI] |", "|---|---|---|---|"] + rows + [""]
    return L


def loop_r_size_table(recs):
    """QUEUE 11/04 18:15 KST (human, record only): r3_loop_rl evaluated with R = 2/4/8/16 loop iterations (R8 = the
    plain r3_loop_rl record) at 10 / 16 / 20 cells: success, mean count on solved, sec per problem."""
    by = merged(recs)
    rows = []
    for n in (10, 16, 20):
        for tech in ("pcb", "pkg", "ip"):
            ms = by.get((tech, n), {})
            if not any(f"r3_loop_rl@R{r}" in ms for r in (2, 4, 16)):
                continue
            cells = []
            for r in (2, 4, 8, 16):
                x = ms.get("r3_loop_rl" if r == 8 else f"r3_loop_rl@R{r}")
                cells.append("–" if x is None else
                             f"{x['ok'].mean():.2f} / {fmt(x['cnt'][x['ok']].mean() if x['ok'].any() else None, 1)} / "
                             f"{fmt(x['sec'], 2)} s ({len(x['ok'])})")
            rows.append(f"| {tech} {n} | " + " | ".join(cells) + " |")
    if not rows:
        return []
    return ["## 루프 반복 수 vs 판 크기 (QUEUE 11/04 18:15 KST, 기록만; `r3_loop_rl` seed 0, R8 = 기본 평가)", "",
            "미리 적은 읽기 (결과 전, 사람): 16·20칸에서 R이 클수록 성공률이 오르거나 개수가 줄면 \"큰 판에서 반복이 쓰인다\"; "
            "평평하면 \"반복의 이득은 관찰되지 않았다\".", "",
            "| 판 | R 2 | R 4 | R 8 | R 16 |", "|---|---|---|---|---|",
            "| (칸 내용) | 성공률 / 개수 (성공 평균) / 문제당 초 (문제 수) | | | |"] + rows + [""]


def repair_ladder_table():
    """PROPOSALS 11/03 R4 (scripts/diag_repair_ladder.py): r2_mott+rep4 vs r0_coord+rep4 per seed and pooled."""
    f = "runs/analysis/diag_repair_ladder.json"
    if not os.path.exists(f):
        return []
    d = json.load(open(f))
    rows = []
    for tech, v in d.items():
        for s, p in v["per_seed"].items():
            rows.append(f"| {tech} 10 | {s} | {p['r2']:.2f} | {p['r0']:.2f} | {p['diff']:+.2f} [{p['ci'][0]:+.2f}, "
                        f"{p['ci'][1]:+.2f}] | {p['count_ratio']:.3f} ({p['n_both']}) | |")
        p = v["pooled"]
        rows.append(f"| {tech} 10 | 묶음 (150 짝) | – | – | {p['diff']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] | "
                    f"{p['count_ratio']:.3f} [{p['count_ratio_ci'][0]:.3f}, {p['count_ratio_ci'][1]:.3f}] | → {v['verdict']} |")
    return ["## 같은 고치기에서 2단 vs 0단, seed 3개 (PROPOSALS 11/03 R4; seed 0·1과 r2 seed 2는 규칙 전에 봤음, "
            "r0 seed 2만 결과 전)", "",
            "| 판 | seed | r2_mott+rep4 | r0_coord+rep4 | 성공률 차 [95% CI] | 개수비 r2/r0 (둘 다 성공, 문제 수) | 판정 |",
            "|---|---|---|---|---|---|---|"] + rows + [""]


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
    lines += zs_chip_table(recs)
    lines += learn_vs_rule_table(recs)
    lines += design_rationale(recs)
    lines += sampling_table(recs)
    lines += repair_table(recs)
    lines += rulestart_table(recs)
    lines += count_gap_table()
    lines += distill_seed_table()
    lines += distill_mech_table()
    lines += rl_noise_table(recs)
    lines += pooled_mod_table(recs)
    lines += scale_curve_table(recs)
    lines += loop_r_size_table(recs)
    lines += repair_ladder_table()
    lines += ckpt_table(recs)
    lines += grid_diag_table(recs)
    lines += train_table()
    lines += teacher_table()
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {a.out}: {len(recs)} eval records")


if __name__ == "__main__":
    main()
