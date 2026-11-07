#!/usr/bin/env bash
# Server 2 (MaskedKD server), human-run, 11/07 KST: paper comparison numbers only (no new method, no training).
# PROPOSALS "11/07 S2 비교 수치" fixed before any run. The server-1 agent does not run or judge these.
#
#   bash scripts/server2_baselines.sh check      # files copied from server 1 present?
#   bash scripts/server2_baselines.sh run        # all tiers (light lane CPU_PAR parallel + one heavy lane), resumable
#   bash scripts/server2_baselines.sh status
#   bash scripts/server2_baselines.sh push       # commit records (runs/s2base/eval) and push
# env: CPU_PAR (light lane, default 2)
#
# Records go to runs/s2base/eval/<set>__<method>[+rep4].json. Seconds/board here are server-2 CPU: compare them only
# with other runs/s2base records. Combinations already recorded on server 1 (runs/repair10, runs/repair16) are skipped
# so no record name appears twice.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
POOL=runs/pool_v1
OUT=runs/s2base
CPU_PAR="${CPU_PAR:-2}"   # g+p 16-cell peaks ~34 GB rss on its own (11/03); keep the pod under its limit
mkdir -p "$OUT/eval" agent/logs/s2base

SETS=""
for n in 10 16 20; do for t in pcb pkg ip; do for k in 0 1; do SETS+="eval_${t}${n}_${k} "; done; done; done
P29=runs/2026-10-29; P28=runs/2026-10-28
# name|eval_policy args          (policies: greedy decoding, --inst_chunk 5 as every other policy eval)
T1="r2pk_rl|--policy r2pk_rl=$P29/r2pk_rl/best.pt --inst_chunk 5
modal_rule|--methods modal_rule
greedy+prune|--methods gp"
T2="r2pk_s1_rl|--policy r2pk_s1_rl=$P29/r2pk_s1_rl/best.pt --inst_chunk 5
r2pk_s2_rl|--policy r2pk_s2_rl=$P29/r2pk_s2_rl/best.pt --inst_chunk 5
r0_coord_rl|--policy r0_coord_rl=$P28/r0_coord_rl/best.pt --inst_chunk 5
r3_loop_rl|--policy r3_loop_rl=$P28/r3_loop_rl/best.pt --inst_chunk 5"
T3=""
for s in "" _s1 _s2; do
  T3+="bp_park${s}_rl|--policy bp_park${s}_rl=$P29/bp_park${s}_rl/best.pt --inst_chunk 5
bm_manoharan_rlonly_l20${s}_rl|--policy bm_manoharan_rlonly_l20${s}_rl=$P29/bm_manoharan_rlonly_l20${s}_rl/best.pt --inst_chunk 5
"
done

done_anywhere() {   # record already exists on server 1 or here
  local f="$1"
  for d in "$OUT" runs/repair10 runs/repair16; do [ -f "$d/eval/$f" ] && return 0; done
  return 1
}

jobs() {   # tier lines -> "lane<TAB>command" for every missing record; prior work also without repair at 16/20
  local tier="$1" norep="$2"
  while IFS='|' read -r name args; do
    [ -n "$name" ] || continue
    for s in $SETS; do
      local heavy=light
      [ "$name" = "greedy+prune" ] && [[ "$s" != *10_* ]] && heavy=heavy
      done_anywhere "${s}__${name}+rep4.json" || \
        printf '%s\t%s\n' "$heavy" "$PY scripts/eval_policy.py --pool $POOL --sets $s $args --repair 4 --workers 1 --out $OUT >> agent/logs/s2base/${name}.log 2>&1"
      if [ "$norep" = 1 ] && [[ "$s" != *10_* ]] && ! done_anywhere "${s}__${name}.json"; then
        printf '%s\t%s\n' light "$PY scripts/eval_policy.py --pool $POOL --sets $s $args --workers 1 --out $OUT >> agent/logs/s2base/${name}.log 2>&1"
      fi
    done
  done <<< "$tier"
}

check() {
  local miss=0
  for s in $SETS; do [ -f "$POOL/$s.pkl" ] || { echo "MISSING $POOL/$s.pkl"; miss=1; }; done
  for p in $(printf '%s\n%s\n%s\n' "$T1" "$T2" "$T3" | grep -o 'runs/[^ ]*best.pt'); do
    [ -f "$p" ] || { echo "MISSING $p"; miss=1; }; done
  [ "$miss" = 0 ] && echo "check ok"
  return $miss
}

case "${1:-}" in
check) check ;;
run)
  check || exit 1
  { jobs "$T1" 0; jobs "$T2" 0; jobs "$T3" 1; } > agent/logs/s2base/jobs.tsv
  echo "$(grep -c '' agent/logs/s2base/jobs.tsv) jobs (tier order T1 -> T2 -> T3); heavy lane: $(grep -c '^heavy' agent/logs/s2base/jobs.tsv)"
  grep '^heavy' agent/logs/s2base/jobs.tsv | cut -f2- > agent/logs/s2base/heavy.txt
  grep '^light' agent/logs/s2base/jobs.tsv | cut -f2- > agent/logs/s2base/light.txt
  ( while IFS= read -r c; do bash -c "$c"; done < agent/logs/s2base/heavy.txt; echo "heavy lane done $(date)" ) &
  xargs -P "$CPU_PAR" -I{} bash -c '{}' < agent/logs/s2base/light.txt
  wait
  echo "all done $(date)"
  ;;
status)
  echo "records: $(ls $OUT/eval/*.json 2>/dev/null | wc -l) / $(grep -c '' agent/logs/s2base/jobs.tsv 2>/dev/null)"
  ls -t $OUT/eval/*.json 2>/dev/null | head -5
  free -g | head -2
  ;;
push)
  git add -f $OUT/eval/*.json
  git commit -m "server 2 (human): comparison records runs/s2base $(date '+%m-%d %H:%M')" || { echo "nothing new"; exit 0; }
  git pull --rebase --autostash r2e main 2>/dev/null || git pull --rebase --autostash origin main
  git push r2e main 2>/dev/null || git push origin main
  ;;
*) sed -n 2,12p "$0" ;;
esac
