#!/usr/bin/env bash
# PROPOSALS 11/09 T (human-run, server 2): clean-as-possible speed benchmark, 16-cell boards, first 10 problems of
# eval_{ip,pcb,pkg}16_0. One process per (method, set, mode), each pinned to one core of CORES (taskset), one torch
# thread, CERTIFY_WORKERS=1; all jobs run at the same time under the same load (supplement trainings keep running on
# the GPU and other cores). Records -> runs/speed_1109/eval/*.json, pushed when all are done.
#   bash scripts/speed_bench.sh check | run | status | table
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; POOL=runs/pool_v1; OUT=runs/speed_1109; L=agent/logs/speed_1109; mkdir -p $L $OUT
CORES=${CORES:-10-15}
MOTT=runs/2026-10-29/r2pk_rl/best.pt
COORD=runs/2026-10-28/r0_coord_rl/best.pt
SETS="eval_ip16_0_k10 eval_pcb16_0_k10 eval_pkg16_0_k10"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CERTIFY_WORKERS=1
jobs() {  # slowest first: greedy+prune (policy only), then coord, rule, MoTT (policy only and + harness)
  for s in eval_pkg16_0_k10 eval_ip16_0_k10 eval_pcb16_0_k10; do echo "$s --methods gp"; done
  for s in $SETS; do echo "$s --policy r0_coord_rl=$COORD --inst_chunk 5"; echo "$s --policy r0_coord_rl=$COORD --inst_chunk 5 --repair 4"; done
  for s in $SETS; do echo "$s --methods modal_rule"; echo "$s --methods modal_rule --repair 4"; done
  for s in $SETS; do echo "$s --policy r2pk_rl=$MOTT --inst_chunk 5"; echo "$s --policy r2pk_rl=$MOTT --inst_chunk 5 --repair 4"; done
}
one() {
  local s=$1; shift
  local tag=$(echo "$s $*" | tr ' =/' '___' | cut -c1-120)
  echo "$(date '+%m-%d %H:%M') start $s $*" >> $L/run.log
  taskset -c $CORES $PY scripts/eval_policy.py --pool $POOL --sets $s "$@" --workers 1 --out $OUT >> $L/$tag.log 2>&1
  echo "$(date '+%m-%d %H:%M') end rc=$? $s $*" >> $L/run.log
}
case "${1:-}" in
check)
  for f in $MOTT $COORD; do [ -f $f ] && echo "ok   $f" || echo "MISS $f"; done
  nproc; command -v taskset >/dev/null && echo "taskset ok" || echo "MISS taskset" ;;
run)
  $PY scripts/speed_subset.py --pool $POOL --sets eval_ip16_0,eval_pcb16_0,eval_pkg16_0 --k 10 >> $L/run.log 2>&1 || exit 1
  N=$(( $(echo $CORES | cut -d- -f2) - $(echo $CORES | cut -d- -f1) + 1 ))
  export -f one; export PY POOL OUT L CORES
  jobs | xargs -P $N -L 1 bash -c 'one "$@"' _
  echo "$(date '+%m-%d %H:%M') all done" >> $L/run.log
  git pull -q --rebase --autostash r2e main; git add -f $OUT/eval/*.json
  git commit -q -m "server 2 (human): speed benchmark 11/09 T" && git pull -q --rebase --autostash r2e main && git push -q r2e main \
    && echo "$(date '+%H:%M') pushed" >> $L/run.log ;;
pcb)   # PROPOSALS 11/09 T addendum: PCB set again, all methods, same conditions, separate records
  OUT=runs/speed_1109b; L=agent/logs/speed_1109b; mkdir -p $OUT $L
  N=$(( $(echo $CORES | cut -d- -f2) - $(echo $CORES | cut -d- -f1) + 1 ))
  export -f one; export PY POOL OUT L CORES
  jobs | grep eval_pcb16_0_k10 | xargs -P $N -L 1 bash -c 'one "$@"' _
  echo "$(date '+%m-%d %H:%M') pcb rerun done" >> $L/run.log
  git pull -q --rebase --autostash r2e main; git add -f $OUT/eval/*.json
  git commit -q -m "server 2 (human): speed PCB rerun 11/09 T" && git pull -q --rebase --autostash r2e main && git push -q r2e main ;;
pcbtable) $PY scripts/speed_table.py runs/speed_1109b/eval ;;
status) tail -6 $L/run.log; echo "records: $(ls $OUT/eval/*.json 2>/dev/null | wc -l) / 21" ;;
table) $PY scripts/speed_table.py $OUT/eval ;;
*) echo "usage: $0 check | run | status | table | pcb | pcbtable" ;;
esac
