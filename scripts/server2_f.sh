#!/usr/bin/env bash
# Server 2, human-run (PROPOSALS "11/08 F"): takes part of the server-1 F queue so it ends sooner. Same command and
# flags as the day2_queue F entries (eval_policy --repair 4 --workers 1 --inst_chunk 5), same output paths, so each
# record pushed from here is the `done` file server 1's driver checks and skips. Two lanes in parallel:
#   lane A = F1 pkg20 (r2pk seeds 0-2 x eval_pkg20_0/1, 6 jobs; server 1 reaches these last in F1)
#   lane B = F2 10-cell seeds 1-2 (r2pk_s1/_s2 x pcb/pkg/ip, 6 jobs)
# A job is skipped if its record already exists (pull first). Each finished record is pushed at once (flock).
#   bash scripts/server2_f.sh run | status
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=agent/logs/s2f; mkdir -p $L
job() {  # run set1[,set2]
  local run=$1 sets=$2 out=runs/2026-10-29/$1 s miss=0
  git pull -q --rebase --autostash r2e main 2>/dev/null
  for s in ${sets//,/ }; do [ -f $out/eval/${s}__${run}+rep4.json ] || miss=1; done
  [ $miss = 0 ] && { echo "$(date '+%H:%M') skip $run $sets (exists)" | tee -a $L/run.log; return; }
  echo "$(date '+%H:%M') start $run $sets" | tee -a $L/run.log
  $PY scripts/eval_policy.py --pool runs/pool_v1 --sets "$sets" --policy $run=$out/best.pt --repair 4 --workers 1 \
      --inst_chunk 5 --out $out >> $L/$run.log 2>&1
  echo "$(date '+%H:%M') end $run $sets rc=$?" | tee -a $L/run.log
  (flock 9
   git add -f $out/eval/*+rep4.json 2>/dev/null
   git commit -q -m "server 2 (human): F records $run $sets $(date '+%m-%d %H:%M')" && \
     git pull -q --rebase --autostash r2e main && git push -q r2e main && echo "$(date '+%H:%M') pushed $run $sets" >> $L/run.log
  ) 9>$L/git.lock
}
laneA() { for r in r2pk_rl r2pk_s1_rl r2pk_s2_rl; do for s in eval_pkg20_0 eval_pkg20_1; do job $r $s; done; done; }
laneB() { for r in r2pk_s1_rl r2pk_s2_rl; do for k in pcb pkg ip; do job $r eval_${k}10_0,eval_${k}10_1; done; done; }
case "${1:-}" in
run)
  laneA > $L/laneA.out 2>&1 &
  laneB > $L/laneB.out 2>&1 &
  wait; echo "$(date '+%H:%M') all done" | tee -a $L/run.log ;;
status)
  tail -6 $L/run.log; ls runs/2026-10-29/r2pk*_rl/eval/*+rep4.json 2>/dev/null | wc -l ;;
*) echo "usage: $0 run | status" ;;
esac
