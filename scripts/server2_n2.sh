#!/usr/bin/env bash
# Server 2, human-run (PROPOSALS "11/09 N2"): B-DevFormer held-out-pkg models, harness on the trained types
# (pcb10, ip10), same flags as F/N. Records pushed as they land. Safe to run next to server2_night.sh (CPU only).
#   bash scripts/server2_n2.sh run | status
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=agent/logs/s2n2; mkdir -p $L
push() {
  (flock 9
   git pull -q --rebase --autostash r2e main 2>/dev/null
   git add -f "$@" 2>/dev/null
   git commit -q -m "server 2 (human): N2 records $(date '+%m-%d %H:%M')" && git pull -q --rebase --autostash r2e main \
     && git push -q r2e main && echo "$(date '+%H:%M') pushed" >> $L/run.log
  ) 9>agent/logs/s2n/git.lock
}
lane() {
  for r in "$@"; do
    d=runs/2026-10-29/$r
    for s in eval_pcb10_0 eval_pcb10_1 eval_ip10_0 eval_ip10_1; do
      [ -f $d/eval/${s}__${r}+rep4.json ] && continue
      echo "$(date '+%m-%d %H:%M') start $r $s" >> $L/run.log
      $PY scripts/eval_policy.py --pool runs/pool_v1 --sets $s --policy $r=$d/best.pt --repair 4 --workers 1 \
          --inst_chunk 5 --out $d >> $L/$r.log 2>&1
      push $d/eval/*.json
    done
  done
}
case "${1:-}" in
run)
  mkdir -p agent/logs/s2n
  lane lo_pkg_bdf_distill > $L/a.out 2>&1 &
  lane lo_pkg_bdf_s1_distill lo_pkg_bdf_s2_distill > $L/b.out 2>&1 &
  wait; echo "$(date '+%m-%d %H:%M') all done" >> $L/run.log ;;
status) tail -5 $L/run.log; ls runs/2026-10-29/lo_pkg_bdf*_distill/eval/eval_{pcb,ip}10_*rep4.json 2>/dev/null | wc -l ;;
*) echo "usage: $0 run | status" ;;
esac
