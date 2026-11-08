#!/usr/bin/env bash
# Server 2, human-run (PROPOSALS "11/09 N2", moved here 11/09 01:40 KST so server 1's single CPU lane only has N left).
# The 48 N2 evals of the server-1 queue (pcb10/ip10 harness for the held-out-pkg MoTT+P1, coord, B-Park,
# B-Manoharan v2 models x 3 seeds), same command and flags as the queue entries. Checkpoints come from server 1:
#   server 1:  cd ~/R2E && tar czf /tmp/n2_ckpts.tgz runs/2026-10-29/lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}{,_s1,_s2}_rl/best.pt
#              scp /tmp/n2_ckpts.tgz USER@SERVER2:/tmp/
#   server 2:  cd ~/R2E-s2 && git pull -q r2e main && tar xzf /tmp/n2_ckpts.tgz && bash scripts/server2_n2b.sh check
#              nohup bash scripts/server2_n2b.sh run > /dev/null 2>&1 &
#   bash scripts/server2_n2b.sh status
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=agent/logs/s2n2b; mkdir -p $L agent/logs/s2n
MODELS="lo_pkg_r2pk_rl lo_pkg_r2pk_s1_rl lo_pkg_r2pk_s2_rl lo_pkg_r0_rl lo_pkg_r0_s1_rl lo_pkg_r0_s2_rl
        lo_pkg_bp_park_rl lo_pkg_bp_park_s1_rl lo_pkg_bp_park_s2_rl lo_pkg_bm2_rlonly_rl lo_pkg_bm2_rlonly_s1_rl lo_pkg_bm2_rlonly_s2_rl"
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
    [ -f $d/best.pt ] || { echo "$(date '+%m-%d %H:%M') MISSING $d/best.pt, skipped" >> $L/run.log; continue; }
    for s in eval_pcb10_0 eval_pcb10_1 eval_ip10_0 eval_ip10_1; do
      [ -f $d/eval/${s}__${r}+rep4.json ] && continue
      git pull -q --rebase --autostash r2e main 2>/dev/null   # server 1 may have pushed it meanwhile
      [ -f $d/eval/${s}__${r}+rep4.json ] && continue
      echo "$(date '+%m-%d %H:%M') start $r $s" >> $L/run.log
      $PY scripts/eval_policy.py --pool runs/pool_v1 --sets $s --policy $r=$d/best.pt --repair 4 --workers 1 \
          --inst_chunk 5 --out $d >> $L/$r.log 2>&1
      push $d/eval/*.json
    done
  done
}
case "${1:-}" in
check) for r in $MODELS; do [ -f runs/2026-10-29/$r/best.pt ] && echo "ok   $r" || echo "MISS $r"; done ;;
run)
  set -- $MODELS
  lane $1 $4 $7 ${10} > $L/a.out 2>&1 &
  lane $2 $5 $8 ${11} > $L/b.out 2>&1 &
  lane $3 $6 $9 ${12} > $L/c.out 2>&1 &
  wait; echo "$(date '+%m-%d %H:%M') all done" >> $L/run.log ;;
status) tail -5 $L/run.log
  echo "N2 records: $(ls runs/2026-10-29/lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*_rl/eval/eval_{pcb,ip}10_*rep4.json 2>/dev/null | wc -l) / 48" ;;
*) echo "usage: $0 check | run | status" ;;
esac
