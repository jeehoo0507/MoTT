#!/usr/bin/env bash
# Server 2, human-run (PROPOSALS "11/08 N", moved here 11/09 13:50 KST): the last 10 server-1 "N harness" evals
# (bm2_rlonly_s1_rl all-types x 8, lo_pkg_r0_s3_rl pkg10 x 2). Runs the exact `cmd` of each entry from
# agent/day2_queue.json (cpu_deferred, moved == server2_nrest) - nothing retyped. Slowest (pkg16) first.
#   checkpoints: git fetch -q r2e n2-ckpts && git archive FETCH_HEAD runs | tar x
#   bash scripts/server2_nrest.sh check | run [lanes, default 4] | status
set -u
cd "$(dirname "$0")/.."
L=agent/logs/s2nrest; mkdir -p $L agent/logs/s2n
list() {  # prints: <done path> \t <cmd>, pkg16 first, then pcb, then the rest
  python3 - <<'PY'
import json
q = json.load(open("agent/day2_queue.json"))
es = [e for e in q["cpu_deferred"] if "server2_nrest" in e.get("moved", "")]
key = lambda e: (0 if "pkg16" in e["name"] else 1 if "pcb10" in e["name"] else 2)
for e in sorted(es, key=key):
    print(e["done"][0] + "\t" + e["cmd"])
PY
}
one() {  # one "<done>\t<cmd>"
  local done=${1%%$'\t'*} cmd=${1#*$'\t'}
  [ -f "$done" ] && return
  echo "$(date '+%m-%d %H:%M') start $(basename $done)" >> $L/run.log
  bash -c "$cmd" >> $L/$(basename $done .json).log 2>&1
  echo "$(date '+%m-%d %H:%M') end rc=$? $(basename $done)" >> $L/run.log
  [ -f "$done" ] || return
  (flock 9
   git pull -q --rebase --autostash r2e main 2>/dev/null
   git add -f "$done" && git commit -q -m "server 2 (human): N records $(date '+%m-%d %H:%M')" \
     && git pull -q --rebase --autostash r2e main && git push -q r2e main && echo "$(date '+%H:%M') pushed" >> $L/run.log
  ) 9>agent/logs/s2n/git.lock
}
case "${1:-}" in
check) list | while IFS=$'\t' read -r d c; do ck=$(echo "$c" | grep -o '=runs/[^ ]*best.pt' | cut -c2-)
         [ -f "$ck" ] && echo "ok   $ck" || echo "MISS $ck"; done | sort -u
       echo "jobs: $(list | wc -l) (expect 10)" ;;
run)  N=${2:-4}; export -f one; export L
      list | tr '\n' '\0' | xargs -0 -P $N -I{} bash -c 'one "$1"' _ {}
      echo "$(date '+%m-%d %H:%M') all done" >> $L/run.log ;;
status) tail -6 $L/run.log; n=0; while IFS=$'\t' read -r d c; do [ -f "$d" ] && n=$((n+1)); done < <(list); echo "done $n / 10" ;;
*) echo "usage: $0 check | run [lanes] | status" ;;
esac
