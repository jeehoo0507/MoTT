#!/usr/bin/env bash
# PROPOSALS 11/09 U (human-run, server 2): speed on 50 fresh 16-cell problems per kind, same conditions as 11/09 T.
#   bash scripts/speed_fresh.sh go      # waits for the 11/09 T PCB rerun, builds the problems, measures, pushes
#   bash scripts/speed_fresh.sh status | table
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; POOL=runs/pool_speed; OUT=runs/speed_1109u; L=agent/logs/speed_1109u; mkdir -p $L $OUT
CORES=${CORES:-10-15}
MOTT=runs/2026-10-29/r2pk_rl/best.pt
COORD=runs/2026-10-28/r0_coord_rl/best.pt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CERTIFY_WORKERS=1
jobs() {   # slowest first: greedy+prune pkg, coord pkg, then the rest
  for t in pkg pcb ip; do for k in 0 1 2 3 4 5 6 7 8 9; do echo "eval_${t}16_7$k --methods gp"; done; done
  for t in pkg pcb ip; do for k in 0 1 2 3 4 5 6 7 8 9; do echo "eval_${t}16_7$k --policy r0_coord_rl=$COORD --inst_chunk 5 --repair 4"; done; done
  for t in pkg pcb ip; do for k in 0 1 2 3 4 5 6 7 8 9; do echo "eval_${t}16_7$k --policy r2pk_rl=$MOTT --inst_chunk 5 --repair 4"; done; done
  for t in pkg pcb ip; do for k in 0 1 2 3 4 5 6 7 8 9; do echo "eval_${t}16_7$k --methods modal_rule --repair 4"; done; done
}
one() {
  local s=$1; shift
  local tag=$(echo "$s $*" | tr ' =/' '___' | cut -c1-120)
  echo "$(date '+%m-%d %H:%M') start $s $*" >> $L/run.log
  local try rc
  for try in 1 2; do   # PROPOSALS 11/09 U: a job over 4 h is a technical hang (0 % CPU seen in 11/09 T) -> same job once more
    timeout 4h taskset -c $CORES $PY scripts/eval_policy.py --pool $POOL --sets $s "$@" --workers 1 --out $OUT >> $L/$tag.log 2>&1
    rc=$?
    echo "$(date '+%m-%d %H:%M') end rc=$rc try=$try $s $*" >> $L/run.log
    [ $rc -ne 124 ] && break
  done
}
case "${1:-}" in
go)
  echo "$(date '+%m-%d %H:%M') build start ($(hostname), cores $CORES)" >> $L/run.log
  N=$(( $(echo $CORES | cut -d- -f2) - $(echo $CORES | cut -d- -f1) + 1 ))
  # one process per problem file (no Pool), 2 h limit, one retry: the Pool build hung at 0 % CPU on server 2 (11/10)
  for t in pkg pcb ip; do for k in 0 1 2 3 4 5 6 7 8 9; do echo eval_${t}16_7$k; done; done | xargs -P $N -I{} bash -c \
    "for try in 1 2; do [ -f $POOL/{}.pkl ] && break; OMP_NUM_THREADS=1 timeout 2h taskset -c $CORES $PY scripts/speed_fresh.py --out $POOL --only {} >> $L/build.log 2>&1; done"
  nb=$(ls $POOL/eval_*16_7?.pkl 2>/dev/null | wc -l)
  echo "$(date '+%m-%d %H:%M') build done: $nb / 30 files" >> $L/run.log
  [ "$nb" -eq 30 ] || { echo "build incomplete, stop" >> $L/run.log; exit 1; }
  export -f one; export PY POOL OUT L CORES
  jobs | xargs -P $N -L 1 bash -c 'one "$@"' _
  echo "$(date '+%m-%d %H:%M') all done" >> $L/run.log
  git pull -q --rebase --autostash r2e main; git add -f $OUT/eval/*.json $POOL/*.json
  git commit -q -m "$(hostname | cut -c1-40) (human): speed fresh 50 problems 11/09 U" && git pull -q --rebase --autostash r2e main \
    && git push -q r2e main && echo "$(date '+%H:%M') pushed" >> $L/run.log ;;
status) tail -4 $L/run.log 2>/dev/null; tail -2 $L/build.log 2>/dev/null; echo "records: $(ls $OUT/eval/*.json 2>/dev/null | wc -l) / 120" ;;
table) $PY - <<'PY'
import glob, json, os, collections
acc = collections.defaultdict(lambda: [0, 0, 0.0, 0.0, 0])
for f in glob.glob("runs/speed_1109u/eval/*.json"):
    r = json.load(open(f)); s, m = os.path.basename(f)[:-5].split("__", 1)
    a = acc[(s.rsplit("_", 1)[0], m)]
    a[0] += r["n"]; a[1] += sum(r["ok"]); a[2] += r["sec_per_inst"] * r["n"]; a[3] += r["sec_total"]; a[4] += 1
print("| kind | method | files | n | success | decision s/problem | total s/problem |")
print("|---|---|---|---|---|---|---|")
for (k, m), (n, ok, dec, tot, nf) in sorted(acc.items()):
    print(f"| {k} | {m} | {nf}/10 | {n} | {ok/n:.2f} | {dec/n:.1f} | {tot/n:.1f} |")
PY
;;
*) echo "usage: $0 go | status | table" ;;
esac
