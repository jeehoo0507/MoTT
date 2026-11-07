#!/usr/bin/env bash
# Server 2, human-run (11/07 14:25 KST): final RL method C1 "verified self-imitation" + matched control
# (PROPOSALS "11/07 C1 재개"). Both continue from server-1 r2pk_rl/best.pt (seed 0) for 1.0 h RL, then 10-cell eval.
#   bash scripts/server2_c1.sh check | run | status | push
# Order: control first (r2pk_ctl), then C1 (r2pk_vsi). Records -> runs/2026-11-07s2/.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; POOL=runs/pool_v1; DAY=2026-11-07s2; INIT=runs/2026-10-29/r2pk_rl/best.pt
mkdir -p agent/logs/s2c1
RUNS=("r2pk_ctl|--model mott --peak_refine" "r2pk_vsi|--model mott --peak_refine --self_repair 4")
case "${1:-}" in
check)
  ls "$INIT" $POOL/train_pcb10_0.pkl $POOL/eval_pkg10_0.pkl >/dev/null && $PY train.py --help | grep -q self_repair && \
  $PY -c "import torch; print('cuda', torch.cuda.is_available())" && echo "check ok" ;;
run)
  for e in "${RUNS[@]}"; do
    name="${e%%|*}"; arm="${e#*|}"; rl="runs/$DAY/${name}_rl"
    if [ ! -f "$rl/metrics.json" ]; then
      echo "$(date '+%m-%d %H:%M') train $name" | tee -a agent/logs/s2c1/run.log
      CERTIFY_WORKERS=2 $PY scripts/run_rung.py --name "$name" --arm "$arm" --day $DAY --pool $POOL --seed 0 \
        --init_from "$INIT" --distill_hours 0 --rl_hours 1.0 --skip_eval >> "agent/logs/s2c1/train_${name}.log" 2>&1 \
        || { echo "train $name FAILED" | tee -a agent/logs/s2c1/run.log; continue; }
    fi
    echo "$(date '+%m-%d %H:%M') eval10 $name" | tee -a agent/logs/s2c1/run.log
    $PY scripts/eval_policy.py --pool $POOL --sets 'eval_*10_*' --policy "${name}_rl=$rl/best.pt" --workers 1 \
      --inst_chunk 5 --out "$rl" >> "agent/logs/s2c1/eval_${name}.log" 2>&1 || echo "eval $name FAILED" | tee -a agent/logs/s2c1/run.log
  done
  echo "all done $(date '+%m-%d %H:%M')" | tee -a agent/logs/s2c1/run.log ;;
status)
  for e in "${RUNS[@]}"; do n="${e%%|*}"; printf '%-10s rl:%s eval:%s\n' "$n" \
    "$([ -f runs/$DAY/${n}_rl/metrics.json ] && echo done || echo -)" "$(ls runs/$DAY/${n}_rl/eval/*.json 2>/dev/null | wc -l)/6"; done
  tail -2 agent/logs/s2c1/run.log 2>/dev/null; tail -1 runs/$DAY/r2pk_*_rl/log.txt 2>/dev/null | cut -c1-120
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader ;;
push)
  git add -f runs/$DAY/*_rl/eval/*.json runs/$DAY/*/metrics.json 2>/dev/null
  git commit -m "server 2 (human): C1/control records $(date '+%m-%d %H:%M')" || { echo "nothing new"; exit 0; }
  git pull --rebase --autostash r2e main && git push r2e main ;;
*) sed -n 2,5p "$0" ;;
esac
