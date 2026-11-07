#!/usr/bin/env bash
# Server 2, human-run (PROPOSALS "11/07 M1"): MoTT+P1+G1+M1 (`--mm_bias`) screening, seed 0, full budget
# (distill 0.75 h + RL 1.25 h, COMMON as run_rung), then 10-cell eval. Order: held-out pkg first, then all kinds.
#   bash scripts/server2_m1.sh check | run | status | push
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; POOL=runs/pool_v1; DAY=2026-11-07s2
mkdir -p agent/logs/s2m1
LO="--pool_glob train_pcb*.pkl,train_ip*.pkl --val_pools val_pcb10_0,val_ip10_0"
RUNS=("lo_pkg_r2pkfcm1|--model mott --peak_refine --fc_bias --mm_bias $LO" "r2pkfcm1|--model mott --peak_refine --fc_bias --mm_bias")
case "${1:-}" in
check)
  ls $POOL/train_pcb10_0.pkl $POOL/eval_pkg10_0.pkl data/teacher >/dev/null && $PY train.py --help | grep -q mm_bias && \
  $PY scripts/smoke_mm_bias.py | tail -1 && $PY -c "import torch; print('cuda', torch.cuda.is_available())" && echo "check ok" ;;
run)
  for e in "${RUNS[@]}"; do
    name="${e%%|*}"; arm="${e#*|}"; rl="runs/$DAY/${name}_rl"
    if [ ! -f "$rl/metrics.json" ]; then
      echo "$(date '+%m-%d %H:%M') train $name" | tee -a agent/logs/s2m1/run.log
      CERTIFY_WORKERS=2 $PY scripts/run_rung.py --name "$name" --arm "$arm" --day $DAY --pool $POOL --seed 0 \
        --skip_eval >> "agent/logs/s2m1/train_${name}.log" 2>&1 \
        || { echo "train $name FAILED" | tee -a agent/logs/s2m1/run.log; continue; }
    fi
    echo "$(date '+%m-%d %H:%M') eval10 $name" | tee -a agent/logs/s2m1/run.log
    $PY scripts/eval_policy.py --pool $POOL --sets 'eval_*10_*' --policy "${name}_rl=$rl/best.pt" --workers 1 \
      --inst_chunk 5 --out "$rl" >> "agent/logs/s2m1/eval_${name}.log" 2>&1 || echo "eval $name FAILED" | tee -a agent/logs/s2m1/run.log
    bash "$0" push >> agent/logs/s2m1/run.log 2>&1 || true
  done
  echo "all done $(date '+%m-%d %H:%M')" | tee -a agent/logs/s2m1/run.log ;;
status)
  for e in "${RUNS[@]}"; do n="${e%%|*}"; printf '%-18s distill:%s rl:%s eval:%s\n' "$n" \
    "$([ -f runs/$DAY/${n}_distill/metrics.json ] && echo done || echo -)" \
    "$([ -f runs/$DAY/${n}_rl/metrics.json ] && echo done || echo -)" "$(ls runs/$DAY/${n}_rl/eval/*.json 2>/dev/null | wc -l)/6"; done
  tail -2 agent/logs/s2m1/run.log 2>/dev/null; tail -1 runs/$DAY/*r2pkfcm1_*/log.txt 2>/dev/null | cut -c1-120
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader ;;
push)
  git add -f runs/$DAY/*r2pkfcm1_rl/eval/*.json runs/$DAY/*r2pkfcm1_*/metrics.json 2>/dev/null
  git commit -q -m "server 2 (human): M1 records $(date '+%m-%d %H:%M')" || { echo "nothing new"; exit 0; }
  git pull -q --rebase --autostash r2e main && git push -q r2e main && echo pushed ;;
*) sed -n 2,4p "$0" ;;
esac
