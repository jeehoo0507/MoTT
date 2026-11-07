#!/usr/bin/env bash
# Server 2, human-run (11/07 KST): RL design ablations A1/A2 (PROPOSALS 11/07 A1·A2). Same budget as r2pk seed 0.
#   bash scripts/server2_ablation.sh check | run | status | push
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; POOL=runs/pool_v1; DAY=2026-11-07s2
mkdir -p agent/logs/s2abl
RUNS=("a1_nogrow|--model mott --peak_refine --grid_growth 0" "a2_noprior|--model mott --peak_refine --no_prior")
case "${1:-}" in
check)
  ls $POOL/train_pcb10_0.pkl $POOL/val_pkg10_0.pkl $POOL/eval_pkg10_0.pkl data/teacher/train_pcb10_0.npz >/dev/null && \
  $PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" && \
  $PY train.py --help | grep -q no_prior && echo "check ok" ;;
run)
  for e in "${RUNS[@]}"; do
    name="${e%%|*}"; arm="${e#*|}"; rl="runs/$DAY/${name}_rl"
    if [ ! -f "$rl/metrics.json" ]; then
      echo "$(date '+%m-%d %H:%M') train $name" | tee -a agent/logs/s2abl/run.log
      CERTIFY_WORKERS=2 $PY scripts/run_rung.py --name "$name" --arm "$arm" --day $DAY --pool $POOL --seed 0 \
        --distill_hours 0.75 --rl_hours 1.25 --skip_eval >> agent/logs/s2abl/train_${name}.log 2>&1 \
        || { echo "train $name FAILED" | tee -a agent/logs/s2abl/run.log; continue; }
    fi
    echo "$(date '+%m-%d %H:%M') eval10 $name" | tee -a agent/logs/s2abl/run.log
    $PY scripts/eval_policy.py --pool $POOL --sets 'eval_*10_*' --policy "${name}_rl=$rl/best.pt" --workers 1 \
      --inst_chunk 5 --out "$rl" >> agent/logs/s2abl/eval_${name}.log 2>&1 || echo "eval $name FAILED" | tee -a agent/logs/s2abl/run.log
  done
  echo "all done $(date '+%m-%d %H:%M')" | tee -a agent/logs/s2abl/run.log ;;
status)
  for e in "${RUNS[@]}"; do n="${e%%|*}"; printf '%-12s distill:%s rl:%s eval:%s\n' "$n" \
    "$([ -f runs/$DAY/${n}_distill/metrics.json ] && echo done || echo -)" \
    "$([ -f runs/$DAY/${n}_rl/metrics.json ] && echo done || echo -)" "$(ls runs/$DAY/${n}_rl/eval/*.json 2>/dev/null | wc -l)/6"; done
  tail -3 agent/logs/s2abl/run.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader ;;
push)
  git add -f runs/$DAY/*_rl/eval/*.json runs/$DAY/*/metrics.json 2>/dev/null
  git commit -m "server 2 (human): A1/A2 ablation records $(date '+%m-%d %H:%M')" || { echo "nothing new"; exit 0; }
  git pull --rebase --autostash r2e main && git push r2e main ;;
*) sed -n 2,3p "$0" ;;
esac
