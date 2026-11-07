#!/usr/bin/env bash
# Server 2, human-run (PROPOSALS "11/07 R"): randomized-physics training kind. Builds the rnd train pool (CPU), runs a
# short RL smoke on it, then trains lo_pkg_rnd seed 0 (MoTT+P1; distill on pcb+ip teacher pools, RL on pcb+ip+rnd;
# pkg never seen) with the usual budget, 10-cell eval, push.   bash scripts/server2_r.sh check | run | status | push
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; POOL=runs/pool_v1; DAY=2026-11-07s2
mkdir -p agent/logs/s2r runs/analysis
LO="--pool_glob train_pcb*.pkl,train_ip*.pkl --val_pools val_pcb10_0,val_ip10_0"
RL="--rl_pool_glob train_pcb*.pkl,train_ip*.pkl,train_rnd*.pkl"
ARM="--model mott --peak_refine $LO $RL"
pool_ok() { [ "$(ls $POOL/train_rnd*.pkl 2>/dev/null | wc -l)" -ge 9 ]; }
build_pool() {
  pool_ok && { echo "rnd pool exists"; return 0; }
  echo "$(date '+%m-%d %H:%M') build rnd pool" | tee -a agent/logs/s2r/run.log
  $PY scripts/build_pools.py --out $POOL --workers 4 --only train --techs rnd >> agent/logs/s2r/build.log 2>&1
  pool_ok && touch $POOL/rnd.ok
}
case "${1:-}" in
check)
  ls $POOL/train_pcb10_0.pkl $POOL/eval_pkg10_0.pkl data/teacher >/dev/null && $PY scripts/smoke_rnd_pool.py | tail -1 && \
  $PY -c "import torch; print('cuda', torch.cuda.is_available())" && echo "check ok" ;;
run)
  build_pool || { echo "pool build FAILED" | tee -a agent/logs/s2r/run.log; exit 1; }
  if [ ! -f runs/analysis/smoke_rnd_rl.ok ]; then   # 2-minute RL from random init on one rnd file: the training path works
    $PY scripts/run_rung.py --name smoke_rnd_rl --arm "--model mott --peak_refine --pool_glob train_rnd8_0.pkl --val_pools val_pcb10_0" \
      --day analysis --pool $POOL --skip_distill --rl_hours 0.03 --skip_eval >> agent/logs/s2r/smoke_rl.log 2>&1 \
      && [ -f runs/analysis/smoke_rnd_rl_rl/metrics.json ] && touch runs/analysis/smoke_rnd_rl.ok
    [ -f runs/analysis/smoke_rnd_rl.ok ] || { echo "RL smoke FAILED (see agent/logs/s2r/smoke_rl.log)" | tee -a agent/logs/s2r/run.log; exit 1; }
  fi
  name=lo_pkg_rnd; rl="runs/$DAY/${name}_rl"
  if [ ! -f "$rl/metrics.json" ]; then
    echo "$(date '+%m-%d %H:%M') train $name" | tee -a agent/logs/s2r/run.log
    CERTIFY_WORKERS=2 $PY scripts/run_rung.py --name "$name" --arm "$ARM" --day $DAY --pool $POOL --seed 0 --skip_eval \
      >> "agent/logs/s2r/train_${name}.log" 2>&1 || { echo "train $name FAILED" | tee -a agent/logs/s2r/run.log; exit 1; }
  fi
  echo "$(date '+%m-%d %H:%M') eval10 $name" | tee -a agent/logs/s2r/run.log
  $PY scripts/eval_policy.py --pool $POOL --sets 'eval_*10_*' --policy "${name}_rl=$rl/best.pt" --workers 1 --inst_chunk 5 \
    --out "$rl" >> "agent/logs/s2r/eval_${name}.log" 2>&1 || echo "eval $name FAILED" | tee -a agent/logs/s2r/run.log
  bash "$0" push >> agent/logs/s2r/run.log 2>&1 || true
  echo "all done $(date '+%m-%d %H:%M')" | tee -a agent/logs/s2r/run.log ;;
status)
  printf 'rnd pool: %s/9 files  smoke_rl: %s\n' "$(ls $POOL/train_rnd*.pkl 2>/dev/null | wc -l)" "$([ -f runs/analysis/smoke_rnd_rl.ok ] && echo ok || echo -)"
  printf 'lo_pkg_rnd  distill:%s rl:%s eval:%s\n' "$([ -f runs/$DAY/lo_pkg_rnd_distill/metrics.json ] && echo done || echo -)" \
    "$([ -f runs/$DAY/lo_pkg_rnd_rl/metrics.json ] && echo done || echo -)" "$(ls runs/$DAY/lo_pkg_rnd_rl/eval/*.json 2>/dev/null | wc -l)/6"
  tail -2 agent/logs/s2r/run.log 2>/dev/null; tail -1 runs/$DAY/lo_pkg_rnd_*/log.txt 2>/dev/null | cut -c1-120
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader ;;
push)
  git add -f runs/$DAY/lo_pkg_rnd_rl/eval/*.json runs/$DAY/lo_pkg_rnd_*/metrics.json $POOL/train_rnd*.json 2>/dev/null
  git commit -q -m "server 2 (human): R records $(date '+%m-%d %H:%M')" || { echo "nothing new"; exit 0; }
  git pull -q --rebase --autostash r2e main && git push -q r2e main && echo pushed ;;
*) sed -n 2,4p "$0" ;;
esac
