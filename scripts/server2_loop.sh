#!/usr/bin/env bash
# Server 2, run by the human (11/05 KST): loop-transformer experiments for the report.
# Pre-registered before any run: PROPOSALS.md "11/05 L1·L2 (서버 2)". The server-1 agent does not run or judge these.
#
#   bash scripts/server2_loop.sh setup   # venv + check copied files
#   bash scripts/server2_loop.sh cpu     # L1: r2/r3 seed 1·2 on 16·20 cells (eval only, CPU)
#   bash scripts/server2_loop.sh gpu     # L2: r3pk (loop + P1) seed 0·1·2, then lo_pcb_r3pk; each evaluated on 10 cells
#   bash scripts/server2_loop.sh push    # commit eval records / metrics and push
#   bash scripts/server2_loop.sh status
# env: NGPU (default: all visible GPUs), CPU_PAR (parallel CPU evals, default 3), CALIB=1 (also train r2pk seed 0 here
#      first, required when server 2's GPU model differs from server 1's: the training budget is wall-clock hours)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
POOL=runs/pool_v1
DAY=2026-11-05s2
CPU_PAR="${CPU_PAR:-3}"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
mkdir -p agent/logs/s2

CKPTS="r2_mott_s1_rl r2_mott_s2_rl r3_loop_s1_rl r3_loop_s2_rl"   # existing server-1 checkpoints (copied)
SETS_BIG=""
for n in 16 20; do for t in pcb pkg ip; do for k in 0 1; do SETS_BIG+="eval_${t}${n}_${k} "; done; done; done

# name|seed|arm   (order = run order; spread round-robin over GPUs)
TRAIN=()
[ "${CALIB:-0}" = 1 ] && TRAIN+=("r2pk_calib|0|--model mott --peak_refine")
TRAIN+=("r3pk|0|--model loop --peak_refine"
        "r3pk_s1|1|--model loop --peak_refine"
        "r3pk_s2|2|--model loop --peak_refine"
        "lo_pcb_r3pk|0|--model loop --peak_refine --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0")

check() {
  local miss=0
  for f in "$POOL"/train_pcb10_0.pkl "$POOL"/val_pkg10_0.pkl; do [ -f "$f" ] || { echo "MISSING $f"; miss=1; }; done
  for s in $SETS_BIG; do [ -f "$POOL/$s.pkl" ] || { echo "MISSING $POOL/$s.pkl"; miss=1; }; done
  for t in pcb pkg ip; do for k in 0 1; do
    [ -f "$POOL/eval_${t}10_${k}.pkl" ] || { echo "MISSING $POOL/eval_${t}10_${k}.pkl"; miss=1; }; done; done
  for c in $CKPTS; do [ -f "runs/2026-10-29/$c/best.pt" ] || { echo "MISSING runs/2026-10-29/$c/best.pt"; miss=1; }; done
  [ -d data/teacher ] || { echo "MISSING data/teacher (git)"; miss=1; }
  return $miss
}

case "${1:-}" in
setup)
  [ -x "$PY" ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
  check && echo "setup ok: GPUs=$NGPU, CPU cores=$(nproc)"
  ;;
cpu)
  check
  : > agent/logs/s2/cpu_jobs.txt
  for c in $CKPTS; do for s in $SETS_BIG; do
    [ -f "runs/2026-10-29/$c/eval/${s}__${c}.json" ] && continue
    echo "$PY scripts/eval_policy.py --pool $POOL --sets $s --policy $c=runs/2026-10-29/$c/best.pt --workers 1 --inst_chunk 5 --out runs/2026-10-29/$c >> agent/logs/s2/cpu_${c}.log 2>&1" >> agent/logs/s2/cpu_jobs.txt
  done; done
  echo "$(wc -l < agent/logs/s2/cpu_jobs.txt) eval jobs, $CPU_PAR in parallel"
  xargs -P "$CPU_PAR" -I{} bash -c '{}' < agent/logs/s2/cpu_jobs.txt
  echo "cpu done $(date)"
  ;;
gpu)
  check
  [ "$NGPU" -ge 1 ] || { echo "no GPU"; exit 1; }
  for g in $(seq 0 $((NGPU - 1))); do
    (
      i=0
      for e in "${TRAIN[@]}"; do
        if [ $((i % NGPU)) -eq "$g" ]; then
          IFS='|' read -r name seed arm <<< "$e"
          rl="runs/$DAY/${name}_rl"
          if [ ! -f "$rl/metrics.json" ]; then
            echo "$(date '+%m-%d %H:%M') GPU$g train $name" | tee -a agent/logs/s2/gpu.log
            CUDA_VISIBLE_DEVICES=$g CERTIFY_WORKERS=2 $PY scripts/run_rung.py --name "$name" --arm "$arm" --day $DAY \
              --pool $POOL --seed "$seed" --distill_hours 0.75 --rl_hours 1.25 --skip_eval \
              >> "agent/logs/s2/train_${name}.log" 2>&1 || { echo "train $name FAILED" | tee -a agent/logs/s2/gpu.log; i=$((i+1)); continue; }
          fi
          echo "$(date '+%m-%d %H:%M') GPU$g eval10 $name" | tee -a agent/logs/s2/gpu.log
          CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_policy.py --pool $POOL --sets 'eval_*10_*' \
            --policy "${name}_rl=$rl/best.pt" --workers 1 --inst_chunk 5 --out "$rl" >> "agent/logs/s2/eval_${name}.log" 2>&1 \
            || echo "eval $name FAILED" | tee -a agent/logs/s2/gpu.log
        fi
        i=$((i+1))
      done
    ) &
  done
  wait
  echo "gpu done $(date)" | tee -a agent/logs/s2/gpu.log
  ;;
push)
  git add -f runs/2026-10-29/r[23]_*_s[12]_rl/eval/eval_*{16,20}_*.json 2>/dev/null || true
  git add -f runs/$DAY/*_rl/eval/*.json runs/$DAY/*/metrics.json 2>/dev/null || true
  git add -f agent/logs/s2/gpu.log 2>/dev/null || true
  git commit -m "server 2 (human): loop L1/L2 eval records $(date '+%m-%d %H:%M')" || { echo "nothing new"; exit 0; }
  git pull --rebase --autostash && git push
  ;;
status)
  echo "CPU evals done: $(ls runs/2026-10-29/r[23]_*_s[12]_rl/eval/eval_*{16,20}_*.json 2>/dev/null | wc -l) / 48"
  for e in "${TRAIN[@]}"; do n="${e%%|*}"
    printf '%-14s distill:%s rl:%s eval10:%s\n' "$n" \
      "$([ -f runs/$DAY/${n}_distill/metrics.json ] && echo done || echo -)" \
      "$([ -f runs/$DAY/${n}_rl/metrics.json ] && echo done || echo -)" \
      "$(ls runs/$DAY/${n}_rl/eval/*.json 2>/dev/null | wc -l) files"; done
  tail -3 agent/logs/s2/gpu.log 2>/dev/null || true
  ;;
*) sed -n 2,13p "$0" ;;
esac
