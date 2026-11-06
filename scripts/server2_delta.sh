#!/usr/bin/env bash
# Separate checkout recommended: clone -> this script; disconnect-safe and resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
case "$(hostname)" in
  *-maskedkd-*) ;;
  *) echo "maskedkd 서버 2 전용입니다. 기존 작업은 변경하지 않았습니다."; exit 1 ;;
esac
mkdir -p agent/logs/s2_delta
exec 9>agent/logs/s2_delta/bootstrap.lock
flock -n 9 || { echo "Setup already running"; exit 1; }
if [ -f agent/logs/s2_delta/runner.pid ] && kill -0 "$(cat agent/logs/s2_delta/runner.pid)" 2>/dev/null; then
  echo "Runner already alive. Check runs/delta_memory_v1/status.json"; exit 0
fi
[ -x .venv/bin/python ] || python3 -m venv .venv
echo "Installing dependencies: agent/logs/s2_delta/install.log"
.venv/bin/python -m pip install -r requirements.txt > agent/logs/s2_delta/install.log 2>&1 || {
  tail -n 25 agent/logs/s2_delta/install.log; exit 1;
}
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
.venv/bin/python -m unittest discover -s tests -p test_delta_memory.py > agent/logs/s2_delta/smoke.log 2>&1 || {
  tail -n 40 agent/logs/s2_delta/smoke.log; exit 1;
}
source_pool="${SOURCE_POOL:-}"
if [ -z "$source_pool" ] && [ -d "$HOME/R2E-loop/runs/pool_s2_fresh" ]; then
  source_pool="$HOME/R2E-loop/runs/pool_s2_fresh"
fi
if [ -z "$source_pool" ] && [ -d runs/pool_s2_fresh ]; then source_pool="$PWD/runs/pool_s2_fresh"; fi
nohup .venv/bin/python -u scripts/server2_delta.py run --source_pool "$source_pool" \
  </dev/null >>agent/logs/s2_delta/launcher.log 2>&1 9>&- &
pid=$!
echo "$pid" > agent/logs/s2_delta/runner.pid
echo "Started launcher PID $pid (may wait for the previous FRESH run)."
echo "Status: .venv/bin/python scripts/server2_delta.py status"
echo "Log: tail -f agent/logs/s2_delta/launcher.log"
echo "Results: runs/delta_memory_v1/summary.csv and runs/delta_memory_v1_results.tgz"
