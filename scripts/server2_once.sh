#!/usr/bin/env bash
# One launch, no agent required. Same GPU model as server 1; see PROPOSALS L3/L4.
# Usage: bash scripts/server2_once.sh /tmp/r2e_s2.tgz
set -euo pipefail
cd "$(dirname "$0")/.."
ARCHIVE="${1:-/tmp/r2e_s2.tgz}"
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'
mkdir -p agent/logs/s2_once
if pgrep -f 'scripts/server2_loop.sh (cpu|gpu)' >/dev/null; then
  echo "Legacy server2_loop.sh is active. Do not launch both runners."
  exit 1
fi
exec 9>agent/logs/s2_once/bootstrap.lock
flock -n 9 || { echo "Another setup is running"; exit 1; }
if [ -f agent/logs/s2_once/runner.pid ] && kill -0 "$(cat agent/logs/s2_once/runner.pid)" 2>/dev/null; then
  echo "Runner already active; use: python3 scripts/server2_once.py status"
  exit 1
fi
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt > agent/logs/s2_once/install.log 2>&1 || {
  tail -30 agent/logs/s2_once/install.log; exit 1;
}
.venv/bin/python scripts/server2_once.py unpack --archive "$ARCHIVE"
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
.venv/bin/python scripts/server2_once.py plan
nohup .venv/bin/python -u scripts/server2_once.py run --cpu-par "${CPU_PAR:-2}" --push \
  </dev/null >> agent/logs/s2_once/launcher.log 2>&1 9>&- &
pid=$!
echo "$pid" > agent/logs/s2_once/runner.pid
echo "Started PID $pid. You may disconnect."
echo "Progress: python3 scripts/server2_once.py status"
echo "Log: tail -f agent/logs/s2_once/launcher.log"
echo "Output: server2_loop_results.tgz (created when the run finishes)"
