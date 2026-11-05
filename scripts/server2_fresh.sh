#!/usr/bin/env bash
# Clone -> local pool generation -> paired training -> evaluation -> result archive.
# No server-1 connection, transferred inputs, Git identity, or agent is required.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'
mkdir -p agent/logs/s2_fresh
exec 9>agent/logs/s2_fresh/bootstrap.lock
flock -n 9 || { echo "Another setup is running"; exit 1; }
if [ -f agent/logs/s2_fresh/runner.pid ] && kill -0 "$(cat agent/logs/s2_fresh/runner.pid)" 2>/dev/null; then
  echo "Already running. Use: python3 scripts/server2_once.py status --fresh"
  exit 1
fi
if pgrep -f 'scripts/server2_loop.sh (cpu|gpu)|scripts/server2_once.py run' >/dev/null; then
  echo "Another server-2 experiment runner is active. Do not launch both."
  exit 1
fi
[ -x .venv/bin/python ] || python3 -m venv .venv
echo "Installing dependencies (log: agent/logs/s2_fresh/install.log)..."
.venv/bin/python -m pip install -r requirements.txt > agent/logs/s2_fresh/install.log 2>&1 || {
  tail -30 agent/logs/s2_fresh/install.log; exit 1;
}
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
.venv/bin/python scripts/server2_once.py plan --fresh
nohup .venv/bin/python -u scripts/server2_once.py run --fresh --cpu-par "${CPU_PAR:-1}" --push \
  </dev/null >> agent/logs/s2_fresh/launcher.log 2>&1 9>&- &
pid=$!
echo "$pid" > agent/logs/s2_fresh/runner.pid
echo "Started PID $pid. You may disconnect. Data generation runs first."
echo "Progress: python3 scripts/server2_once.py status --fresh"
echo "Log: tail -f agent/logs/s2_fresh/launcher.log"
echo "Output: server2_fresh_results.tgz (no GitHub login required to produce it)"
