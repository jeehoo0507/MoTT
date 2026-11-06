#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd -P)
[[ "$root" == "$HOME/R2E-posttrain" ]] || { echo '별도 R2E-posttrain clone에서 실행해 주세요.'; exit 1; }
case "$(hostname)" in
  *-com-loop-*) server=1; py="$HOME/R2E/.venv/bin/python" ;;
  *-maskedkd-*) server=2; py="$HOME/R2E-delta/.venv/bin/python"
    [[ -x "$py" ]] || py="$HOME/R2E-loop/.venv/bin/python" ;;
  *) echo '서버1(loop) 또는 서버2(maskedkd)가 아닙니다.'; exit 1 ;;
esac
[[ -x "$py" ]] || { echo "Python 환경 없음: $py"; exit 1; }
mkdir -p runs/transition agent/logs
exec 9>runs/transition/bootstrap.lock
flock -n 9 || exit 0
if [[ -f runs/transition/status.json ]]; then
  if "$py" -c 'import json,sys;from pathlib import Path;p=Path("/proc")/str(json.load(open("runs/transition/status.json"))["pid"]);sys.exit(0 if p.exists() and b"switch_post_training.py" in (p/"cmdline").read_bytes() else 1)'; then
    echo '전환 실행기가 이미 동작 중입니다.'; exit 0
  fi
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CERTIFY_WORKERS=1
nohup "$py" -u scripts/switch_post_training.py "$server" </dev/null >>"agent/logs/server${server}_posttrain.log" 2>&1 9>&- &
pid=$!;sleep 2
kill -0 "$pid" 2>/dev/null || { tail -n 25 "agent/logs/server${server}_posttrain.log"; exit 1; }
echo "서버 $server 전환 실행기 PID: $pid"
echo "상태: cat $root/runs/transition/status.json"
echo "로그: tail -n 15 $root/agent/logs/server${server}_posttrain.log"
