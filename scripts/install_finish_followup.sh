#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd -P)
case "$(hostname)" in
  *-com-loop-*) task=server1; py="$HOME/R2E/.venv/bin/python"; out=pkg_guard_parallel_v1 ;;
  *-maskedkd-*) task=server2; py="$HOME/R2E-delta/.venv/bin/python"; out=report_finish_supervisor ;;
  *) echo '서버1(loop) 또는 서버2(maskedkd)에서 실행해 주세요.'; exit 1 ;;
esac
if [[ "$task" == server2 && ! -x "$py" ]]; then py="$HOME/R2E-loop/.venv/bin/python"; fi
[[ "$root" == "$HOME/R2E-followups" ]] || { echo '별도 R2E-followups clone에서 실행해 주세요.'; exit 1; }
[[ -x "$py" ]] || { echo "Python 없음: $py"; exit 1; }
mkdir -p "runs/$out" agent/logs
exec 9>"runs/$out/bootstrap.lock"
flock -n 9 || exit 0
if [[ -f "runs/$out/followup_status.json" ]]; then
  if "$py" -c 'import json,os,sys; from pathlib import Path; d=json.load(open(sys.argv[1])); p=Path("/proc")/str(d["pid"]); sys.exit(0 if p.exists() and b"finish_followup.py" in (p/"cmdline").read_bytes() else 1)' "runs/$out/followup_status.json"; then
    echo '후속 실행기가 이미 동작 중입니다.'; exit 0
  fi
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CERTIFY_WORKERS=1
nohup "$py" -u scripts/finish_followup.py "$task" </dev/null >>"agent/logs/${task}_followup.log" 2>&1 9>&- &
pid=$!
sleep 2
kill -0 "$pid" 2>/dev/null || { tail -n 20 "agent/logs/${task}_followup.log"; exit 1; }
echo "후속 실행기 PID: $pid — 현재 실행 종료 후 자동으로 이어집니다."
echo "로그: tail -n 20 $root/agent/logs/${task}_followup.log"
echo "상태: cat $root/runs/$out/followup_status.json"
