#!/usr/bin/env bash
set -euo pipefail
case "$(hostname)" in *-maskedkd-*) ;; *) echo 'maskedkd 서버에서 실행해 주세요.'; exit 1;; esac
cd "$(dirname "$0")/.."
root=$(pwd -P)
for old in "$HOME/R2E-loop" "$HOME/R2E-delta"; do
  [[ "$root" != "$old" && "$root" != "$old/"* ]] || { echo '별도 R2E-report clone이 필요합니다.'; exit 1; }
done
py=''
for candidate in "$HOME/R2E-delta/.venv/bin/python" "$HOME/R2E-loop/.venv/bin/python"; do
  if [[ -x "$candidate" ]]; then py="$candidate"; break; fi
done
[[ -n "$py" ]] || { echo '기존 서버2 Python 환경을 찾지 못했습니다.'; exit 1; }
mkdir -p runs/report_validation_v1 agent/logs inbox
exec 9>runs/report_validation_v1/bootstrap.lock
flock -n 9 || { echo '이미 시작 중입니다.'; exit 0; }
if [[ -f runs/report_validation_v1/launcher.pid ]]; then
  read -r previous < runs/report_validation_v1/launcher.pid
  if [[ "$previous" =~ ^[0-9]+$ ]] && [[ -r "/proc/$previous/cmdline" ]] &&
     tr '\0' ' ' < "/proc/$previous/cmdline" | grep -q 'report_validation.py run'; then
    echo "이미 실행 중: $previous"; "$py" scripts/report_validation.py status; exit 0
  fi
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CERTIFY_WORKERS=1
"$py" -m unittest discover -s tests -p test_report_validation.py -q
nohup "$py" -u scripts/report_validation.py run </dev/null >>agent/logs/report_validation.log 2>&1 9>&- &
launcher=$!; printf '%s\n' "$launcher" > runs/report_validation_v1/launcher.pid
sleep 2
if ! kill -0 "$launcher" 2>/dev/null; then tail -n 20 agent/logs/report_validation.log; exit 1; fi
printf '보고서 검증 PID: %s. CPU 1작업, 대기 포함 24시간 제한.\n' "$launcher"
printf '모델: report-models-pkg-guard-v1 브랜치에서 2분 간격으로 자동 수신합니다.\n'
printf '진행: %s scripts/report_validation.py status\n' "$py"
printf '결과: %s/runs/report_validation_v1/REPORT.md\n' "$root"
