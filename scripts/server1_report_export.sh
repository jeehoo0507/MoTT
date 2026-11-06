#!/usr/bin/env bash
set -euo pipefail
case "$(hostname)" in *-com-loop-*) ;; *) echo '서버1(loop)에서 실행해 주세요.'; exit 1;; esac
cd "$(dirname "$0")/.."
root=$(pwd -P)
for old in "$HOME/R2E" "$HOME/R2E-pkg-fix" "$HOME/R2E-codex"; do
  [[ "$root" != "$old" && "$root" != "$old/"* ]] || { echo '별도 R2E-report-transfer clone에서 실행해 주세요.'; exit 1; }
done
py="$HOME/R2E/.venv/bin/python"
mkdir -p agent/logs
# The Python exporter holds its own lock, so repeated launches cannot publish concurrently.
nohup "$py" -u scripts/report_model_bundle.py --wait --publish \
  </dev/null >>agent/logs/report_export.log 2>&1 &
exporter=$!
sleep 2
if ! kill -0 "$exporter" 2>/dev/null; then tail -n 15 agent/logs/report_export.log; exit 1; fi
printf '모델 전달기 PID: %s\n' "$exporter"
printf '공개 GitHub 전용 브랜치: report-models-pkg-guard-v1\n'
printf '완료된 가중치/설정만 전송. 기존 학습·평가·큐는 변경하지 않습니다.\n'
printf '로그: tail -n 20 %s/agent/logs/report_export.log\n' "$root"
