#!/usr/bin/env bash
# Run from a NEW clone. Original checkout is input-only; no package installation or queue edits.
set -euo pipefail
case "$(hostname)" in
  *-com-loop-*) ;;
  *) echo '서버 1(loop)이 아니므로 중단합니다.'; exit 1 ;;
esac
cd "$(dirname "$0")/.."
root=$(pwd -P)
source_root="$HOME/R2E"
py="$source_root/.venv/bin/python"
if [[ "$root" == "$source_root" || "$root" == "$source_root/"* ]]; then
  echo '기존 R2E 밖의 별도 clone에서 실행해 주세요.'; exit 1
fi
[[ -x "$py" ]] || { echo "기존 Python 환경 없음: $py"; exit 1; }
mkdir -p agent/logs runs/pkg_guard_v1
exec 9>runs/pkg_guard_v1/bootstrap.lock
flock -n 9 || { echo '이미 시작 명령을 처리 중입니다.'; exit 0; }
if [[ -f runs/pkg_guard_v1/launcher.pid ]]; then
  read -r previous < runs/pkg_guard_v1/launcher.pid
  if [[ "$previous" =~ ^[0-9]+$ ]] && [[ -r "/proc/$previous/cmdline" ]] &&
      tr '\0' ' ' < "/proc/$previous/cmdline" | grep -q 'pkg_guard_study.py run'; then
    echo "이미 실행 중: PID $previous"; "$py" scripts/pkg_guard_study.py status; exit 0
  fi
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CERTIFY_WORKERS=1
"$py" -c 'import torch, numpy, scipy; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
"$py" -m unittest discover -s tests -p test_pkg_guard.py -q
nohup "$py" -u scripts/pkg_guard_study.py run --source "$source_root" \
  </dev/null >>agent/logs/pkg_guard_launcher.log 2>&1 9>&- &
launcher=$!
printf '%s\n' "$launcher" > runs/pkg_guard_v1/launcher.pid
sleep 2
if ! kill -0 "$launcher" 2>/dev/null; then
  tail -n 20 agent/logs/pkg_guard_launcher.log; exit 1
fi
printf '새 비교 실험 실행기 PID: %s\n' "$launcher"
printf '기존 학습·평가·큐는 변경하지 않습니다. GPU 사용 중이면 대기합니다.\n'
printf '학습 6 x 45분 + 평가, 전체 대기 포함 12시간 제한(미완료 시 부분 결과 저장).\n'
printf '상태: %s scripts/pkg_guard_study.py status\n' "$py"
printf '로그: tail -n 20 %s/agent/logs/pkg_guard_launcher.log\n' "$root"
printf '결과: %s/runs/pkg_guard_v1/REPORT.md\n' "$root"
printf '압축: %s/runs/pkg_guard_v1_results.tgz\n' "$root"
