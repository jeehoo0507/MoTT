#!/usr/bin/env bash
# Install only the diagnostic files; preserve the live checkout and running jobs.
set -euo pipefail
case "$(hostname)" in
  *-com-loop-*) ;;
  *) echo "서버 1(loop)이 아니므로 중단" >&2; exit 1 ;;
esac
cd "$HOME/R2E"
git fetch origin main
install_ref=$(git rev-parse origin/main)
install_tmp=$(mktemp -d)
trap 'rm -rf "$install_tmp"' EXIT
for name in diag_peak_mask.py queue_peak_mask_diag.py; do
  git show "$install_ref:scripts/$name" > "$install_tmp/$name"
done
.venv/bin/python -m py_compile "$install_tmp/diag_peak_mask.py" "$install_tmp/queue_peak_mask_diag.py"
.venv/bin/python - <<'PY'
from pathlib import Path
import json
q = json.loads(Path('agent/day2_queue.json').read_text())
assert isinstance(q['cpu'], list) and isinstance(q['train'], list)
required = [f'runs/pool_v1/{p}.pkl' for p in
            ('val_pkg10_0', 'eval_pkg16_0', 'eval_pcb16_0', 'eval_ip16_0')]
required += [f'runs/2026-10-29/r2pk{s}_rl/best.pt' for s in ('', '_s1', '_s2')]
missing = [p for p in required if not Path(p).is_file()]
if missing:
    raise SystemExit('Missing inputs; queue unchanged: ' + ', '.join(missing))
PY
install_backup="agent/backups/peakmask_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$install_backup"
cp agent/day2_queue.json "$install_backup/day2_queue.json"
for name in diag_peak_mask.py queue_peak_mask_diag.py; do
  if [ -f "scripts/$name" ]; then cp "scripts/$name" "$install_backup/$name"; fi
  cp "$install_tmp/$name" "scripts/$name"
done
.venv/bin/python scripts/diag_peak_mask.py --help >/dev/null
.venv/bin/python scripts/queue_peak_mask_diag.py
.venv/bin/python - <<'PY'
import json
from pathlib import Path
q = json.loads(Path('agent/day2_queue.json').read_text())
jobs = [j for j in q['cpu'] if j['name'].startswith('peakmask ')]
assert len(jobs) == 6, f'Unexpected diagnostic job count: {len(jobs)}'
for job in jobs:
    print(job['name'])
PY
echo "Installed from $install_ref; backup: $install_backup"
echo "Queue ready. Existing driver picks it up after active CPU work and pending model evaluations."
echo "No training or evaluation process was started, stopped, or restarted by this installer."
