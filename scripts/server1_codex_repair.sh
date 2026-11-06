#!/usr/bin/env bash
# Complete the CLI runtime installation; restart only a verified blocked management invocation.
set -euo pipefail
case "$(hostname)" in *-com-loop-*) ;; *) echo "Server 1 only"; exit 1;; esac
cd "$HOME/R2E"
git fetch origin main
repair_ref=$(git rev-parse origin/main)
for name in scripts/codex_install_helpers.py scripts/codex_pkg_round.py scripts/server1_codex_now.sh; do
  git show "$repair_ref:$name" > "$name.repair"
  mv "$name.repair" "$name"
done
if command -v codex >/dev/null; then codex_bin=$(command -v codex)
else codex_bin="$HOME/.local/share/r2e-codex/node_modules/.bin/codex"; fi
.venv/bin/python scripts/codex_install_helpers.py --codex "$codex_bin"
.venv/bin/python - <<'PY'
import json,os,signal,time
from pathlib import Path
status=Path('agent/CODEX_STATUS.json')
if not status.exists():raise SystemExit(0)
d=json.loads(status.read_text())
if d.get('stage')!='running':raise SystemExit(0)
parent=int(d['pid']); log=Path(d['log'])
if not log.exists() or 'failed to spawn code-mode host' not in log.read_text(errors='replace'):
    print('Current round has no verified missing-host failure; leaving it alone')
    raise SystemExit(0)
children=Path(f'/proc/{parent}/task/{parent}/children')
if not children.exists():raise SystemExit(0)
for pid in map(int,children.read_text().split()):
    try:args=[os.fsdecode(x) for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    except FileNotFoundError:continue
    if (args and Path(args[0]).name.startswith('codex') and 'exec' in args and '--cd' in args
            and args[args.index('--cd')+1]==str(Path.home()/'R2E-codex')):
        os.kill(pid,signal.SIGTERM)
        print('Stopped blocked Codex CLI only, PID',pid,'; no training/evaluation process signalled',flush=True)
for _ in range(20):
    if not Path(f'/proc/{parent}').exists():break
    time.sleep(1)
PY
bash scripts/server1_codex_now.sh
