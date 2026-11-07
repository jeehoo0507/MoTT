#!/usr/bin/env bash
# Refresh management-only files and request one round now, without touching research jobs.
set -euo pipefail
case "$(hostname)" in *-com-loop-*) ;; *) echo "Server 1 only"; exit 1;; esac
cd "$HOME/R2E"
git fetch origin main
now_ref=$(git rev-parse origin/main)
[ -d "$HOME/R2E-codex/agent" ] || { echo "Run Codex switch installer first"; exit 1; }
for name in scripts/codex_pkg_round.py agent/CODEX_PKG_TASK.md; do
  git show "$now_ref:$name" > "$name.refresh"
  mv "$name.refresh" "$name"
done
cp agent/CODEX_PKG_TASK.md "$HOME/R2E-codex/agent/CODEX_PKG_TASK.md"
.venv/bin/python - <<'PY'
import os
from pathlib import Path
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args=[os.fsdecode(x) for x in p.read_bytes().split(b'\0') if x]
        if len(args)>1 and Path(args[1]).name=='codex_pkg_round.py':
            print('An existing Codex round is active, PID',p.parent.name,'; no duplicate started')
            raise SystemExit(3)
    except OSError:pass
PY
if command -v codex >/dev/null; then
  codex_bin=$(command -v codex)
else
  codex_bin="$HOME/.local/share/r2e-codex/node_modules/.bin/codex"
fi
# Python Popen uses an argument list; no shell interpolation of the CLI path.
.venv/bin/python - "$codex_bin" <<'PY'
import subprocess,sys
from pathlib import Path
root=Path.cwd()
with (root/'agent/logs/codex_now.log').open('a') as log:
    p=subprocess.Popen([str(root/'.venv/bin/python'),'scripts/codex_pkg_round.py','--codex',sys.argv[1]],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Requested immediate Codex round, PID',p.pid)
print('Original training/evaluation untouched. Inspect agent/logs/codex_now.log and agent/CODEX_STATUS.json.')
PY
