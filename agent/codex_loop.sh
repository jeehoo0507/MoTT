#!/usr/bin/env bash
# Invoked through agent/loop.sh so the existing resume watcher still recognizes the manager.
set -u
cd "$(dirname "$0")/.."
mkdir -p agent/logs
exec 8>agent/.codex_manager.lock
flock -n 8 || { echo "Codex manager already active"; exit 0; }
if command -v codex >/dev/null; then
  codex_bin=$(command -v codex)
else
  codex_bin="$HOME/.local/share/r2e-codex/node_modules/.bin/codex"
fi
[ -x "$codex_bin" ] || { echo "Codex CLI missing; run switch installer"; exit 1; }
"$codex_bin" login status || exit 1
while [ ! -f agent/STOP ] && [[ "$(TZ=Asia/Seoul date +%F)" < "${END_DATE:-2026-11-11}" ]]; do
  echo "[codex] round start $(date -u +%FT%TZ)" | tee -a agent/logs/codex_loop.log
  .venv/bin/python scripts/codex_pkg_round.py --codex "$codex_bin" 8>&- >>agent/logs/codex_loop.log 2>&1
  echo "[codex] round end rc=$? $(date -u +%FT%TZ)" | tee -a agent/logs/codex_loop.log
  sleep "${ROUND_SLEEP:-900}"
done
