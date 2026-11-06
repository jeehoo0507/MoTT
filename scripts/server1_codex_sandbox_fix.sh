#!/usr/bin/env bash
# Capability-preserving diagnosis/repair. Never change global security settings or research jobs.
set -euo pipefail
case "$(hostname)" in *-com-loop-*) ;; *) echo "Server 1 only"; exit 1;; esac
cd "$HOME/R2E"
git fetch origin main
fix_ref=$(git rev-parse origin/main)
for name in scripts/codex_sandbox_probe.py scripts/codex_pkg_round.py scripts/server1_codex_now.sh; do
  git show "$fix_ref:$name" > "$name.fix"
  mv "$name.fix" "$name"
done
if command -v codex >/dev/null; then codex_bin=$(command -v codex)
else codex_bin="$HOME/.local/share/r2e-codex/node_modules/.bin/codex"; fi
.venv/bin/python scripts/codex_sandbox_probe.py --codex "$codex_bin"
# Only a positively verified existing sandbox policy reaches this line.
bash scripts/server1_codex_now.sh
