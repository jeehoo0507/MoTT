#!/usr/bin/env bash
# Human, 2026-11-07: stop the Codex-era managers and studies on this server. Results on disk are kept.
# Run from the R2E checkout on the server:
#   bash scripts/stop_codex.sh --dry        # only list what would be stopped
#   bash scripts/stop_codex.sh              # stop Codex managers/studies (server 1: keeps the R2E driver, F1/B1)
#   bash scripts/stop_codex.sh --all-r2e    # also stop every R2E process (driver, loop, training, eval) — server 2
set -u
cd "$(dirname "$0")/.."
PATS=(agent/codex_loop.sh codex_pkg_round.py "codex exec" "codex --search"
      server2_once.py server2_delta.py server2_fresh.sh
      pkg_guard_study.py pkg_guard_train.py
      switch_curriculum.py curriculum_campaign.py train_reserve_curriculum.py
      switch_post_training.py posttrain_campaign.py train_post_training.py
      report_validation.py report_model_bundle.py finish_followup.py
      diag_peak_mask.py server1_peak_mask_priority.py)
DRY=0
for a in "$@"; do
  case "$a" in
    --dry) DRY=1 ;;
    --all-r2e) PATS+=("bash agent/loop.sh" pipeline_day2.py run_rung.py train.py eval_policy.py side_slot.py
                      report_watch.py resume_agent.py ensure_running.py build_pools.py run_queue_entries.py) ;;
  esac
done

list() { for p in "${PATS[@]}"; do pgrep -af -- "$p"; done | grep -v "stop_codex.sh" | sort -u; }
killtree() { local c; for c in $(pgrep -P "$1"); do killtree "$c"; done; kill -TERM "$1" 2>/dev/null; }

echo "== matching processes"; list
[ "$DRY" = 1 ] && exit 0
for p in "${PATS[@]}"; do
  for pid in $(pgrep -f -- "$p"); do [ "$pid" != $$ ] && killtree "$pid"; done
done
sleep 5
echo "== still alive after TERM (kill -9 these by hand if they stay)"; list
if [ -f agent/PROVIDER ]; then mv agent/PROVIDER "agent/PROVIDER.stopped_$(date +%m%d%H%M)"; echo "agent/PROVIDER moved aside -> loop.sh uses Claude"; fi
echo "GPU:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
