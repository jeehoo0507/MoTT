#!/usr/bin/env bash
# Server 2, human-run (PROPOSALS "11/08 N", QUEUE 11/08 11:30 KST). One GPU lane + three CPU lanes; every record is
# committed and pushed as soon as it exists, so server 1 / the laptop see it. Idempotent: every step skips existing
# outputs, so re-running after a crash continues where it stopped.
#   scripts/server2_night.sh smoke    # step 0 only: code smokes (writes runs/analysis/smoke_*.ok) + --bl_seed check
#   scripts/server2_night.sh run      # smoke, then all lanes (detach with nohup/tmux; ~20 h)
#   scripts/server2_night.sh status   # last log lines + record counts
# GPU lane (sequential, 2 h each, order = priority):
#   lo_pkg_bdf s0 s1 s2  (B-DevFormer, pkg held out, distill-only 2.0 h)   -> policy eval_*10_* + pkg10 rep4
#   bdf s0 s1 s2         (B-DevFormer, all types)                          -> eval_*10_* + pcb/pkg/ip10 + pkg16 rep4
#   lo_pkg_r2pk_s3       (MoTT+P1, pkg held out, extra seed; supplement)    -> eval_*10_* + pkg10 rep4
#   r2pk_s3              (MoTT+P1, all types, extra seed; supplement)       -> eval_*10_* + pcb/pkg/ip10 + pkg16 rep4
#   r0_coord_s3          (coord, all types, extra seed; supplement)         -> eval_*10_* + pkg10 + pkg16 rep4
# CPU lanes (start at once): C = coord s1/s2 x pkg16/pkg20 rep4; G = GA seeds 1,2 (10-cell); L = LS seeds 1,2 (10-cell).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; DAY=2026-10-29; POOL=runs/pool_v1; L=agent/logs/s2n; mkdir -p $L
export CERTIFY_WORKERS=2
LOPKG="--pool_glob train_pcb*.pkl,train_ip*.pkl --val_pools val_pcb10_0,val_ip10_0"
DF="--model devformer --df_perm 4 --df_self 5"
SETS10="eval_pcb10_0,eval_pcb10_1,eval_pkg10_0,eval_pkg10_1,eval_ip10_0,eval_ip10_1"
PKG10="eval_pkg10_0,eval_pkg10_1"
PKG16="eval_pkg16_0,eval_pkg16_1"

say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $L/run.log; }
sync() { git pull -q --rebase --autostash r2e main 2>/dev/null; }
push() {  # push <paths...>  (json records only; *.pt stays out of git as before)
  (flock 9
   sync
   git add -f "$@" 2>/dev/null
   git commit -q -m "server 2 (human): N records $(date '+%m-%d %H:%M')" && sync && git push -q r2e main \
     && echo "$(date '+%H:%M') pushed $*" >> $L/run.log
  ) 9>$L/git.lock
}
ev() {  # ev <tag> <dir> <sets> [extra eval_policy flags]  -> <dir>/eval/<set>__<tag>[+rep4].json (skips existing)
  local tag=$1 dir=$2 sets=$3; shift 3
  $PY scripts/eval_policy.py --pool $POOL --sets "$sets" --policy $tag=$dir/best.pt --workers 1 --inst_chunk 5 "$@" \
      --out $dir >> $L/$tag.eval.log 2>&1
  push $dir/eval/*.json
}
train() {  # train <name> <seed> <arm> <std|distill>   (prints the result dir)
  local name=$1 seed=$2 arm=$3 mode=$4 stage=rl hours="--distill_hours 0.75 --rl_hours 1.25"
  [ "$mode" = distill ] && { stage=distill; hours="--distill_hours 2.0 --skip_rl"; }
  local dir=runs/$DAY/${name}_$stage
  if [ -f $dir/metrics.json ]; then say "skip train $name (exists)"; else
    say "start train $name ($mode, seed $seed)"
    $PY scripts/run_rung.py --name $name --arm "$arm" --day $DAY --pool $POOL --seed $seed $hours --skip_eval \
        >> $L/$name.log 2>&1
    say "end train $name rc=$?"
    push $dir/metrics.json $dir/log.csv 2>/dev/null
  fi
  echo $dir
}
post() {  # post <tag> <dir> <rep4 sets>: policy-only 10-cell eval, then the harness (--repair 4) sets; in the background
  local tag=$1 dir=$2 sets=$3
  [ -f $dir/best.pt ] || { say "no best.pt for $tag, eval skipped"; return; }
  ( ev $tag $dir "$SETS10"; ev $tag $dir "$sets" --repair 4; say "eval done $tag" ) &
}

smoke() {
  sync
  $PY scripts/smoke_devformer.py > $L/smoke_devformer.log 2>&1 || { say "smoke_devformer FAILED (see $L/smoke_devformer.log)"; return 1; }
  $PY scripts/smoke_manoharan2.py > $L/smoke_manoharan2.log 2>&1 || { say "smoke_manoharan2 FAILED"; return 1; }
  # --bl_seed (new 11/08): GA 4x2 on one set with seed 1 must write <set>__genetic@s1.json
  rm -rf runs/analysis/smoke_blseed
  $PY scripts/eval_policy.py --pool $POOL --sets eval_ip10_1 --methods ga --ga 4x2 --bl_seed 1 --workers 1 \
      --out runs/analysis/smoke_blseed > $L/smoke_blseed.log 2>&1
  [ -f runs/analysis/smoke_blseed/eval/eval_ip10_1__genetic@s1.json ] || { say "bl_seed smoke FAILED"; return 1; }
  rm -rf runs/analysis/smoke_blseed
  say "smokes ok (devformer, manoharan2, bl_seed)"
}

gpu_lane() {
  local d
  for s in 0 1 2; do sfx=""; [ $s = 0 ] || sfx=_s$s
    d=$(train lo_pkg_bdf$sfx $s "$DF $LOPKG" distill | tail -1); post lo_pkg_bdf${sfx}_distill $d "$PKG10"
  done
  for s in 0 1 2; do sfx=""; [ $s = 0 ] || sfx=_s$s
    d=$(train bdf$sfx $s "$DF" distill | tail -1); post bdf${sfx}_distill $d "$SETS10,$PKG16"
  done
  d=$(train lo_pkg_r2pk_s3 3 "--model mott --peak_refine $LOPKG" std | tail -1); post lo_pkg_r2pk_s3_rl $d "$PKG10"
  d=$(train r2pk_s3 3 "--model mott --peak_refine" std | tail -1);           post r2pk_s3_rl $d "$SETS10,$PKG16"
  d=$(train r0_coord_s3 3 "--feat coord" std | tail -1);                     post r0_coord_s3_rl $d "$PKG10,$PKG16"
  say "gpu lane done"
}
laneC() {  # coord seeds 1,2 with the harness on the big pkg boards (seed 0 exists: runs/repair16, 2026-10-28)
  for r in r0_coord_s1_rl r0_coord_s2_rl; do
    [ -f runs/$DAY/$r/best.pt ] || { say "laneC: no $r/best.pt on this server -> server 1 queue has it (QUEUE 11/08 11:55)"; continue; }
    for s in eval_pkg16_0 eval_pkg16_1 eval_pkg20_0 eval_pkg20_1; do
      [ -f runs/$DAY/$r/eval/${s}__${r}+rep4.json ] && continue
      say "laneC start $r $s"; ev $r runs/$DAY/$r $s --repair 4; say "laneC end $r $s"
    done
  done
  say "laneC done"
}
laneB() {  # laneB <ga|ls>: stochastic search baselines, seeds 1 and 2, 10-cell sets, same config as the seed-0 records
  local m=$1
  for k in 1 2; do
    say "lane$m start seed $k"
    $PY scripts/eval_policy.py --pool $POOL --sets 'eval_*10_*' --methods $m --bl_seed $k --workers 2 \
        --out runs/2026-10-28/baselines >> $L/base10_${m}_s$k.log 2>&1
    push runs/2026-10-28/baselines/eval/*.json
    say "lane$m end seed $k"
  done
}

case "${1:-}" in
smoke) smoke ;;
run)
  smoke || { say "ABORT: smoke failed, nothing started"; exit 1; }
  gpu_lane > $L/gpu.out 2>&1 &
  laneC   > $L/laneC.out 2>&1 &
  laneB ga > $L/laneG.out 2>&1 &
  laneB ls > $L/laneL.out 2>&1 &
  wait; say "all lanes done" ;;
status)
  tail -8 $L/run.log
  echo "rep4 records: $(ls runs/$DAY/{lo_pkg_bdf,bdf,lo_pkg_r2pk_s3,r2pk_s3,r0_coord_s3,r0_coord_s1,r0_coord_s2}*/eval/*+rep4.json 2>/dev/null | wc -l)"
  echo "GA/LS seeded: $(ls runs/2026-10-28/baselines/eval/*@s[12].json 2>/dev/null | wc -l) / 24"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null ;;
*) echo "usage: $0 smoke | run | status" ;;
esac
