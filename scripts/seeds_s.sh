#!/usr/bin/env bash
# PROPOSALS "11/09 S" (human-run): supplement seeds for the unstable / failed settings, all reported.
# One GPU lane per server (one training at a time: two at once OOM the 24 GB card, 10/28), evals in the background.
#   bash scripts/seeds_s.sh s1 run      # server 1 (~/R2E): lo_pkg_bdf_s3, lo_pkg_bdf_s4, bdf_s4, r2pk_s4, r2pk_s5
#   bash scripts/seeds_s.sh s2 run      # server 2 (~/R2E-s2): bm2_rlonly_s3, r0_coord_s4, r0_coord_s5, bdf_s3
#   bash scripts/seeds_s.sh s1|s2 status
# Server 1: evals run one after another (pod memory: the agent driver is also there); server 2: in parallel.
# Idempotent: re-running skips finished trainings / records.
set -u
cd "$(dirname "$0")/.."
WHO=${1:-}; CMD=${2:-}
PY=.venv/bin/python; DAY=2026-10-29; POOL=runs/pool_v1; L=agent/logs/seeds_s_$WHO; mkdir -p $L
export CERTIFY_WORKERS=2
LOPKG="--pool_glob train_pcb*.pkl,train_ip*.pkl --val_pools val_pcb10_0,val_ip10_0"
DF="--model devformer --df_perm 4 --df_self 5"
SETS10="eval_pcb10_0,eval_pcb10_1,eval_pkg10_0,eval_pkg10_1,eval_ip10_0,eval_ip10_1"
PKG16="eval_pkg16_0,eval_pkg16_1"
N2SETS="eval_pkg10_0,eval_pkg10_1,eval_pcb10_0,eval_pcb10_1,eval_ip10_0,eval_ip10_1"

say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $L/run.log; }
push() {  # json records only; retried because server 1's agent also commits in this checkout
  for k in 1 2 3 4 5; do
    ( flock 9
      git pull -q --rebase --autostash r2e main 2>/dev/null
      git add -f "$@" 2>/dev/null
      git diff --cached --quiet && exit 0
      git commit -q -m "$WHO (human): S records $(date '+%m-%d %H:%M')" && git pull -q --rebase --autostash r2e main \
        && git push -q r2e main ) 9>$L/git.lock && { echo "$(date '+%H:%M') pushed" >> $L/run.log; return; }
    sleep 60
  done
  say "push failed 5x: $*"
}
ev() {  # ev <tag> <dir> <sets> [flags]
  local tag=$1 dir=$2 sets=$3; shift 3
  $PY scripts/eval_policy.py --pool $POOL --sets "$sets" --policy $tag=$dir/best.pt --workers 1 --inst_chunk 5 "$@" \
      --out $dir >> $L/$tag.eval.log 2>&1
  push $dir/eval/*.json
}
train() {  # train <name> <seed> <arm> <std|distill|rlonly>  -> prints result dir
  local name=$1 seed=$2 arm=$3 mode=$4 stage=rl hours="--distill_hours 0.75 --rl_hours 1.25"
  [ "$mode" = distill ] && { stage=distill; hours="--distill_hours 2.0 --skip_rl"; }
  [ "$mode" = rlonly ] && hours="--distill_hours 0 --rl_hours 2.0 --skip_distill"
  local dir=runs/$DAY/${name}_$stage
  if [ -f $dir/metrics.json ]; then say "skip train $name (exists)"; else
    say "start train $name ($mode, seed $seed)"
    $PY scripts/run_rung.py --name $name --arm "$arm" --day $DAY --pool $POOL --seed $seed $hours --skip_eval \
        >> $L/$name.log 2>&1
    say "end train $name rc=$?"
    [ -f $dir/metrics.json ] && push $dir/metrics.json $dir/log.csv
  fi
  echo $dir
}
post() {  # post <tag> <dir> <rep4 sets>: policy-only 10-cell, then harness
  local tag=$1 dir=$2 sets=$3
  [ -f $dir/metrics.json ] || { say "no metrics.json for $tag (training failed) -> no eval, reported as failed"; return; }
  if [ "$WHO" = s1 ]; then   # serial evals (one at a time, lock), GPU lane does not wait
    ( flock 8; ev $tag $dir "$SETS10"; ev $tag $dir "$sets" --repair 4; say "eval done $tag" ) 8>$L/eval.lock &
  else
    ( ev $tag $dir "$SETS10"; ev $tag $dir "$sets" --repair 4; say "eval done $tag" ) &
  fi
}
smoke() {
  git pull -q --rebase --autostash r2e main 2>/dev/null
  $PY scripts/smoke_devformer.py > $L/smoke_devformer.log 2>&1 || { say "smoke_devformer FAILED"; return 1; }
  $PY scripts/smoke_manoharan2.py > $L/smoke_manoharan2.log 2>&1 || { say "smoke_manoharan2 FAILED"; return 1; }
  say "smokes ok"
}
lane_s1() {
  local d
  for s in 3 4; do d=$(train lo_pkg_bdf_s$s $s "$DF $LOPKG" distill | tail -1); post lo_pkg_bdf_s${s}_distill $d "$N2SETS"; done
  d=$(train bdf_s4 4 "$DF" distill | tail -1);                       post bdf_s4_distill $d "$SETS10,$PKG16"
  for s in 4 5; do d=$(train r2pk_s$s $s "--model mott --peak_refine" std | tail -1); post r2pk_s${s}_rl $d "$SETS10,$PKG16"; done
}
lane_s2() {
  local d
  d=$(train bm2_rlonly_s3 3 "--model manoharan2 --imit 0 --reward manoharan" rlonly | tail -1); post bm2_rlonly_s3_rl $d "$SETS10,$PKG16"
  for s in 4 5; do d=$(train r0_coord_s$s $s "--feat coord" std | tail -1); post r0_coord_s${s}_rl $d "$SETS10,$PKG16"; done
  d=$(train bdf_s3 3 "$DF" distill | tail -1);                       post bdf_s3_distill $d "$SETS10,$PKG16"
}
case "$WHO:$CMD" in
s1:run|s2:run)
  smoke || { say "ABORT: smoke failed"; exit 1; }
  lane_$WHO; say "gpu lane done, waiting for evals"; wait; say "all done" ;;
s1:status|s2:status)
  tail -8 $L/run.log
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null ;;
*) echo "usage: $0 s1|s2 run|status" ;;
esac
