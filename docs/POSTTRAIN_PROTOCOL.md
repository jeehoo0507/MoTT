# Three-phase post-training v1 (frozen before results)

Primary deliverable: an exploratory, auditable seed-0 comparison of four equal-budget
arms. More seeds assess reproducibility; one seed does not establish a reliable gain.
This is post-training of the existing MoTT+P1, not new pretraining or a new architecture.

## Actual phases, not just a changing reserve

Each arm runs THREE separate 15-minute RL budgets, with fresh AdamW and failure
multiplier per phase, initialized from the previous phase's best validation checkpoint.
Optimizer reset and checkpoint chaining apply to ALL arms. This makes each phase run
and validate even if a preceding phase consumed substantial validation time. Loading,
validation and certification overhead is recorded; nominal RL time is 45 min/arm.

Use all 27 original training pools (three types, sizes 8/10/12), original teachers,
same source checkpoint per seed, same three size-10 validation pools, group-normalized
REINFORCE, B=4/M=8/chunk=1/lr=1e-4/initial lambda=20/imitation=0/certify=2/max_steps=64.
Every candidate eligible on the ORIGINAL working-grid goal receives certification,
irrespective of reserve. Original goal defines violation and failure reward. Keep
the original physical targets, 700-point plus peak checker and frequency-growth cap.

1. Feasibility: empty-start RL with count coefficient 0.1; original-target violation
   and exact-failure penalty. The intent is to acquire successful placements before
   emphasizing sparsity. It is a hypothesis, not an improvement guarantee.
2. Robustness: empty-start RL with count coefficient 0.3. For each physical type/size,
   q90 positive exact-minus-working margin (conditional on checked candidates) updates
   an EMA (0.8 old + 0.2 clipped estimate in [0,0.1], initial 0.1). This controls the
   working reserve, linearly reduced to zero during phase 2. Observation/action masks
   use the same reserve. Final success/violation reward always uses the original goal.
3. Sparsity: count coefficient 1, reserve zero. Roughly half of sampled INSTANCES
   (iteration+instance parity) start all M episodes from the best certified training
   placement archived for that exact board; the others start empty to retain construction
   behavior. Missing archive entries can bootstrap from an existing teacher ONLY after
   re-certification. Otherwise fall back to empty and record that fallback. Phase-3 reward
   is normalized (initial count - final count) minus original violation/failure penalty.
   All M samples of an instance share the initial state, so the count offset does not
   bias within-group advantages. Refinement uses the existing remove/type-change/stop
   policy and physics legality masks. Update archives only with independently certified,
   smaller placements. No postprocessing or archive is used at test time.

Archive identity hashes original board masks, entries, probes, targets and physical
configuration/size, not the growing working grid. Keep validation/test boards OUT.
Feedback, archive, stats and RNG/optimizer resume state are persisted. Archive success
does not guarantee that the final empty-start policy retains success; measure it.

## Four matched ablations (all 3 x 15 min)

- pt_control: standard original-goal count coefficient 1 throughout; empty starts.
- pt_feasible: stage 1 objective throughout; empty starts.
- pt_robust: stage 1, then stage 2, then continue its coefficient-0.3 objective with
  zero reserve/empty starts for the last 15 minutes (no stage 3).
- pt_full: stages 1, 2 and the actual archive-start/empty-start stage 3.

This isolates stage-3 as a combined training intervention (reward weight and start
distribution). It does not isolate the individual subcomponents of stage 3; that needs
another ablation. Stage checkpoints along one trajectory have unequal elapsed budgets;
use equal-budget final arms for attribution and do not claim all differences are causal.

Validate all arms from EMPTY with zero reserve and original exact success <=1.01 Zt.
Select within each phase by total validation failure count, worst-type failure count,
then bounded normalized successful count. Same rule for all arms. Validate every 25
iterations and at phase completion when necessary. Validation does not modify archive
or feedback. Never optimize from server-2 evaluation outputs.

## Diagnostics and checkpoints

There are iterations/elapsed seconds, not meaningful epochs over a fixed dataset.
Keep model-only checkpoints at phase start, every five minutes, and EVERY validation,
plus phase best/last and rolling resumable optimizer checkpoint. Each validation
checkpoint has a JSON sidecar with fixed-board success, count, exact margin, working
margin, gap, worst frequency and add/remove/replace/stop counts. trace.jsonl also records
training certification gaps, archive successes/improvements, refinement/construction
episode counts and reserve. Preserve original raw logs, grid caps/OOMs and real timings.
Checkpoints live on the server; final sanitized arm weights go to the dedicated Git branch.

learning_curves.csv / REPORT.md summarize the histories without inventing reasons.
Smaller grid/exact gap alongside fewer certified failures supports the robustness
mechanism; unchanged success plus fewer jointly-solved-board parts and archive-start
reductions supports refinement. Neither pattern alone proves causality; use ablations.

Finish and publish ALL seed-0 arms and seed0_complete.json before more seeds. Expand
ALL arms to seeds 1/2 only if full-vs-control AND full-vs-stage12 each have no validation
type success loss, at least one jointly solved board/type, joint count ratio <=1.05
everywhere, and some success gain or >=5% count reduction. Require nonzero actual archive
refinement episodes. Publish failure/no-expansion as well. Conditional expansion is
exploratory selection, not an unconditional three-seed statistical result.

## Server transition / independent validation

Use a NEW R2E-posttrain clone. Cancel only the explicitly superseded reserve-only
curriculum scheduler/training under R2E-curriculum and the old parallel-repeat scheduler.
Preserve their artifacts. Let original R2E-pkg-fix fixed-reserve seed 0 finish all six
package evaluations and both training runs, then stop its manager (seed1+ superseded).
Do not touch the original pipeline_day2 or unrelated jobs. Maintain original GPU priority,
memory start <=28 GiB, stop/retry owned jobs above 32 GiB, fixed CPU worker count per arm.

Server 2 finishes the current original baseline comparison first, migrating managers
between cases. Original guard baseline/robust seeds 1/2 are intentionally removed from
required models; all other baseline seeds remain. Then use a SECOND NEW test plan,
seed offset +6,000,000, distinct from both original and previous curriculum tests:
20 boards/type at sizes 10/16, 8/type at size 20. All models use the same fresh boards.
Evaluate four new arms and all original references, per seed actually trained.
Report paired success and jointly solved count, and total decoding/certification time.
No extra test-time refinement for pt_full. These are seen-type / unseen-size tests,
not leave-one-type-out. Same original checker and per-method action semantics remain.

Git branch: report-models-mott-posttrain-v1. Preserve original-init and phase-chain
hashes; never overwrite an evaluated identity. Server 2 waits for a final inventory,
uses no overall/case timeout, retries errors. Permanent errors/missing weights/credentials
can still prevent completion. No unattended process can guarantee a wall-clock finish.
The 11/07 10:00 target concerns the seed-0 training comparison, not all extra seeds
or the whole independent server-2 baseline table.

## Report-only memory guard correction

Report evaluation discounts inactive_file minus dirty/writeback bytes from cgroup-v2
memory.current, retaining the existing 28 GiB start / 32 GiB stop working-set bounds.
It additionally requires 4 GiB raw headroom to start and 2 GiB while running against
memory.max. Missing statistics fall back to raw usage. This is scheduling only;
case plans, model weights, success criteria and CPU worker counts are unchanged.
The server2 reload helper pauses only this clone's supervisor, waits for any active
report case to finish, then replaces that report runner via its parent's retry loop.
FRESH/data generation and server1 training are not signalled. No cache is flushed.
