# Prospective certification-gap curriculum (v1)

Hypothesis, not a confirmed improvement: a fixed conservative target can purchase
feasibility with excessive capacitors. Use the measured certification/working-grid
gap to guide early exploration, then remove that reserve so the final policy learns
the actual count-constrained task. This is a general RL training rule tested first
by matched continuation, not a new transformer architecture or full from-scratch
training result. No claim of algorithmic novelty before literature/ablation checks.

## Frozen three-arm comparison

Use the same original MoTT+P1 checkpoint per seed and all 27 original train pools
(PCB/package/interposer, sides 8/10/12), the same three validation pools at side 10.
Existing equal pools already give equal type/size sampling probabilities. Do not
sample or tune from evaluation cases. Continue each arm for 45 minutes, reset AdamW,
B=4, M=8, chunk=1, lr=1e-4, initial lambda=20, imitation=0, certify=2, max_steps=64.
Keep the group-normalized REINFORCE and adaptive failure multiplier. All arms use
the same fixed number of CPU certification workers (1–4), each one thread. Wall time
is matched; record iterations, OOM skips and elapsed hours. Do not combine these
weights/results with the earlier serial or cancelled parallel-repeat campaign.

- control: zero reserve throughout.
- annealed: reserve 0.1 for the first third of the RL time budget, linear 0.1→0
  during the middle third, zero for the final third.
- adaptive: same schedule, but replace 0.1 by a feedback estimate for each physical
  configuration and board size. Initialize at 0.1. On each training certification
  call, take q90(max(0, exact_margin - working_grid_margin)), clip to [0,0.1],
  update EMA = 0.8 previous + 0.2 estimate. This conditional estimate covers certified
  original-grid-feasible samples only; it is not a calibrated universal error bound.

Reserve consistently affects observations, physical action scores and legal actions.
For EVERY arm, terminal feasibility and violation-area reward use the ORIGINAL goal.
Original-grid-feasible samples are eligible for independent certification even if
they miss the reserved goal. Certification failures supply cutting-plane frequencies
under the same existing growth cap. Neither physical targets nor checker are relaxed.
Evaluation uses zero reserve, 700 check points plus peak search, success <=1.01 Zt.
No postprocessing is attached selectively to the candidate.

Validate at iteration 50 and every 50 thereafter using zero reserve and the original
exact checker for all arms. Select lexicographically by total failed validation cases,
then worst-group failed cases, then normalized count on successes (bounded <1 term).
All three arms share this new selection rule; comparisons to earlier training must
acknowledge that change. Save per-case validation records and curriculum/resume state.
The old pretraining/distillation is unchanged; this is an exploratory continuation
test of a reusable RL rule, not proof that full training from scratch generalizes.

## Prospective expansion, not success cherry-picking

Train all three arms for seed 0. Expand to seeds 1 and 2 for ALL three arms iff adaptive
vs matched control on the selected validation records has no success loss in ANY
type, at least one jointly solved case per type, joint capacitor-count ratio <=1.05
in each type, and some success gain or >=5% count reduction in a type. This is an
engineering screening rule, not a significance test. Always publish seed-0 failures
and the gate decision. Report that extra seeds were conditional on seed-0 screening;
do not claim an unconditional three-seed estimate. Never use server-2 outcomes to
decide this gate. Reverse arm order for seed 1 to reduce a fixed time-order effect.

## Scoped transition and handoff

Server 1 cancels only the named parallel-repeat follow-up. Let the earlier fixed
reserve study finish frozen/baseline/robust seed-0 package10/16 evaluations. Then
terminate its named manager via its existing graceful handler; any seed-1 job that
has already started is cancelled as superseded and its files preserved. Original
pipeline_day2, training/evaluation queues, unrelated jobs and environments stay intact.
Inputs are hashed copies. Existing GPU jobs retain priority; start below 28 GiB,
yield owned jobs above 32 GiB or when another GPU job appears. No overall campaign
deadline. Each training arm still has its fixed 45-minute RL budget plus validation.

Publish sanitized completed weights to report-models-mott-curriculum-v1. Record the
original init SHA and frozen protocol SHA. The original report branch also receives
the completed seed-0 guard weights. Credentials and optimizer states are not exported.

Server 2 migrates its named report manager between cases, preserving every completed
case. Finish its ORIGINAL case plan and all original paper/classical/internal
baselines first; only the deliberately cancelled guard baseline/robust seeds 1/2 are
removed from its required-model list. All three frozen and other baseline seeds remain.
The scheduler handoff briefly pauses only managers to inspect child state; it never
terminates an active evaluation case. Original report outputs remain separate.

Then evaluate old references and new training arms on a SECOND fresh common case plan:
original report generator defaults, problem seed offset +3,000,000, 20 cases per
type/side 10 and 16, 8 per type at side 20. Same checker and action semantics, CPU
one job/one thread. These cases do not train/select models. Sides 16/20 test size
transfer; all types were in training, so this is NOT leave-one-type-out generalization.
Wait for the campaign_final manifest before declaring completion; respect its gated
seed inventory. Never substitute weights under an existing identity. Preserve failures,
paired rescued/harmed cases, jointly solved count, unconditional success and total
runtime. Do not compare different fresh/historical problem sets as paired gains.

Server 2 uses no overall/case timeout, retries recorded errors, and waits for missing
models. Permanent errors, missing credentials or missing weights can still block
completion and must remain visible. It performs validation, not adaptive architecture
search on the test set. All scripts use the existing server Python environments.
