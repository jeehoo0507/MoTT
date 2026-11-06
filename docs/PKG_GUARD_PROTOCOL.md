# MoTT working-constraint reserve pilot (pkg_guard_v1)

Prospective engineering pilot. No improvement or architectural novelty is assumed.
This protocol is fixed before the new server training/evaluation. Earlier package-type
failures informed the hypothesis, so these existing evaluation pools are exploratory,
not a fresh confirmatory test set.

## Hypothesis and intervention

Some solutions satisfy the sampled working grid but fail the independent dense-grid
and peak-refinement check. Opening the action mask alone rescued none of the eight
previously inspected package instances; it did not establish a primary cause.
Test whether a small conservative working constraint improves robustness instead:

- Baseline working target: Zt(f).
- Candidate working target: Zt(f) / 10**0.1 (approximately 0.794 Zt).
- Use that same target for mode/cell features, one-action physical effects, legal
  actions, working feasibility and the RL violation penalty. This is not the old
  reward-only `--feas_margin` flag.
- Keep PDNBatch.Zt, target parameters, impedance equations and decap library intact.
  Independent certification and checkpoint selection use the ORIGINAL physical target.
- The transformer architecture is unchanged (original MoTT + P1). This tests a
  constraint/training change, not a new architecture or a learned physics solver.

## Matched continued training

Copy the original 27 train pools (all three types, sides 8/10/12), original teacher
files, three validation pools and r2pk seed 0/1/2 best.pt from server 1. Hash all copies.
Do not modify the source checkout, original queue or existing results.

For each seed, initialize both arms from the identical pretrained r2pk checkpoint,
reset AdamW and continue RL for 45 minutes per arm. Training uses group-normalized
REINFORCE as in the repository, not clipped GRPO. All shared settings are fixed:
B=4, M=8, chunk=1, lr=1e-4, initial lambda=20, imitation weight=0, certify=2,
max_steps=64, validation every 50 updates, resume checkpoint every 5 minutes.
Architecture dimensions and optional frequency/mode heads are copied from the source.
Both arms use the same data, seed, optimizer setup and original-target validation rule.
Imitation is disabled for both because original teacher trajectories need not be legal
under the conservative constraint. Teacher counts still normalize the RL objective.

Total nominal RL time: 6 x 45 minutes = 4.5 hours, excluding pool loading, final
validation and evaluation. This matches wall-clock budget, not optimizer updates/FLOPs;
record actual iterations and elapsed time. There is no learning from evaluation pools.
Model selection averages the same original-target validation score over all three types;
package validation does not receive a special selection weight.

## Evaluation and priority

Use the first 8 instances, fixed by index, from eval_{pkg,pcb,ip}{10,16}_1.pkl.
Three training seeds reuse the same boards; do not count them as independent boards.
Each method gets a fresh per-instance bank and working grid. Exact success is worst
|Z|/Zt <= 1.01 on the original 700-point checker plus peak refinement. No checker
parameters, physical targets or component counts are changed for a favorable result.
Up to four cutting-plane restarts use the same original-target refinement rule/cap.
A working-grid failure is still independently checked against the original target.

Publish these primary package results across all seeds BEFORE extra controls:

1. Frozen original MoTT + P1 decoded at reserve 0.
2. Continued baseline decoded at reserve 0.
3. Continued candidate decoded at reserve 0.1.

Next complete the package 3 x 2 factorial controls: frozen, baseline, robust-trained,
each decoded with reserve 0 and 0.1. This separates reserve-only decoding from learning:
robust@0.1 vs baseline@0.1, robust@0 vs baseline@0, frozen@0.1 vs frozen@0.
Then evaluate the three primary methods on PCB/IP sides 10 and 16 as regression checks.
These secondary/control stages may remain incomplete at the deadline.

Report paired rescued/harmed boards per seed, success rate, count on successful cases,
count on JOINTLY successful cases, final exact margin, steps/restarts and seconds.
Time includes decoding, physical effects and certification, but excludes initial
bank construction; include bank construction separately. CPU concurrency may affect
absolute speed, so this pilot cannot establish a speedup over older server timings.
Report all completed seeds/cases, including regressions. No significance/novelty claim
from this small pilot. Consider a candidate promising only if package improvements
repeat across seeds and are not explained solely by more decaps or reserve-only decoding;
confirm it later on fresh boards and complete PCB/IP checks before a general claim.

## Isolation, resources and deadline

Run ordinary Python in a separate clone; no Codex/Claude login or sandbox change needed.
Use server 1's existing Python environment without installing/changing its packages.
Input files are copied; all new outputs live in runs/pkg_guard_v1 in the new clone.
Never signal original processes, edit original queues or replace existing experiment code.

Only launch GPU continuation when no enabled original GPU queue entry is pending and
nvidia-smi shows no compute process. One new child job at a time, CPU exact checks use
one thread and no worker pool. Start only below 28 GiB cgroup memory. If total cgroup
memory exceeds 32 GiB, stop ONLY this study's own child process group and save partial
results. The original 45 GiB memory guard is not changed. Stop a new GPU child if another
GPU process appears; original jobs retain priority. This is a best-effort resource
boundary, not a guarantee that other workloads or instantaneous memory spikes cannot
compete. A short priority CPU evaluation can run alongside the original CPU experiment.

12-hour overall deadline from first launch, including waiting; each evaluation job has
a 40-minute limit. Save each evaluated board atomically and refresh reports/archive at
job boundaries and termination. If resources/deadlines prevent completion, clearly
label partial results; do not call the three-seed study complete. Rerunning resumes
existing rows/checkpoints under the same frozen settings and the same original deadline.
No automatic upload from server 1; output tgz can be downloaded without GitHub credentials.

## Commands

New clone: `bash scripts/server1_pkg_guard.sh`

Status: `$HOME/R2E/.venv/bin/python scripts/pkg_guard_study.py status`

Log: `tail -n 20 agent/logs/pkg_guard_launcher.log`

Tables: `runs/pkg_guard_v1/REPORT.md`, `summary.csv`, `paired.json`.

Archive: `runs/pkg_guard_v1_results.tgz` (no model weights).
