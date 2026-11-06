# Matched CPU-parallel follow-up

The original pkg_guard_v1 study completes without interruption. A separate clone and
pkg_guard_parallel_v1 output then repeat the frozen PKG_GUARD_PROTOCOL.md comparison.
Do not merge the two campaigns or overwrite their exported checkpoint identities.

Only training certification execution changes: 2–4 spawn processes, each using one
CPU thread and the identical safe_verify._job physics, 700-point grid and peak search.
The fixed worker count applies to both continuation arms, including validation.
Source checkpoints, data, seeds, optimizer, architecture and 45-minute budgets remain
matched. More iterations within the same budget are expected; a shorter total runtime
is not assumed. Record attempted iterations, OOM skips and certification profiles.
Evaluation remains serial so neither model receives a different timing protocol.

Before training, compare serial and parallel checks on the first eight existing
package training cases with teacher placements. Check numerical and success parity;
continue only with measured steady-state speedup >1.05. This is a runtime gate, not
selection by validation/test performance. Reserve one CPU core, cap workers at four,
and retain existing memory/resource guards. Insufficient cores stop only this follow-up.
The study's 12-hour deadline begins after waiting for the original run and benchmarking.

Server 2 completion mode retains its frozen case plan, models, checker and baselines.
After the original runner releases its lock, resume existing outputs with no overall
or per-case time limit. Archive and retry failed jobs; wait for missing checkpoints.
Existing CPU jobs are never signalled. Memory guards remain and may retry owned jobs.
Unexpected runner failures retry after 60 seconds. Permanent errors or missing models
can prevent completion and remain visible in logs; elapsed time alone never declares
success. completion_policy.json and retry_history.jsonl record the operational change.

## Deployment

Use a separate `$HOME/R2E-followups` clone on each server and run
`bash scripts/install_finish_followup.sh`. Hostname selects the server-specific task.
No package installation or original checkout changes are made. The supervisor is
detached from the terminal; do not pull/change this clone while it is active.
Server 1 outputs `runs/pkg_guard_parallel_v1/REPORT.md` in this clone.
Server 2 continues `$HOME/R2E-report/runs/report_validation_v1/REPORT.md`.
The original report runner retains its already-loaded limits until it exits; the
supervisor then resumes its records and retries timeout jobs without those limits.
Model export on server 1 must continue supplying the original study's checkpoints;
parallel follow-up weights are not silently substituted into the server-2 comparison.
