# Report validation v1 — frozen before follow-up results

## Decision

Server 1 makes a candidate; server 2 validates completed checkpoints on fresh synthetic
problem definitions. No new architecture training, LoRA or automatic model adoption.
All old processes, original queues, environments and artifacts stay untouched. Server 2
uses one new CPU subprocess at a time; it does not use the GPU.

Primary comparison: original MoTT+P1, continued baseline, reserved-training candidate.
All three source seeds (0/1/2); candidate vs continued baseline uses the identical original
checkpoint and continuation budget. Record source SHA256 and verify pairing on import.
Candidate decoding uses 0.1 log10 target reserve; other models use their original settings.
Original physical target, decap library and 700-frequency + golden-section checker apply
to every method. Success is worst |Z|/Zt <= 1.01. Four cutting-plane restarts at most.
The reserved target affects working constraints only, never final certification.

The new method is a training/decoding intervention. Coordinate/modal models have not received
matched reserve continuation: this table alone cannot establish MoTT-specific structural
benefits. If adopting this training improvement, a later matched coordinate training control
is needed for that attribution. No automatic adoption decision is made from this report.

## Frozen data and priority

20 fresh boards per type/size for PCB, package, interposer at 10/16; 8/type at 20.
Seed = 9,100,000 + type_index*100,000 + side*1,000 + instance, type order PCB/package/IP.
Use existing TECHS/DIFFICULTY and sample_batch defaults, including exact target construction
and bare/filled resonance seed frequencies. Do not pre-refine pools with successful policies
or greedy cutting planes, and do not select boards by outcomes. This is a NEW common
working-grid protocol, different from the historical greedy-refined pools. Report separately
from RESULTS.md; a difference against old percentages is not a paired gain.

Priority: pkg16, pkg10, pcb10, ip10, pcb16, ip16, then pkg/pcb/ip20. Finish received primary checkpoints on ALL 10/16 cases before secondary jobs. Next run
classical references, paper-inspired methods, internal controls, size-20 checks and RL-only appendix.
Within each phase, package cases precede the other types.
Fresh case generation and classical references run before model arrival. No evaluation output
feeds back into training or parameter selection. Lock reserve/settings before exporting models.
Same boards repeat across training seeds, so 20 boards x 3 seeds is not 60 independent boards.

## Baselines and what the comparisons mean

Primary references, on all sizes: greedy+prune, resonance rule. Internal representation
controls: existing coordinate-input and modal-feature checkpoints, all seeds, all sizes.

Paper-inspired comparisons, all seeds at 10/16:

- B-Park: the existing adapted attention/pointer model with distillation + rollout-baseline RL.
  The original HBM task differs (fixed-count placement / impedance objective); current code
  uses our multi-type add/remove/stop action space, current observations, and shared synthetic
  environment. It is not an exact reproduction of Park's paper and is not DevFormer.
- B-Manoharan: existing adapted fixed-port-order / decap-type attention model with distillation
  + group-baseline RL. Our port-priority frequency, input, training schedule and RL algorithm
  differ from the original post-layout method (including critic/replay). Label explicitly.
- Appendix at size 10: both existing RL-only variants, included prospectively rather than
  selecting whichever variant scores worse. The code name of Manoharan's RL-only run includes
  l20; record it as a distinct adapted training variant.

Do not claim these limited-budget adaptations beat the original published methods. Historical
baselines used roughly two training hours; continued MoTT adds 45 minutes. Record available
checkpoint training metadata and distinguish comparison of final systems from equal-budget
attribution. The matched continued-baseline control isolates the new intervention.

DevFormer belongs in related work, not the numerical table: this repository's B-Park lacks
DevFormer's specific context/symmetry learning setup. A faithful DevFormer comparison is a
separate implementation task. Do not relabel B-Park as DevFormer.

No new GA/local-search runs in this deadline-limited validation. Existing historical GA/LS
results can be shown only in their original protocol table. No percentages from different
problem sets, training data or checkers are combined.

References:
- Park et al.: https://arxiv.org/abs/2203.15722
- DevFormer: https://proceedings.mlr.press/v202/kim23h.html
- Manoharan et al.: https://doi.org/10.1109/TSIPI.2026.3691276

## Measurement and interpretation

CPU one thread, new per-case bank for every method, greedy argmax for learned policies.
64 actions per rollout for MoTT/coordinate/modal/Park and classical methods. Manoharan retains
its native fixed-port traversal budget (one decision per candidate port + stop); do not silently
truncate its original action semantics. Record this difference in the methods section.

Save every completed board: exact margin, component count, placement, history and timings.
Placement time includes policy and physics calculations. Total includes certification and
frequency refinement; initial bank construction is recorded separately. Do not replace total
runtime with the fastest/final cutting-plane round. CPU contention with old jobs makes these
exploratory timings, not a controlled speed benchmark against earlier hardware measurements.

Report per seed/type/size and paired rescued/harmed counts. Component differences use only
JOINTLY solved boards, alongside success rates and unconditional denominators. Small n=20/8
requires cautious conclusions; no significance claim is automated. Failures/timeouts/missing
weights are missing evaluations, not successes or physical failures. Never silently drop a seed.
Fresh-board transfer across size is not leave-one-type-out generalization: the candidate has
seen all three types during pretraining. That claim requires separately held-out-type training.

## Transport and execution

No credentials, data pools or optimizer states are exported. report_model_bundle.py creates
sanitized weights/configuration plus source hashes and training metadata. Existing source
files are only read. With --wait it refreshes a local archive as checkpoints finish, until all
nine primary and eighteen optional checkpoints arrive or its 24-hour deadline expires.
With explicit --publish it sends only these sanitized artifacts to the dedicated public Git branch
report-models-pkg-guard-v1, using a separate Git directory; main and source checkouts are untouched.

Server 2 polls that branch every two minutes and imports the allowlisted files via inbox/models.tgz.
--transport manual also accepts a manually copied archive at inbox/models.tgz. Copy to models.tgz.part,
then rename to models.tgz atomically; a new manifest is installed only after file allowlisting,
size/hash checks and matched-initialization checks. Replacing an already evaluated checkpoint
under the same identity is rejected. Importing the archive automatically enables pending jobs.

Server 1 must run the exporter once with --wait --publish, with working Git push credentials.
Server 2 requires only public Git read access and then receives completed models automatically.
No SSH addresses, credential copies or login settings are assumed. Existing server-2 FRESH/Delta model weights are not substituted for server-1 pairs.

24-hour deadline from the first server-2 launch, including model waiting. Each case/method has
a 15-minute timeout. New child starts below 28 GiB cgroup memory and is stopped/retried if total
memory exceeds 32 GiB. Only the newly spawned process group may be signalled. Existing
experiments are never stopped. These conservative guards reduce contention, not guarantee
isolation from all instantaneous memory spikes. Save partial reports at every job and stop.
No promise that all secondary baselines/large cases finish before the deadline.

Outputs: runs/report_validation_v1/REPORT.md, summary.csv, paired.json, errors.json,
case_plan.json, artifacts/manifest.json, eval/*.json; runs/report_validation_v1_results.tgz.
The result archive contains records, not weights or raw pools.
