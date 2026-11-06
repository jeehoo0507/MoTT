# Delta-memory MoTT: prospective exploratory pilot, 2026-11-06

This experiment is separate from the main MoTT and server-2 FRESH experiments. No previously
reported checkpoint, result, or final-model selection rule is changed. Synthetic PDNs only.
The implementation is inspired by gated delta-rule memory, not a reproduction of Gated DeltaNet
or an FLA kernel benchmark. No novelty claim is established by the implementation alone.

## Hypotheses and attribution

H1: Carrying a temporal memory of placements helps a cached-encoder MoTT maintain placement
quality, compared with initial-encoding reuse without history (`gru`, `delta` versus `cached`).
H2: A normalized-key, gated delta update improves the quality/cost tradeoff relative to a
row-shared GRU with the same d*d recurrent state (`delta` versus `gru`). Parameters and FLOPs
are not identical; report actual parameter counts, neural latency, and end-to-end latency.
H3: Predicting the next physical spectral envelope improves the representation
(`delta_aux` versus `delta`, and `gru_aux` versus `gru`). An aux benefit in both architectures
is evidence for physical supervision, not delta-specific novelty.
H4: The combined model preserves quality on held-out 16x16 boards while reducing repeated
encoder calls (`delta_aux` versus `mott`). Train only 8x8 and 10x10; validate only 10x10.
This is size transfer within the three trained synthetic types, NOT unseen-type generalization.

Six arms: original three-block MoTT; cached initial encoder + current local features, no history;
GRU memory; delta memory; GRU + physical auxiliary; delta + physical auxiliary.
All use P1 inputs, the same exact candidate-action effects, original legality masks, and original
physics prior. Initial full encoder is identical across arms for each seed. Cached arms compute
one full encoder per episode; they still embed CURRENT node/edge/mode/decap observations.
No cached representation of the initial board is substituted for current physics observations.

Delta memory: k is L2-normalized, v is tanh-bounded; Mbar = sigmoid(alpha+3) M;
Mnew = Mbar + sigmoid(beta) k (v - k^T Mbar)^T. Cell queries read q^T Mnew.
The write input is current pooled context, its temporal difference, and the previous selected
action embedding. Each episode starts with zero memory. Memory has placement-time order;
there is no causal scan over candidate-cell or peak-rank order. Cell permutation equivariance
is tested. No learned physical update replaces an exact circuit solve.

## Fixed data and training

Default seeds 0,1,2. Train: first 6 boards from each of three types at 8 and 10 (36 total).
Validation: first 4/type at 10 (12). Evaluation: first 6/type at 10 and 16 (36).
When available, use the fixed first n instance DEFINITIONS from corresponding FRESH pool
files, recording source SHA256 and grids; drop saved outcomes. Never select by success.
Missing source files are generated independently with disjoint fixed split seeds, using the
same tech configurations and 700-point target construction, without pool cutting-plane
refinement. Every arm sees the exact same frozen resulting data. Source/generated differences
are recorded and results are not merged with existing experiments. No teacher cross-split reuse.

Build one deterministic greedy+prune training trajectory per board, retain both successful and
failed trajectories, and share these across every arm. No local-search teacher or expert iteration.
300 imitation updates (one whole trajectory/update), then 30 group-normalized REINFORCE updates
(one board, 4 sampled trajectories/update). Fixed update counts, not equal wall-clock/FLOPs.
Reward = -(count/50 + 20*working-grid failure + 10*working-grid violation area).
Standardize reward within each group (population std, epsilon floor). This is not full GRPO.
LR 3e-4, AdamW without weight decay, gradient norm cap 1. Shared d64/3 blocks/4 heads, float32.
All arms use chronological replay with TBPTT 8; detach recurrent AND cached encoder states
at boundaries. No independent-step shuffle for policy-gradient replay. Cached arms' initial
encoder gradients therefore come from the first chunk; report this limitation when interpreting
quality, and do not claim an architecture-independent comparison to the full main training.
Aux = .1 * Huber(next log10(max_probe |Z|/Zt) at 16 fixed relative log-frequency queries,
clipped to [-6,6]); interpolate labels from the working grid. No auxiliary loss in other arms.
These are next-state spectral targets, not directly supplied as a full vector in the current
policy input. Do NOT use next V/margin as the target: current exact action-effect features
already encode these scalars. Current inputs still contain related per-peak after-action values;
this experiment does not establish learning a physics solver without action-effect features.
Report next-envelope MSE on the free-running policy states. Auxiliary diagnostic overhead is
excluded from placement/policy timing and logged separately.

Select best checkpoint using validation mean(count/50 + 20*grid failure + 10*V), after imitation
and after RL updates 15 and 30, identically for all arms. No test-based checkpoint selection.
Arm order rotates by seed. Resumable optimizer/RNG checkpoints; existing completed runs skipped.
The short-budget original-MoTT arm is retrained; its scores are not the main paper's MoTT scores.

## Evaluation and claim rules

Greedy argmax, max 64 actions, single original working-grid rollout. Independently certify each
final placement using 700 log frequencies plus the repository's golden-section peak refinement;
success = worst ratio <= 1.01. No cutting-plane restart or post-hoc repair for any arm.
This protocol is shared across arms, but differs from main cutting-plane evaluation.
Greedy+prune is an additional exact-checked same-instance reference, not an architecture control.

Report every type, size and seed; success rates; rescued/harmed paired cases; decap difference
on JOINTLY successful cases; model-only, full placement, and independent certification time.
Initialization/bank building are separately part of setup, not placement latency. Timing uses
one CPU thread by default; device is recorded. First-episode effects are included. Timing is an
exploratory measurement, with short episodes and no repeated timing/thermal calibration;
small speed differences need a dedicated repeated benchmark before a speedup claim.
Report parameters and recurrent memory size. Retained cached token states also consume memory;
d*d alone is not total memory use. Exact action-effects costs are retained, so inference savings
do not imply simulator compression or proportional end-to-end acceleration.

Use paired same-instance comparisons within a seed; do not treat seed repetitions as new boards.
An exploratory crossed bootstrap resamples seeds and shared board IDs; with only 3 seeds and
6 boards/cell, intervals are unstable. Report per-seed contrasts and rescued/harmed counts.
No automatic final-model replacement or hard claim from this pilot. Any noninferiority margin
or superiority test for a larger follow-up must be fixed before fresh follow-up results.

## Run and isolation

Use a separate checkout on maskedkd, not server 1. Launcher rejects the loop server. It waits
for a live `server2_once.py run` to finish, then requires >=12 GiB GPU headroom; it never kills
existing jobs. One new GPU run at a time, no new parallel CPU worker pools. Old FRESH results
are only read for pool definitions. Outputs: runs/delta_memory_v1; archive:
runs/delta_memory_v1_results.tgz. No automatic Git push or login is required.

References: https://arxiv.org/abs/2412.06464 (gated delta rule);
https://github.com/fla-org/flash-linear-attention (implementation ecosystem, not a dependency).
