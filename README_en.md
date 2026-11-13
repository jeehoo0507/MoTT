<div align="center">

# MoTT: Modal-Token Transformer

**A Modal-Token Transformer for Generalizable Decoupling Capacitor Placement**

[한국어](README.md) | English

[Introduction](#introduction) · [Main results](#main-results) · [Experiments](#experiments) · [Usage](#usage) · [Checking the numbers](#checking-the-numbers)

<img src="assets/mott_architecture.png" width="500">

</div>

## Introduction

A power distribution network (PDN) resonates at certain frequencies, where its impedance rises. Decoupling capacitors (decaps) at the right cells lower these peaks, but there are far too many combinations of cells and decap types to enumerate.

Prior reinforcement-learning methods decide placements quickly but mostly take cell coordinates as input, and need retraining when the kind of board (PCB, package, interposer) changes. MoTT describes a board with **resonant-mode, cell and decap-type tokens** instead of coordinates, so one policy works on board kinds and sizes it has not seen in training.

<div align="center">
<img src="assets/boards.png" width="620">

<sub>MoTT placements on the three board kinds (10×10). Bottom: impedance before (dashed) and after (solid)</sub>
</div>

## Main results

| Question | Result |
|:--|:--|
| Does it solve an unseen kind of board? | PCB 0.89, interposer 0.99 (coordinate model 0.64, 0.65). Package 0.99 with the verification harness |
| Does it solve an unseen board size? | 0.94 or higher on all three kinds at 16×16 and 20×20 (with harness) |
| Does it use fewer decaps? | About 30% fewer than the resonance rule on interposer boards |
| Is it faster than search? | About 4× (interposer), 4–7× (package), 1.5× (PCB) faster than greedy+prune |

Success rate is the fraction of problems passing the exact check (700 frequency points plus refinement around peaks). All methods are evaluated on the same problems with the same check.

## Experiments

Unless noted: 10×10 boards, 3 seeds × 50 problems.

### 1. Input analysis

Before training, we measured by AUC whether an input that separates good cells on one kind of board still does so on another (0.5 is chance).

| Input | PCB → PCB | PCB → interposer | Interposer → PCB |
|:--|:-:|:-:|:-:|
| Distance to the probe | 0.74 | 0.46 | 0.23 |
| Loop inductance + fixed-frequency impedance | 0.88 | 0.45 | 0.59 |
| Resonance features | 0.86 | 0.59 | 0.32 |
| **Effect of placing one decap** | 0.73 | **0.98** | **0.74** |

Inputs used in prior work fall below chance when the kind of board changes. The model therefore uses the one-decap effect, which carries over, as its base score and resonance features only as a correction.

<details>
<summary>Records and decision rules</summary>

- Analysis code and results: [analysis folder on the `research-log` branch](../../tree/research-log/research_notes/%ED%86%A0%ED%81%B0%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%EC%84%9D)

</details>

### 2. Unseen kind of board

One of the three kinds is removed from training and used only for evaluation.

| Held-out kind | MoTT | Coordinate input |
|:--|:-:|:-:|
| PCB | **0.89** | 0.64 |
| Interposer | **0.99** | 0.65 |
| Package | 0.71 | **0.80** |
| Package + verification harness | **0.99** | 0.92 |

The policy alone is below the coordinate model on held-out package boards. The cause was peaks hidden between the 32 frequencies used for decisions. With the **verification harness**, which finds the failing frequency by the exact check and lets the policy fix the placement, 43 of the 44 failed problems were solved without adding decaps.

The same harness was applied to the re-implemented prior methods (models trained without package boards).

| Method | Package (held out) | PCB | Interposer | Decaps (MoTT = 1) |
|:--|:-:|:-:|:-:|:-:|
| **MoTT** | **0.99** | **0.99** | 1.00 | 1.0 |
| Coordinate input | 0.92 | 0.94 | 1.00 | 1.0 |
| B-Park | 0.91 | 0.04 | 1.00 | 2.8 |
| B-DevFormer | 0.45 | 0.79 | 0.99 | 4.5 |
| B-Manoharan | 0.89 | 0.03 | 1.00 | 4.3 |

<details>
<summary>Records and decision rules</summary>

- MoTT: `lo_<kind>_r2pk*_rl` under [`runs/2026-10-29/`](runs/2026-10-29)
- Coordinate input: `lo_<kind>_r0*_rl`
- Re-implementations: `lo_pkg_bp_park*`, `lo_pkg_bdf*`, `lo_pkg_bm2_rlonly*`
- Harness results are the files ending in `+rep4`
- Decision rules: F1 [`ed4ced5`](https://github.com/jeehoo0507/R2E/commit/ed4ced5), N [`ce4422f`](https://github.com/jeehoo0507/R2E/commit/ce4422f)
- The package harness evaluation was added after seeing the policy result: [`451ec4d`](https://github.com/jeehoo0507/R2E/commit/451ec4d)

</details>

### 3. Unseen board size

Models trained on 8–12 cell boards are evaluated on 16×16 and 20×20.

| Size | Kind | Policy | + harness |
|:--|:--|:-:|:-:|
| 16×16 | Interposer | 1.00 | 1.00 |
| | PCB | 0.92 | 0.99 |
| | Package | 0.56 | 0.94 |
| 20×20 | Interposer | 1.00 | 1.00 |
| | PCB | 0.92 | 1.00 |
| | Package | 0.50 | 0.96 |

<div align="center">
<img src="assets/pkg16.png" width="500">

<sub>Success rate and decap count on 16×16 package boards, the hardest case (with harness)</sub>
</div>

<details>
<summary>Records and decision rules</summary>

- `r2pk*_rl/eval/eval_<kind>16_*` and `eval_<kind>20_*` under [`runs/2026-10-29/`](runs/2026-10-29)
- Decision rule: F [`0242e56`](https://github.com/jeehoo0507/R2E/commit/0242e56)

</details>

### 4. Effect of the input and the architecture

From the coordinate model to MoTT, one change at a time.

| Model | Input | Architecture | Unseen PCB | Unseen interposer | Decaps (interposer 10 / 16) |
|:--|:--|:--|:-:|:-:|:-:|
| Coordinate input | coordinates | baseline | 0.64 | 0.65 | 11.2 / 20.6 |
| Resonance input | coordinates + resonance | baseline | 0.84 | 0.89 | 9.8 / 17.9 |
| MoTT (no P1) | resonance | MoTT | **0.90** | 0.89 | 8.0 / 14.6 |
| **MoTT** | resonance + P1 | MoTT | 0.89 | **0.99** | **8.0 / 13.9** |

Adding resonance features raises the success rate on unseen kinds; changing the architecture to MoTT lowers the decap count. P1 is a peak-height correction added to the mode tokens.

<details>
<summary>Records and decision rules</summary>

- `r0_coord*` (coordinate input), `r1_modal*` (resonance input), `r2_mott*` (no P1), `r2pk*` (MoTT) and their `lo_<kind>_*` runs
- Decision rule: P1 [`4b496a5`](https://github.com/jeehoo0507/R2E/commit/4b496a5)

</details>

### 5. Trained on all three kinds

<div align="center">
<img src="assets/all_kinds.png" width="800">

<sub>Success rate (%, 10×10, harness applied to every method). Bars: mean over seeds; whiskers: range over seeds</sub>
</div>

MoTT is the only learned method at or above 0.99 on all three kinds. The re-implemented prior methods drop to 0.02 on PCB boards for some seeds.

<details>
<summary>Records and decision rules</summary>

- `r2pk*`, `r0_coord*`, `bp_park*`, `bdf*`, `bm2_rlonly*` under [`runs/2026-10-29/`](runs/2026-10-29)
- Decision rules: B1 [`e70de68`](https://github.com/jeehoo0507/R2E/commit/e70de68), N [`ce4422f`](https://github.com/jeehoo0507/R2E/commit/ce4422f)
- The harness evaluation of B-Park was added after seeing its policy result (reported as post hoc). The plan was recorded before the run: [`bd69f36`](https://github.com/jeehoo0507/MoTT/commit/bd69f36), results [`aa92f10`](https://github.com/jeehoo0507/MoTT/commit/aa92f10)

</details>

### 6. Decap count and speed

<div align="center">
<img src="assets/decap_speed.png" width="800">
</div>

**Decap count.** About 30% fewer than the resonance rule on interposer boards.

| Size | MoTT | Resonance rule |
|:--|:-:|:-:|
| 10×10 | **8.4** | 12.4 |
| 16×16 | **13.8** | 20.3 |
| 20×20 | **21.8** | 31.2 |

**Speed.** Measured three times on different problem sets. Values are the total time of greedy+prune divided by that of MoTT with the harness (16×16).

| Measurement | Interposer | PCB | Package |
|:--|:-:|:-:|:-:|
| 1 (50 problems) | 4.1× | 1.7× | 7.4× |
| 2 (150 problems) | 4.4× | 1.4× | 4.3× |
| 3 (150 problems, another server) | 3.9× | 1.7× | 6.0× |

<details>
<summary>Records and decision rules</summary>

- Measurement 1: [`runs/speed_1109u/`](runs/speed_1109u); measurement 2: `runs/speed_1110v_*/`
- Records of measurement 3 (server 2) will be added
- Conditions: 11/09 U and 11/10 V in [docs/PROPOSALS.md](docs/PROPOSALS.md)

</details>

## Usage

```bash
pip install -r requirements.txt
python sanity_check.py
```

**1. Problems and teacher solutions**

```bash
python scripts/build_pools.py --out runs/pool_v1
python scripts/make_teacher.py --pool runs/pool_v1 --out data/teacher
```

**2. Training** (imitation 0.75 h, then reinforcement learning 1.25 h)

```bash
python scripts/run_rung.py --name r2pk --arm "--model mott --peak_refine" \
    --distill_hours 0.75 --rl_hours 1.25 --seed 0
```

**3. Evaluation** (`--repair 4` turns on the verification harness)

```bash
python scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*' \
    --policy mott=runs/<day>/r2pk_rl/best.pt --repair 4 --out runs/<day>/r2pk_rl
```

| Path | Contents |
|:--|:--|
| `decap_loop/` | PDN simulator, exact check, MoTT model, search baselines |
| `baselines_ml/` | re-implementations of prior work (B-Park, B-DevFormer, B-Manoharan) |
| `train.py`, `scripts/` | training, problem generation, evaluation, result tables, speed measurement |
| `runs/` | evaluation records |
| `docs/` | decision rules, pre-registration timeline, result tables, agent operation |

## Checking the numbers

| What to check | Where |
|:--|:--|
| Where a number comes from | evaluation records in [`runs/`](runs) ([format](runs/README.md), [full tables](docs/RESULTS.md)) |
| Whether the rule came before the result | [pre-registration timeline](docs/PREREGISTRATION.md) |
| The decision rules | [docs/PROPOSALS.md](docs/PROPOSALS.md) |
| The full record before cleanup | [`research-log`](../../tree/research-log) branch, [`experiments-final`](../../tree/experiments-final) tag |

Open "Records and decision rules" under each experiment for its folders and commits.

## Prior work

**Re-implemented and compared**

Methods named B- carry the architecture and inputs of the papers below into the problem setting of this repository. The setting and the training budget (2 h) differ from the papers, so they are not reproductions of the original results.

- **B-Park**: Park et al., "Transformer network-based reinforcement learning method for power distribution network (PDN) optimization of high bandwidth memory (HBM)," *IEEE Trans. Microwave Theory and Techniques*, 70(11), 2022. [doi](https://doi.org/10.1109/TMTT.2022.3202221)
- **B-DevFormer**: Kim et al., "DevFormer: A symmetric transformer for context-aware device placement," *ICML*, 2023. [paper](https://proceedings.mlr.press/v202/kim23h.html)
- **B-Manoharan**: Manoharan et al., "Reinforcement learning for decoupling capacitor optimization in postlayout PDNs using transformer architecture," *IEEE Trans. Signal and Power Integrity*, 5, 2026. [doi](https://doi.org/10.1109/TSIPI.2026.3691276)

**Other work on decap placement**

- Zhang et al., "Decoupling optimization for complex PDN structures using deep reinforcement learning," *IEEE Trans. Microwave Theory and Techniques*, 71(9), 2023. [doi](https://doi.org/10.1109/TMTT.2023.3248237) (trained separately for each board, so not included in the comparison)
- Park et al., "Deep reinforcement learning-based optimal decoupling capacitor design method for silicon interposer-based 2.5-D/3-D ICs," *IEEE Trans. Components, Packaging and Manufacturing Technology*, 10(3), 2020. [doi](https://doi.org/10.1109/TCPMT.2020.2972019)
- Jiang et al., "A novel physics-assisted genetic algorithm for decoupling capacitor optimization," *IEEE Trans. Microwave Theory and Techniques*, 72(8), 2024. [doi](https://doi.org/10.1109/TMTT.2024.3356575) (search-based)
- Anderson et al., "Methodology for determining the placement of decoupling capacitors in a power distribution system," US Patent 6,789,241 B2, 2004. [patent](https://patents.google.com/patent/US6789241B2/en) (places decaps matched to resonance peaks; the resonance rule here is of this family)

**Learning methods**

- Vaswani et al., "Attention is all you need," *NeurIPS*, 2017.
- Shao et al., "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," arXiv:2402.03300, 2024. (group-normalized advantage, GRPO)
- Kwon et al., "POMO: Policy optimization with multiple optima for reinforcement learning," *NeurIPS*, 2020. (mean over several solutions of the same problem as the baseline)
- Anthony et al., "Thinking fast and slow with deep learning and tree search," *NeurIPS*, 2017. (imitation, then replacing the teacher with better solutions)

## Limitations

Synthetic boards only, up to 20×20 cells and three decap types per kind.

## Environment

| Item | Spec |
|---|---|
| GPU | NVIDIA RTX A5000 (24 GB), driver 595.99.02 |
| CPU | Intel Xeon Gold 5218R @ 2.10 GHz (40 threads) |
| RAM | 123 GB |
| OS | Ubuntu 26.04.1 LTS |
| Software | Python 3.14.4, PyTorch 2.14.1 (CUDA 13.0), NumPy 2.5.4 |

The per-problem times in the report and in `docs/RESULTS.md` were measured on this server (server 1). Records from server 2, which has a different CPU, have their times left blank in `docs/RESULTS.md`.
