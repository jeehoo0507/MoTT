# MoTT: Modal-Token Transformer

<p align="center">
  <img src="assets/mott_architecture.png" width="520">
</p>

Code and evaluation records for the report *MoTT: A Modal-Token Transformer for Generalizable Decoupling Capacitor Placement* (Decap 배치 범용화를 위한 모달 토큰 트랜스포머), 2026 KOFAC R&E.

MoTT places decoupling capacitors on a power distribution network (PDN) so that the impedance stays below the target with as few capacitors as possible. A board is given to the policy as resonant-mode tokens, cell tokens and decap-type tokens instead of coordinates, and cells exchange information only through the mode tokens. The same policy is then used on board kinds and sizes that were not in the training set.

<p align="center">
  <img src="assets/boards.png" width="640">
</p>

## Dependencies

* Python 3
* PyTorch >= 2.2
* NumPy, SciPy
* Matplotlib (plots only)

```bash
pip install -r requirements.txt
python sanity_check.py
```

## Usage

### Generating problems

Problems are generated from fixed seeds for three board kinds (PCB, package, interposer). Every problem is checked to be solvable.

```bash
python scripts/build_pools.py --out runs/pool_v1
```

### Teacher solutions

Greedy search with pruning, followed by local search.

```bash
python scripts/make_teacher.py --pool runs/pool_v1 --out data/teacher
```

### Training

Imitation of the teacher for 0.75 h, then reinforcement learning for 1.25 h, and evaluation of both checkpoints.

```bash
python scripts/run_rung.py --name r2pk --arm "--model mott --peak_refine" \
    --distill_hours 0.75 --rl_hours 1.25 --seed 0
```

To hold out one board kind, see the `lo_<kind>_*` runs in [docs/PROPOSALS.md](docs/PROPOSALS.md).

### Evaluation

All methods are evaluated on the same problems with the same exact check (700 frequency points plus refinement around peaks). `--repair 4` turns on the verification harness.

```bash
python scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*' \
    --policy mott=runs/<day>/r2pk_rl/best.pt --repair 4 --out runs/<day>/r2pk_rl
```

Baselines run through the same script, e.g. `--methods gp,ls` for greedy+prune and local search.

```bash
python scripts/collect_results.py --out docs/RESULTS.md
```

## Results

Success rate under the exact check. 10×10 boards, 3 seeds × 50 problems, unless noted.

Unseen board kind (one kind removed from training):

| Held-out kind | MoTT | Coordinate input |
|:--|:-:|:-:|
| PCB | 0.89 | 0.64 |
| Interposer | 0.99 | 0.65 |
| Package | 0.71 | 0.80 |
| Package, with harness | 0.99 | 0.92 |

Unseen board size (trained on 8–12, with harness):

| Size | Interposer | PCB | Package |
|:--|:-:|:-:|:-:|
| 16×16 | 1.00 | 0.99 | 0.94 |
| 20×20 | 1.00 | 1.00 | 0.96 |

On interposer boards MoTT uses about 30% fewer decaps than the resonance rule. Including the harness, it is about 4× (interposer), 4–7× (package) and 1.5× (PCB) faster than greedy+prune on 16×16 boards.

Every number comes from the JSON records under [runs/](runs); see [runs/README.md](runs/README.md) for the format and [docs/RESULTS.md](docs/RESULTS.md) for the full tables.

## Baselines

| Name | Description |
|:--|:--|
| greedy+prune, local search, GA | search, `decap_loop/baselines.py` |
| resonance rule | SRF-matched decap at the most strongly coupled cell |
| coordinate input | same training as MoTT, coordinate-based input |
| B-Park, B-DevFormer, B-Manoharan | re-implementations of [Park et al. 2022](https://doi.org/10.1109/TMTT.2022.3202221), [Kim et al. 2023](https://proceedings.mlr.press/v202/kim23h.html), [Manoharan et al. 2026](https://doi.org/10.1109/TSIPI.2026.3691276), `baselines_ml/` |

The re-implementations follow the structure and inputs of each paper but use this repository's problem setting and a 2 h training budget, so they are not reproductions of the original results.

## Experiment log

Decision rules were written to [docs/PROPOSALS.md](docs/PROPOSALS.md) and pushed before the corresponding results existed. [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) lists, for each experiment, the commit that registered the rule and the commit with the first result. Analyses added after seeing results are marked as such.

`main` was cleaned up for submission. The complete record, including agent logs and notes, is kept unchanged on the [`research-log`](../../tree/research-log) branch and the [`experiments-final`](../../tree/experiments-final) tag.

## Limitations

Synthetic boards only, up to 20×20 cells and three decap types per kind.
