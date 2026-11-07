"""Three board kinds (PCB-like, package-like, interposer-like) as PDNConfig variants.

The environment files (pdn.py, sim.py, verify.py, bank.py) are not touched: a kind is just a
PDNConfig with different physical constants and a decap library. All kinds keep T = 3 decap
types and the same working-grid size (n_freq = 32, n_feat_freq = 8), so one policy can be
trained on a mixture of kinds.

Kinds differ in what matters electrically:
  - plane geometry (cell size, dielectric thickness, metal) -> plane C per cell, edge L/R,
    first cavity resonance f1 = c / (2 * N * cell * sqrt(eps_r))
  - VRM / power-entry path (R + L)
  - decap library (C, ESL, ESR): SRF = 1 / (2 pi sqrt(ESL C))

PCB is the repository default (all earlier analysis used it). IP is the interposer-like config of
build_pool.py (2026-10-20). PKG sits between them: thicker build-up dielectric than the PCB
default, lower-ESL on-package capacitors.

Difficulty ranges follow build_pool.py: the interposer library (tiny capacitors) only meets hard
targets with almost every cell filled, so it uses the easier range.
"""
import dataclasses

from .pdn import PDNConfig

PCB = PDNConfig(dense_max_nodes=900)

PKG = dataclasses.replace(
    PCB,
    cell_size=400e-6, diel_thickness=35e-6, eps_r=3.4, tan_d=0.01, metal_thickness=15e-6,
    vrm_r=3e-3, vrm_l=100e-12,
    decaps=((1e-9, 40e-12, 0.03), (220e-12, 15e-12, 0.05), (47e-12, 8e-12, 0.10)))

IP = dataclasses.replace(
    PCB,
    cell_size=200e-6, diel_thickness=1e-6, metal_thickness=1e-6, vrm_l=50e-12,
    decaps=((100e-12, 5e-12, 0.05), (20e-12, 2e-12, 0.1), (5e-12, 1e-12, 0.2)))

TECHS = {"pcb": PCB, "pkg": PKG, "ip": IP}
HARD, EASY = (0.02, 0.2), (0.2, 0.5)
DIFFICULTY = {"pcb": HARD, "pkg": HARD, "ip": EASY}

# Fourth kind, never in training or val (AGENT_PROMPT bonus: zero-shot to an unseen kind; fixed in
# PROPOSALS 2026-11-03 before any policy result). On-die-like grid: small cells, very thin and
# lossy metal (R_edge ~4x IP), short power-entry path seen from the bumps, MOS-style decaps with
# low ESL and high ESR. Kept out of TECHS so pool builders / smoke tests over TECHS do not change.
CHIP = dataclasses.replace(
    PCB,
    cell_size=100e-6, diel_thickness=0.2e-6, eps_r=3.9, metal_thickness=0.5e-6, vrm_r=2e-3, vrm_l=30e-12,
    decaps=((1e-9, 3e-12, 0.05), (300e-12, 1.5e-12, 0.1), (50e-12, 0.5e-12, 0.3)))
ZS_TECHS = {"chip": CHIP}
ALL_TECHS = {**TECHS, **ZS_TECHS}

# Randomized-physics kind "rnd" (PROPOSALS 11/07 R, human): every electrical constant that shapes the resonance
# structure is drawn log-uniformly between its PCB and IP value, extended by RND_MARGIN of the log-gap on both
# sides, independently per constant; the decap library is interpolated (log) between the PCB and IP libraries on
# its own draw `lam` (lam 0 = PCB library, 1 = IP library, outside = extrapolated). The PKG preset is never used,
# so a policy trained on pcb + ip + rnd has still never seen a package board. Difficulty follows the library
# (tiny IP-like capacitors only meet easy targets, as techs.DIFFICULTY): HARD when lam < RND_EASY_LAM else EASY.
RND_MARGIN, RND_EASY_LAM = 0.3, 0.6


def sample_rnd(rng):
    """(PDNConfig, difficulty) for one randomized-physics pool file. rng: numpy Generator (seeded per file)."""
    import numpy as np

    def lu(a, b, margin=RND_MARGIN):
        lo, hi = sorted((np.log(a), np.log(b)))
        g = hi - lo
        return float(np.exp(rng.uniform(lo - margin * g, hi + margin * g)))

    lam = float(rng.uniform(-RND_MARGIN, 1.0 + RND_MARGIN))
    lib = tuple(tuple(float(np.exp(np.log(x0) + lam * (np.log(x1) - np.log(x0)))) for x0, x1 in zip(d0, d1))
                for d0, d1 in zip(PCB.decaps, IP.decaps))
    cfg = dataclasses.replace(
        PCB,
        cell_size=lu(PCB.cell_size, IP.cell_size), diel_thickness=lu(PCB.diel_thickness, IP.diel_thickness),
        eps_r=float(rng.uniform(3.0, 4.5)), tan_d=lu(0.01, 0.02, 0.0),
        metal_thickness=lu(PCB.metal_thickness, IP.metal_thickness),
        vrm_r=lu(3e-3, 5e-3, 0.0), vrm_l=lu(PCB.vrm_l, IP.vrm_l), decaps=lib)
    return cfg, (HARD if lam < RND_EASY_LAM else EASY)


def describe(name):
    cfg = ALL_TECHS[name]
    import numpy as np
    srf = [1.0 / (2 * np.pi * np.sqrt(esl * c)) for c, esl, _ in cfg.decaps]
    return (f"{name}: cell {cfg.cell_size*1e6:.0f}um diel {cfg.diel_thickness*1e6:.0f}um "
            f"C_cell {cfg.c_cell*1e12:.3f}pF L_edge {cfg.l_edge*1e12:.2f}pH R_edge {cfg.r_edge*1e3:.2f}mohm "
            f"VRM {cfg.vrm_r*1e3:.0f}mohm+{cfg.vrm_l*1e12:.0f}pH f1(10x10) {cfg.f1(10,10)/1e9:.1f}GHz "
            f"f1(20x20) {cfg.f1(20,20)/1e9:.1f}GHz SRF {[f'{s/1e9:.2f}GHz' for s in srf]}")


if __name__ == "__main__":
    for k in ALL_TECHS:
        print(describe(k))
