"""`--dedup_modes` (PROPOSALS 11/04 D8b -> candidate P2): MoTT mode tokens picked from distinct working-grid frequencies.

D8 / D8b (round 37): PDNBatch.add_freqs (pdn.py) pads every instance's appended seed peaks to a common length by
repeating its last appended frequency, so the working grid holds runs of identical frequencies (pkg10_0 bare boards:
58-167 distinct of 236). sim.features_mott marks a point as a peak when it is >= both sorted neighbours, so every member
of a run on a plateau counts as a peak and the K = 8 mode tokens are mostly copies of the same frequency (median 3
distinct tokens in the failed r2 final states), while the distinct curve has ~4-5 local maxima; every "uncovered"
r2 pkg failure of D7 sits at a distinct top-ranked peak that the copies pushed out.
This env keeps sim.py unchanged (environment file) and re-implements features_mott with one difference: a run of equal
frequencies counts as one point (its first member), a peak when it is >= the nearest *distinct* neighbours. Unused
token slots are invalid (masked), exactly as sim does when a state has fewer than K peaks. Dims, channels, edges,
decap tokens and the node part are bit-identical to sim.features_mott for the same token set.
"""
import numpy as np
import torch

from .sim import EPS, FEAS_TOL, K_MODES, LOG_CLIP, Env


def distinct_peaks(lvs, lfs):
    """lvs, lfs [E,F] sorted level / frequency -> bool [E,F]: first member of each equal-frequency run whose (common)
    level is >= the levels of the neighbouring distinct frequencies (band edges count as -inf)."""
    E, F = lvs.shape
    first = torch.ones(E, F, dtype=torch.bool, device=lvs.device)
    first[:, 1:] = lfs[:, 1:] != lfs[:, :-1]
    ar = torch.arange(F, device=lvs.device).expand(E, F)
    fidx = torch.where(first, ar, torch.zeros_like(ar)).cummax(1).values      # index of the run's first member
    lr = lvs.gather(1, fidx)                                                     # one level per run
    ninf = torch.full_like(lr[:, :1], -float("inf"))
    left_ok = lr >= torch.cat([ninf, lr[:, :-1]], 1)
    right_ok = lr >= torch.cat([lr[:, 1:], ninf], 1)
    bad = (~(left_ok & right_ok)).long()
    rid = first.long().cumsum(1) - 1                                             # run id
    run_bad = torch.zeros(E, F, dtype=torch.long, device=lvs.device).scatter_add_(1, rid, bad)
    return first & (run_bad.gather(1, rid) == 0)


class DedupEnv(Env):
    def features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas):
        return self.features_mott(z_ip, z_ii, z_pp, V, Vn, mn, feas)

    def features_mott(self, z_ip, z_ii, z_pp, V, Vn, mn, feas):
        """sim.Env.features_mott with the distinct-frequency peak rule (see module doc)."""
        b, inst, E, n, T = self.b, self.inst, self.E, self.n, self.T
        dev = z_ip.device
        K = min(K_MODES, z_pp.shape[1])
        lvp = torch.log10(z_pp.abs() + EPS) - self.logZt[..., None]                    # [E,F,P]
        lv, pk = lvp.max(-1)
        lvs = lv.gather(1, self.forder)
        peak = distinct_peaks(lvs, self.freqs_t.gather(1, self.forder))
        top, j = torch.where(peak, lvs, -float("inf")).topk(K, 1)
        mvalid = torch.isfinite(top)
        fk = self.forder.gather(1, j)                                                   # grid index [E,K]
        pr = pk.gather(1, fk)                                                           # worst probe
        P = z_pp.shape[2]
        zpp = z_pp.gather(1, fk[..., None].expand(E, K, P)).gather(2, pr[..., None])[..., 0]          # [E,K]
        zip_ = z_ip.gather(1, fk[:, :, None, None].expand(E, K, n, P))
        zip_ = zip_.gather(3, pr[:, :, None, None].expand(E, K, n, 1))[..., 0]                         # [E,K,n]
        zii = z_ii.gather(1, fk[..., None].expand(E, K, n))                                             # [E,K,n]
        lzt = self.logZt.gather(1, fk)                                                                  # [E,K]
        zdk = self.zd.gather(2, fk[:, None, :].expand(E, T, K))                                         # [E,T,K]
        g = zip_.abs() / torch.sqrt(zpp.abs()[..., None] * zii.abs() + EPS)
        ed = [torch.log10(g + 1e-6), torch.log10(zii.abs() + EPS) - lzt[..., None]]
        for t in range(T):
            after = zpp[..., None] - zip_ ** 2 / (zii + zdk[:, t, :, None])
            ed.append(torch.log10(after.abs() + EPS) - lzt[..., None])
        edge = torch.stack(ed, -1).clamp(-LOG_CLIP, LOG_CLIP).float() * mvalid[..., None, None]       # [E,K,n,De]
        edge = edge.permute(0, 2, 1, 3).reshape(E, n, -1)
        cfg = self.b.cfg
        fc = float(np.sqrt(cfg.f_min * cfg.f_max))
        fkv = self.freqs_t.gather(1, fk)
        mode = torch.stack([torch.log10(fkv / fc), top.clamp(-LOG_CLIP, LOG_CLIP), (top > FEAS_TOL).double(),
                            mvalid.double()], -1).float()
        mode = torch.where(mvalid[..., None], mode, torch.zeros_like(mode))
        if K < K_MODES:
            mode = torch.cat([mode, mode.new_zeros(E, K_MODES - K, mode.shape[-1])], 1)
            edge = torch.cat([edge, edge.new_zeros(E, n, (K_MODES - K) * (2 + T))], 2)
        ztc = 10 ** self.logZt.median(1).values                                                       # [E]
        wc = 2 * np.pi * fc
        dec = []
        for c, esl, esr in cfg.decaps:
            srf = 1.0 / (2 * np.pi * np.sqrt(esl * c))
            dec.append(torch.stack([torch.full_like(ztc, np.log10(srf / fc)), torch.log10(wc * esl / ztc),
                                    torch.log10(esr / ztc), torch.log10(1.0 / (wc * c) / ztc)], -1))
        dec = torch.stack(dec, 1).float()                                                               # [E,T,4]
        vf = self.valid.float()
        flags = torch.stack([vf, b.keepout[inst].float(), b.probe_mask()[inst].float(), b.entry[inst].float()], -1)
        occ_oh = torch.zeros(E, n, T, device=dev)
        pl = self.occ >= 0
        occ_oh[pl] = torch.nn.functional.one_hot(self.occ[pl], T).float()
        glob = torch.stack([feas.float(), 1 - self.steps.float() / self.max_steps, pl.sum(1).float() / vf.sum(1)], -1)
        dV = (V[:, None, None] - Vn).float().clamp(-LOG_CLIP, LOG_CLIP)
        node = torch.cat([flags, occ_oh, glob[:, None, :].expand(E, n, 3), dV, mn.clamp(-3, 3).float()], -1) * vf[..., None]
        return torch.cat([node, edge, mode.reshape(E, 1, -1).expand(E, n, -1), dec.reshape(E, 1, -1).expand(E, n, -1)], -1)
