"""Consistent conservative working constraints; original physical target/checker unchanged.

This is a robustness intervention, not a new transformer architecture. A positive reserve lowers
only Env.logZt, so features, candidate effects, legal actions and working-grid rewards agree.
PDNBatch.Zt, zt_par, library, impedance equations and independent certification are untouched.
"""
import torch
from .peak_env import PkEnv


class GuardedPkEnv(PkEnv):
    def __init__(self,*args,reserve_decades=0.,**kwargs):
        self.reserve_decades=float(reserve_decades)
        if not 0 <= self.reserve_decades <= .3:raise ValueError('reserve_decades outside [0,.3]')
        super().__init__(*args,**kwargs)
        self._working_target()

    def _working_target(self):
        self.logZt=torch.log10(self.b.Zt[self.inst])-self.reserve_decades

    def refresh(self):
        super().refresh()
        self._working_target()  # Recompute from original target; never accumulate the reserve.
