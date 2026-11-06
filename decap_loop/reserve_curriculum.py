"""Training-only gap feedback. Original target remains the task definition."""
from dataclasses import asdict
import hashlib
import json
import numpy as np


class ReserveCurriculum:
    def __init__(self,arm,hours):
        if arm not in ('control','annealed','adaptive'):raise ValueError(arm)
        self.arm=arm;self.seconds=hours*3600;self.elapsed=0.;self.gaps={};self.observations={}

    def key(self,batch):
        return hashlib.sha256(json.dumps(dict(cfg=asdict(batch.cfg),H=batch.H,W=batch.W),sort_keys=True).encode()).hexdigest()[:16]

    def reserve(self,batch):
        # Full exploration reserve for first third, anneal over second third, zero thereafter.
        factor=float(np.clip(2-3*self.elapsed/self.seconds,0,1))
        gap=.1 if self.arm=='annealed' else self.gaps.get(self.key(batch),.1)
        return 0. if self.arm=='control' else factor*gap

    def update(self,batch,grid,exact):
        values=np.maximum(0,np.asarray(exact)-np.asarray(grid))
        if not len(values):return
        if not np.isfinite(values).all():raise ValueError('Nonfinite certification gap')
        key=self.key(batch);estimate=float(np.clip(np.quantile(values,.9),0,.1))
        self.gaps[key]=.8*self.gaps.get(key,.1)+.2*estimate
        self.observations[key]=self.observations.get(key,0)+len(values)

    def state(self):return dict(arm=self.arm,seconds=self.seconds,elapsed=self.elapsed,gaps=self.gaps,observations=self.observations)

    def restore(self,state):
        if not state:return
        if state['arm']!=self.arm or state['seconds']!=self.seconds:raise ValueError('Curriculum resume mismatch')
        self.elapsed=state['elapsed'];self.gaps=state['gaps'];self.observations=state['observations']
