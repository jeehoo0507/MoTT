"""Bounded CPU-only spawn workers for the unchanged exact impedance checker."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from .safe_verify import _cfg_dict, _job, pin_threads
from .verify import _inst_data


def _worker_init():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


class ExactVerifier:
    def __init__(self,workers):
        if not 1<=workers<=4:raise ValueError('Use 1 to 4 CPU workers')
        self.workers=workers;self.pool=None
        self.profile=dict(workers=workers,start_method='spawn',calls=0,jobs=0,
                          parallel_calls=0,wall_seconds=0.)

    def check_jobs(self,jobs):
        start=time.monotonic();pin_threads()
        if self.workers==1 or len(jobs)<=1:
            result=[_job(j) for j in jobs]
        else:
            if self.pool is None:
                # Spawn fresh interpreters: never fork the trainer's CUDA/BLAS state.
                for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
                    os.environ[key]='1'
                self.pool=ProcessPoolExecutor(max_workers=self.workers,
                    mp_context=mp.get_context('spawn'),initializer=_worker_init)
            result=list(self.pool.map(_job,jobs,chunksize=1))
            self.profile['parallel_calls']+=1
        self.profile['calls']+=1;self.profile['jobs']+=len(jobs)
        self.profile['wall_seconds']+=time.monotonic()-start
        return result

    @torch.no_grad()
    def __call__(self,batch,inst,occ,cfg,tol=1e-3,k_peaks=3,rel=3e-3,span=3,workers=None):
        if workers is not None and workers!=self.workers:raise ValueError('Worker count must stay fixed')
        indices=inst.cpu().numpy();placements=occ.cpu().numpy();config=_cfg_dict(cfg)
        jobs=[]
        for e,b in enumerate(indices):
            mask,entries,probes=_inst_data(batch,b)
            jobs.append((mask,entries,probes,placements[e],config,batch.zt_par[b:b+1],tol,k_peaks,rel,span))
        result=self.check_jobs(jobs)
        add=[[] for _ in range(batch.B)]
        for e,b in enumerate(indices):add[b]+=result[e][2]
        return np.array([r[0] for r in result]),np.array([r[1] for r in result]),add

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True,cancel_futures=True);self.pool=None

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
