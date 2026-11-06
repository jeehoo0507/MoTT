"""Standalone adapter around the original trainer. Never patches original files on disk."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import train as base
from decap_loop.pkg_guard_env import GuardedPkEnv
from decap_loop.safe_verify import certify_parallel


def main():
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--reserve_decades',type=float,default=0.)
    extra,rest=ap.parse_known_args();sys.argv=[sys.argv[0]]+rest
    original_parse=base.parse;original_env=base.make_env
    def parse():
        a=original_parse();a.reserve_decades=extra.reserve_decades
        if a.model!='mott' or not a.peak_refine or a.dedup_modes:raise ValueError('This study requires baseline MoTT + P1')
        return a
    def env_factory(batch,inst,cfg,feat_mode,**kw):
        if feat_mode=='mott_pk':
            return GuardedPkEnv(batch,inst,cfg,feat_mode=feat_mode,reserve_decades=extra.reserve_decades,**kw)
        return original_env(batch,inst,cfg,feat_mode,**kw)
    def certify(*args,**kw):
        kw['workers']=1
        return certify_parallel(*args,**kw)
    @torch.no_grad()
    def validate(policy,val_sets,args,cfg,dev):
        # Identical checkpoint selection for both arms: ORIGINAL target, independent exact check.
        policy.eval();result={}
        for name,(batch,ref) in val_sets.items():
            indices=torch.arange(batch.B,device=dev)
            env=env_factory(batch,indices,batch.cfg,args.feat,max_steps=args.max_steps,allow_prune=True)
            out=base.rollout(policy,env,greedy=True,amp=not args.no_amp)
            worst,_,_=certify(batch,indices,out['occ'],batch.cfg)
            feasible=torch.from_numpy(worst<=base.TOL1).to(dev)
            jointly=feasible & ref['feasible']
            ratio=(out['count'][jointly].float()/ref['count'][jointly].clamp(min=1).float()).mean().item() if jointly.any() else float('nan')
            result[name]=(ratio,1-feasible.float().mean().item())
        policy.train();return result
    base.parse=parse;base.make_env=env_factory;base.certify_parallel=certify;base.validate=validate
    base.main()


if __name__=='__main__':main()
