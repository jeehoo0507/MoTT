"""Matched RL adapter; all arms use original-target rewards and zero-reserve validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import train as base
from decap_loop.parallel_verify import ExactVerifier
from decap_loop.pkg_guard_env import GuardedPkEnv
from decap_loop.reserve_curriculum import ReserveCurriculum
from decap_loop.sim import _probe_diag,violation,FEAS_TOL
from decap_loop.verify import grid_worst


def original_goal_outcome(env,result):
    v,m=violation(_probe_diag(env.z_ip,env.b.probes[env.inst]),torch.log10(env.b.Zt[env.inst]))
    result.update(V=v,m=m,feasible=m<=FEAS_TOL)
    return result


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--curriculum',choices=('control','annealed','adaptive'),required=True)
    p.add_argument('--certify_workers',type=int,default=2)
    extra,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
    old_parse=base.parse;old_rollout=base.rollout
    args=None;curriculum=None;validating=False;validation_rows={};iteration=0
    verifier=ExactVerifier(extra.certify_workers)
    def parse():
        nonlocal args,curriculum
        args=old_parse();args.curriculum=extra.curriculum;args.certify_workers=extra.certify_workers;args.reserve_decades=0.
        if args.stage!='rl' or args.model!='mott' or not args.peak_refine or args.dedup_modes or args.imit!=0 or args.baseline!='group' or args.certify!=2:
            raise ValueError('Requires RL MoTT+P1, group baseline, imitation=0, certify=2')
        curriculum=ReserveCurriculum(extra.curriculum,args.hours if args.hours else 1.)
        return args
    def env_factory(batch,inst,cfg,feat_mode,**kw):
        reserve=0. if validating else curriculum.reserve(batch)
        return GuardedPkEnv(batch,inst,cfg,feat_mode=feat_mode,reserve_decades=reserve,**kw)
    def rollout(policy,env,**kw):
        result=old_rollout(policy,env,**kw)
        # A sample can miss the reserved goal yet meet the actual goal: do not discard it.
        return original_goal_outcome(env,result)
    def certify(batch,inst,occ,cfg,**kw):
        result=verifier(batch,inst,occ,cfg,**kw)
        if not validating:curriculum.update(batch,grid_worst(batch,inst,occ,cfg),result[0])
        return result
    @torch.no_grad()
    def validate(policy,val_sets,a,cfg,dev):
        nonlocal validating,validation_rows
        validating=True;policy.eval();result={};validation_rows={}
        try:
            for name,(batch,ref) in val_sets.items():
                idx=torch.arange(batch.B,device=dev)
                env=env_factory(batch,idx,batch.cfg,a.feat,max_steps=a.max_steps,allow_prune=True)
                out=rollout(policy,env,greedy=True,amp=not a.no_amp)
                worst,_,_=certify(batch,idx,out['occ'],batch.cfg)
                ok=torch.from_numpy(worst<=base.TOL1).to(dev);joint=ok & ref['feasible']
                ratio=(out['count'][joint].float()/ref['count'][joint].clamp(min=1)).mean().item() if joint.any() else float('nan')
                result[name]=(ratio,1-ok.float().mean().item())
                validation_rows[name]=[dict(instance=i,success=bool(ok[i]),count=int(out['count'][i]),worst=float(worst[i]),max_count=batch.n) for i in range(batch.B)]
            folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
            (folder/'validation_last.json').write_text(json.dumps(validation_rows,indent=2)+'\n')
            return result
        finally:validating=False;policy.train()
    def score(values):
        # Failure count dominates all count variation; worst group is the second criterion.
        groups=list(validation_rows.values());failed=[sum(not r['success'] for r in rows) for rows in groups]
        size=sum(len(rows) for rows in groups)
        count=sum(r['count']/r['max_count'] for rows in groups for r in rows if r['success'])/max(1,size)
        value=sum(failed)*1e6+max(failed,default=0)*1000+count
        path=Path(args.out)/'selection_score.json';best=json.loads(path.read_text())['score'] if path.exists() else float('inf')
        if value<best:
            (Path(args.out)/'validation_best.json').write_text(json.dumps(validation_rows,indent=2)+'\n')
            path.write_text(json.dumps(dict(score=value,iteration=iteration))+'\n')
        return value
    def step(it,elapsed):
        nonlocal iteration
        iteration=it;curriculum.elapsed=elapsed
        if it%25==0:
            print('CURRICULUM '+json.dumps(curriculum.state()),flush=True)
    base.parse=parse;base.make_env=env_factory;base.rollout=rollout;base.certify_parallel=certify;base.validate=validate
    base.validation_score=score;base.study_iteration_hook=step
    base.study_state_hook=lambda:curriculum.state();base.study_restore_hook=lambda state:curriculum.restore(state)
    try:base.main()
    finally:
        verifier.close()
        if args is not None and Path(args.out).exists():
            (Path(args.out)/'curriculum_state.json').write_text(json.dumps(curriculum.state(),indent=2)+'\n')
            (Path(args.out)/'certify_profile.json').write_text(json.dumps(verifier.profile,indent=2)+'\n')


if __name__=='__main__':main()
