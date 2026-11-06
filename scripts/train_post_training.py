"""Matched RL adapter; all arms use original-target rewards and zero-reserve validation."""
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import train as base
from decap_loop.parallel_verify import ExactVerifier
from decap_loop.pkg_guard_env import GuardedPkEnv
from decap_loop.post_training import PostTraining,ARMS
from decap_loop.sim import _probe_diag,violation,FEAS_TOL
from decap_loop.verify import grid_worst


def original_goal_outcome(env,result):
    v,m=violation(_probe_diag(env.z_ip,env.b.probes[env.inst]),torch.log10(env.b.Zt[env.inst]))
    result.update(V=v,m=m,feasible=m<=FEAS_TOL)
    return result


def action_counts(initial,actions,n_types):
    results=[]
    for occ,sequence in zip(initial.cpu().tolist(),actions.cpu().tolist()):
        row=dict(add=0,remove=0,replace=0,stop=0)
        for action in sequence:
            if action<0:continue
            if action==len(occ)*(n_types+1):row['stop']+=1;continue
            cell,kind=divmod(action,n_types+1)
            if kind==n_types:row['remove']+=1;occ[cell]=-1
            else:
                row['add' if occ[cell]<0 else 'replace']+=1;occ[cell]=kind
        results.append(row)
    return results


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--post_arm',choices=ARMS,required=True)
    p.add_argument('--post_phase',type=int,required=True);p.add_argument('--post_state',default='');p.add_argument('--root_init',required=True)
    p.add_argument('--certify_workers',type=int,default=2)
    extra,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
    old_parse=base.parse;old_rollout=base.rollout;old_teacher=base.load_teacher;old_make_policy=base.make_policy
    teachers={};teacher_checked=set()
    args=None;curriculum=None;validating=False;validation_rows={};iteration=0;model=None;last_snapshot=-300.;phase_clock=None
    verifier=ExactVerifier(extra.certify_workers)
    def record(kind,**kw):
        folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'trace.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind,iteration=iteration,phase=extra.post_phase,stage=curriculum.stage,
                elapsed=time.monotonic()-phase_clock if phase_clock is not None else 0.,time=time.time(),**kw))+'\n')
    def snapshot(label):
        folder=Path(args.out)/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'{label}_iter{iteration:07d}_{time.time_ns()}.pt'
        completed=max(0,iteration-1) if label=='periodic' else iteration
        torch.save(dict(model=model.state_dict(),args=vars(args),iteration=iteration,iter=completed,
            completed_iterations=completed,phase=extra.post_phase),path)
        return str(path.relative_to(Path(args.out)))
    def make_policy(cfg,a):
        nonlocal model
        model=old_make_policy(cfg,a);return model
    def parse():
        nonlocal args,curriculum
        args=old_parse();args.post_arm=extra.post_arm;args.post_phase=extra.post_phase;args.root_init=extra.root_init;args.certify_workers=extra.certify_workers;args.reserve_decades=0.;args.force_final_validation=True
        if args.stage!='rl' or args.model!='mott' or not args.peak_refine or args.dedup_modes or args.imit!=0 or args.baseline!='group' or args.certify!=2:
            raise ValueError('Requires RL MoTT+P1, group baseline, imitation=0, certify=2')
        curriculum=PostTraining(extra.post_arm,extra.post_phase,args.hours if args.hours else 1.)
        if extra.post_state:curriculum.restore(json.loads(Path(extra.post_state).read_text()),transfer=True)
        return args
    def load_teacher(path,batch,dev,log):
        result=old_teacher(path,batch,dev,log);teachers[curriculum.key(batch)]=result
        return result
    def env_factory(batch,inst,cfg,feat_mode,**kw):
        reserve=0. if validating else curriculum.reserve(batch)
        if not validating:
            key=curriculum.key(batch);bank=curriculum.archive.get(key,{})
            missing=[i for i in sorted(set(inst.cpu().tolist())) if curriculum.wants_refinement(i)
                     and str(i) not in bank and (key,i) not in teacher_checked]
            if missing and key in teachers:
                teacher=teachers[key];missing=[i for i in missing if bool(teacher['ok'][i])]
                if missing:
                    indices=torch.tensor(missing,device=inst.device)
                    exact,_,_=verifier(batch,indices,teacher['occ'][indices],cfg)
                    curriculum.add(batch,indices,teacher['occ'][indices],exact)
                    curriculum.stats['teacher_checks']+=len(missing)
                    teacher_checked.update((key,i) for i in missing)
            kw['occ0']=curriculum.initial(batch,inst)
        return GuardedPkEnv(batch,inst,cfg,feat_mode=feat_mode,reserve_decades=reserve,**kw)
    def rollout(policy,env,**kw):
        start=env.count().clone();initial=env.occ.clone()
        result=old_rollout(policy,env,**kw);result["start_count"]=start
        # A sample can miss the reserved goal yet meet the actual goal: do not discard it.
        result=original_goal_outcome(env,result)
        if validating or iteration%25==0:
            moves=action_counts(initial,result['actions'],env.T)
            result['move_counts']=moves
            if not validating:record('rollout',start_count=start.cpu().tolist(),final_count=result['count'].cpu().tolist(),
                grid_margin=result['m'].cpu().tolist(),moves=moves,reserve=env.reserve_decades)
        return result
    def certify(batch,inst,occ,cfg,**kw):
        result=verifier(batch,inst,occ,cfg,**kw)
        if not validating:
            grid=grid_worst(batch,inst,occ,cfg)
            curriculum.feedback.update(batch,grid,result[0])
            curriculum.add(batch,inst,occ,result[0])
            record('certificate',instances=inst.cpu().tolist(),grid_margin=grid.tolist(),exact_margin=result[0].tolist(),
                gap=(result[0]-grid).tolist(),archive_stats=curriculum.stats)
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
                worst,frequency,_=certify(batch,idx,out['occ'],batch.cfg)
                ok=torch.from_numpy(worst<=base.TOL1).to(dev);joint=ok & ref['feasible']
                ratio=(out['count'][joint].float()/ref['count'][joint].clamp(min=1)).mean().item() if joint.any() else float('nan')
                result[name]=(ratio,1-ok.float().mean().item())
                validation_rows[name]=[dict(instance=i,success=bool(ok[i]),count=int(out['count'][i]),worst=float(worst[i]),max_count=batch.n,
                    grid_margin=float(out['m'][i]),gap=float(worst[i])-float(out['m'][i]),peak_hz=float(frequency[i]),moves=out['move_counts'][i]) for i in range(batch.B)]
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
        checkpoint=snapshot('validation')
        saved=Path(args.out)/checkpoint.replace('.pt','.json');saved.write_text(json.dumps(validation_rows,indent=2)+'\n')
        record('validation',checkpoint=checkpoint,selection_score=value,rows=validation_rows)
        path=Path(args.out)/'selection_score.json';best=json.loads(path.read_text())['score'] if path.exists() else float('inf')
        if value<best:
            (Path(args.out)/'validation_best.json').write_text(json.dumps(validation_rows,indent=2)+'\n')
            path.write_text(json.dumps(dict(score=value,iteration=iteration))+'\n')
        return value
    def step(it,elapsed):
        nonlocal iteration,last_snapshot,phase_clock
        if phase_clock is None:phase_clock=time.monotonic()-elapsed
        iteration=it;curriculum.iteration=it;curriculum.feedback.elapsed=elapsed
        if elapsed-last_snapshot>=300:
            record('checkpoint',checkpoint=snapshot('periodic'),stats=curriculum.stats);last_snapshot=elapsed
        if it%25==0:
            print('POST_TRAIN '+json.dumps(dict(stage=curriculum.stage,phase=curriculum.phase,count_weight=curriculum.count_weight,stats=curriculum.stats,elapsed=elapsed)),flush=True)
    def curriculum_reward(*a):return curriculum.reward(*a)
    base.parse=parse;base.make_policy=make_policy;base.load_teacher=load_teacher;base.make_env=env_factory;base.rollout=rollout;base.certify_parallel=certify;base.validate=validate
    base.validation_score=score;base.study_iteration_hook=step;base.study_reward_hook=curriculum_reward

    def restore_state(state):
        nonlocal iteration,phase_clock
        curriculum.restore(state);iteration=curriculum.iteration;phase_clock=time.monotonic()-curriculum.feedback.elapsed
    base.study_state_hook=lambda:curriculum.state();base.study_restore_hook=restore_state
    completed=False
    try:
        base.main()
        record('phase_end',checkpoint=snapshot('phase_end'),stats=curriculum.stats)
        completed=True
    finally:
        verifier.close()
        if args is not None and Path(args.out).exists():
            (Path(args.out)/'post_state.json').write_text(json.dumps(curriculum.state(),indent=2)+'\n')
            (Path(args.out)/'certify_profile.json').write_text(json.dumps(verifier.profile,indent=2)+'\n')
            if completed:(Path(args.out)/'phase_complete.json').write_text(json.dumps(dict(completed=time.time(),phase=extra.post_phase))+'\n')


if __name__=='__main__':main()
