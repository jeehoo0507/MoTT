"""Standalone matched memory-MoTT experiment. See docs/DELTA_MEMORY_PROTOCOL.md.
No main training/rollout changes; no server-1 assets, Git authentication, or FLA dependency.
"""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from decap_loop.bank import load_bank, save_bank
from decap_loop.baselines import greedy
from decap_loop.delta_memory import ARMS, MemoryMoTT, detach_state
from decap_loop.modal_env import make_env
from decap_loop.pdn import sample_batch
from decap_loop.techs import TECHS, DIFFICULTY
from decap_loop.verify import certify
from decap_loop.sim import _probe_diag


def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def save_torch(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specs(a):
    jobs=[]
    for ti,tech in enumerate(('pcb','pkg','ip')):
        for split,sizes,n,offset in (('train',(8,10),a.train_n,11000),('val',(10,),a.val_n,22000),
                                     ('eval',(10,16),a.eval_n,33000)):
            for size in sizes:
                jobs.append(dict(name=f'{split}_{tech}{size}_0',tech=tech,size=size,n=n,
                                 seed=734000+ti*100000+offset+size))
    return jobs


def sync(dev):
    if torch.device(dev).type=='cuda': torch.cuda.synchronize()


def paths(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    return out,out/'pool',out/'traces'


def physical_target(env):
    """Next loaded spectral envelope on 16 fixed log-frequency queries, working-grid labels."""
    zpp=_probe_diag(env.z_ip,env.b.probes[env.inst])
    levels=(torch.log10(zpp.abs()+1e-30)-env.logZt[...,None]).amax(-1).gather(1,env.forder)
    xf=torch.log10(env.freqs_t).gather(1,env.forder).contiguous()
    queries=torch.linspace(np.log10(env.cfg.f_min),np.log10(env.cfg.f_max),16,
                           device=xf.device,dtype=xf.dtype).expand(env.E,-1).contiguous()
    hi=torch.searchsorted(xf,queries).clamp(1,xf.shape[1]-1);lo=hi-1
    x0,x1=xf.gather(1,lo),xf.gather(1,hi)
    y0,y1=levels.gather(1,lo),levels.gather(1,hi)
    fraction=((queries-x0)/(x1-x0).clamp(min=1e-12)).clamp(0,1)
    return (y0+fraction*(y1-y0)).float().clamp(-6,6)


def prepare(a):
    out,pool,traces=paths(a); pool.mkdir(exist_ok=True); traces.mkdir(exist_ok=True)
    for job in specs(a):
        name=job['name']; path=pool/f'{name}.pkl'
        if not path.exists():
            src=Path(a.source_pool)/f'{name}.pkl' if a.source_pool else None
            if src and src.is_file():
                with src.open('rb') as f: d=pickle.load(f)
                if len(d['valid']) < job['n']: raise ValueError(f'Too few source instances: {src}')
                for key in ('valid','keepout','probes','entry','zt_par','freqs','shapes','targets'):
                    d[key]=d[key][:job['n']]
                d['results']={}
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f: pickle.dump(d,f)
                tmp.replace(path)
                source=dict(path=str(src.resolve()),sha256=digest(src),selection='first n, no outcome filtering')
            else:
                cfg=TECHS[job['tech']]
                batch=sample_batch(job['n'],job['size'],job['size'],cfg,np.random.default_rng(job['seed']),
                                   difficulty=DIFFICULTY[job['tech']])
                tmp=path.with_suffix('.tmp'); save_bank(tmp,batch); tmp.replace(path)
                source=dict(generated_seed=job['seed'],cutting_plane=False)
            save_json(path.with_suffix('.json'),dict(**job,source=source,sha256=digest(path)))
            print('POOL ready',name,flush=True)
        if not name.startswith('train'): continue
        target=traces/f'{name}.pt'
        if target.exists():
            existing=torch.load(target,weights_only=False)
            if existing.get('schema')==2 and existing['pool_sha256']==digest(path): continue
        batch,_=load_bank(path,'cpu')
        sequences=[]
        for i in range(batch.B):
            env=make_env(batch,torch.tensor([i]),batch.cfg,'mott_pk',max_steps=a.max_steps)
            obs=env.observe(); seq=[]
            while not bool(env.done.all()):
                legal=obs['mask'][:,:-1].reshape(1,env.n,env.T+1).clone()
                legal[...,:env.T] &= (env.occ<0)[...,None]  # identical greedy+prune teacher, no type swaps
                score=torch.where(obs['Vn']<=0,obs['mn']-100,obs['Vn']).clone()
                score[...,env.T]=obs['mn'][...,env.T]
                best,action=score.masked_fill(~legal,float('inf')).reshape(1,-1).min(1)
                action=torch.where(torch.isfinite(best),action,torch.full_like(action,env.stop_idx))
                row=dict(x=obs['x'].float().cpu(),valid=env.valid.cpu(),mask=obs['mask'].cpu(),a=action.cpu(),active=torch.ones(1,dtype=torch.bool))
                env.step(action); obs=env.observe()
                row['physics']=physical_target(env).cpu()
                seq.append(row)
            if seq: sequences.append(seq)
        if not sequences: raise ValueError(f'No teacher actions in {name}')
        save_torch(target,dict(schema=2,pool_sha256=digest(path),sequences=sequences))
        print('TRACES ready',name,len(sequences),flush=True)
        del batch
    save_json(out/'data_manifest.json',[dict(**j,sha256=digest(pool/f'{j["name"]}.pkl')) for j in specs(a)])


def move(row,device):
    return {k:v.to(device) for k,v in row.items()}


def fit_sequence(model,seq,optimizer,a,weight=None):
    """Replay in chronological order. TBPTT at chunk boundaries; never shuffle time steps."""
    state=None; optimizer.zero_grad(set_to_none=True); segment=0.; total=0.
    count=sum(int(s['active'].sum()) for s in seq)
    for t,row in enumerate(seq):
        s=move(row,a.device)
        logits,state=model.step(s['x'],s['valid'],state)
        logits=logits.masked_fill(~s['mask'],float('-inf'))
        lp=torch.log_softmax(logits,-1).gather(1,s['a'][:,None])[:,0]
        if weight is None:
            loss=-(lp*s['active']).sum()/max(1,count)
        else:
            loss=-(lp*s['active']*weight).sum()/len(weight)
        if model.use_aux:
            err=F.smooth_l1_loss(model.physics(state,s['a']),s['physics'],reduction='none').mean(-1)
            loss=loss+a.aux_weight*(err*s['active']).sum()/max(1,count)
        segment=segment+loss
        state=model.advance(state,s['a'])
        if (t+1)%a.tbptt==0 or t+1==len(seq):
            if not torch.isfinite(segment): raise ValueError('Non-finite training loss')
            segment.backward(); total+=float(segment.detach()); segment=0.
            state=detach_state(state)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return total


@torch.no_grad()
def episode(model,batch,indices,a,sample=False,store=False,generator=None):
    env=make_env(batch,indices,batch.cfg,'mott_pk',max_steps=a.max_steps)
    state=None; seq=[]; model_seconds=0.; steps=0; encoder_calls=0; physical_errors=[]; auxiliary_seconds=0.
    sync(a.device); started=time.perf_counter(); obs=env.observe()
    while not bool(env.done.all()):
        active=~env.done
        mask=obs['mask'].clone(); mask[~active]=False; mask[~active,env.stop_idx]=True
        sync(a.device); begin=time.perf_counter()
        logits,state=model.step(obs['x'],env.valid,state)
        logits=logits.masked_fill(~mask,float('-inf'))
        action=(torch.multinomial(torch.softmax(logits,-1),1,generator=generator)[:,0] if sample else logits.argmax(-1))
        predicted_state=state
        state=model.advance(state,action)
        sync(a.device); model_seconds+=time.perf_counter()-begin
        aux_begin=time.perf_counter()
        predicted=model.physics(predicted_state,action) if model.use_aux else None
        sync(a.device); auxiliary_seconds+=time.perf_counter()-aux_begin
        encoder_calls+=int(model.kind=='mott' or steps==0)
        if store:
            row=dict(x=obs['x'].float().cpu(),valid=env.valid.cpu(),mask=mask.cpu(),a=action.cpu(),active=active.cpu())
        env.step(action); obs=env.observe(); steps+=1
        if predicted is not None:
            aux_begin=time.perf_counter()
            actual=physical_target(env)
            physical_errors.extend(((predicted-actual).square().mean(-1)[active]).tolist())
            sync(a.device); auxiliary_seconds+=time.perf_counter()-aux_begin
        if store:
            row['physics']=physical_target(env).cpu()
            seq.append(row)
    sync(a.device)
    return dict(occ=env.occ.clone(),count=env.count().float(),feasible=env.feas.clone(),V=obs['V'].float(),
                margin=obs['m'].float(),seq=seq,seconds=time.perf_counter()-started-auxiliary_seconds,model_seconds=model_seconds,auxiliary_seconds=auxiliary_seconds,
                steps=steps,encoder_calls=encoder_calls,physics_mse=float(np.mean(physical_errors)) if physical_errors else None)


@torch.no_grad()
def validate(model,a):
    was=model.training; model.eval(); vals=[]
    for path in sorted((Path(a.out)/'pool').glob('val_*.pkl')):
        batch,_=load_bank(path,a.device)
        r=episode(model,batch,torch.arange(batch.B,device=a.device),a)
        vals.extend((r['count']/50+20*(~r['feasible'])+10*r['V']).tolist())
        del batch
    model.train(was)
    return float(np.mean(vals))


def train(a):
    out,pool,traces=paths(a); folder=out/'models'/f'{a.arm}_s{a.seed}'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): return
    torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    rng=np.random.default_rng(1000+a.seed)
    gen=torch.Generator(device=a.device).manual_seed(2000+a.seed)
    model=MemoryMoTT(a.arm,d=a.d,layers=a.layers,heads=a.heads).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.)
    seqs=[]
    for path in sorted(traces.glob('train_*.pt')):
        d=torch.load(path,weights_only=False)
        if d.get('schema')!=2 or d['pool_sha256']!=digest(pool/f'{path.stem}.pkl'): raise ValueError('Trace/pool mismatch')
        seqs.extend(d['sequences'])
    if not seqs: raise ValueError('No training traces')
    bc,rl,best,elapsed=0,0,float('inf'),0.
    resume=folder/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,map_location=a.device,weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        bc,rl,best,elapsed=ck['bc'],ck['rl'],ck['best'],ck['elapsed']
        rng.bit_generator.state=ck['rng']; gen.set_state(ck['generator'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu())
        if a.device.startswith('cuda'): torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
    started=time.monotonic(); loss=None
    def checkpoint(select=False):
        nonlocal best
        if select:
            score=validate(model,a)
            if score<best:
                best=score
                save_torch(folder/'best.pt',dict(model=model.state_dict(),args=vars(a),score=score,bc=bc,rl=rl))
        save_torch(resume,dict(model=model.state_dict(),optimizer=opt.state_dict(),bc=bc,rl=rl,best=best,
            elapsed=elapsed+time.monotonic()-started,rng=rng.bit_generator.state,generator=gen.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []))
        print(json.dumps(dict(arm=a.arm,seed=a.seed,bc=bc,rl=rl,loss=loss,best=best,elapsed=elapsed+time.monotonic()-started)),flush=True)
    model.train()
    while bc<a.bc_updates:
        loss=fit_sequence(model,seqs[int(rng.integers(len(seqs)))],opt,a); bc+=1
        if bc%a.checkpoint_every==0: checkpoint()
    if rl==0: checkpoint(select=True)
    train_paths=sorted(pool.glob('train_*.pkl')); current=None; batch=None
    while rl<a.rl_updates:
        path=train_paths[int(rng.integers(len(train_paths)))]
        if path!=current:
            del batch
            batch,_=load_bank(path,a.device); current=path
        i=int(rng.integers(batch.B)); indices=torch.full((a.group,),i,device=a.device,dtype=torch.long)
        r=episode(model,batch,indices,a,sample=True,store=True,generator=gen)
        reward=-(r['count']/50+20*(~r['feasible'])+10*r['V'])
        advantage=(reward-reward.mean())/reward.std(unbiased=False).clamp(min=1e-6)
        loss=fit_sequence(model,r['seq'],opt,a,weight=advantage.detach()); rl+=1
        if rl%a.rl_val_every==0 or rl==a.rl_updates: checkpoint(select=True)
        elif rl%a.checkpoint_every==0: checkpoint()
    if not (folder/'best.pt').exists(): checkpoint(select=True)
    save_json(folder/'complete.json',dict(arm=a.arm,seed=a.seed,bc=bc,rl=rl,best=best,
        parameters=sum(p.numel() for p in model.parameters()),memory_elements=0 if model.kind in ('mott','cached') else a.d*a.d,
        train_seconds=elapsed+time.monotonic()-started,args=vars(a),peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0))


def evaluate(a):
    out,pool,_=paths(a); models=sorted((out/'models').glob('*/complete.json'))
    for path in sorted(pool.glob('eval_*.pkl')):
        batch,_=load_bank(path,a.eval_device)
        old=a.device; a.device=a.eval_device
        try:
            for marker in models:
                dest=out/'eval'/f'{path.stem}__{marker.parent.name}.json'
                if dest.exists(): continue
                ck=torch.load(marker.parent/'best.pt',map_location=a.device,weights_only=False)
                args=ck['args']; model=MemoryMoTT(args['arm'],d=args['d'],layers=args['layers'],heads=args['heads']).to(a.device).eval()
                rows=[]
                for i in range(batch.B):
                    indices=torch.tensor([i],device=a.device)
                    r=episode(model,batch,indices,a)
                    begin=time.perf_counter(); worst,_,_=certify(batch,indices,r['occ'],batch.cfg)
                    rows.append(dict(instance=i,success=bool(worst[0]<=np.log10(1.01)),worst=float(worst[0]),
                        count=int(r['count'][0]),steps=r['steps'],encoder_calls=r['encoder_calls'],physics_mse=r['physics_mse'],
                        policy_seconds=r['model_seconds'],placement_seconds=r['seconds'],
                        certification_seconds=time.perf_counter()-begin,auxiliary_diagnostic_seconds=r['auxiliary_seconds'],occ=r['occ'][0].cpu().tolist()))
                save_json(dest,dict(pool=path.stem,pool_sha256=digest(path),arm=args['arm'],seed=args['seed'],
                    checkpoint_sha256=digest(marker.parent/'best.pt'),device=a.device,rows=rows,
                    note='Single working-grid rollout + independent 700-point refined certification; no cutting-plane repair'))
                print('EVAL ready',dest.name,flush=True)
                del model
            # Same-instance nonlearning reference, separate from the learned-arm paired contrasts.
            dest=out/'eval'/f'{path.stem}__greedy.json'
            if not dest.exists():
                rows=[]
                for i in range(batch.B):
                    idx=torch.tensor([i],device=a.device); begin=time.perf_counter()
                    r=greedy(batch,batch.cfg,inst=idx,max_steps=a.max_steps); sec=time.perf_counter()-begin
                    worst,_,_=certify(batch,idx,r['occ'],batch.cfg)
                    rows.append(dict(instance=i,success=bool(worst[0]<=np.log10(1.01)),worst=float(worst[0]),
                                     count=int(r['count'][0]),placement_seconds=sec))
                save_json(dest,dict(pool=path.stem,pool_sha256=digest(path),arm='greedy',seed=None,rows=rows))
        finally: a.device=old
        del batch


def report(a):
    import csv
    out,_,_=paths(a); rows=[]; groups={}
    for path in sorted((out/'eval').glob('*.json')):
        d=json.loads(path.read_text()); tech=''.join(c for c in d['pool'].split('_')[1] if c.isalpha())
        size=int(''.join(c for c in d['pool'].split('_')[1] if c.isdigit()))
        key=(tech,size,d['arm']); groups.setdefault(key,[]).append(d)
        ok=[r for r in d['rows'] if r['success']]
        # Count is conditional on success; pairwise count comparisons below use joint successes only.
        rows.append(dict(pool=d['pool'],arm=d['arm'],seed=d['seed'],n=len(d['rows']),
            success_rate=len(ok)/len(d['rows']),count_success=np.mean([r['count'] for r in ok]).item() if ok else None,
            placement_seconds=np.mean([r['placement_seconds'] for r in d['rows']]).item(),
            policy_seconds=np.mean([r['policy_seconds'] for r in d['rows']]).item() if d['arm']!='greedy' else None,
            physics_mse=float(np.mean([r['physics_mse'] for r in d['rows'] if r.get('physics_mse') is not None]))
                if any(r.get('physics_mse') is not None for r in d['rows']) else None))
    out.mkdir(exist_ok=True)
    if rows:
        with (out/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    contrasts=[]
    pairs=(('cached','mott'),('gru','cached'),('delta','cached'),('delta','gru'),
           ('delta_aux','delta'),('gru_aux','gru'),('delta_aux','gru_aux'),('delta_aux','mott'))
    for tech,size in sorted({(k[0],k[1]) for k in groups}):
        for treated,base in pairs:
            left={v['seed']:v for v in groups.get((tech,size,treated),[])}
            right={v['seed']:v for v in groups.get((tech,size,base),[])}
            per_seed=[]; differences=[]
            for seed in sorted(left.keys()&right.keys()):
                l,r=left[seed],right[seed]
                assert l['pool_sha256']==r['pool_sha256']
                paired=list(zip(l['rows'],r['rows']))
                assert all(x['instance']==y['instance'] for x,y in paired) and len(l['rows'])==len(r['rows'])
                diff=np.array([int(x['success'])-int(y['success']) for x,y in paired]);differences.append(diff)
                jointly=[(x,y) for x,y in paired if x['success'] and y['success']]
                per_seed.append(dict(seed=seed,success_difference=float(diff.mean()),
                    rescued=int((diff>0).sum()),harmed=int((diff<0).sum()),joint_success_n=len(jointly),
                    count_difference=float(np.mean([x['count']-y['count'] for x,y in jointly])) if jointly else None,
                    placement_time_ratio=float(np.mean([x['placement_seconds'] for x,y in paired])/np.mean([y['placement_seconds'] for x,y in paired]))))
            ci=None
            if len(differences)>=3:
                # Crossed bootstrap: resample seeds and common board IDs, retaining paired arms.
                arr=np.stack(differences);rng=np.random.default_rng(734);boots=[]
                for _ in range(2000):
                    si=rng.integers(arr.shape[0],size=arr.shape[0]);bi=rng.integers(arr.shape[1],size=arr.shape[1])
                    boots.append(arr[si][:,bi].mean())
                ci=np.quantile(boots,[.025,.975]).tolist()
            contrasts.append(dict(tech=tech,size=size,treated=treated,base=base,per_seed=per_seed,
                                  exploratory_crossed_bootstrap_95ci=ci))
    save_json(out/'paired_comparisons.json',dict(contrasts=contrasts,note='Exploratory pilot; 3 seeds and few boards give limited precision. No automatic main-model replacement.'))
    text=['# Delta memory MoTT pilot', '',
        'Exploratory short-budget comparison. Exact action effects/masks retained; no simulator replacement claim.',
        'Success: 700-point + peak-refined certification. Times below: placement only, excluding final certification.',
        'Do not merge these results with main MoTT results. Paired rescue/harm/count contrasts: paired_comparisons.json.',
        '', '| Type | Side | Arm | Seeds complete | Mean success | Placement s/board |',
        '|---|---:|---|---:|---:|---:|']
    for (tech,size,arm),items in sorted(groups.items()):
        rate=np.mean([np.mean([r['success'] for r in d['rows']]) for d in items])
        seconds=np.mean([np.mean([r['placement_seconds'] for r in d['rows']]) for d in items])
        text.append(f'| {tech} | {size} | {arm} | {len(items)} | {rate:.3f} | {seconds:.4f} |')
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    archive=out.parent/(out.name+'_results.tgz')
    with tarfile.open(archive,'w:gz') as tar:
        for pattern in ('*.json','*.csv','*.md','eval/*.json','models/*/complete.json','models/*/best.pt','logs/*.log','pool/*.pkl','pool/*.json'):
            for path in out.glob(pattern):tar.add(path,arcname=str(path.relative_to(out.parent)))
    print('REPORT',out/'summary.csv','ARCHIVE',archive,flush=True)


def old_training_complete(root):
    names=('r2pk','r2pk_s1','r2pk_s2','r3pk','r3pk_s1','r3pk_s2')
    for name in names:
        folder=Path(root)/'runs/2026-11-05fresh'/f'{name}_rl'
        try: metrics=json.loads((folder/'metrics.json').read_text())
        except (OSError,ValueError):return False
        if not isinstance(metrics,dict) or not (folder/'best.pt').is_file():return False
    return True


def active_old_runner():
    if not Path('/proc').exists():return False
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=[x.decode(errors='replace') for x in p.read_bytes().split(b'\0') if x]
        except OSError:continue
        if any(Path(x).name=='server2_once.py' for x in args) and 'run' in args:
            # Once all six GPU runs have completed, the old runner only schedules CPU evals.
            # It is safe to release the GPU gate without terminating that runner.
            try: root=(p.parent/'cwd').resolve(strict=True)
            except OSError:return True
            if '--fresh' not in args or not old_training_complete(root):return True
    return False


def restart_waiting(a):
    """Stop only this checkout's idle launcher so a new version can be started safely."""
    status=Path(a.out)/'status.json'
    info=json.loads(status.read_text())
    if info.get('stage')!='waiting_existing_server2_run':
        raise RuntimeError('Launcher is not waiting; no process stopped')
    pid=int(info['pid']);proc=Path(f'/proc/{pid}')
    args=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if (proc/'cwd').resolve()!=ROOT or 'run' not in args or not any(Path(x).name=='server2_delta.py' for x in args):
        raise RuntimeError('PID identity mismatch; no process stopped')
    os.kill(pid,signal.SIGSTOP)
    try:
        for _ in range(100):
            state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
            if state=='T':break
            time.sleep(.02)
        else:raise RuntimeError('Launcher did not pause; no termination attempted')
        if json.loads(status.read_text()).get('stage')!='waiting_existing_server2_run' or (proc/f'task/{pid}/children').read_text().strip():
            raise RuntimeError('Launcher started work; resuming it without termination')
        os.kill(pid,signal.SIGTERM)
    finally:
        os.kill(pid,signal.SIGCONT)
    for _ in range(100):
        if not proc.exists():break
        state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        if state=='Z':break
        time.sleep(.05)
    else:raise RuntimeError('Waiter did not exit; do not launch a duplicate')
    print(f'Stopped idle Delta launcher {pid} only. Original FRESH runner untouched.',flush=True)


def run(a):
    out,_,_=paths(a);(out/'logs').mkdir(exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Never share the GPU with the original fresh experiment.
        while active_old_runner():
            save_json(out/'status.json',dict(stage='waiting_existing_server2_run',pid=os.getpid()))
            print('WAIT original server-2 GPU training not complete; nothing interrupted',flush=True)
            time.sleep(60)
        if a.device.startswith('cuda'):
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            free,total=torch.cuda.mem_get_info()
            if free<12*2**30:raise RuntimeError('Less than 12 GiB GPU headroom; existing jobs untouched. Retry later.')
        config={k:v for k,v in vars(a).items() if k not in ('stage','arm','seed')}
        config['arms']=list(ARMS); config['git']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        config['protocol_sha256']=digest(ROOT/'docs/DELTA_MEMORY_PROTOCOL.md')
        config['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None)
        frozen=out/'config.json'
        if frozen.exists() and json.loads(frozen.read_text())!=config:
            raise ValueError('Configuration changed; use a new --out directory')
        save_json(frozen,config)
        # Separate child processes release all model/optimizer/GPU allocations between arms.
        def execute(stage,arm='mott',seed=0):
            cmd=[sys.executable,str(Path(__file__).resolve()),stage]
            for k,v in vars(a).items():
                if k in ('stage','arm','seed'):continue
                if v is not None:cmd.extend(['--'+k,str(v)])
            cmd.extend(['--arm',arm,'--seed',str(seed)])
            save_json(out/'status.json',dict(stage=stage,arm=arm,seed=seed,pid=os.getpid(),final=False))
            print('START',stage,arm,seed,flush=True)
            with (out/'logs'/f'{stage}_{arm}_s{seed}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    rc=child.wait()
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM); child.wait(); raise
            if rc:raise RuntimeError(f'{stage} {arm} s{seed} failed rc={rc}; see logs')
        execute('prepare')
        for seed in map(int,a.seeds.split(',')):
            # Rotate order across seeds to reduce fixed order/cache/thermal confounding.
            arms=list(ARMS);shift=seed%len(arms);arms=arms[shift:]+arms[:shift]
            for arm in arms:execute('train',arm,seed)
            execute('evaluate');report(a)
        save_json(out/'status.json',dict(stage='complete',pid=os.getpid(),final=True))


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','prepare','train','evaluate','report','status','restart_waiting'])
    defaults=dict(out='runs/delta_memory_v1',source_pool='',device='cuda',eval_device='cpu',seeds='0,1,2',
        arm='mott',seed=0,d=64,layers=3,heads=4,train_n=6,val_n=4,eval_n=6,max_steps=64,
        bc_updates=300,rl_updates=30,group=4,lr=.0003,aux_weight=.1,tbptt=8,checkpoint_every=25,rl_val_every=15)
    for key,value in defaults.items():p.add_argument('--'+key,type=type(value),default=value)
    a=p.parse_args()
    if a.bc_updates<1 or a.group<2 or min(a.train_n,a.val_n,a.eval_n)<1: p.error('Invalid sample/update/group count')
    return a


if __name__=='__main__':
    torch.set_num_threads(1)
    a=parse()
    if a.stage=='status':
        path=Path(a.out)/'status.json';print(path.read_text() if path.exists() else 'Not started')
    else:
        try:
            globals()[a.stage](a)
        except Exception as error:
            if a.stage=='run':
                save_json(Path(a.out)/'status.json',dict(stage='failed',final=False,pid=os.getpid(),error=str(error)))
                try: report(a)
                except Exception: pass
            raise
