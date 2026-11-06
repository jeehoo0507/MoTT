"""Server-1 independent fixed-budget study: frozen / continued baseline / robust MoTT.
Uses only new output files, copied inputs, one GPU job or one CPU evaluation at a time.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from decap_loop.pdn import PDNBatch,PDNConfig,ZBank,build_Y0,target_at
from decap_loop.pkg_guard_env import GuardedPkEnv
from decap_loop.rollout import rollout
from decap_loop.verify import certify
from train import make_policy


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def single(d,i):
    cfg=PDNConfig(**d['cfg']);h,w=d['H'],d['W']
    freqs=d['freqs'][i:i+1].copy();par=d['zt_par'][i:i+1].copy()
    probes=torch.from_numpy(d['probes'][i:i+1].copy())
    bank=ZBank([build_Y0(d['valid'][i].reshape(h,w),np.flatnonzero(d['entry'][i]),cfg,freqs[0])],probes,'cpu',True)
    return PDNBatch(h,w,torch.from_numpy(d['valid'][i:i+1].copy()),torch.from_numpy(d['keepout'][i:i+1].copy()),
        probes,torch.from_numpy(d['entry'][i:i+1].copy()),torch.from_numpy(target_at(par,freqs[0])),bank,
        [d['shapes'][i]],[d['targets'][i]],freqs=freqs,zt_par=par,cfg=cfg)


@torch.no_grad()
def evaluate_case(policy,batch,reserve,max_steps,rounds):
    idx=torch.tensor([0]);history=[];policy_seconds=0.;started=time.monotonic();occ=None
    for rnd in range(rounds+1):
        env=GuardedPkEnv(batch,idx,batch.cfg,feat_mode='mott_pk',reserve_decades=reserve,max_steps=max_steps)
        begin=time.monotonic();r=rollout(policy,env,greedy=True,amp=False)
        # Includes feature generation and rank-one updates: call this decoding time, not pure neural latency.
        policy_seconds+=time.monotonic()-begin;occ=r['occ']
        begin=time.monotonic();worst,frequency,add=certify(batch,idx,occ,batch.cfg)
        actual_grid_margin=float(r['m'][0])-reserve
        history.append(dict(round=rnd,working_feasible=bool(r['feasible'][0]),working_margin=float(r['m'][0]),
            original_grid_margin=actual_grid_margin,exact_margin=float(worst[0]),frequency=float(frequency[0]),
            count=int(r['count'][0]),steps=int((r['actions'][0]>=0).sum()),check_seconds=time.monotonic()-begin,
            stop_selected=bool(r['actions'].numel() and int(r['actions'][0,-1])==env.stop_idx)))
        # Same restart/point-add rule and cap for all arms, with the original physical target.
        bad=bool(r['feasible'][0]) and worst[0]>1e-3
        gap=bool(r['feasible'][0]) and not bad and worst[0]-actual_grid_margin>np.log10(1.05)
        if gap:
            add[0]+=list(float(frequency[0])*np.exp(.003*np.arange(-3,4)))
        if not (bad or gap) or rnd==rounds:break
        if batch.add_freqs(add)==0:break
    return dict(success=bool(worst[0]<=np.log10(1.01)),worst=float(worst[0]),count=int((occ[0]>=0).sum()),
        seconds=time.monotonic()-started,decoding_seconds=policy_seconds,history=history,occ=occ[0].tolist())


def evaluate(a):
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    with open(a.pool,'rb') as f:d=pickle.load(f)
    policy=make_policy(PDNConfig(**d['cfg']),ck['args']).eval();policy.load_state_dict(ck['model'])
    out=Path(a.result)
    identity=dict(pool=Path(a.pool).stem,pool_sha256=sha(a.pool),checkpoint_sha256=sha(a.ckpt),
        seed=a.seed,arm=a.arm,eval_reserve=a.eval_reserve,eval_n=a.eval_n,max_steps=a.max_steps,
        rounds=a.rounds,checker_points=d['cfg']['n_check'])
    result=json.loads(out.read_text()) if out.exists() else dict(**identity,rows=[],complete=False)
    if any(result.get(k)!=v for k,v in identity.items()):raise ValueError('Result input/settings mismatch')
    done={r['instance'] for r in result['rows']}
    for i in range(min(a.eval_n,len(d['valid']))):
        if i in done:continue
        begin=time.monotonic();batch=single(d,i);bank_seconds=time.monotonic()-begin
        row=evaluate_case(policy,batch,a.eval_reserve,a.max_steps,a.rounds)
        row['bank_seconds']=bank_seconds
        result['rows'].append(dict(instance=i,**row));save(out,result)
        print(json.dumps(dict(instance=i,success=row['success'],count=row['count'],seconds=row['seconds'])),flush=True)
    result['complete']=True;save(out,result)


def report(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);evaluations=[];flat=[]
    for path in sorted((out/'eval').glob('*.json')):
        d=json.loads(path.read_text());evaluations.append(d)
        if not d['rows']:continue
        success=[r for r in d['rows'] if r['success']]
        flat.append(dict(pool=d['pool'],seed=d['seed'],arm=d['arm'],reserve=d['eval_reserve'],complete=d['complete'],
            n=len(d['rows']),success_rate=len(success)/len(d['rows']),count_success=sum(r['count'] for r in success)/len(success) if success else None,
            mean_seconds=sum(r['seconds'] for r in d['rows'])/len(d['rows']),
            mean_bank_seconds=sum(r.get('bank_seconds',0.) for r in d['rows'])/len(d['rows'])))
    if flat:
        with (out/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    contrasts=[]
    pairs=[(('robust',a.reserve),('baseline',0.)),(('robust',a.reserve),('frozen',0.)),
           (('robust',a.reserve),('baseline',a.reserve)),
           (('robust',0.),('baseline',0.)),(('frozen',a.reserve),('frozen',0.)),
           (('baseline',0.),('frozen',0.))]
    for pool in sorted({d['pool'] for d in evaluations}):
        for treated,control in pairs:
            seeds=[]
            for seed in map(int,a.seeds.split(',')):
                left=next((d for d in evaluations if d['pool']==pool and d['seed']==seed and (d['arm'],d['eval_reserve'])==treated),None)
                right=next((d for d in evaluations if d['pool']==pool and d['seed']==seed and (d['arm'],d['eval_reserve'])==control),None)
                if not left or not right:continue
                assert left['pool_sha256']==right['pool_sha256']
                l={r['instance']:r for r in left['rows']};r={r['instance']:r for r in right['rows']}
                paired=[(l[i],r[i]) for i in sorted(l.keys()&r.keys())]
                if not paired:continue
                joint=[(x,y) for x,y in paired if x['success'] and y['success']]
                seeds.append(dict(seed=seed,n=len(paired),success_difference=sum(int(x['success'])-int(y['success']) for x,y in paired)/len(paired),
                    rescued=sum(x['success'] and not y['success'] for x,y in paired),harmed=sum(y['success'] and not x['success'] for x,y in paired),
                    joint_success_n=len(joint),joint_count_difference=sum(x['count']-y['count'] for x,y in joint)/len(joint) if joint else None))
            contrasts.append(dict(pool=pool,treated=list(treated),control=list(control),per_seed=seeds))
    save(out/'paired.json',contrasts)
    training=[]
    attempts=[json.loads(line) for line in (out/'attempts.jsonl').read_text().splitlines()] if (out/'attempts.jsonl').exists() else []
    for path in sorted((out/'models').glob('*/metrics.json')):
        d=json.loads(path.read_text());training.append(dict(model=path.parent.name,iterations=d['iters'],
            elapsed_hours=sum(e['seconds'] for e in attempts if e['job']=='train_'+path.parent.name)/3600 if attempts else d['hours'],
            best_iteration=d['best_iter'],oom_skipped=d.get('cuda_oom_skipped',0)))
    lines=['# MoTT working-constraint robustness pilot','',
        'Exploratory continued training; no architecture novelty or confirmed gain claimed. Original physical target/certifier unchanged.',
        'Paired results distinguish reserve-only decoding from training. Time includes all restarts and certification.',
        'Rows may be partial after resource/deadline stops; same board repeated across seeds is not a new board.',
        f'Primary package comparison complete: {(out/"primary_complete.json").exists()}.','',
        '| Pool | Seed | Model | Working reserve | n | Success | Successful count | Seconds/board | Complete |',
        '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for r in flat:
        count=f'{r["count_success"]:.2f}' if r['count_success'] is not None else 'NA'
        lines.append(f'| {r["pool"]} | {r["seed"]} | {r["arm"]} | {r["reserve"]} | {r["n"]} | {r["success_rate"]:.3f} | {count} | {r["mean_seconds"]:.2f} | {r["complete"]} |')
    lines+=['','Training: wall-clock matching does not imply equal optimizer updates.','',
        '| Model | Updates | Hours incl. loading/validation | Best update | OOM skipped |',
        '|---|---:|---:|---:|---:|']
    for d in training:
        lines.append(f'| {d["model"]} | {d["iterations"]} | {d["elapsed_hours"]:.3f} | {d["best_iteration"]} | {d["oom_skipped"]} |')
    lines+=['','Paired comparisons (treatment minus control; joint count uses boards both solve).','',
        '| Pool | Comparison | Seed | Paired n | Success difference (pp) | Rescued | Harmed | Joint n | Joint count difference |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in contrasts:
        label=f'{c["treated"][0]}@{c["treated"][1]} vs {c["control"][0]}@{c["control"][1]}'
        for s in c['per_seed']:
            diff=f'{s["joint_count_difference"]:.2f}' if s['joint_count_difference'] is not None else 'NA'
            lines.append(f'| {c["pool"]} | {label} | {s["seed"]} | {s["n"]} | {100*s["success_difference"]:.1f} | {s["rescued"]} | {s["harmed"]} | {s["joint_success_n"]} | {diff} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    archive=out.parent/(out.name+'_results.tgz');temp=archive.with_suffix('.tmp')
    with tarfile.open(temp,'w:gz') as tar:
        for pattern in ('*.json','*.jsonl','*.csv','*.md','eval/*.json','models/*/metrics.json','models/*/config.json'):
            for path in out.glob(pattern):tar.add(path,arcname=str(path.relative_to(out.parent)))
    temp.replace(archive)


def memory():
    p=Path('/sys/fs/cgroup/memory.current')
    return int(p.read_text())/2**30 if p.exists() else 0.


def old_pending(source):
    q=json.loads((source/'agent/day2_queue.json').read_text())
    return [j['name'] for j in q['train'] if not (source/f'runs/2026-10-29/{j["name"]}_rl/metrics.json').exists()
            and (not j.get('after') or (source/j['after']).exists())]


def gpu_pids():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    return [line.strip() for line in r.stdout.splitlines() if line.strip().isdigit()]


def stop_own_child(child):
    """Only for the Popen child we created with start_new_session=True."""
    if child.poll() is not None:return
    try:
        if os.getpgid(child.pid)!=child.pid:raise RuntimeError('Refuse to signal a shared process group')
        os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()


def inputs(a):
    out=Path(a.out);source=Path(a.source);dest=out/'inputs';pool=dest/'pool';teacher=dest/'teacher'
    pool.mkdir(parents=True,exist_ok=True);teacher.mkdir(exist_ok=True);manifest=[]
    required=sorted((source/'runs/pool_v1').glob('train_*.pkl'))
    required += [source/f'runs/pool_v1/val_{tech}10_0.pkl' for tech in ('pcb','pkg','ip')]
    required += [source/f'runs/pool_v1/eval_{tech}{size}_1.pkl' for tech in ('pcb','pkg','ip') for size in (10,16)]
    required += [source/f'runs/2026-10-29/r2pk{"" if seed==0 else "_s"+str(seed)}_rl/best.pt' for seed in map(int,a.seeds.split(','))]
    if not required or len(list((source/'runs/pool_v1').glob('train_*.pkl')))!=27:raise ValueError('Expected the original 27 training pools')
    for p in required:
        if not p.is_file():raise FileNotFoundError(p)
        if p.suffix=='.pkl':
            with p.open('rb') as f:metadata=pickle.load(f)
            if metadata['cfg']['n_check']!=700:raise ValueError(f'Original study requires a 700-point checker: {p}')
            target=pool/p.name
        else:target=dest/p.parent.name/p.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(p,target)
        if sha(target)!=sha(p):raise ValueError(f'Input changed since copy: {p}')
        manifest.append(dict(source=str(p),copy=str(target),sha256=sha(target)))
    for poolfile in sorted(pool.glob('train_*.pkl')):
        p=source/'data/teacher'/poolfile.with_suffix('.npz').name
        if not p.is_file():raise FileNotFoundError(p)
        target=teacher/p.name
        if not target.exists():shutil.copy2(p,target)
        if sha(target)!=sha(p):raise ValueError('Teacher changed')
        manifest.append(dict(source=str(p),copy=str(target),sha256=sha(target)))
    if len(list(teacher.glob('train_*.npz')))!=27:raise ValueError('Expected 27 existing teacher files; refuse expensive regeneration')
    save(out/'inputs.json',manifest)


def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    source=Path(a.source)
    if ROOT.resolve()==source.resolve() or out.resolve().is_relative_to(source.resolve()):
        raise ValueError('Run in a separate clone with output outside the original checkout')
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config={k:v for k,v in vars(a).items() if k not in ('stage','arm','seed','ckpt','pool','result','eval_reserve')}
        config.update(git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),protocol_sha256=sha(ROOT/'docs/PKG_GUARD_PROTOCOL.md'))
        configpath=out/'config.json'
        if configpath.exists() and json.loads(configpath.read_text())!=config:raise ValueError('Configuration changed; use another --out')
        save(configpath,config)
        statepath=out/'status.json'
        clockpath=out/'clock.json'
        if not clockpath.exists():save(clockpath,dict(started=time.time(),deadline=time.time()+a.wall_hours*3600))
        clock=json.loads(clockpath.read_text());start=time.monotonic()-(time.time()-clock['started'])
        inputs(a);report(a)
        env=dict(os.environ,CERTIFY_WORKERS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        def status(stage,**kw):
            save(statepath,dict(stage=stage,pid=os.getpid(),updated=time.time(),
                elapsed=time.monotonic()-start,deadline=clock['deadline'],final=stage=='complete',**kw))
            print(json.dumps(dict(stage=stage,elapsed_hours=round((time.monotonic()-start)/3600,3),**kw)),flush=True)
        def execute(cmd,tag,timeout):
            if memory()>28:status('waiting_memory',job=tag,memory_gib=memory());return 'wait'
            if time.monotonic()-start > a.wall_hours*3600-300:
                status('deadline_partial',job=tag);return 'deadline'
            status('running',job=tag,command=cmd)
            with (out/'logs'/f'{tag}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                t=time.monotonic();reason=None;last_gpu_check=t
                try:
                    while child.poll() is None:
                        if memory()>a.memory_limit:
                            reason='memory_limit';stop_own_child(child)
                            break
                        if time.monotonic()-t>timeout or time.monotonic()-start>a.wall_hours*3600:
                            reason='time_limit';stop_own_child(child)
                            break
                        if tag.startswith('train_') and time.monotonic()-last_gpu_check>15:
                            last_gpu_check=time.monotonic()
                            # This study has one CUDA process. Count rather than compare host/container PIDs.
                            if len(gpu_pids())>1 or old_pending(source):
                                reason='yielded_to_existing_gpu_job';stop_own_child(child);break
                        time.sleep(.5)
                except BaseException:
                    stop_own_child(child)
                    raise
            with (out/'attempts.jsonl').open('a') as audit:
                audit.write(json.dumps(dict(job=tag,seconds=time.monotonic()-t,returncode=child.returncode,reason=reason))+'\n')
            report(a)
            if reason in ('memory_limit','yielded_to_existing_gpu_job'):
                status('resource_wait_partial',job=tag,reason=reason);return 'wait'
            if child.returncode or reason:
                status('failed_or_partial',job=tag,rc=child.returncode,reason=reason);return 'failed'
            status('job_done',job=tag);return 'done'
        def execute_wait(cmd,tag,timeout):
            while True:
                if tag.startswith('train_') and (old_pending(source) or gpu_pids()):
                    status('waiting_existing_jobs',job=tag)
                    if time.monotonic()-start>a.wall_hours*3600:
                        status('deadline_partial',job=tag);return 'deadline'
                    time.sleep(30);continue
                r=execute(cmd,tag,timeout)
                if r!='wait':return r
                if time.monotonic()-start>a.wall_hours*3600:
                    status('deadline_partial',job=tag);return 'deadline'
                time.sleep(30)
        def eval_jobs(seed,arm,ckpt,primary,controls=False):
            kinds=('pkg',) if primary else ('pcb','ip')
            for size in (16,10):
                for tech in kinds:
                    pool=out/'inputs/pool'/f'eval_{tech}{size}_1.pkl'
                    reserves=(a.reserve,) if arm=='robust' else (0.,)
                    if controls:reserves=(0.,) if arm=='robust' else (a.reserve,)
                    for reserve in reserves:
                        tag=f'{arm}_s{seed}_{tech}{size}_g{reserve}'
                        result=out/'eval'/f'{tag}.json'
                        if result.exists() and json.loads(result.read_text()).get('complete'):continue
                        cmd=[sys.executable,__file__,'evaluate','--ckpt',str(ckpt),'--pool',str(pool),'--result',str(result),
                             '--arm',arm,'--seed',str(seed),'--eval_reserve',str(reserve),'--eval_n',str(a.eval_n),
                             '--rounds',str(a.rounds),'--max_steps',str(a.max_steps)]
                        if execute_wait(cmd,tag,a.eval_timeout)!='done':return False
            return True
        trained=[]
        for seed in map(int,a.seeds.split(',')):
            src=out/f'inputs/r2pk{"" if seed==0 else "_s"+str(seed)}_rl/best.pt'
            ck=torch.load(src,map_location='cpu',weights_only=False);args=ck['args']
            if args['model']!='mott' or args['feat']!='mott_pk':raise ValueError('Source must be original MoTT+P1')
            if not eval_jobs(seed,'frozen',src,True):return
            order=(('baseline',0.),('robust',a.reserve)) if seed%2==0 else (('robust',a.reserve),('baseline',0.))
            for arm,reserve in order:
                folder=out/'models'/f'{arm}_s{seed}'
                if not (folder/'metrics.json').exists():
                    while True:
                        pending=old_pending(source);active=gpu_pids()
                        if not pending and not active and memory()<=28:break
                        status('waiting_existing_jobs',pending_original_training=pending,gpu_pids=active,memory_gib=memory())
                        if time.monotonic()-start>a.wall_hours*3600:
                            status('deadline_partial',job=f'train_{arm}_s{seed}');report(a);return
                        time.sleep(30)
                    cmd=[sys.executable,'scripts/pkg_guard_train.py','--stage','rl','--model','mott','--peak_refine',
                         '--reserve_decades',str(reserve),'--init',str(src),'--out',str(folder),'--pool',str(out/'inputs/pool'),
                         '--teacher_dir',str(out/'inputs/teacher'),'--seed',str(seed),'--hours',str(a.train_hours),
                         '--iters','1000000','--B','4','--M','8','--chunk','1','--lr','0.0001','--lam','20',
                         '--imit','0','--certify','2','--val_every','50','--ckpt_min','5','--max_steps',str(a.max_steps)]
                    for key in ('d','layers','heads'):cmd+=['--'+key,str(args[key])]
                    for flag in ('fc_bias','mode_head'):
                        if args.get(flag):cmd+=['--'+flag]
                    if execute_wait(cmd,f'train_{arm}_s{seed}',a.train_hours*3600+2400)!='done':return
                    if not (folder/'metrics.json').is_file() or not (folder/'best.pt').is_file():
                        raise RuntimeError(f'Trainer returned without complete artifacts: {folder}')
                trained.append((seed,arm,folder/'best.pt'))
                if not eval_jobs(seed,arm,folder/'best.pt',True):return
        # Package comparison is published before any secondary evaluation.
        save(out/'primary_complete.json',dict(completed=time.time(),seeds=a.seeds,eval_n=a.eval_n))
        status('package_comparison_complete');report(a)
        for seed in map(int,a.seeds.split(',')):
            src=out/f'inputs/r2pk{"" if seed==0 else "_s"+str(seed)}_rl/best.pt'
            if not eval_jobs(seed,'frozen',src,True,controls=True):return
        for seed,arm,ckpt in trained:
            if not eval_jobs(seed,arm,ckpt,True,controls=True):return
        for seed in map(int,a.seeds.split(',')):
            src=out/f'inputs/r2pk{"" if seed==0 else "_s"+str(seed)}_rl/best.pt'
            if not eval_jobs(seed,'frozen',src,False):return
        for seed,arm,ckpt in trained:
            if not eval_jobs(seed,arm,ckpt,False):return
        status('complete');report(a)


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('run','evaluate','report','status'))
    defaults=dict(source=str(Path.home()/'R2E'),out=str(ROOT/'runs/pkg_guard_v1'),seeds='0,1,2',reserve=.1,
        train_hours=.75,wall_hours=12.,eval_n=8,rounds=4,max_steps=64,memory_limit=32.,eval_timeout=2400,
        arm='frozen',seed=0,ckpt='',pool='',result='',eval_reserve=0.)
    for key,value in defaults.items():p.add_argument('--'+key,type=type(value),default=value)
    a=p.parse_args()
    if (not 0<a.reserve<=.3 or a.eval_n<1 or a.wall_hours<=0 or a.train_hours<=0
        or not 28<a.memory_limit<=40 or a.max_steps<1 or a.rounds<0 or a.eval_timeout<1):p.error('Invalid study settings')
    seeds=list(map(int,a.seeds.split(',')))
    if len(set(seeds))!=len(seeds) or any(s not in (0,1,2) for s in seeds):p.error('Use unique source seeds 0,1,2')
    return a


if __name__=='__main__':
    torch.set_num_threads(1);a=parse()
    if a.stage=='status':
        path=Path(a.out)/'status.json';print(path.read_text() if path.exists() else 'Not started')
    else:
        if a.stage=='run':
            def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
            signal.signal(signal.SIGTERM,interrupted)
        try:globals()[a.stage](a)
        except BlockingIOError:
            raise SystemExit('This study already has a running launcher; no status or process changed')
        except (Exception,KeyboardInterrupt) as error:
            if a.stage=='run':
                save(Path(a.out)/'status.json',dict(stage='failed_or_interrupted',final=False,error=str(error),pid=os.getpid()))
                report(a)
            raise
        else:
            if a.stage=='run':report(a)
