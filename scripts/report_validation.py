"""Frozen fresh-board report validation. CPU only, isolated outputs, Git model handoff."""
import argparse
import csv
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tarfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from decap_loop.bank import save_bank
from decap_loop.baselines import greedy,modal_heuristic
from decap_loop.modal_env import make_env
from decap_loop.pdn import sample_batch
from decap_loop.pkg_guard_env import GuardedPkEnv
from decap_loop.rollout import rollout
from decap_loop.techs import TECHS,DIFFICULTY
from decap_loop.verify import certify
from train import make_policy
from scripts.pkg_guard_study import save,sha,single,memory,stop_own_child
from scripts.report_model_bundle import receive,fetch_bundle

ROLES=('baseline','robust','frozen','park','manoharan','coord','modal','park_rlonly','manoharan_rlonly')
LABELS={'frozen':'Original MoTT + P1','baseline':'Continued MoTT + P1','robust':'MoTT + P1, reserved training/decoding',
        'gp':'Greedy + prune','rule':'Resonance rule','park':'B-Park (adapted AM)',
        'manoharan':'B-Manoharan (adapted, distill + RL)','park_rlonly':'B-Park (adapted, RL only)',
        'manoharan_rlonly':'B-Manoharan (adapted, RL only)','coord':'Coordinate model','modal':'Modal-feature model'}


def cases(n,large_n):
    cells=[('pkg',16),('pkg',10),('pcb',10),('ip',10),('pcb',16),('ip',16),('pkg',20),('pcb',20),('ip',20)]
    return [dict(id=f'{tech}{size}_{i:03d}',pool=f'{tech}{size}',tech=tech,size=size,index=i,
                 problem_seed=9100000+('pcb','pkg','ip').index(tech)*100000+size*1000+i)
            for tech,size in cells for i in range(large_n if size==20 else n)]


def prepare(a):
    case=json.loads(a.case);cfg=TECHS[case['tech']];torch.manual_seed(case['problem_seed'])
    b=sample_batch(1,case['size'],case['size'],cfg,np.random.default_rng(case['problem_seed']),difficulty=DIFFICULTY[case['tech']])
    path=Path(a.pool);path.parent.mkdir(parents=True,exist_ok=True)
    save_bank(path.with_suffix('.tmp'),b);path.with_suffix('.tmp').replace(path)
    save(path.with_suffix('.json'),dict(**case,sha256=sha(path),cfg=asdict(cfg),
         grid='bare/filled seeded working grid; no outcome-driven pool refinement'))


@torch.no_grad()
def evaluate(a):
    with open(a.pool,'rb') as f:d=pickle.load(f)
    begin=time.monotonic();b=single(d,0);bank_seconds=time.monotonic()-begin
    idx=torch.tensor([0]);pol=None;model_args={};weight_sha=None
    if a.ckpt:
        ck=torch.load(a.ckpt,map_location='cpu',weights_only=True);model_args=ck['args'];weight_sha=sha(a.ckpt)
        pol=make_policy(b.cfg,model_args).eval();pol.load_state_dict(ck['model'])
    reserve=.1 if a.role=='robust' else 0.;history=[];placement_seconds=0.;started=time.monotonic()
    for iteration in range(5):
        t=time.monotonic()
        if a.role=='gp':r=greedy(b,b.cfg,max_steps=64)
        elif a.role=='rule':r=modal_heuristic(b,b.cfg,max_steps=64)
        else:
            env=(GuardedPkEnv(b,idx,b.cfg,feat_mode='mott_pk',reserve_decades=reserve,max_steps=64)
                 if a.role=='robust' else make_env(b,idx,b.cfg,model_args['feat'],max_steps=64))
            r=rollout(pol,env,greedy=True,amp=False)
        placement_seconds+=time.monotonic()-t
        t=time.monotonic();worst,frequency,add=certify(b,idx,r['occ'],b.cfg)
        # Recompute ORIGINAL working-grid margin identically for every method.
        from decap_loop.verify import grid_worst
        margin=float(grid_worst(b,idx,r['occ'],b.cfg)[0])
        claimed=bool(r['feasible'][0]);bad=claimed and worst[0]>1e-3
        gap=claimed and not bad and worst[0]-margin>np.log10(1.05)
        history.append(dict(round=iteration,claimed=claimed,grid_margin=margin,exact_margin=float(worst[0]),
                            frequency=float(frequency[0]),check_seconds=time.monotonic()-t))
        if gap:add[0]+=list(float(frequency[0])*np.exp(.003*np.arange(-3,4)))
        if not (bad or gap) or iteration==4 or b.add_freqs(add)==0:break
    case=json.loads(a.case)
    save(a.result,dict(**case,role=a.role,seed=a.seed,pool_sha256=sha(a.pool),checkpoint_sha256=weight_sha,
        success=bool(worst[0]<=np.log10(1.01)),worst=float(worst[0]),count=int((r['occ'][0]>=0).sum()),
        bank_seconds=bank_seconds,placement_seconds=placement_seconds,total_seconds=time.monotonic()-started,
        occ=r['occ'][0].tolist(),history=history,checker_points=b.cfg.n_check))


def report(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(p.read_text()) for p in sorted((out/'eval').glob('*.json'))]
    manifest=json.loads((out/'artifacts/manifest.json').read_text()) if (out/'artifacts/manifest.json').exists() else {'models':[]}
    groups={}
    for r in rows:groups.setdefault((r['pool'],r['role'],r['seed']),[]).append(r)
    lines=['# 보고서용 새 문제 검증','',
        '부분 결과를 포함합니다. 같은 보드의 시드 반복을 독립적인 새 문제로 세지 않습니다.',
        '이 표는 새 평가 프로토콜의 결과이며 기존 RESULTS.md 수치와 합치지 않습니다.',
        'B-Park/B-Manoharan은 합성 환경에 맞춘 재구현입니다. 원 논문 성능 재현이나 DevFormer 비교가 아닙니다.',
        '개선/continued MoTT는 사전학습에 45분을 더한 예산입니다. 다른 학습 모델의 추가 예산은 같지 않습니다.',
        '시간은 같은 CPU 1스레드에서 측정하지만 기존 작업과 CPU를 공유할 수 있으므로 역사적 속도비와 비교하지 않습니다.','',
        '| 유형/크기 | 모델 | 시드 | 완료 문제 | 성공률 | 성공 시 부품 수 | 배치 초 | 검사 포함 초 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    flat=[]
    for (pool,role,seed),rs in sorted(groups.items()):
        ok=[r for r in rs if r['success']];count=float(np.mean([r['count'] for r in ok])) if ok else None
        row=dict(pool=pool,role=role,seed=seed,n=len(rs),success_rate=len(ok)/len(rs),count_success=count,
                 placement_seconds=float(np.mean([r['placement_seconds'] for r in rs])),total_seconds=float(np.mean([r['total_seconds'] for r in rs])))
        flat.append(row);c=f'{count:.2f}' if count is not None else 'NA'
        lines.append(f'| {pool} | {LABELS[role]} | {seed} | {len(rs)} | {row["success_rate"]:.3f} | {c} | {row["placement_seconds"]:.2f} | {row["total_seconds"]:.2f} |')
    if flat:
        with (out/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    paired=[]
    lines+=['','## 짝 비교','', '| 유형/크기 | 비교 | 시드 | 짝 수 | 성공 차 %p | 구제/악화 | 동시 성공 부품 차 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for pool in sorted({r['pool'] for r in rows}):
        for control in ('baseline','frozen','coord','modal','park','manoharan','gp','rule'):
            for seed in range(3):
                l={r['index']:r for r in groups.get((pool,'robust',seed),[])}
                rr={r['index']:r for r in groups.get((pool,control,-1 if control in ('gp','rule') else seed),[])}
                ids=sorted(l.keys()&rr.keys())
                if not ids:continue
                if any(l[i]['pool_sha256']!=rr[i]['pool_sha256'] for i in ids):raise ValueError('Unpaired data')
                joint=[i for i in ids if l[i]['success'] and rr[i]['success']]
                p=dict(pool=pool,control=control,seed=seed,n=len(ids),success_difference=float(np.mean([int(l[i]['success'])-int(rr[i]['success']) for i in ids])),
                       rescued=sum(l[i]['success'] and not rr[i]['success'] for i in ids),harmed=sum(rr[i]['success'] and not l[i]['success'] for i in ids),
                       joint_n=len(joint),joint_count_difference=float(np.mean([l[i]['count']-rr[i]['count'] for i in joint])) if joint else None)
                paired.append(p);diff='NA' if not joint else f'{p["joint_count_difference"]:.2f}'
                lines.append(f'| {pool} | robust − {control} | {seed} | {len(ids)} | {100*p["success_difference"]:.1f} | {p["rescued"]}/{p["harmed"]} | {diff} |')
    save(out/'paired.json',paired)
    available={m['id'] for m in manifest['models']}
    missing=[f'{role}_s{s}' for role in ROLES for s in range(3) if f'{role}_s{s}' not in available]
    errors=json.loads((out/'errors.json').read_text()) if (out/'errors.json').exists() else {}
    cfg=json.loads((out/'config.json').read_text()) if (out/'config.json').exists() else dict(n=a.n,large_n=a.large_n)
    primary_done=sum(r['role'] in ('frozen','baseline','robust') and r['size'] in (10,16) for r in rows)
    lines+=['',f'10/16칸 핵심 비교 완료: {primary_done}/{cfg["n"]*6*9}개 평가. 처리 오류/시간 초과: {len(errors)}개 작업.',
        '미수신 모델: '+(', '.join(missing) or '없음'),'',
        '전체 학습된 체크포인트의 새 보드 평가는 미학습 유형(leave-one-type-out) 실험이 아닙니다.',
        f'조건당 {cfg["n"]}개(20×20은 {cfg["large_n"]}개)와 시드 3개가 계획값이며, 표의 완료 문제 수가 실제 분모입니다.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    archive=out.parent/(out.name+'_results.tgz')
    with tarfile.open(archive.with_suffix('.tmp'),'w:gz') as tar:
        for pattern in ('*.json','*.csv','*.md','eval/*.json','pool/*.json','artifacts/manifest.json'):
            for p in out.glob(pattern):tar.add(p,arcname=str(p.relative_to(out.parent)))
    archive.with_suffix('.tmp').replace(archive)


def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=dict(n=a.n,large_n=a.large_n,hours=a.hours,inbox=str(Path(a.inbox).resolve()),transport=a.transport,
                 git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 protocol_sha256=sha(ROOT/'docs/REPORT_VALIDATION_PROTOCOL.md'))
        p=out/'config.json'
        if p.exists():
            frozen=json.loads(p.read_text())
            # Completion mode changes scheduling only; preserve the original provenance.
            keys=('n','large_n','inbox','transport','hours') if a.finish_all else tuple(cfg)
            if any(frozen.get(k)!=cfg[k] for k in keys):raise ValueError('Frozen configuration changed; use another output directory')
        else:save(p,cfg)
        if a.finish_all:
            save(out/'completion_policy.json',dict(finish_all=True,activated=time.time(),
                code_sha256=sha(__file__),original_config_sha256=sha(p),case_timeout=None))
        if not (out/'clock.json').exists():save(out/'clock.json',dict(deadline=time.time()+a.hours*3600))
        deadline=None if a.finish_all else json.loads((out/'clock.json').read_text())['deadline']
        def expired():return deadline is not None and time.time()>=deadline
        errors=json.loads((out/'errors.json').read_text()) if (out/'errors.json').exists() else {}
        if a.finish_all and errors:
            with (out/'retry_history.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),errors=errors))+'\n')
            errors={};save(out/'errors.json',errors)
        received=None;all_cases=cases(a.n,a.large_n);last_fetch=0.
        save(out/'case_plan.json',all_cases)
        def state(stage,**kw):
            save(out/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),deadline=deadline,**kw))
            print(json.dumps(dict(stage=stage,**kw),ensure_ascii=False),flush=True)
        def execute(cmd,key):
            while memory()>28 and not expired():
                state('waiting_memory',memory_gib=memory());time.sleep(30)
            if expired():return False
            state('running',job=key)
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CERTIFY_WORKERS='1',CUDA_VISIBLE_DEVICES='')
            with (out/'logs'/f'{key}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                start=time.monotonic();reason=None
                try:
                    while child.poll() is None:
                        if memory()>32:reason='memory';stop_own_child(child);break
                        if (not a.finish_all and time.monotonic()-start>900) or expired():reason='timeout';stop_own_child(child);break
                        time.sleep(.5)
                except BaseException:stop_own_child(child);raise
            if reason=='memory':time.sleep(30);return False
            if child.returncode or reason:
                errors[key]=dict(rc=child.returncode,reason=reason);save(out/'errors.json',errors)
            report(a);return True
        while not expired():
            if a.transport=='git' and time.time()-last_fetch>120:
                last_fetch=time.time()
                try:fetch_bundle(out/'download',a.inbox)
                except (subprocess.SubprocessError,OSError) as e:state('model_fetch_retry',error=str(e))
            inbox=Path(a.inbox)
            if inbox.exists() and sha(inbox)!=received:
                try:
                    receive(inbox,out/'artifacts');received=sha(inbox);state('models_received');report(a)
                except (tarfile.TarError,EOFError,OSError) as e:
                    state('waiting_complete_transfer',error=str(e));time.sleep(30);continue
            mp=out/'artifacts/manifest.json';manifest=json.loads(mp.read_text()) if mp.exists() else {'models':[]}
            models=sorted(manifest['models'],key=lambda m:(ROLES.index(m['role']),m['seed']))
            chosen=None
            # Finish the main paired comparison before secondary/appendix jobs.
            primary=[(m['role'],m['seed'],out/'artifacts'/m['path']) for m in models if m['role'] in ('baseline','robust','frozen')]
            paper=[(m['role'],m['seed'],out/'artifacts'/m['path']) for m in models if m['role'] in ('park','manoharan')]
            internal=[(m['role'],m['seed'],out/'artifacts'/m['path']) for m in models if m['role'] in ('coord','modal')]
            appendix=[(m['role'],m['seed'],out/'artifacts'/m['path']) for m in models if m['role'].endswith('rlonly')]
            classical=[('gp',-1,None),('rule',-1,None)]
            core=[c for c in all_cases if c['size']!=20];large=[c for c in all_cases if c['size']==20]
            phases=[(core,primary),(core,classical),(core,paper),(core,internal),
                    (large,primary+classical+internal),([c for c in core if c['size']==10],appendix)]
            for selected_cases,jobs in phases:
                if not jobs:continue
                for case in selected_cases:
                    pool=out/'pool'/f'{case["id"]}.pkl'
                    if not pool.exists():
                        key='prepare_'+case['id']
                        if key not in errors:chosen=([sys.executable,__file__,'prepare','--pool',str(pool),'--case',json.dumps(case)],key);break
                        continue
                    for role,seed,ck in jobs:
                        key=f'{case["id"]}__{role}_s{seed}';result=out/'eval'/f'{key}.json'
                        if result.exists() or key in errors:continue
                        cmd=[sys.executable,__file__,'evaluate','--pool',str(pool),'--case',json.dumps(case),'--role',role,
                             '--seed',str(seed),'--result',str(result)]
                        if ck:cmd+=['--ckpt',str(ck)]
                        chosen=cmd,key;break
                    if chosen:break
                if chosen:break
            if chosen:execute(*chosen);continue
            present={m['id'] for m in models};expected={f'{role}_s{s}' for role in ROLES for s in range(3)}
            if a.finish_all and errors:
                state('retrying_failed_jobs',failed_jobs=len(errors));report(a)
                with (out/'retry_history.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),errors=errors))+'\n')
                time.sleep(60);errors={};save(out/'errors.json',errors);continue
            if expected<=present:
                state('complete' if not errors else 'finished_with_missing_jobs',failed_jobs=len(errors));report(a);return
            state('waiting_models',missing=sorted(expected-present));report(a);time.sleep(30)
        state('deadline_partial',failed_jobs=len(errors));report(a)


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('run','prepare','evaluate','report','status'))
    for k,v in dict(out=str(ROOT/'runs/report_validation_v1'),inbox=str(ROOT/'inbox/models.tgz'),transport='git',n=20,large_n=8,hours=24.,
                    case='{}',pool='',role='frozen',seed=0,ckpt='',result='').items():p.add_argument('--'+k,type=type(v),default=v)
    p.add_argument('--finish_all',action='store_true',help='Resume until every planned comparison completes; no wall/case timeout')
    a=p.parse_args()
    if a.n<1 or a.large_n<0 or a.hours<=0 or a.transport not in ('git','manual'):p.error('Invalid settings')
    return a


if __name__=='__main__':
    torch.set_num_threads(1);a=parse()
    if a.stage=='status':
        p=Path(a.out)/'status.json';print(p.read_text() if p.exists() else 'Not started')
    else:
        if a.stage=='run':
            def stop(signum,frame):raise KeyboardInterrupt()
            signal.signal(signal.SIGTERM,stop)
        try:globals()[a.stage](a)
        except BlockingIOError:raise SystemExit('Already running')
        except BaseException as e:
            if a.stage=='run':
                save(Path(a.out)/'status.json',dict(stage='failed_or_interrupted',error=str(e),pid=os.getpid()));report(a)
            raise
