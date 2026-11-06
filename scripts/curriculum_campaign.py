"""Prospective matched continuation; validation-only expansion and sanitized Git handoff."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from scripts import pkg_guard_study as study
from scripts import report_model_bundle as bridge
from scripts.finish_followup import workers

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/mott_curriculum_v1'
BRANCH='report-models-mott-curriculum-v1'
ARMS=('control','annealed','adaptive')
PROTOCOL='MOTT_CURRICULUM_PROTOCOL.md'
TRAIN_ARM_HOOK=None
LEGACY=[f'{role}_s{s}' for role in ('frozen',)+bridge.FAMILIES for s in range(3)]+['baseline_s0','robust_s0']


def expansion_gate(control,candidate):
    comparisons=[]
    for name,rows in control.items():
        other=candidate[name]
        if [r['instance'] for r in rows]!=[r['instance'] for r in other]:raise ValueError('Unpaired validation')
        joint=[(x,y) for x,y in zip(rows,other) if x['success'] and y['success']]
        diff=sum(y['success']-x['success'] for x,y in zip(rows,other))
        left=sum(x['count'] for x,y in joint);right=sum(y['count'] for x,y in joint)
        ratio=(right/left if left else (1. if right==0 else None)) if joint else None
        comparisons.append(dict(group=name,success_count_difference=diff,joint_n=len(joint),joint_count_ratio=ratio))
    passed=all(r['success_count_difference']>=0 and r['joint_n']>0 and r['joint_count_ratio'] is not None and r['joint_count_ratio']<=1.05 for r in comparisons)
    passed=passed and any(r['success_count_difference']>0 or r['joint_count_ratio']<=.95 for r in comparisons)
    return dict(expand=bool(passed),comparisons=comparisons,rule='No validation-group success loss; joint count <=1.05; some success gain or >=5% count reduction')


def owned(cmd,tag,gpu=False):
    source=Path.home()/'R2E';(OUT/'logs').mkdir(parents=True,exist_ok=True)
    while True:
        while study.memory()>28 or (gpu and (study.old_pending(source) or study.gpu_pids())):
            study.save(OUT/'status.json',dict(stage='waiting_resources',job=tag,updated=time.time()));time.sleep(30)
        study.save(OUT/'status.json',dict(stage='running',job=tag,pid=os.getpid(),updated=time.time(),command=cmd))
        with (OUT/'logs'/f'{tag}.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            reason=None;last=time.monotonic();started=last
            try:
                while child.poll() is None:
                    if study.memory()>32:reason='memory'
                    if gpu and time.monotonic()-last>15:
                        last=time.monotonic()
                        if len(study.gpu_pids())>1 or study.old_pending(source):reason='yield_to_original_gpu'
                    if reason:study.stop_own_child(child);break
                    time.sleep(1)
            except BaseException:study.stop_own_child(child);raise
        with (OUT/'attempts.jsonl').open('a') as audit:
            audit.write(json.dumps(dict(job=tag,seconds=time.monotonic()-started,returncode=child.returncode,reason=reason))+'\n')
        if reason:time.sleep(30);continue
        if child.returncode:raise RuntimeError(f'{tag} failed ({child.returncode}); inspect {OUT}/logs/{tag}.log')
        return


def publish(final=False,seeds=(0,),decision=None):
    export=OUT/'export';export.mkdir(parents=True,exist_ok=True)
    legacy=OUT/'legacy_export'
    bridge.export(Path.home()/'R2E',Path.home()/'R2E-pkg-fix/runs/pkg_guard_v1',legacy)
    old=json.loads((legacy/'manifest.json').read_text());entries=[]
    for m in old['models']:
        if m['id'] not in LEGACY:continue
        dest=export/m['path'];dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copy2(legacy/m['path'],dest)
        entries.append(m)
    for seed in seeds:
        src=OUT/f'inputs/r2pk{"" if seed==0 else "_s"+str(seed)}_rl/best.pt'
        for arm in ARMS:
            folder=OUT/'models'/f'{arm}_s{seed}';ck=folder/'best.pt'
            if not (folder/'metrics.json').exists():continue
            dest=export/'models'/f'{arm}_s{seed}.pt'
            data=torch.load(ck,map_location='cpu',weights_only=False)
            if study.sha(data['args'].get('root_init',data['args']['init']))!=study.sha(src):raise ValueError('Unmatched initialization')
            stamp=dest.with_suffix('.json');source_hash=study.sha(ck)
            if dest.exists() and (not stamp.exists() or json.loads(stamp.read_text())['source_sha256']!=source_hash):
                raise ValueError('Previously exported campaign identity changed')
            if not dest.exists():
                torch.save(dict(model=data['model'],args={k:data['args'][k] for k in bridge.ARGS if k in data['args']}),dest)
                study.save(stamp,dict(source_sha256=source_hash))
            metrics=json.loads((folder/'metrics.json').read_text())
            entries.append(dict(id=f'{arm}_s{seed}',role=arm,seed=seed,path=f'models/{dest.name}',sha256=study.sha(dest),
                source_sha256=study.sha(ck),init_sha256=study.sha(src),training={k:metrics.get(k) for k in ('iters','hours','best_iter','cuda_oom_skipped','chain_sha256')}))
    expected=LEGACY+[f'{arm}_s{s}' for s in seeds for arm in ARMS]
    study.save(export/'manifest.json',dict(schema=1,models=entries,expected=expected,
        campaign_final=final,missing=sorted(set(expected)-{e['id'] for e in entries}),
        expansion=decision,protocol_sha256=study.sha(ROOT/'docs'/PROTOCOL)))
    bridge.publish(export,branch=BRANCH)
    study.save(OUT/'publish_status.json',dict(ok=True,updated=time.time(),final=final))


def safe_publish(**kw):
    try:publish(**kw);return True
    except (OSError,subprocess.SubprocessError) as e:
        study.save(OUT/'publish_status.json',dict(ok=False,error=str(e),updated=time.time()));return False


def gate_from_models():
    paths=[OUT/f'models/{arm}_s0/validation_best.json' for arm in ('control','adaptive')]
    decision=expansion_gate(*(json.loads(p.read_text()) for p in paths))
    decision['validation_sha256']=[study.sha(p) for p in paths]
    return decision


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        setting=OUT/'settings.json'
        if setting.exists():cfg=json.loads(setting.read_text())
        else:
            cfg=dict(workers=workers(),train_hours=.75,arms=list(ARMS),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                     protocol_sha256=study.sha(ROOT/'docs'/PROTOCOL))
            study.save(setting,cfg)
        if cfg['git']!=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip():raise ValueError('Changed campaign code; use original commit')
        study.inputs(argparse.Namespace(out=str(OUT),source=str(Path.home()/'R2E'),seeds='0,1,2'))
        seeds=[0];decision=None
        for seed in (0,1,2):
            if seed and not decision['expand']:break
            if seed not in seeds:seeds.append(seed)
            order=ARMS if seed%2==0 else tuple(reversed(ARMS))
            for arm in order:
                folder=OUT/'models'/f'{arm}_s{seed}'
                if not (folder/'metrics.json').exists():
                    src=OUT/f'inputs/r2pk{"" if seed==0 else "_s"+str(seed)}_rl/best.pt'
                    ck=torch.load(src,map_location='cpu',weights_only=False);a=ck['args']
                    if TRAIN_ARM_HOOK is not None:TRAIN_ARM_HOOK(seed,arm,folder,cfg,src,a)
                    else:
                        cmd=[sys.executable,'scripts/train_reserve_curriculum.py','--curriculum',arm,'--certify_workers',str(cfg['workers']),
                            '--stage','rl','--model','mott','--peak_refine','--init',str(src),'--out',str(folder),
                            '--pool',str(OUT/'inputs/pool'),'--teacher_dir',str(OUT/'inputs/teacher'),'--seed',str(seed),
                            '--hours',str(cfg['train_hours']),'--iters','1000000','--B','4','--M','8','--chunk','1','--lr','0.0001',
                            '--lam','20','--imit','0','--certify','2','--val_every','50','--ckpt_min','5','--max_steps','64']
                        for key in ('d','layers','heads'):cmd+=['--'+key,str(a[key])]
                        for flag in ('fc_bias','mode_head'):
                            if a.get(flag):cmd+=['--'+flag]
                        owned(cmd,f'{arm}_s{seed}',gpu=True)
                safe_publish(seeds=tuple(seeds),decision=decision)
            if seed==0:
                decision=gate_from_models();study.save(OUT/'expansion.json',decision)
        while not safe_publish(final=True,seeds=tuple(seeds),decision=decision):time.sleep(60)
        study.save(OUT/'status.json',dict(stage='complete',seeds=seeds,expansion=decision,updated=time.time()))


if __name__=='__main__':
    import signal
    def stop(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    try:run()
    except BaseException as e:
        study.save(OUT/'status.json',dict(stage='failed_or_interrupted',error=str(e),updated=time.time()));raise
