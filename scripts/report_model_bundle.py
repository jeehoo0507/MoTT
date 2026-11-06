"""Export completed research checkpoints; opt-in publishing uses an isolated Git branch."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.pkg_guard_study import save,sha

SCHEMA=1
ARGS=('model','feat','d','layers','heads','peak_refine','dedup_modes','fc_bias','mode_head',
      'glob','prior','loop_train','loop_eval','seed','reserve_decades','baseline')
FAMILIES=('coord','modal','park','manoharan','park_rlonly','manoharan_rlonly')
BASES=dict(coord='r0_coord',modal='r1_modal',park='bp_park',manoharan='bm_manoharan',
           park_rlonly='bp_park_rlonly',manoharan_rlonly='bm_manoharan_rlonly_l20')
REPO='https://github.com/jeehoo0507/R2E.git'
BRANCH='report-models-pkg-guard-v1'


def git(args,cwd):
    import os
    return subprocess.run(['git',*args],cwd=cwd,text=True,capture_output=True,check=True,timeout=60,
                          env=dict(os.environ,GIT_TERMINAL_PROMPT='0')).stdout.strip()


def publish(out):
    out=Path(out);work=out/'publish';work.mkdir(exist_ok=True)
    if not (work/'.git').exists():
        git(['init','-b',BRANCH],work);git(['remote','add','origin',REPO],work)
        if git(['ls-remote','--heads','origin',BRANCH],work):
            git(['fetch','--depth','1','origin',BRANCH],work);git(['checkout','-B',BRANCH,'FETCH_HEAD'],work)
    manifest=json.loads((out/'manifest.json').read_text())
    for e in manifest['models']:
        src=out/e['path'];dst=work/e['path'];dst.parent.mkdir(exist_ok=True)
        if src.stat().st_size>=95*1024**2:raise ValueError('Checkpoint exceeds Git file limit')
        if not dst.exists() or sha(dst)!=e['sha256']:shutil.copy2(src,dst)
    shutil.copy2(out/'manifest.json',work/'manifest.json')
    git(['add','manifest.json','models'],work)
    if git(['status','--porcelain'],work):
        git(['-c','user.name=R2E artifact exporter','-c','user.email=r2e-artifacts@users.noreply.github.com',
             'commit','-m','Update completed report checkpoints'],work)
    # Always retry push, including after an earlier authentication/network failure.
    git(['push','origin',f'HEAD:refs/heads/{BRANCH}'],work)


def fetch_bundle(cache,archive):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    if not (cache/'.git').exists():
        git(['init'],cache);git(['remote','add','origin',REPO],cache)
    if not git(['ls-remote','--heads','origin',BRANCH],cache):return False
    try:previous=git(['rev-parse','HEAD'],cache)
    except subprocess.CalledProcessError:previous=None
    git(['fetch','--depth','1','origin',BRANCH],cache)
    current=git(['rev-parse','FETCH_HEAD'],cache)
    if previous==current and Path(archive).exists():return True
    git(['checkout','--detach','FETCH_HEAD'],cache)
    manifest=json.loads((cache/'manifest.json').read_text())
    # Keep the same strict tar importer for both manual and Git-based transfer.
    archive=Path(archive);archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive.with_suffix('.part'),'w:gz') as tar:
        tar.add(cache/'manifest.json',arcname='manifest.json')
        for e in manifest['models']:
            if not re.fullmatch(r'models/[a-z_]+_s[012]\.pt',e['path']):raise ValueError('Unexpected published model path')
            tar.add(cache/e['path'],arcname=e['path'])
    archive.with_suffix('.part').replace(archive);return True


def export(source,study,out):
    source,study,out=map(Path,(source,study,out))
    if out.resolve().is_relative_to(source.resolve()) or out.resolve().is_relative_to(study.resolve()):
        raise ValueError('Export outside the original/study directories')
    out.mkdir(parents=True,exist_ok=True)
    entries=[];missing=[]
    def add(role,seed,ckpath,metrics=None,init=None):
        name=f'{role}_s{seed}'
        if not ckpath.is_file() or (metrics is not None and not metrics.is_file()):missing.append(name);return
        ck=torch.load(ckpath,map_location='cpu',weights_only=False);a=ck['args']
        if int(a.get('seed',seed))!=seed:raise ValueError(f'Seed mismatch: {name}')
        if role in ('baseline','robust'):
            if a.get('model')!='mott' or a.get('feat')!='mott_pk':raise ValueError(name)
            if abs(a.get('reserve_decades',-1)-(0.1 if role=='robust' else 0.))>1e-9:raise ValueError('Reserve mismatch')
            if not Path(a['init']).is_file() or sha(a['init'])!=sha(init):raise ValueError('Unmatched source initialization')
        target=out/'models'/f'{name}.pt';target.parent.mkdir(exist_ok=True)
        origin_sha=sha(ckpath);stamp=target.with_suffix('.json')
        if not stamp.exists() or json.loads(stamp.read_text()).get('source_sha256')!=origin_sha:
            # Only weights and model configuration; never optimizer, credentials or original pools.
            torch.save(dict(model=ck['model'],args={k:a[k] for k in ARGS if k in a}),target.with_suffix('.tmp'))
            target.with_suffix('.tmp').replace(target);save(stamp,dict(source_sha256=origin_sha))
        training={k:a.get(k) for k in ('stage','hours','B','M','lr','certify','imit','baseline')}
        if metrics and metrics.exists():
            m=json.loads(metrics.read_text());training.update(updates=m.get('iters'),process_hours=m.get('hours'))
        entries.append(dict(id=name,role=role,seed=seed,path=f'models/{name}.pt',sha256=sha(target),
            source_sha256=origin_sha,init_sha256=sha(init) if init else None,training=training))
    for seed in range(3):
        suffix='' if seed==0 else f'_s{seed}'
        src=source/f'runs/2026-10-29/r2pk{suffix}_rl/best.pt'
        add('frozen',seed,src,src.parent/'metrics.json')
        for role in ('baseline','robust'):
            folder=study/'models'/f'{role}_s{seed}'
            add(role,seed,folder/'best.pt',folder/'metrics.json',src)
        for role,base in BASES.items():
            day='2026-10-28' if role in ('coord','modal') and seed==0 else '2026-10-29'
            ck=source/f'runs/{day}/{base}{suffix}_rl/best.pt'
            add(role,seed,ck,ck.parent/'metrics.json')
    main_complete=all(any(e['id']==f'{role}_s{s}' for e in entries) for role in ('frozen','baseline','robust') for s in range(3))
    manifest=dict(schema=SCHEMA,main_complete=main_complete,models=entries,missing=missing)
    save(out/'manifest.json',manifest)
    archive=out/'models.tgz';temp=out/'models.tgz.tmp'
    with tarfile.open(temp,'w:gz') as tar:
        tar.add(out/'manifest.json',arcname='manifest.json')
        for entry in entries:tar.add(out/entry['path'],arcname=entry['path'])
    temp.replace(archive)
    print(json.dumps(dict(archive=str(archive),main_complete=main_complete,models=len(entries),missing=missing)),flush=True)
    return main_complete


def receive(archive,dest):
    """Allowlisted extraction, verify every weight hash, publish manifest last."""
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)>100 or sum(m.size for m in members)>2*1024**3:raise ValueError('Oversized bundle')
        if any(not m.isfile() or not re.fullmatch(r'(manifest.json|models/[a-z_]+_s[012]\.pt)',m.name) for m in members):
            raise ValueError('Unexpected bundle member')
        if len({m.name for m in members})!=len(members):raise ValueError('Duplicate bundle member')
        manifest=json.load(tar.extractfile('manifest.json'))
        if manifest['schema']!=SCHEMA:raise ValueError('Unknown bundle schema')
        expected={'manifest.json'}|{e['path'] for e in manifest['models']}
        if {m.name for m in members}!=expected:raise ValueError('Bundle inventory mismatch')
        existing=json.loads((dest/'manifest.json').read_text()) if (dest/'manifest.json').exists() else {'models':[]}
        old={e['id']:e for e in existing['models']};incoming={e['id']:e for e in manifest['models']}
        if len(incoming)!=len(manifest['models']):raise ValueError('Duplicate model ID')
        for key,entry in old.items():
            if key not in incoming or incoming[key]['sha256']!=entry['sha256']:raise ValueError('Previously evaluated model changed/removed')
        import hashlib
        for entry in manifest['models']:
            if entry['role'] not in ('frozen','baseline','robust')+FAMILIES or entry['seed'] not in (0,1,2):raise ValueError('Unknown model')
            if entry['id']!=f"{entry['role']}_s{entry['seed']}" or entry['path']!=f"models/{entry['id']}.pt":raise ValueError('Model path mismatch')
            data=tar.extractfile(entry['path']).read()
            if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Weight hash mismatch')
            target=dest/entry['path'];target.parent.mkdir(exist_ok=True);target.with_suffix('.tmp').write_bytes(data);target.with_suffix('.tmp').replace(target)
        for entry in manifest['models']:
            if entry['role'] in ('baseline','robust'):
                frozen=incoming.get(f"frozen_s{entry['seed']}")
                if not frozen or frozen['source_sha256']!=entry['init_sha256']:raise ValueError('Unmatched MoTT initialization')
        save(dest/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--source',default=str(Path.home()/'R2E'))
    p.add_argument('--study',default=str(Path.home()/'R2E-pkg-fix/runs/pkg_guard_v1'))
    p.add_argument('--out',default=str(Path.home()/'R2E-report-export'));p.add_argument('--wait',action='store_true')
    p.add_argument('--publish',action='store_true');p.add_argument('--hours',type=float,default=24.)
    a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True)
    with (Path(a.out)/'export.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        until=time.time()+a.hours*3600;last=None
        while time.time()<until:
            complete=export(a.source,a.study,a.out)
            manifest=json.loads((Path(a.out)/'manifest.json').read_text());published=True
            if a.publish and manifest['models'] and sha(Path(a.out)/'manifest.json')!=last:
                try:publish(a.out);last=sha(Path(a.out)/'manifest.json');print('PUBLISHED '+BRANCH,flush=True)
                except (subprocess.SubprocessError,OSError) as e:
                    published=False;print('PUBLISH RETRY: '+str(e)+' '+(getattr(e,'stderr','') or '')[-1000:],flush=True)
            if (complete and not manifest['missing'] and published) or not a.wait:break
            time.sleep(30)
