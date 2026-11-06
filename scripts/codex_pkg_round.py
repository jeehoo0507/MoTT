"""One Codex research-management round in an isolated worktree; no original job control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=Path.home()/'R2E'
WORK=Path.home()/'R2E-codex'
ALLOWED=('scripts/pkg_','decap_loop/pkg_','reports/pkg_base/','docs/PKG_BASE','PROGRESS.md','PROPOSALS.md')


def git(*args,root=None,check=True):
    root=WORK if root is None else root
    return subprocess.run(['git',*args],cwd=root,text=True,capture_output=True,check=check,
                          env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),timeout=180)


def snapshot():
    result={'time_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),'processes':[]}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if any(x in args for x in ('pipeline_day','train.py','run_rung.py','eval_policy.py','mem_guard.py','side_slot','diag_peak_mask.py')):
                result['processes'].append({'pid':int(path.parent.name),'command':args})
        except OSError:pass
    result['queue']=json.loads((ORIGINAL/'agent/day2_queue.json').read_text())
    for name in ('memory.current','memory.max','memory.events'):
        p=Path('/sys/fs/cgroup')/name
        if p.exists():result[name]=p.read_text()
    gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=20,check=False)
    result['nvidia_smi']=gpu.stdout or gpu.stderr
    return result


def candidates():
    tracked=git('diff','--name-only','HEAD').stdout.splitlines()
    untracked=git('ls-files','--others','--exclude-standard').stdout.splitlines()
    return sorted(p for p in set(tracked+untracked) if p.startswith(ALLOWED) and (WORK/p).is_file())


def main():
    p=argparse.ArgumentParser();p.add_argument('--codex',required=True);a=p.parse_args()
    if not WORK.is_dir():raise SystemExit('Isolated worktree missing; run the switch installer')
    report=WORK/'reports/pkg_base';report.mkdir(parents=True,exist_ok=True)
    (report/'live_snapshot.json').write_text(json.dumps(snapshot(),indent=2)+'\n')
    ts=time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    logfile=ORIGINAL/'agent/logs'/f'codex_round_{ts}.log'
    prompt=(WORK/'agent/CODEX_PKG_TASK.md').read_text()
    prompt+='\n이번 라운드는 기존 실험 보호를 지키면서 위 작업을 이어서 실행하라. 결과/검증/미완료를 명확히 남겨라.\n'
    before={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for name in
            ('decap_loop/pdn.py','decap_loop/sim.py','decap_loop/verify.py','decap_loop/bank.py','train.py','decap_loop/model.py','agent/day2_queue.json') if (p:=WORK/name).exists()}
    cmd=[a.codex,'--ask-for-approval','never','exec','--sandbox','workspace-write','--cd',str(WORK),'-']
    status=ORIGINAL/'agent/CODEX_STATUS.json'
    status.write_text(json.dumps(dict(stage='running',pid=os.getpid(),worktree=str(WORK),log=str(logfile))))
    with logfile.open('a') as log:
        try:
            rc=subprocess.run(cmd,input=prompt,text=True,stdout=log,stderr=subprocess.STDOUT,timeout=1500).returncode
        except subprocess.TimeoutExpired:rc=124
    changed=[name for name,digest in before.items() if not (WORK/name).exists() or hashlib.sha256((WORK/name).read_bytes()).hexdigest()!=digest]
    if changed:raise RuntimeError(f'Protected worktree files changed; publication stopped: {changed}')
    files=candidates()
    publish='no changes'
    if files:
        git('add','--',*files)
        if git('diff','--cached','--quiet',check=False).returncode:
            git('-c','user.name=R2E Codex agent','-c','user.email=codex-agent@localhost','commit','-m',f'Codex: package MoTT analysis {ts}')
        branch=git('branch','--show-current').stdout.strip()
        remote='origin' if 'origin' in git('remote').stdout.splitlines() else 'r2e'
        pushed=git('push',remote,f'HEAD:refs/heads/{branch}',check=False)
        publish='pushed '+branch if pushed.returncode==0 else 'push failed: '+pushed.stderr[-400:]
    progress=WORK/'PROGRESS.md'
    if progress.exists():shutil.copyfile(progress,ORIGINAL/'agent/CODEX_PROGRESS.md')
    status.write_text(json.dumps(dict(stage='round_done',rc=rc,worktree=str(WORK),publish=publish,log=str(logfile)),indent=2))
    print(f'Codex round rc={rc}; {publish}; log={logfile}',flush=True)


if __name__=='__main__':main()
