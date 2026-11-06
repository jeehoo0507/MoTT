"""Wait for the existing study; own only new children and resume saved report results."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import pkg_guard_study as study


def busy(root,proc_root=Path('/proc')):
    lock=root/'runs'/('pkg_guard_v1' if root.name=='R2E-pkg-fix' else 'report_validation_v1')/'runner.lock'
    if lock.exists():
        with lock.open('r') as f:
            try:fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:return True
    for p in proc_root.iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if (p/'cwd').resolve()!=root.resolve():continue
            if any(Path(os.fsdecode(arg)).name in ('pkg_guard_study.py','pkg_guard_train.py','report_validation.py') for arg in args):return True
        except (OSError,ValueError):continue
    return False


def workers():
    cpus=len(os.sched_getaffinity(0))
    quota=Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        limit,period=quota.read_text().split()
        if limit!='max':cpus=min(cpus,int(limit)//int(period))
    return max(1,min(4,cpus-1))


def benchmark(source,n_workers,path):
    import numpy as np
    from decap_loop.parallel_verify import ExactVerifier
    from decap_loop.safe_verify import _job,pin_threads
    pool=source/'runs/pool_v1/train_pkg12_0.pkl'
    teacher=source/'data/teacher/train_pkg12_0.npz'
    with pool.open('rb') as f:d=pickle.load(f)
    if d['cfg']['n_check']!=700:raise ValueError('Expected unchanged 700-point checker')
    placements=np.load(teacher)['occ'];jobs=[]
    for i in range(min(8,len(d['valid']))):
        jobs.append((d['valid'][i].reshape(d['H'],d['W']),np.flatnonzero(d['entry'][i]),d['probes'][i],
                     placements[i],d['cfg'],d['zt_par'][i:i+1],1e-3,3,.003,3))
    if len(jobs)<n_workers:raise ValueError('Too few benchmark cases')
    pin_threads()
    with ExactVerifier(n_workers) as verifier:
        verifier.check_jobs(jobs[:n_workers])  # exclude interpreter startup from steady-state timing
        start=time.monotonic();serial=[_job(j) for j in jobs];serial_sec=time.monotonic()-start
        start=time.monotonic();parallel=verifier.check_jobs(jobs);parallel_sec=time.monotonic()-start
    for left,right in zip(serial,parallel):
        for x,y in zip(left,right):np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
        for tol in (1e-3,np.log10(1.01)):
            if (left[0]<=tol)!=(right[0]<=tol):raise ValueError('Success decision changed')
    result=dict(workers=n_workers,serial_seconds=serial_sec,parallel_seconds=parallel_sec,
        speedup=serial_sec/parallel_sec,parity=True,pool_sha256=study.sha(pool),teacher_sha256=study.sha(teacher))
    study.save(path,result);print(json.dumps(result),flush=True)


def run(server):
    home=Path.home();old=home/('R2E-pkg-fix' if server==1 else 'R2E-report')
    out=ROOT/'runs'/('pkg_guard_parallel_v1' if server==1 else 'report_finish_supervisor')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def state(stage,**kw):
            record=dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw)
            study.save(out/'followup_status.json',record);print(json.dumps(record),flush=True)
        if server==1 and not (old/'runs/pkg_guard_v1/config.json').exists():raise ValueError('Original server-1 study not found')
        while busy(old):
            state('waiting_current_experiment',root=str(old));time.sleep(30)
        if server==1:
            settings=out/'worker_settings.json'
            count=json.loads(settings.read_text())['workers'] if settings.exists() else workers()
            if count<2:raise RuntimeError('At least 3 available CPU cores required; original experiments unchanged')
            study.save(settings,dict(workers=count))
            while study.memory()>28:state('waiting_memory');time.sleep(30)
            if not (out/'benchmark.json').exists():
                # Use an owned subprocess so memory/time limits include the benchmark workers.
                cmd=[sys.executable,__file__,'benchmark','--workers',str(count)]
                with (out/'benchmark.log').open('a') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    start=time.monotonic()
                    try:
                        while child.poll() is None:
                            if study.memory()>32 or time.monotonic()-start>900:
                                study.stop_own_child(child);raise RuntimeError('Benchmark resource limit; see benchmark.log')
                            time.sleep(1)
                    except BaseException:study.stop_own_child(child);raise
                if child.returncode:raise RuntimeError('Benchmark failed; see benchmark.log')
            if json.loads((out/'benchmark.json').read_text())['speedup']<=1.05:
                state('no_measured_speedup');return
            state('starting_parallel_comparison',workers=count)
            a=study.parse(['run','--source',str(home/'R2E'),'--out',str(out),'--certify_workers',str(count)])
            try:study.run(a)
            finally:study.report(a)
            state('study_returned',study_status=json.loads((out/'status.json').read_text()))
        else:
            # Existing records, pools and checkpoint identities are reused under their runner lock.
            target=old/'runs/report_validation_v1'
            frozen=json.loads((target/'config.json').read_text()) if (target/'config.json').exists() else {}
            cmd=[sys.executable,'-u',str(ROOT/'scripts/report_validation.py'),'run','--finish_all',
                 '--out',str(target),'--inbox',frozen.get('inbox',str(old/'inbox/models.tgz'))]
            for key in ('n','large_n','hours','transport'):
                if key in frozen:cmd+=['--'+key,str(frozen[key])]
            while True:
                while busy(old):state('waiting_current_report');time.sleep(30)
                state('continuing_report')
                child=subprocess.Popen(cmd,cwd=ROOT,start_new_session=True)
                try:rc=child.wait()
                except BaseException:study.stop_own_child(child);raise
                status=target/'status.json'
                if rc==0 and status.exists() and json.loads(status.read_text()).get('stage')=='complete':
                    state('complete');return
                state('report_retry',rc=rc);time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=('server1','server2','benchmark'));p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    def stop(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    if a.task=='benchmark':benchmark(Path.home()/'R2E',a.workers,ROOT/'runs/pkg_guard_parallel_v1/benchmark.json')
    else:run(1 if a.task=='server1' else 2)
