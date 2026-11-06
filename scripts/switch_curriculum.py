"""Scoped scheduler handoff. Preserve seed-0 artifacts and finish the active report case."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import pkg_guard_study as study
from scripts import report_model_bundle as bridge
from scripts.curriculum_campaign import LEGACY,BRANCH
IDENTITY={}


def birth(pid):return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def processes(script,roots):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            pid=int(p.name);started=birth(pid)
            cmd=(p/'cmdline').read_bytes().split(b'\0');cwd=(p/'cwd').resolve()
            if cwd in roots and any(Path(os.fsdecode(x)).name==script for x in cmd) and birth(pid)==started:
                IDENTITY[pid]=started;result.append(pid)
        except (OSError,ValueError):pass
    return result


def alive(pid):
    try:
        if pid in IDENTITY and birth(pid)!=IDENTITY[pid]:return False
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def send(pid,sig):
    # Hold a pidfd so a recycled PID cannot receive this signal.
    try:
        if pid not in IDENTITY:raise ValueError('Refuse to signal an unverified manager')
        fd=os.pidfd_open(pid)
        try:
            if birth(pid)==IDENTITY[pid]:signal.pidfd_send_signal(fd,sig)
        finally:os.close(fd)
    except (ProcessLookupError,FileNotFoundError):pass


def stop_manager(pid):
    send(pid,signal.SIGTERM);send(pid,signal.SIGCONT)
    while alive(pid):time.sleep(.5)


def stop_between_cases(pid):
    while alive(pid):
        send(pid,signal.SIGSTOP)
        try:
            children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
            running=any(alive(int(c)) for c in children)
            if not running:stop_manager(pid);return
        except FileNotFoundError:return
        finally:send(pid,signal.SIGCONT)
        time.sleep(1)


def seed_zero_complete(out):
    for arm,reserve in (('frozen',0.),('baseline',0.),('robust',.1)):
        for size in (16,10):
            path=out/f'eval/{arm}_s0_pkg{size}_g{reserve}.json'
            if not path.exists() or not json.loads(path.read_text()).get('complete'):return False
    return all((out/f'models/{arm}_s0/metrics.json').exists() for arm in ('baseline','robust'))


def child(cmd,log):
    with log.open('a') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=proc.wait()
        except BaseException:study.stop_own_child(proc);raise
    if rc:raise RuntimeError(f'Command failed: see {log}')


def run(server):
    home=Path.home();out=ROOT/'runs/transition';out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def state(stage,**kw):
            r=dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw);study.save(out/'status.json',r);print(json.dumps(r),flush=True)
        old_follow=home/'R2E-followups'
        if server==1:
            # This replaces the explicitly cancelled parallel-repeat scheduler, never the original driver.
            for pid in processes('finish_followup.py',{old_follow}):
                state('cancelling_parallel_repeat',pid_to_stop=pid);stop_manager(pid)
            old=home/'R2E-pkg-fix';oldout=old/'runs/pkg_guard_v1'
            while not seed_zero_complete(oldout):
                if not processes('pkg_guard_study.py',{old}):raise RuntimeError('Seed 0 incomplete and original study is not running')
                state('waiting_seed0_training_and_evaluation');time.sleep(15)
            for pid in processes('pkg_guard_study.py',{old}):
                state('stopping_guard_after_seed0',pid_to_stop=pid);stop_manager(pid)
            # Publish the completed seed-0 guard artifacts before the replacement campaign.
            export=out/'legacy_export'
            while True:
                try:
                    bridge.export(home/'R2E',oldout,export);bridge.publish(export);break
                except (OSError,subprocess.SubprocessError) as e:state('legacy_publish_retry',error=str(e));time.sleep(60)
            state('starting_new_training')
            child([sys.executable,'-u','scripts/curriculum_campaign.py'],out/'campaign.log')
            state('complete')
        else:
            # Pause only the old follow-up scheduler so it cannot relaunch a migrated runner.
            supervisors=processes('finish_followup.py',{old_follow})
            for pid in supervisors:send(pid,signal.SIGSTOP)
            try:
                for pid in processes('report_validation.py',{home/'R2E-report',old_follow}):
                    args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                    if b'run' in args:
                        state('handover_after_current_case',pid_to_stop=pid);stop_between_cases(pid)
                for pid in supervisors:stop_manager(pid)
            except BaseException:
                for pid in supervisors:send(pid,signal.SIGCONT)
                raise
            target=home/'R2E-report/runs/report_validation_v1'
            if not (target/'config.json').exists():raise RuntimeError('Original report config not found; no report results changed')
            frozen=json.loads((target/'config.json').read_text())
            plan=out/'legacy_plan.json';study.save(plan,dict(expected=LEGACY,reason='Original guard campaign intentionally ends after seed 0'))
            cmd=[sys.executable,'-u','scripts/report_validation.py','run','--finish_all','--out',str(target),'--inbox',frozen['inbox'],'--plan',str(plan)]
            for key in ('n','large_n','hours','transport'):cmd+=['--'+key,str(frozen[key])]
            def finish(cmd,folder,log):
                while True:
                    try:child(cmd,log)
                    except RuntimeError as e:state('report_retry',error=str(e));time.sleep(60);continue
                    path=folder/'status.json'
                    if path.exists() and json.loads(path.read_text()).get('stage')=='complete':return
                    time.sleep(60)
            state('finishing_existing_baseline_comparison')
            finish(cmd,target,out/'legacy_report.log')
            # New independent problem seeds; do not tune from the existing report outcomes.
            fresh=ROOT/'runs/curriculum_validation_v1'
            plan=out/'new_plan.json';study.save(plan,dict(from_manifest=True,seed_offset=3000000))
            cmd=[sys.executable,'-u','scripts/report_validation.py','run','--finish_all','--out',str(fresh),
                 '--inbox',str(ROOT/'inbox/curriculum.tgz'),'--plan',str(plan),'--bundle_branch',BRANCH]
            state('validating_new_training_on_fresh_boards')
            finish(cmd,fresh,out/'new_report.log');state('complete')


if __name__=='__main__':
    def stop(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    try:run(int(sys.argv[1]))
    except BaseException as e:
        study.save(ROOT/'runs/transition/status.json',dict(stage='failed_or_interrupted',error=str(e),pid=os.getpid(),updated=time.time()));raise
