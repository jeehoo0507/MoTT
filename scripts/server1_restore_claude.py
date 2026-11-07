"""Restore the existing Claude manager; leave all research workers untouched."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

ROOT=Path.home()/'R2E'


def birth(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def find(names,roots):
    found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            pid=int(p.name);stamp=birth(pid)
            args=[os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if (p/'cwd').resolve() in roots and len(args)>1 and Path(args[1]).name in names and birth(pid)==stamp:
                found[pid]=stamp
        except (OSError,ValueError):pass
    return found


def send(pid,stamp,sig):
    try:
        fd=os.pidfd_open(pid)
        try:
            if birth(pid)==stamp:signal.pidfd_send_signal(fd,sig)
        finally:os.close(fd)
    except (FileNotFoundError,ProcessLookupError):pass


def main():
    if '-com-loop-' not in socket.gethostname():raise SystemExit('Server1 LOOP only')
    os.chdir(ROOT)
    claude=shutil.which('claude')
    if not claude:
        for p in (Path.home()/'.local/bin/claude',Path.home()/'.claude/local/claude'):
            if os.access(p,os.X_OK):claude=str(p);break
    if not claude:raise SystemExit('Claude CLI missing; no manager changed')
    loop=ROOT/'agent/loop.sh'
    if 'claude -p' not in loop.read_text():raise SystemExit('Existing Claude loop missing; no manager changed')
    (ROOT/'agent/logs').mkdir(parents=True,exist_ok=True)
    with (ROOT/'agent/.restore_claude.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stamp=time.strftime('%Y%m%d_%H%M%S',time.gmtime())
        backup=ROOT/'agent/backups'/('restore_claude_'+stamp);backup.mkdir(parents=True)
        for name in ('PROVIDER','STOP','MODEL','CODEX_STATUS.json'):
            path=ROOT/'agent'/name
            if path.exists():shutil.copy2(path,backup/name)
        managers=find({'loop.sh','codex_loop.sh'},{ROOT})
        # Freeze launchers only; allow any active management round to finish.
        for pid,start in managers.items():send(pid,start,signal.SIGSTOP)
        try:
            while find({'codex_pkg_round.py'},{ROOT,Path.home()/'R2E-codex'}):
                print('Waiting for active Codex management round; training/evaluation unchanged',flush=True)
                time.sleep(10)
            for pid,start in managers.items():
                send(pid,start,signal.SIGTERM);send(pid,start,signal.SIGCONT)
            for _ in range(100):
                alive=False
                for pid,start in managers.items():
                    try:
                        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
                        if birth(pid)==start and stat[0]!='Z':alive=True
                    except (FileNotFoundError,ProcessLookupError):pass
                if not alive:break
                time.sleep(.1)
            else:raise RuntimeError('Manager did not exit; Claude not launched')
            (ROOT/'agent/PROVIDER').write_text('claude\n')
            stop=ROOT/'agent/STOP'
            if stop.exists():stop.unlink()  # already backed up; user explicitly requested resumption
            env=dict(os.environ);env['PATH']=str(Path(claude).parent)+os.pathsep+env.get('PATH','')
            with (ROOT/'agent/logs/claude_restored.log').open('a') as log:
                p=subprocess.Popen(['bash','agent/loop.sh'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            receipt=dict(provider='claude',pid=p.pid,backup=str(backup),claude=claude,updated=time.time())
            (ROOT/'agent/CLAUDE_RESTORE_STATUS.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps(receipt),flush=True)
        except BaseException:
            for pid,start in managers.items():send(pid,start,signal.SIGCONT)
            raise


if __name__=='__main__':main()
