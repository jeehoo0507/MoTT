"""Reload only this clone's report runner at a case boundary; its parent retries."""
import fcntl
import os
from pathlib import Path
import signal
import socket
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import switch_curriculum as switch
from scripts.report_memory import snapshot, blocked


def main():
    root = Path(__file__).resolve().parents[1]
    if '-maskedkd-' not in socket.gethostname() or root != Path.home()/'R2E-posttrain':
        raise SystemExit('Run only in maskedkd ~/R2E-posttrain')
    with (root/'runs/transition/memory_reload.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parents = switch.processes('switch_post_training.py', {root})
        if len(parents) != 1:
            raise SystemExit('Expected one active posttrain supervisor; no processes changed')
        parent = parents[0]
        switch.send(parent, signal.SIGSTOP)
        try:
            runners = switch.processes('report_validation.py', {root})
            for pid in runners:
                args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                if b'run' not in args or int(stat[1]) != parent:
                    continue
                print(f'Waiting for active case to finish, then reloading report runner {pid}', flush=True)
                switch.stop_between_cases(pid)
        finally:
            switch.send(parent, signal.SIGCONT)
        info = snapshot()
        print(dict(memory=info, start_blocked=blocked(info)), flush=True)
        print('Supervisor resumed. Report retries automatically (normally within 60 seconds). FRESH untouched.')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    main()
