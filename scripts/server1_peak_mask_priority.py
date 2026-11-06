"""Temporarily pause the CPU LS16 subtree for two small diagnostic pilots, then resume.

No driver restart, no GPU schedule changes, no discarded LS computation. Run on server 1 only.
Pilot outputs are separate from the six queued diagnostic jobs and main experiment metrics.
"""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path.home() / "R2E"
OUT = ROOT / "reports/peak_mask_priority_20261106"


def process(pid):
    p = Path(f"/proc/{pid}")
    try:
        stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
        args = [v.decode(errors="replace") for v in (p / "cmdline").read_bytes().split(b"\0") if v]
        return dict(pid=pid, ppid=int(stat[1]), born=stat[19], state=stat[0], args=args)
    except (OSError, ValueError, IndexError):
        return None


def memory():
    return int(Path('/sys/fs/cgroup/memory.current').read_text()) / 2**30


def main():
    if "-com-loop-" not in socket.gethostname():
        raise SystemExit("Server 1 (loop) only")
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'runner.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Priority pilot already running")
        if (OUT / 'paused.json').exists() and not (OUT / 'resumed.txt').exists():
            raise SystemExit('Previous pilot has no resume receipt; inspect paused.json before rerunning')
        jobs = [('val_pkg10_0', 900), ('eval_pkg16_0', 1800)]
        required = [ROOT / 'scripts/diag_peak_mask.py', ROOT / 'runs/2026-10-29/r2pk_rl/best.pt']
        required += [ROOT / f'runs/pool_v1/{pool}.pkl' for pool, _ in jobs]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise SystemExit(f"Missing inputs: {missing}")
        if memory() > 30:
            raise SystemExit(f"Memory {memory():.1f} GiB > 30 GiB: no processes paused; pilot not started")
        snapshot = [p for pid in os.listdir('/proc') if pid.isdigit() and (p := process(int(pid)))]
        guards = [p for p in snapshot if any(a.endswith('/mem_guard.py') for a in p['args'])
                  and '--ls_chunk' in p['args'] and 'ls' in p['args']
                  and any('eval_' in a and '16_' in a for a in p['args'])]
        if len(guards) != 1:
            raise SystemExit(f"Expected exactly one LS16 memory guard, found {len(guards)}; nothing paused")
        paused, child = [], None

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            (OUT / 'resumed.txt').unlink(missing_ok=True)
            pending = [guards[0]['pid']]
            while pending:
                pid = pending.pop()
                p = process(pid)
                if p is None:
                    continue
                if p['state'] in ('T', 't'):
                    raise RuntimeError(f'PID {pid} was already stopped; aborting without taking ownership')
                # Persist identities before stopping, for recovery after an uncatchable kill.
                paused.append(dict(pid=pid, born=p['born']))
                (OUT / 'paused.json').write_text(json.dumps(paused, indent=2))
                os.kill(pid, signal.SIGSTOP)
                for _ in range(100):
                    state = process(pid)
                    if state is None or state['state'] in ('T', 't', 'Z'):
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError(f'PID {pid} did not stop promptly')
                # Parent is frozen before reading descendants, including the forkserver worker.
                try:
                    pending.extend(map(int, Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
                except FileNotFoundError:
                    pass
            print(f'PAUSED LS16 subtree: {[p["pid"] for p in paused]}; memory={memory():.1f} GiB', flush=True)
            for pool, budget in jobs:
                path = OUT / f'{pool}_s0.json'
                if path.with_suffix('.done').exists():
                    print(f'SKIP complete {pool}', flush=True)
                    continue
                cmd = [str(ROOT / '.venv/bin/python'), '-u', 'scripts/diag_peak_mask.py',
                       '--pool', f'runs/pool_v1/{pool}.pkl', '--ckpt',
                       'runs/2026-10-29/r2pk_rl/best.pt', '--n', '4',
                       '--max_seconds', str(budget), '--out', str(path)]
                if memory() > 30:
                    print('Memory headroom insufficient; ending pilot', flush=True)
                    break
                print(f'START {pool}, first 4 instances, soft budget {budget}s', flush=True)
                with open(OUT / f'{pool}.log', 'a') as log:
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    started = time.monotonic()
                    while child.poll() is None:
                        if memory() > 42 or time.monotonic() - started > budget + 300:
                            print('Pilot memory/time guard: terminating diagnostic only', flush=True)
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                            break
                        time.sleep(2)
                rc = child.returncode
                child = None
                print(f'END {pool} rc={rc}; output={path}', flush=True)
                if rc != 0:
                    break
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            for saved in reversed(paused):
                p = process(saved['pid'])
                if p is not None and p['born'] == saved['born']:
                    try:
                        os.kill(saved['pid'], signal.SIGCONT)
                    except ProcessLookupError:
                        pass
            (OUT / 'resumed.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
            print('RESUMED original LS16 subtree; GPU/driver schedule unchanged', flush=True)


if __name__ == '__main__':
    main()
