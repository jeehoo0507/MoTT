"""Merge server-2 pushes into server 1 often, so the driver sees their `done` files before it reaches the same F
entries (QUEUE 11/08 01:40 KST: server 2 runs F1 pkg20 + F2 via scripts/server2_f.sh and pushes per record; the driver
only checks files on disk and never pulls). Every PERIOD s: fetch; if behind, merge r2e/main (no stash, no rebase: the
live logs stay put, see PROGRESS round 99) unless an incoming path already exists here untracked or modified (then
log it and wait for the human). Ends when all 12 server-2 records are here, or after MAX_H hours.
Run: .venv/bin/python scripts/bg.py agent/logs/pull_s2.log '.venv/bin/python scripts/pull_s2_records.py'
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERIOD, MAX_H = 180, 8
ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0")
RUNS = ("r2pk_rl", "r2pk_s1_rl", "r2pk_s2_rl")
WANT = [f"runs/2026-10-29/{r}/eval/eval_pkg20_{i}__{r}+rep4.json" for r in RUNS for i in (0, 1)] + \
       [f"runs/2026-10-29/{r}/eval/eval_{k}10_{i}__{r}+rep4.json" for r in RUNS[1:] for k in ("pcb", "pkg", "ip") for i in (0, 1)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=ENV,
                       timeout=300)
    return r.returncode, (r.stdout + r.stderr).strip()


t0 = time.time()
log(f"start: waiting for {len(WANT)} server-2 F records")
while time.time() - t0 < MAX_H * 3600:
    have = sum(os.path.exists(os.path.join(ROOT, p)) for p in WANT)
    if have == len(WANT):
        log("all server-2 F records here, done")
        break
    try:
        rc, out = git("fetch", "-q", "r2e", "main")
        rc, n = git("rev-list", "--count", "HEAD..r2e/main")
        if rc == 0 and int(n or 0) > 0:
            _, inc = git("diff", "--name-only", "HEAD...r2e/main")
            _, dirty = git("status", "--porcelain", "--untracked-files=all", "--ignored", "--", *inc.split()) if inc else (0, "")
            if dirty:
                log(f"not merging ({n} behind): incoming paths exist here: {dirty.splitlines()[:5]}")
            else:
                rc, out = git("merge", "--no-edit", "r2e/main")
                log(f"merge {n} commits rc={rc}: {out.splitlines()[-1] if out else ''} | records {have} -> "
                    f"{sum(os.path.exists(os.path.join(ROOT, p)) for p in WANT)}/{len(WANT)}")
                if rc != 0:
                    git("merge", "--abort")
    except (subprocess.TimeoutExpired, ValueError) as e:
        log(f"git error: {e}")
    time.sleep(PERIOD)
else:
    log("timeout")
