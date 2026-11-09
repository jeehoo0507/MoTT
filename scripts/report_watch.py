"""Results auto-reporter (QUEUE 2026-10-29 14:20 item 5), so results keep reaching the remote after the agent
loop stops. Every 30 min: if an eval record under runs/**/eval/*.json is new (untracked) or changed, regenerate
RESULTS.md (scripts/collect_results.py), commit only RESULTS.md and those eval records ("results: auto (lo_*)",
`git add -f` because runs/ is gitignored), pull --rebase --autostash, push to r2e main. Unpushed commits from
a failed cycle are pushed on the next one. Ends itself after the cycle that has all 27 leave-one-type-out rungs
(lo_<t>_r{0,1,2}, seeds 0-2) evaluated on every 10×10 eval pool and pushed, or after MAX_H hours. Never touches the
driver or any training/eval process; a failed cycle is logged and retried.
Run: .venv/bin/python scripts/bg.py agent/logs/report_watch.log '.venv/bin/python scripts/report_watch.py'
"""
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DAY = "2026-10-29"
POOL = "runs/pool_v1"
REMOTE, BRANCH = "r2e", "main"
PERIOD_S = 1800
MAX_H = 96
# QUEUE 10/31: seeds 1-2 of the 9 rungs too (run names lo_<t>_<r>_s<seed>), so it waits for 27 rungs
LO = [f"lo_{t}_{r}{s}" for s in ("", "_s1", "_s2") for t in ("ip", "pcb", "pkg") for r in ("r0", "r1", "r2")]
# git never prompts for credentials and times out: after the 10/30 workspace restart a push waited 32 h for a
# prompt nobody could see
GIT_TIMEOUT_S = 300


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}] {msg}", flush=True)


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       timeout=GIT_TIMEOUT_S, env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="true"))
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} rc={r.returncode}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout


def new_evals():
    files = sorted(os.path.relpath(f, ROOT) for f in glob.glob(os.path.join(ROOT, "runs/**/eval/*.json"), recursive=True))
    tracked = set(git("ls-files", "runs").split())
    changed = set(git("diff", "--name-only", "--", "runs").split())
    return [f for f in files if f not in tracked or f in changed]


def all_lo_done():
    sets = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, POOL, "eval_*10_*.pkl")))
    return bool(sets) and all(os.path.exists(os.path.join(ROOT, f"runs/{DAY}/{n}_rl/eval/{s}__{n}_rl.json"))
                              for n in LO for s in sets)


def ahead():
    git("fetch", "-q", REMOTE, BRANCH)
    return int(git("rev-list", "--count", f"{REMOTE}/{BRANCH}..HEAD").strip() or 0)


def cycle():
    """One cycle; returns True when everything committed so far is on the remote."""
    new = new_evals()
    if new:
        r = subprocess.run([PY, "scripts/collect_results.py"], cwd=ROOT, capture_output=True, text=True)
        log(f"collect_results rc={r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}")
        paths = new + (["RESULTS.md"] if r.returncode == 0 else [])
        git("add", "-f", "--", *paths)
        git("commit", "-q", "-m", "results: auto (lo_*)",
            "-m", f"{len(new)} eval records: " + ", ".join(os.path.basename(p) for p in new[:12])
            + (" ..." if len(new) > 12 else ""), "--", *paths)
        log(f"committed {len(new)} eval records" + (" + RESULTS.md" if r.returncode == 0 else ""))
    n = ahead()
    if not n:
        if not new:
            log("변화 없음 (no new eval json, nothing to push)")
        return True
    git("pull", "-q", "--rebase", "--autostash", REMOTE, BRANCH)
    git("push", "-q", REMOTE, f"HEAD:{BRANCH}")
    log(f"pushed {n} commit(s)")
    return True


def main():
    log(f"report_watch start (period {PERIOD_S} s, waits for {len(LO)} lo_* rungs)")
    t0 = time.time()
    while True:
        done = all_lo_done()          # read before the cycle: the cycle then commits the last records
        try:
            pushed = cycle()
        except Exception as e:  # noqa: BLE001 — log and retry next cycle; never stop the driver
            pushed = False
            log(f"cycle failed: {e}")
            if os.path.isdir(os.path.join(ROOT, ".git/rebase-merge")) or os.path.isdir(os.path.join(ROOT, ".git/rebase-apply")):
                git("rebase", "--abort", check=False)
                log("rebase aborted")
        if done and pushed:
            log("all lo_* rungs evaluated and pushed; exit")
            return
        if time.time() - t0 > MAX_H * 3600:
            log(f"{MAX_H} h limit; exit")
            return
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    main()
