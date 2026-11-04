"""Run one command; kill it if the pod goes above the 45 GiB rule while it runs (QUEUE 11/04 18:20 KST item 1:
20-cell greedy+prune may exceed it -> skip that file and say "메모리로 생략" in RESULTS).

Usage: mem_guard.py <skip_marker.json> <cmd> [args...]
  - marker already exists -> exit 3 at once (the driver then skips the entry until it restarts)
  - pod memory.current > 45 GiB while the command runs -> kill its process group, write the marker
    {"reason": "memory", "mem_gib", "limit_gib", "peak_child_rss_gib", "time"}, exit 2
  - otherwise exit with the command's return code
collect_results reads `*.json.memskip` markers next to the eval records (not `*.json`, so record globs skip them).
"""
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem import LIMIT_GIB, mem_gib, mem_line  # noqa: E402


def main():
    marker, cmd = sys.argv[1], sys.argv[2:]
    if os.path.exists(marker):
        print(f"skip marker exists: {marker}", flush=True)
        sys.exit(3)
    print(f"mem_guard start | {mem_line()} | {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, start_new_session=True)
    peak = 0.0
    while p.poll() is None:
        m = mem_gib()
        peak = max(peak, m)
        if m > LIMIT_GIB:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            os.makedirs(os.path.dirname(marker) or ".", exist_ok=True)
            json.dump(dict(reason="memory", mem_gib=round(m, 2), limit_gib=LIMIT_GIB, peak_pod_gib=round(peak, 2),
                           cmd=cmd, time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())),
                      open(marker, "w"), indent=1)
            print(f"mem_guard KILLED at {m:.1f} GiB > {LIMIT_GIB} | {mem_line()} -> {marker}", flush=True)
            sys.exit(2)
        time.sleep(2)
    print(f"mem_guard end rc={p.returncode} peak pod {peak:.1f} GiB | {mem_line()}", flush=True)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
