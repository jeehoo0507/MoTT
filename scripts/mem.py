"""Pod memory guard (agent/QUEUE.md 2026-10-28 14:20: pod limit 60 GiB, suspected OOM restarts).

  mem_line()        "mem 12.3 GiB / 60.0 GiB, oom_kill 0" from the cgroup files
  wait_mem(log)     block while memory.current > LIMIT_GIB (45), logging every 5 min
Run directly to print the line: .venv/bin/python scripts/mem.py
"""
import time

LIMIT_GIB = 45.0
CG = "/sys/fs/cgroup"


def _read(name):
    try:
        return open(f"{CG}/{name}").read()
    except OSError:
        return ""


def mem_gib():
    s = _read("memory.current").strip()
    return int(s) / 2**30 if s.isdigit() else 0.0


def oom_kills():
    for line in _read("memory.events").splitlines():
        k, _, v = line.partition(" ")
        if k == "oom_kill":
            return int(v)
    return -1


def mem_line():
    mx = _read("memory.max").strip()
    mx = f"{int(mx) / 2**30:.1f} GiB" if mx.isdigit() else mx
    return f"mem {mem_gib():.1f} GiB / {mx}, oom_kill {oom_kills()}"


def wait_mem(log=print, limit=LIMIT_GIB):
    """Do not start a new process while the pod is above `limit` GiB."""
    t = 0.0
    while mem_gib() > limit:
        if time.time() - t > 300:
            log(f"memory above {limit} GiB, waiting: {mem_line()}")
            t = time.time()
        time.sleep(30)


if __name__ == "__main__":
    print(mem_line())
