"""Run pending driver CPU-lane entries (agent/day2_queue.json "cpu") in the free 4th CPU slot, in queue order, until a
UTC deadline. Same cmd / log / done paths as the driver, so the driver later finds them done and skips them.
Only entries whose name starts with one of the given prefixes; one at a time; no new entry above 45 GiB (wait_mem)
or after the deadline. Usage: run_queue_entries.py HH:MM <name prefix> [...]"""
import calendar
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem import mem_line, wait_mem  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def main():
    hh, mm = map(int, sys.argv[1].split(":"))
    prefixes = sys.argv[2:]
    t = time.gmtime()
    deadline = calendar.timegm((t.tm_year, t.tm_mon, t.tm_mday, hh, mm, 0))
    while True:
        q = json.load(open("agent/day2_queue.json"))
        todo = [e for e in q["cpu"] if any(e["name"].startswith(p) for p in prefixes)
                and not all(os.path.exists(p) for p in e["done"])]
        if not todo:
            log("nothing left")
            return
        if time.time() > deadline:
            log(f"deadline {sys.argv[1]} UTC reached, {len(todo)} left")
            return
        e = todo[0]
        wait_mem(log)
        log(f"START {e['name']} | {mem_line()}")
        with open(e.get("log", "agent/logs/day2_cpu.log"), "a") as f:
            rc = subprocess.run(e["cmd"], shell=True, stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"END {e['name']} rc={rc} | {mem_line()}")
        if rc != 0 or not all(os.path.exists(p) for p in e["done"]):
            log("failed or missing outputs, stop")
            return


if __name__ == "__main__":
    main()
