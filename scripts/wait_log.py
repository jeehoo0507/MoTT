"""Block until a log file contains one of the given strings (or timeout), then print its tail.
Run: .venv/bin/python scripts/wait_log.py <log> <timeout_s> <pattern> [<pattern> ...]
"""
import sys
import time

log, timeout, pats = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
t0 = time.time()
while time.time() - t0 < timeout:
    try:
        s = open(log).read()
    except OSError:
        s = ""
    hit = [p for p in pats if p in s]
    if hit:
        print("HIT", hit)
        break
    time.sleep(30)
else:
    print("TIMEOUT")
print("\n".join(s.splitlines()[-15:]))
