"""SIGTERM every process whose command line contains one of the patterns (the agent has no kill/pkill).
Run: .venv/bin/python scripts/kill_pat.py '<pattern>' [...]   (prints what it killed)
"""
import os
import signal
import sys

me = str(os.getpid())
for pid in filter(str.isdigit, os.listdir("/proc")):
    if pid == me:
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        continue
    if "kill_pat.py" in cmd:
        continue
    if any(p in cmd for p in sys.argv[1:]):
        try:
            os.kill(int(pid), signal.SIGTERM)
            print("killed", pid, cmd[:150])
        except OSError as e:
            print("fail", pid, e)
