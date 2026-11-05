"""Print the full command line of processes matching a pattern (scripts/ps.py cuts it at 110 chars).
Run: python3 scripts/cmdline.py <pattern>"""
import os
import sys

pat = sys.argv[1]
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        continue
    if pat in cmd and "cmdline.py" not in cmd:
        print(pid, cmd)
