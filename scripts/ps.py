"""List python processes with CPU time, elapsed time and RSS (ps/top are not in the tool allowlist).
Run: .venv/bin/python scripts/ps.py [pattern]"""
import os
import sys
import time

pat = sys.argv[1] if len(sys.argv) > 1 else "python"
clk = os.sysconf("SC_CLK_TCK")
boot = float([l for l in open("/proc/stat") if l.startswith("btime")][0].split()[1])
rows = []
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        if pat not in cmd:
            continue
        st = open(f"/proc/{pid}/stat").read().split(")")[-1].split()
        cpu = (int(st[11]) + int(st[12])) / clk
        start = boot + int(st[19]) / clk
        rss = int(st[21]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        rows.append((int(pid), cpu, time.time() - start, rss, cmd[:110]))
    except Exception:  # noqa: BLE001
        pass
print(f"load {open('/proc/loadavg').read().split()[:3]}")
for pid, cpu, el, rss, cmd in sorted(rows, key=lambda r: -r[1]):
    print(f"{pid:7d} cpu {cpu:7.0f}s  elapsed {el:6.0f}s  util {min(100, 100*cpu/max(el,1)):4.0f}%  rss {rss:6.0f}MB  {cmd}")
