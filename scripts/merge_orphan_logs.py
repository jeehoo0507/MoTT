"""Keep the orphaned stdout of several live processes and append their lines to the on-disk log once they exit.

restore_proc_log.py copies one process's stream over the disk file; that loses lines when several processes share one
log (11/10: speed_fresh.sh U build, 6 processes `>> build.log`, the file swapped by loop.sh `git pull --autostash`).
Here we open /proc/<pid>/fd/1 ourselves, so the deleted inode stays readable after the writer exits, wait for all
writers, then append every line that the disk file does not already have (in stream order).

Run: .venv/bin/python scripts/merge_orphan_logs.py <log> <pid> [<pid> ...]
"""
import os
import sys
import time

dst, pids = os.path.abspath(sys.argv[1]), sys.argv[2:]
held = {}
for pid in pids:
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError as e:
        print(f"{pid}: {e}", flush=True)
        continue
    if target == dst + " (deleted)":
        held[pid] = open(f"/proc/{pid}/fd/1", "rb")
    print(f"{pid}: fd/1 -> {target}{'  (held)' if pid in held else ''}", flush=True)
while any(os.path.exists(f"/proc/{p}") for p in held):
    time.sleep(30)
disk = set(open(dst, "rb").read().splitlines()) if os.path.exists(dst) else set()
new = []
for f in held.values():
    f.seek(0)
    for line in f.read().splitlines():
        if line not in disk:
            disk.add(line)
            new.append(line)
with open(dst, "ab") as out:
    for line in new:
        out.write(line + b"\n")
print(f"{time.strftime('%m-%d %H:%M')} appended {len(new)} lines to {dst}", flush=True)
