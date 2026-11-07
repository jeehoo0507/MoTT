"""Copy a live process's stdout (/proc/<pid>/fd/1) over its on-disk log when the disk file was orphaned
(agent/loop.sh `git pull --autostash` recreates tracked logs; the process keeps writing to the old inode).

Run: .venv/bin/python scripts/restore_proc_log.py <pid> <log>   (copies only if the live stream is longer)
"""
import os
import sys

pid, dst = sys.argv[1], sys.argv[2]
src = f"/proc/{pid}/fd/1"
target = os.readlink(src)
live = open(src, "rb").read()
disk = open(dst, "rb").read() if os.path.exists(dst) else b""
print(f"fd/1 -> {target}: live {len(live)} B, disk {len(disk)} B")
if "(deleted)" not in target and os.path.samefile(src, dst):
    print("same inode, nothing to do")
elif len(live) > len(disk) and live.startswith(disk[:1000]):
    open(dst, "wb").write(live)
    print(f"restored {dst} ({len(live) - len(disk)} B new)")
else:
    print("not copied (live stream not longer or different head)")
