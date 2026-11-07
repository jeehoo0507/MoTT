"""Move CPU-lane entries of agent/day2_queue.json into "cpu_deferred" (the driver never reads that key), or back to
the end of "cpu" with --restore. QUEUE 11/07 10:45 KST step 1: LS16 entries wait until the method search is done.

  .venv/bin/python scripts/queue_defer.py "<name>" ["<name>" ...]
  .venv/bin/python scripts/queue_defer.py --restore "<name>" ["<name>" ...]
"""
import json
import os
import sys

QUEUE = "agent/day2_queue.json"


def main():
    restore = sys.argv[1] == "--restore"
    names = sys.argv[2:] if restore else sys.argv[1:]
    q = json.load(open(QUEUE))
    src, dst = ("cpu_deferred", "cpu") if restore else ("cpu", "cpu_deferred")
    q.setdefault("cpu_deferred", [])
    moved = [j for n in names for j in q[src] if j["name"] == n]
    if len(moved) != len(names):
        sys.exit(f"not found in {src}: {set(names) - {j['name'] for j in moved}}")
    q[src] = [j for j in q[src] if j["name"] not in names]
    q[dst] = q[dst] + moved
    tmp = QUEUE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(q, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, QUEUE)
    print(f"{len(moved)} moved {src} -> {dst}; cpu {len(q['cpu'])}, cpu_deferred {len(q['cpu_deferred'])}")


if __name__ == "__main__":
    main()
