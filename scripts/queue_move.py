"""Move CPU-lane entries of agent/day2_queue.json to just after another entry (the driver reads the file before
every step, so this reorders what runs next without restarting it).

  .venv/bin/python scripts/queue_move.py "<after name>" "<name>" ["<name>" ...]
"""
import json
import os
import sys

QUEUE = "agent/day2_queue.json"


def main():
    after, names = sys.argv[1], sys.argv[2:]
    q = json.load(open(QUEUE))
    cpu = q["cpu"]
    moved = [j for n in names for j in cpu if j["name"] == n]
    if len(moved) != len(names):
        sys.exit(f"not found: {set(names) - {j['name'] for j in moved}}")
    rest = [j for j in cpu if j["name"] not in names]
    idx = [j["name"] for j in rest].index(after) + 1
    q["cpu"] = rest[:idx] + moved + rest[idx:]
    tmp = QUEUE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(q, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, QUEUE)
    for i, j in enumerate(q["cpu"][idx - 1: idx + len(moved) + 1], idx - 1):
        print(i, j["name"])


if __name__ == "__main__":
    main()
