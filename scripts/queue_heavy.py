"""PROGRESS 11/03 round 27: move the heavy CPU-queue entries (6 `ls16 *` LS 16-cell files, 6 `base20 * greedy+prune`
20-cell g+p files) from `cpu` (the driver's lane) to a separate `heavy` list that only scripts/side_lane16.py runs,
one at a time. Why: with the GPU idle, side_lane16 uses the 2 free exact-check slots; if the driver and the lane both
took entries from the same list they could start the same heavy job twice (16-cell g+p/LS rss 19-34 GB each).
Order: ls16 pcb -> ip -> pkg (as queued in round 24), then base20 g+p pcb -> pkg -> ip (each needs its 20-cell file).
base20 g+p entries get "solo": the lane waits until no other eval_policy / build_pools process is alive before
starting one (20-cell g+p memory is not measured yet)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
q = json.load(open(Q))


def is_heavy(e):
    return e["name"].startswith("ls16 ") or (e["name"].startswith("base20 ") and e["name"].endswith(" greedy+prune"))


moved = [e for e in q["cpu"] if is_heavy(e)]
q["cpu"] = [e for e in q["cpu"] if not is_heavy(e)]
for e in moved:
    if e["name"].startswith("base20 "):
        e["solo"] = True
heavy = q.setdefault("heavy", [])
have = {e["name"] for e in heavy}
heavy += [e for e in moved if e["name"] not in have]
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print("heavy:", [e["name"] for e in heavy])
print("cpu left:", len(q["cpu"]))
