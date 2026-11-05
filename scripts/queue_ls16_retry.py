"""Round 50 (11/05): retry ls16c eval_pcb16_0 / eval_pcb16_1. Both were stopped by mem_guard (pod > 45 GiB) within
40 s of a 4th-slot `ls10 scale-eval eval_pcb16_* r2pk_s2_rl` start (12:16:04 -> kill 12:16:41, 12:33:44 -> kill
12:34:12; agent/logs/side_slot.log, pipeline_day2.log), while the chunked LS worker itself sat flat at ~10 GB rss for
4 h. The overshoot came from the concurrent filler, not from LS, so the "메모리로 생략" markers do not describe LS16.
side_slot.py now starts no scale-eval filler while a driver LS16 job is alive.

For each of the two entries (record still missing):
  - keep the marker as history: `<record>.json.memskip` -> `<record>.json.memskip_concurrent` (out of the RESULTS glob)
  - rename the entry `ls16r ...` (the live driver skips failed names until it restarts); command unchanged
    (same --ls_chunk 8, same mem_guard 45 GiB rule and marker path -> a second overshoot still writes the marker)
Idempotent; atomic queue write."""
import json
import os

P = "agent/day2_queue.json"
SETS = ("eval_pcb16_0", "eval_pcb16_1")
q = json.load(open(P))
for e in q["cpu"]:
    if not e["name"].startswith(("ls16c ", "ls16r ")) or e["name"].split()[1] not in SETS:
        continue
    rec = e["done"][0]
    if os.path.exists(rec):
        continue
    mk = rec + ".memskip"
    if os.path.exists(mk):
        os.replace(mk, mk + "_concurrent")
    if e["name"].startswith("ls16c "):
        e["name"] = "ls16r " + e["name"][len("ls16c "):]
tmp = P + ".tmp"
open(tmp, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
os.replace(tmp, P)
for e in q["cpu"]:
    if e["name"].startswith(("ls16c ", "ls16r ")):
        print(e["name"], "|", e["cmd"][-100:])
