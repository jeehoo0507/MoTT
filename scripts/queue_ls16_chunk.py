"""Round 46 (11/05): LS16 memory fix. Every unchunked `ls16 ...` entry so far went above 45 GiB (pcb16_0 kernel OOM x2,
pcb16_1 / ip16_0 / ip16_1 stopped by mem_guard): each destroy-and-repair pass runs greedy on max(32, placed) copies of
one 16-cell instance at once (~1 GB each). `eval_policy.py --ls_chunk 8` repairs the copies (and runs the starting
greedy+prune) 8 at a time -- memory only: on eval_ip10_0 the chunked run gives the recorded LS solutions bit for bit
(scripts/check_ls_chunk.py: ok / count / worst / occ 25/25, peak rss 2.9 vs 4.4 GB).

For each `ls16 ...` CPU entry whose record does not exist yet:
  - keep an old marker as history: `<record>.json.memskip` -> `<record>.json.memskip_unchunked` (out of the RESULTS glob)
  - rename the entry `ls16c ...` (the live driver skips failed names until it restarts) and add `--ls_chunk 8` to its
    command; still wrapped in mem_guard (same 45 GiB rule, same marker path)
Idempotent; atomic queue write."""
import json
import os

P = "agent/day2_queue.json"
q = json.load(open(P))
for e in q["cpu"]:
    if not e["name"].startswith(("ls16 ", "ls16c ")):
        continue
    rec = e["done"][0]
    if os.path.exists(rec):
        continue
    mk = rec + ".memskip"
    if os.path.exists(mk):
        os.replace(mk, mk + "_unchunked")
    if e["name"].startswith("ls16 "):
        e["name"] = "ls16c " + e["name"][len("ls16 "):]
    if "--ls_chunk" not in e["cmd"]:
        e["cmd"] = e["cmd"].replace("--methods ls ", "--methods ls --ls_chunk 8 ")
        assert "--ls_chunk 8" in e["cmd"], e["cmd"]
tmp = P + ".tmp"
open(tmp, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
os.replace(tmp, P)
for e in q["cpu"]:
    if e["name"].startswith(("ls16 ", "ls16c ")):
        print(e["name"], "|", e["cmd"][-120:])
