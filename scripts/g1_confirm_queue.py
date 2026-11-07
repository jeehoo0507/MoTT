"""G1 confirmation runs (QUEUE 11/07 10:50 KST step 3, PROPOSALS G1): append the 7 runs for the picked candidate X to
the end of "train" in agent/day2_queue.json, in the PROPOSALS order. Arms are X's seed-0 arm (and its held-out
variants) with only the seed changed. Prints the entries without --apply; use only after g1_screen.py says "pass".

  .venv/bin/python scripts/g1_confirm_queue.py r2pkfc            # dry run
  .venv/bin/python scripts/g1_confirm_queue.py r2pkfc --apply
"""
import json
import os
import sys

QUEUE = "agent/day2_queue.json"
HOLD_OUT = {  # same pool/val options as the lo_pcb_* / lo_pkg_* ladder entries
    "pcb": "--pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0",
    "pkg": "--pool_glob train_pcb*.pkl,train_ip*.pkl --val_pools val_pcb10_0,val_ip10_0",
}


def entries(q, x):
    base = next(j for j in q["train"] if j["name"] == x)
    held = next(j for j in q["train"] if j["name"] == f"lo_pkg_{x}")
    assert held["arm"] == f"{base['arm']} {HOLD_OUT['pkg']}", held["arm"]
    order = [(f"lo_pkg_{x}_s1", "pkg", 1), (f"lo_pkg_{x}_s2", "pkg", 2), (f"{x}_s1", None, 1), (f"{x}_s2", None, 2),
             (f"lo_pcb_{x}", "pcb", 0), (f"lo_pcb_{x}_s1", "pcb", 1), (f"lo_pcb_{x}_s2", "pcb", 2)]
    out = []
    for name, t, seed in order:
        e = {"name": name, "arm": base["arm"] if t is None else f"{base['arm']} {HOLD_OUT[t]}", "seed": seed}
        for k in ("after", "rl_only"):
            if k in base:
                e[k] = base[k]
        out.append(e)
    return out


def main():
    x = sys.argv[1]
    q = json.load(open(QUEUE))
    new = entries(q, x)
    have = {j["name"] for j in q["train"]}
    new = [e for e in new if e["name"] not in have]
    print(json.dumps(new, indent=1, ensure_ascii=False))
    if "--apply" not in sys.argv:
        print(f"dry run: {len(new)} entries not written")
        return
    q["train"] = q["train"] + new
    tmp = QUEUE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(q, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, QUEUE)
    print(f"{len(new)} appended; train {len(q['train'])}")


if __name__ == "__main__":
    main()
