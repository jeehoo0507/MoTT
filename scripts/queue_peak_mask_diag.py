"""Prioritize the user-requested CPU diagnostic after the current running job; idempotent."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = "reports/peak_mask_diag_20261106"


def entries():
    specs = [("val_pkg10_0", 0), ("eval_pkg16_0", 0), ("eval_pcb16_0", 0),
             ("eval_ip16_0", 0), ("eval_pkg16_0", 1), ("eval_pkg16_0", 2)]
    jobs = []
    for pool, seed in specs:
        suffix = "" if seed == 0 else f"_s{seed}"
        tag = f"{pool}_s{seed}"
        jobs.append(dict(name=f"peakmask {tag}",
            cmd=f".venv/bin/python scripts/diag_peak_mask.py --pool runs/pool_v1/{pool}.pkl "
                f"--ckpt runs/2026-10-29/r2pk{suffix}_rl/best.pt --n 8 --max_seconds 3600 "
                f"--out {OUT}/{tag}.json",
            done=[f"{OUT}/{tag}.done"], log=f"agent/logs/peakmask_{tag}.log"))
    return jobs


def main():
    p = ROOT / "agent/day2_queue.json"
    q = json.loads(p.read_text())
    names = {j["name"] for j in q["cpu"]}
    new = [j for j in entries() if j["name"] not in names]
    q["cpu"] = new + q["cpu"]
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(p)
    print("Queued", len(new), "CPU diagnostics. GPU queue untouched.")


if __name__ == "__main__":
    main()
