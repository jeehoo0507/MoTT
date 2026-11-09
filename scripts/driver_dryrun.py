"""Dry run of scripts/pipeline_day2.py's next picks (no process started): what a freshly started driver would do
first in each lane, and whether it would ATTACH to a live job. Check before restarting the driver.
Run: .venv/bin/python scripts/driver_dryrun.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_day2 as p  # noqa: E402


def main():
    q = p.queue()
    todo = [t for t in q["train"] if not p.rl_done(t["name"]) and (not t.get("after") or p.exists(t["after"]))]
    if todo:
        t = todo[0]
        print("GPU todo[0]:", t["name"], "rl_only" if t.get("rl_only") else "", "| live -> ATTACH" if p.alive(f"run_rung.py --name {t['name']} ") else "| NOT live -> would START")
    print("GPU todo:", [t["name"] for t in todo])
    jobs = p.eval_jobs() + q.get("cpu", [])
    cj = [j for j in jobs if not all(p.exists(x) for x in j["done"])]
    if cj:
        j = cj[0]
        print("CPU todo[0]:", j["name"], "| live -> ATTACH" if p.alive(j["cmd"].replace("'", "")) else "| NOT live -> would START")
    print("CPU todo (first 6):", [j["name"] for j in cj[:6]])


if __name__ == "__main__":
    main()
