"""One-off queue edit (11/02 07:10): rl-only prior-work baselines start from lam0 = 20 instead of 1.

bm_manoharan_rlonly (lam0 1) collapsed to all-skip: val fail 1.00 at it 100 / 200, loss 0.000 from it 75 (every sample
of a group fails with 0 decaps -> zero advantage, cannot recover). With lam0 1, "place nothing and fail" (R ~ -1.5) beats
a feasible random-init solution (count/ref 2.7 for Manoharan, 9.0 for Park in smoke_rlonly). The collapsed entry moves to
"train_removed" (the driver only reads "train", so a restart does not resume it); bm_manoharan_rlonly_l20 goes before
bp_park_rlonly, which gets --lam 20 too.
Run: .venv/bin/python scripts/queue_rlonly_lam.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "agent/day2_queue.json")
q = json.load(open(p))
tr = q["train"]
names = [t["name"] for t in tr]
if "bm_manoharan_rlonly_l20" in names:
    raise SystemExit("already applied")
old = tr.pop(names.index("bm_manoharan_rlonly"))
old["removed"] = ("11/02 07:10: collapsed to all-skip (val fail 1.00 it 100/200, loss 0.000 from it 75 = zero advantage); "
                  "lam0 1 makes 'place nothing and fail' beat a feasible start. Replaced by bm_manoharan_rlonly_l20.")
q.setdefault("train_removed", []).append(old)
j = [t["name"] for t in tr].index("bp_park_rlonly")
tr[j]["arm"] += " --lam 20"
tr.insert(j, dict(name="bm_manoharan_rlonly_l20", arm="--model manoharan --imit 0 --lam 20", seed=0, rl_only=True,
                  after="runs/analysis/smoke_rlonly.ok"))
json.dump(q, open(p, "w"), indent=1, ensure_ascii=False)
for t in tr[j - 1:j + 3]:
    print(t)
