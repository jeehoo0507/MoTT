"""Check collect_results.prior_seed_table before any B1 run exists: use r2_mott (seeds 0-2 evaluated) as a stand-in
"re-implementation", so the table must show r2_mott's per-seed success and r2pk - r2_mott per seed (these per-seed values
equal the P1 rows of RESULTS). Prints the table; writes nothing.
Run: .venv/bin/python scripts/test_prior_seed_table.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_results as c  # noqa: E402

recs = c.load_evals()
print("real bases:", len(c.prior_seed_table(recs)), "lines (expect 0 before B1 runs)")
c.PRIOR_SEED_BASES = ("r2_mott",)
print("\n".join(c.prior_seed_table(recs)))
