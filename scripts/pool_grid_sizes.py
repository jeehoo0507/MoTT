"""Working-grid size (frequency points per board) of pool files.
Run: .venv/bin/python scripts/pool_grid_sizes.py <glob> [<glob> ...]   (globs relative to runs/pool_v1)
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from decap_loop.bank import load_bank  # noqa: E402

for g in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(ROOT, "runs/pool_v1", g))):
        b, _ = load_bank(p)
        print(os.path.basename(p), tuple(b.freqs.shape))
