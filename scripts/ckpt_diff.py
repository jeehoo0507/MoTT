"""Max |weight difference| between two checkpoints (and their iters).
Run: .venv/bin/python scripts/ckpt_diff.py <a.pt> <b.pt>
"""
import sys

import torch

a, b = (torch.load(p, map_location="cpu", weights_only=False) for p in sys.argv[1:3])
d = max((a["model"][k].float() - b["model"][k].float()).abs().max().item() for k in a["model"])
print(f"iter {a.get('iter')} vs {b.get('iter')}  max|diff| {d}")
