"""Reproduce the 'Intel oneMKL ERROR: Parameter 6 was incorrect on entry to ZLASWP' seen in
verify._probe_z_exact (batched complex128 solve on CPU) and test workarounds.

Run: .venv/bin/python scripts/mkl_repro.py [threads] [batch] [n]
"""
import os
import sys
import time

if len(sys.argv) > 4:                       # e.g. MKL_NUM_THREADS=1,MKL_THREADING_LAYER=GNU
    for kv in sys.argv[4].split(","):
        k, v = kv.split("=")
        os.environ[k] = v
import torch  # noqa: E402

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 256
n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
torch.set_num_threads(threads)
print(f"torch {torch.__version__} threads {torch.get_num_threads()} batch {batch} n {n}", flush=True)
print(torch.__config__.parallel_info(), flush=True)
g = torch.Generator().manual_seed(0)
A = torch.randn(batch, n, n, dtype=torch.complex128, generator=g) + n * torch.eye(n, dtype=torch.complex128)
rhs = torch.randn(n, 3, dtype=torch.complex128, generator=g)
for bs in (batch, 64, 16, 1):
    t = time.time()
    try:
        xs = [torch.linalg.solve(A[i:i + bs], rhs.expand(min(bs, batch - i), n, 3)) for i in range(0, batch, bs)]
        x = torch.cat(xs)
        res = (A @ x - rhs).abs().max().item()
        print(f"  chunk {bs:4d}: ok, residual {res:.1e} ({time.time()-t:.2f}s)", flush=True)
    except Exception as e:
        print(f"  chunk {bs:4d}: FAIL {type(e).__name__}: {e}", flush=True)
