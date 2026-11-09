"""Inspect the public DevFormer DPP data (QUEUE 10/29 03:30 item B) without executing any of its code.

Reads only .npy arrays (allow_pickle=False) and the zip directory listing. Run with `python -I`.
Usage: python -I scripts/ext_devformer_inspect.py /home/coder/ext_devformer/devformer
"""
import sys
import zipfile
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
sim = root / "data" / "dpp" / "simulator"
for name in ["10x10_pkg_chip.npy", "01nF_decap.npy", "freq_201.npy"]:
    a = np.load(sim / name, allow_pickle=False)
    print(name, a.shape, a.dtype, "min", np.abs(a).min(), "max", np.abs(a).max())
    if np.iscomplexobj(a):
        print("  complex; mean |imag|/|real|", float(np.abs(a.imag).mean() / (np.abs(a.real).mean() + 1e-30)))
f = np.load(sim / "freq_201.npy", allow_pickle=False)
print("freq first/last", f[0], f[-1])
z = np.load(sim / "10x10_pkg_chip.npy", allow_pickle=False)
print("symmetric max rel err", float(np.abs(z - np.transpose(z, (0, 2, 1))).max() / np.abs(z).max()))
d = np.load(sim / "01nF_decap.npy", allow_pickle=False).reshape(-1)
print("decap |Z| at f idx 0, 50, 100, 200:", [complex(d[i]) for i in (0, 50, 100, 200)])

with zipfile.ZipFile(root / "data" / "data.zip") as zf:
    for info in zf.infolist():
        print(f"  zip: {info.filename}  {info.file_size}")
    with zf.open("test.npz") as fh:
        t = np.load(fh, allow_pickle=False)
        for k in t.files:
            print("  test.npz", k, t[k].shape, t[k].dtype, t[k].reshape(-1)[:8])
