"""Restore tracked files that one commit deleted by accident.

usage: restore_mass_delete.py BAD_COMMIT [--apply]
Lists files deleted in BAD_COMMIT (vs its parent) that are still absent at HEAD,
and files later commits touched (those are skipped). --apply checks the rest out
from the parent into index + working tree (no commit).
"""
import collections
import subprocess
import sys


def git(*a):
    return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout


bad = sys.argv[1]
apply = "--apply" in sys.argv
parent = git("rev-parse", bad + "^").strip()
deleted = [f for f in git("diff", "-z", "--name-only", "--diff-filter=D", parent, bad).split("\0") if f]
other = git("diff", "--name-status", "--diff-filter=AM", parent, bad)
later = set(git("diff", "-z", "--name-only", bad, "HEAD").split("\0"))
head = set(git("ls-tree", "-z", "-r", "--name-only", "HEAD").split("\0"))
todo = [f for f in deleted if f not in head and f not in later]
skip = [f for f in deleted if f in later or f in head]
print(f"parent {parent[:8]}  deleted {len(deleted)}  to restore {len(todo)}  skipped {len(skip)}")
print("added/modified in bad commit:\n" + other)
for f in skip:
    print("skip (touched later or present):", f)
c = collections.Counter("/".join(f.split("/")[:2]) for f in todo)
for k, n in c.most_common(30):
    print(f"{n:5d} {k}")
if apply:
    for i in range(0, len(todo), 200):
        git("checkout", parent, "--", *todo[i:i + 200])
    print("restored", len(todo))
