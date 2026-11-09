"""Start a shell command detached from the agent session (new session, own log), so it survives the round end.

The agent's tool allowlist has python but not nohup/setsid; this is the replacement.
Usage: .venv/bin/python scripts/bg.py <log> '<command>'
Prints the pid; the command runs with the repo root as cwd.
"""
import os
import subprocess
import sys

log, cmd = sys.argv[1], sys.argv[2]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.dirname(os.path.join(root, log)) or ".", exist_ok=True)
with open(os.path.join(root, log), "a") as f:
    p = subprocess.Popen(["bash", "-c", cmd], cwd=root, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
print(p.pid)
