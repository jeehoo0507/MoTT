"""Unattended server-2 loop experiments. Standard-library orchestration; no protocol changes.

pack is safe to run while server 1 is training: only the fixed pool and seven completed
checkpoints are read. run calls the existing training/evaluation entrypoints unchanged.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

DAY = "2026-11-05s2"
OLD = Path("runs/2026-10-29")
NEW = Path("runs") / DAY
LOG = Path("agent/logs/s2_once")
BASE = ["r2pk", "r2pk_s1", "r2pk_s2"]
LOOP = ["r3pk", "r3pk_s1", "r3pk_s2"]
LEGACY = ["r2_mott_s1", "r2_mott_s2", "r3_loop_s1", "r3_loop_s2"]
CHILDREN = set()
CHILD_LOCK = threading.Lock()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sets(sizes):
    return [f"eval_{t}{n}_{k}" for n in sizes for t in ("pcb", "pkg", "ip") for k in (0, 1)]


def plan():
    jobs = []

    def add(phase, name, folder, sizes, out=None, repeat=None):
        method = name + "_rl" + (f"@R{repeat}" if repeat else "")
        for pool in sets(sizes):
            dst = Path(out or folder) / "eval" / f"{pool}__{method}.json"
            jobs.append(dict(phase=phase, method=method, pool=pool,
                             ckpt=str(Path(folder) / "best.pt"), out=str(Path(out or folder)),
                             result=str(dst), state="pending"))

    for name in LEGACY:
        add("L1", name, OLD / (name + "_rl"), (16, 20))
    for name in LOOP + ["lo_pcb_r3pk"]:
        add("L2", name, NEW / (name + "_rl"), (10,))
    for name in BASE:
        add("L3", name, OLD / (name + "_rl"), (16, 20), out=NEW / ("l3_" + name))
    for name in LOOP:
        folder = NEW / (name + "_rl")
        add("L3", name, folder, (16, 20))  # Default R=8; reused for L4.
        for repeat in (2, 4):
            add("L4", name, folder, (16, 20), repeat=repeat)
    return jobs


def complete(job):
    r = read_json(job["result"])
    if not isinstance(r, dict):
        return False
    n = r.get("n", 0)
    return (isinstance(n, int) and n > 0 and r.get("set") == job["pool"]
            and r.get("method") == job["method"] and r.get("ckpt") == job["ckpt"]
            and r.get("samples") == 1 and r.get("inst_chunk") == 5
            and all(isinstance(r.get(k), list) and len(r[k]) == n for k in ("ok", "count", "worst", "occ")))


def trained(name):
    folder = NEW / (name + "_rl")
    return isinstance(read_json(folder / "metrics.json"), dict) and (folder / "best.pt").is_file()


def pack(archive):
    files = sorted(p for p in Path("runs/pool_v1").rglob("*") if p.is_file())
    files += [OLD / (n + "_rl") / "best.pt" for n in LEGACY + BASE]
    if not files or not Path("runs/pool_v1/train_pcb10_0.pkl").is_file():
        raise RuntimeError("runs/pool_v1 is missing")
    for p in files:
        if not p.is_file():
            raise RuntimeError(f"Missing transfer file: {p}")
    manifest = {str(p): sha(p) for p in files}
    archive = Path(archive).resolve()
    with tarfile.open(str(archive) + ".partial", "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p), recursive=False)
        raw = json.dumps(manifest, indent=2).encode()
        entry = tarfile.TarInfo("server2_transfer_manifest.json")
        entry.size = len(raw)
        tar.addfile(entry, io.BytesIO(raw))
    Path(str(archive) + ".partial").replace(archive)
    print(f"Packed {len(files)} files -> {archive}", flush=True)


def unpack(archive):
    import shutil
    with tarfile.open(archive, "r:gz") as tar:
        manifest = json.load(tar.extractfile("server2_transfer_manifest.json"))
        for member in tar.getmembers():
            p = Path(member.name)
            allowed = (member.name == "server2_transfer_manifest.json" or
                       (member.name in manifest and (p.is_relative_to("runs/pool_v1") or
                        member.name in {str(OLD / (n + "_rl") / "best.pt") for n in LEGACY + BASE})))
            if p.is_absolute() or ".." in p.parts or not member.isfile() or not allowed:
                raise RuntimeError(f"Unexpected archive member: {member.name}")
            if p.resolve() != Path.cwd().joinpath(p).absolute():
                raise RuntimeError(f"Refusing extraction through a symlink: {p}")
        for member in tar.getmembers():
            p = Path(member.name)
            if p.exists() and p.name != "server2_transfer_manifest.json":
                if sha(p) != manifest[member.name]:
                    raise RuntimeError(f"Existing input differs: {p}; use a fresh checkout")
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, p.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print("Inputs extracted; full checksums will be verified before training", flush=True)


def execute(cmd, log, env=None, timeout=8 * 3600):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write("\n" + json.dumps(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, **(env or {})}, start_new_session=True)
        with CHILD_LOCK:
            CHILDREN.add(p)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return 124
        finally:
            with CHILD_LOCK:
                CHILDREN.discard(p)


def gpu_worker(gpu, names):
    for name, seed in names:
        if trained(name):
            continue
        arm = "--model loop --peak_refine"
        if name.startswith("lo_"):
            arm += " --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0"
        cmd = [sys.executable, "scripts/run_rung.py", "--name", name, "--arm", arm,
               "--day", DAY, "--pool", "runs/pool_v1", "--seed", str(seed),
               "--distill_hours", "0.75", "--rl_hours", "1.25", "--skip_eval"]
        print(f"TRAIN {name}: starting on GPU {gpu}", flush=True)
        rc = execute(cmd, LOG / f"train_{name}.log", {"CUDA_VISIBLE_DEVICES": str(gpu), "CERTIFY_WORKERS": "2"})
        print(f"TRAIN {name}: {'done' if rc == 0 and trained(name) else 'FAILED'}", flush=True)


def evaluate(job):
    dst = Path(job["result"])
    if complete(job):
        return "done"
    if dst.exists():  # The legacy evaluator skips any existing path, even truncated JSON.
        dst.rename(str(dst) + f".invalid.{time.time_ns()}")
    cmd = [sys.executable, "scripts/eval_policy.py", "--pool", "runs/pool_v1",
           "--sets", job["pool"], "--policy", f"{job['method']}={job['ckpt']}",
           "--workers", "1", "--inst_chunk", "5", "--out", job["out"]]
    rc = execute(cmd, LOG / f"{job['phase']}_{job['pool']}__{job['method']}.log", timeout=4 * 3600)
    return "done" if rc == 0 and complete(job) else "failed"


def available_gib():
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    # Respect common container cgroup limits in addition to host RAM.
    for limit, used in [("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")]:
        try:
            available = min(available, int(Path(limit).read_text()) - int(Path(used).read_text()))
        except (OSError, ValueError):
            pass
    return available / 1024 ** 3


def preflight():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to train on CPU")
    manifest = read_json("server2_transfer_manifest.json")
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError("Missing transfer manifest; use the pack command")
    print(f"Verifying {len(manifest)} transferred files...", flush=True)
    for path, expected in manifest.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError(f"Transfer checksum mismatch: {path}")
    required = [Path("runs/pool_v1") / (s + ".pkl") for s in sets((10, 16, 20))]
    required += [OLD / (n + "_rl") / "best.pt" for n in LEGACY + BASE]
    required += [Path("runs/pool_v1") / f"{split}_{t}{n}_{k}.pkl"
                 for split, sizes, copies in [("train", (8, 10, 12), range(3)), ("val", (10,), range(1))]
                 for t in ("pcb", "pkg", "ip") for n in sizes for k in copies]
    for p in required:
        if not p.is_file():
            raise RuntimeError(f"Missing input: {p}")
        if p.name.startswith("train_") and p.suffix == ".pkl":
            teacher = Path("data/teacher") / (p.stem + ".npz")
            if not teacher.is_file():
                raise RuntimeError(f"Missing teacher: {teacher}; refusing a silent greedy fallback")
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def receipt(jobs, metadata, final=False):
    counts = {s: sum(j["state"] == s for j in jobs) for s in ("pending", "running", "done", "failed", "blocked")}
    save_json(LOG / "status.json", dict(metadata, updated=time.time(), final=final, counts=counts, jobs=jobs))
    with (LOG / "summary.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "pool", "method", "n", "success_rate", "count_success_mean", "sec_total", "status"])
        for j in jobs:
            r = read_json(j["result"]) if j["state"] == "done" else None
            if r:
                solved = [c for c, ok in zip(r["count"], r["ok"]) if ok]
                writer.writerow([j["phase"], j["pool"], j["method"], r["n"], sum(r["ok"]) / r["n"],
                                 sum(solved) / len(solved) if solved else "", r.get("sec_total"), j["state"]])
            else:
                writer.writerow([j["phase"], j["pool"], j["method"], "", "", "", "", j["state"]])
    return counts


def bundle(jobs):
    files = {Path(j["result"]) for j in jobs if complete(j)}
    files.update(LOG.glob("*.json"))
    files.add(LOG / "summary.csv")
    files.update(NEW.glob("*/metrics.json"))
    files.update(NEW.glob("*/driver.log"))
    for p in files:
        if not p.is_file():
            raise RuntimeError(f"Missing result artifact: {p}")
    dst = Path("server2_loop_results.tgz")
    with tarfile.open(str(dst) + ".partial", "w:gz") as tar:
        for p in sorted(files | set(LOG.glob("*.log"))):
            tar.add(p, arcname=str(p), recursive=False)
    Path(str(dst) + ".partial").replace(dst)
    return sorted(files)


def publish(files):
    # Publish only allowlisted outputs using an isolated checkout. Never stage user changes.
    import shutil
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    with tempfile.TemporaryDirectory(prefix="r2e-s2-publish-") as temp:
        def git(*args):
            return subprocess.run(["git", "-C", temp, *args], env=env, check=True, timeout=180)
        subprocess.run(["git", "clone", "--depth", "1", origin, temp], env=env, check=True, timeout=300)
        for p in files:
            target = Path(temp) / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        git("add", "-f", "--", *map(str, files))
        if subprocess.run(["git", "-C", temp, "diff", "--cached", "--quiet"]).returncode == 0:
            return
        git("-c", "user.name=R2E server 2", "-c", "user.email=server2@localhost", "commit", "-m",
            "server2: unattended loop experiment results (L1-L4)")
        for attempt in range(3):
            try:
                git("push", "origin", "HEAD:main")
                return
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                git("fetch", "origin", "main")
                git("rebase", "FETCH_HEAD")


def run(cpu_par, push):
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / "runner.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A server2_once runner is already active")
        gpus = preflight()
        print("Preflight OK:", gpus, flush=True)
        jobs = plan()
        train = list(zip(LOOP + ["lo_pcb_r3pk"], (0, 1, 2, 0)))
        metadata = dict(pid=os.getpid(), started=time.time(), gpus=gpus, cpu_parallel=cpu_par,
                        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        note="L3/L4 supplementary; existing pools; no change to L1/L2 verdicts; timings are server 2 only")
        for j in jobs:
            if complete(j):
                j["state"] = "done"
        try:
            with ThreadPoolExecutor(max_workers=len(gpus)) as trainers, ThreadPoolExecutor(max_workers=cpu_par) as evaluators:
                training = [trainers.submit(gpu_worker, g, train[g::len(gpus)]) for g in range(min(len(gpus), len(train)))]
                active = {}
                idle_since = None
                milestones = set()
                while any(j["state"] in ("pending", "running") for j in jobs) or not all(t.done() for t in training):
                    for f in list(active):
                        if f.done():
                            j = active.pop(f)
                            try:
                                j["state"] = f.result()
                            except Exception as e:
                                j.update(state="failed", error=str(e))
                            print(f"{j['phase']} {j['pool']} {j['method']}: {j['state']}", flush=True)
                    finished = all(t.done() for t in training)
                    for j in sorted(jobs, key=lambda x: {"L2": 0, "L1": 1, "L3": 2, "L4": 3}[x["phase"]]):
                        if j["state"] != "pending":
                            continue
                        if j["phase"] == "L4" and (not finished or any(
                                q["phase"] != "L4" and q["state"] in ("pending", "running") for q in jobs)):
                            continue
                        ckpt = Path(j["ckpt"])
                        needs_training = ckpt.is_relative_to(NEW)
                        ready = (trained(ckpt.parent.name.removesuffix("_rl")) if needs_training else ckpt.is_file())
                        if not ready:
                            if finished:
                                j["state"] = "blocked"
                            continue
                        if len(active) >= cpu_par or available_gib() < 12:
                            break
                        j["state"] = "running"
                        active[evaluators.submit(evaluate, j)] = j
                    if finished and not active and any(j["state"] == "pending" for j in jobs):
                        idle_since = idle_since or time.monotonic()
                        if time.monotonic() - idle_since > 600:
                            for j in jobs:
                                if j["state"] == "pending":
                                    j.update(state="blocked", error="Insufficient available RAM for 10 minutes")
                    else:
                        idle_since = None
                    receipt(jobs, metadata)
                    for phase in ("L1", "L2", "L3", "L4"):
                        group = [j for j in jobs if j["phase"] == phase]
                        if group and phase not in milestones and all(j["state"] not in ("pending", "running") for j in group):
                            bundle(jobs)
                            milestones.add(phase)
                            print(f"{phase} finished; result archive refreshed", flush=True)
                    time.sleep(5)
                for t in training:
                    t.result()
        finally:
            counts = receipt(jobs, metadata, final=True)
            files = bundle(jobs)
            print("Results:", counts, "archive: server2_loop_results.tgz", flush=True)
        if push:
            try:
                publish(files)
                print("GitHub upload complete", flush=True)
            except Exception as e:
                print(f"GitHub upload failed: {e}. Local archive is intact; use publish later.", flush=True)
        return 0 if counts["done"] == len(jobs) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("pack", "unpack", "run", "plan", "status", "publish"))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--archive", default="/tmp/r2e_s2.tgz")
    ap.add_argument("--cpu-par", type=int, default=2)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    os.chdir(a.root)
    if a.cpu_par < 1:
        ap.error("--cpu-par must be positive")
    if a.action == "pack":
        pack(a.archive)
    elif a.action == "unpack":
        unpack(a.archive)
    elif a.action == "plan":
        jobs = plan()
        print(json.dumps({"training": 4, "eval_jobs": len(jobs), "phases": {p: sum(j['phase'] == p for j in jobs) for p in ('L1', 'L2', 'L3', 'L4')}}, indent=2))
    elif a.action == "status":
        s = read_json(LOG / "status.json")
        print(json.dumps({k: v for k, v in (s or {}).items() if k != "jobs"}, indent=2))
    elif a.action == "publish":
        publish(bundle(plan()))
    else:
        raise SystemExit(run(a.cpu_par, a.push))


if __name__ == "__main__":
    main()
