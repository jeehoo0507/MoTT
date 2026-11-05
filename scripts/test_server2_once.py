"""Standard-library orchestration checks; does not train or require CUDA."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("server2_once.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


@contextlib.contextmanager
def workspace():
    before = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(before)


def result(job):
    r.save_json(job["result"], dict(n=1, set=job["pool"], method=job["method"], ckpt=job["ckpt"],
                                  samples=1, inst_chunk=5, ok=[True], count=[3], worst=[0], occ=[[0]], sec_total=1))


@contextlib.contextmanager
def fresh_mode():
    names = ("FRESH", "DAY", "NEW", "LOG", "POOL", "PHASES", "RESULT_ARCHIVE")
    saved = {name: getattr(r, name) for name in names}
    r.configure_fresh()
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(r, name, value)


class RunnerTests(unittest.TestCase):
    def test_fresh_plan_has_no_server1_dependencies(self):
        with fresh_mode():
            jobs = r.plan()
            self.assertEqual(len(jobs), 180)
            self.assertEqual(len({j["result"] for j in jobs}), 180)
            self.assertEqual(len(r.train_specs()), 6)
            self.assertEqual([seed for _, seed in r.train_specs()], [0, 0, 1, 1, 2, 2])
            self.assertTrue(all(Path(j["ckpt"]).is_relative_to(r.NEW) for j in jobs))
            self.assertFalse(any("2026-10-29" in j["ckpt"] for j in jobs))

    def test_fresh_training_selects_both_architectures(self):
        with fresh_mode(), workspace(), patch.object(r, "trained", return_value=False), \
                patch.object(r, "execute", return_value=0) as execute:
            r.gpu_worker(0, [("r2pk", 0), ("r3pk", 0)])
            cmds = [c.args[0] for c in execute.call_args_list]
            self.assertEqual([c[c.index("--arm") + 1] for c in cmds],
                             ["--model mott --peak_refine", "--model loop --peak_refine"])
            self.assertTrue(all(c[c.index("--pool") + 1] == str(r.POOL) for c in cmds))

    def test_pool_metadata_alone_cannot_mark_a_partial_pickle_ready(self):
        with fresh_mode(), workspace():
            name = "eval_pcb10_0"
            p = r.POOL / (name + ".pkl")
            r.save_json(p.with_suffix(".json"), {"name": name, "n": 25, "size": 10})
            p.write_bytes(b"partial")
            self.assertFalse(r.usable_pool(name))
            with p.open("wb") as f:
                pickle.dump(dict(valid=SimpleNamespace(shape=(25, 100)), H=10, W=10,
                                 results={"greedy+prune": {}}), f)
            self.assertTrue(r.usable_pool(name))

    def test_fresh_scheduler_completes_without_transfer_files(self):
        with fresh_mode(), workspace():
            full = r.plan()
            jobs = [full[0], next(j for j in full if j["phase"] == "FR")]
            for j in jobs:
                p = Path(j["ckpt"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"checkpoint")
                r.save_json(p.parent / "metrics.json", {"complete": True})
            def evaluate(j):
                result(j)
                return "done"
            with patch.object(r, "plan", return_value=jobs), patch.object(r, "preflight", return_value=["fake GPU"]), \
                 patch.object(r, "gpu_worker"), patch.object(r, "build_fresh_evaluation"), \
                 patch.object(r, "usable_pool", return_value=True), patch.object(r, "available_gib", return_value=64), \
                 patch.object(r.subprocess, "check_output", return_value="fake-git"), \
                 patch.object(r.time, "sleep"), patch.object(r, "evaluate", side_effect=evaluate):
                self.assertEqual(r.run(1, False), 0)
            self.assertEqual(r.read_json(r.LOG / "status.json")["counts"]["done"], 2)
            self.assertTrue(Path("server2_fresh_results.tgz").exists())
            self.assertFalse(Path("server2_transfer_manifest.json").exists())

    def test_plan_is_complete_and_unique(self):
        jobs = r.plan()
        self.assertEqual(len(jobs), 216)
        self.assertEqual(len({j["result"] for j in jobs}), 216)
        self.assertEqual({p: sum(j["phase"] == p for j in jobs) for p in ("L1", "L2", "L3", "L4")},
                         {"L1": 48, "L2": 24, "L3": 72, "L4": 72})
        for j in jobs:
            if j["phase"] == "L3" and "r2pk" in j["method"]:
                self.assertTrue(j["out"].startswith(str(r.NEW)))
            if j["phase"] == "L4":
                self.assertTrue(j["method"].endswith(("@R2", "@R4")))

    def test_invalid_result_is_not_a_completed_job(self):
        with workspace():
            j = r.plan()[0]
            self.assertFalse(r.complete(j))
            result(j)
            self.assertTrue(r.complete(j))
            Path(j["result"]).write_text('{"n":')
            self.assertFalse(r.complete(j))
            result(j)
            bad = r.read_json(j["result"])
            bad["ckpt"] = "wrong.pt"
            r.save_json(j["result"], bad)
            self.assertFalse(r.complete(j))

    def test_pack_roundtrip_and_existing_input_protection(self):
        with workspace() as root:
            p = Path("runs/pool_v1/train_pcb10_0.pkl")
            p.parent.mkdir(parents=True)
            p.write_bytes(b"pool")
            for n in r.LEGACY + r.BASE:
                q = r.OLD / (n + "_rl") / "best.pt"
                q.parent.mkdir(parents=True)
                q.write_bytes(n.encode())
            archive = root / "inputs.tgz"
            r.pack(archive)
            target = root / "target"
            target.mkdir()
            os.chdir(target)
            r.unpack(archive)
            manifest = r.read_json("server2_transfer_manifest.json")
            self.assertEqual(len(manifest), 8)
            self.assertTrue(all(r.sha(p) == h for p, h in manifest.items()))
            p.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Existing input differs"):
                r.unpack(archive)

    def test_scheduler_dependency_failure_and_resume(self):
        with workspace():
            jobs = [r.plan()[0], next(j for j in r.plan() if j["phase"] == "L2"),
                    next(j for j in r.plan() if j["phase"] == "L4")]
            p = Path(jobs[0]["ckpt"])
            p.parent.mkdir(parents=True)
            p.write_bytes(b"ckpt")
            result(jobs[0])  # Must be reused, not executed again.
            with patch.object(r, "plan", return_value=jobs), patch.object(r, "preflight", return_value=["fake GPU"]), \
                 patch.object(r, "gpu_worker"), patch.object(r, "available_gib", return_value=64), \
                 patch.object(r.subprocess, "check_output", return_value="fake-git"), \
                 patch.object(r.time, "sleep"), patch.object(r, "evaluate") as ev:
                self.assertEqual(r.run(1, False), 1)
                ev.assert_not_called()
            status = r.read_json(r.LOG / "status.json")
            self.assertEqual(status["counts"]["done"], 1)
            self.assertEqual(status["counts"]["blocked"], 2)
            self.assertTrue(Path("server2_loop_results.tgz").exists())

    def test_scheduler_finishes_ready_jobs_and_archives(self):
        with workspace():
            jobs = [r.plan()[0]]
            p = Path(jobs[0]["ckpt"])
            p.parent.mkdir(parents=True)
            p.write_bytes(b"ckpt")
            def evaluate(j):
                result(j)
                return "done"
            with patch.object(r, "plan", return_value=jobs), patch.object(r, "preflight", return_value=["fake GPU"]), \
                 patch.object(r, "gpu_worker"), patch.object(r, "available_gib", return_value=64), \
                 patch.object(r.subprocess, "check_output", return_value="fake-git"), \
                 patch.object(r.time, "sleep"), patch.object(r, "evaluate", side_effect=evaluate):
                self.assertEqual(r.run(1, False), 0)
            self.assertEqual(r.read_json(r.LOG / "status.json")["counts"]["done"], 1)

    def test_publish_does_not_commit_existing_staged_files(self):
        with workspace() as root:
            def git(*args):
                return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT)
            git("init", "-b", "main")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@localhost")
            Path("README").write_text("test repository")
            git("add", "README")
            git("commit", "-m", "initial")
            remote = root / "remote.git"
            git("init", "--bare", "-b", "main", str(remote))
            git("remote", "add", "origin", str(remote))
            git("push", "origin", "main")
            Path("unrelated.txt").write_text("must not publish")
            git("add", "unrelated.txt")
            output = Path("runs/test/result.json")
            output.parent.mkdir(parents=True)
            output.write_text('{"ok": true}')
            r.publish([output])
            self.assertIn('"ok": true', git("--git-dir", str(remote), "show", "main:runs/test/result.json"))
            self.assertNotIn("unrelated.txt", git("--git-dir", str(remote), "ls-tree", "--name-only", "main"))
            self.assertIn("unrelated.txt", git("diff", "--cached", "--name-only"))


if __name__ == "__main__":
    unittest.main()
