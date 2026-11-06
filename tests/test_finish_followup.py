import fcntl
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from decap_loop.pdn import PDNConfig,sample_batch
from decap_loop.bank import save_bank
from decap_loop.parallel_verify import ExactVerifier
from decap_loop.safe_verify import certify_parallel
from scripts import finish_followup as follow
from scripts import report_validation as report


class FollowupTests(unittest.TestCase):
    def test_both_training_arms_finish_with_parallel_validation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pool=root/'pool';pool.mkdir();teacher=root/'teacher';teacher.mkdir()
            batch=sample_batch(2,4,4,PDNConfig(n_freq=12,n_check=700,seed_peaks=False),np.random.default_rng(13),n_probe=(1,1))
            ref={'greedy+prune':dict(count=np.ones(2),ok=np.ones(2,dtype=bool))}
            save_bank(pool/'train_pkg4_0.pkl',batch,ref);save_bank(pool/'val_pkg10_0.pkl',batch,ref)
            np.savez(teacher/'train_pkg4_0.npz',occ=np.full((2,16),-1),count=np.ones(2),ok=np.ones(2,dtype=bool),gp_count=np.ones(2))
            for reserve in (0.,.1):
                out=root/str(reserve)
                cmd=[sys.executable,str(follow.ROOT/'scripts/pkg_guard_train.py'),'--stage','rl','--model','mott','--peak_refine',
                    '--device','cpu','--pool',str(pool),'--teacher_dir',str(teacher),'--val_pools','val_pkg10_0',
                    '--out',str(out),'--iters','1','--B','1','--M','2','--chunk','1','--max_steps','2',
                    '--d','16','--layers','1','--heads','2','--imit','0','--certify','2','--certify_workers','2',
                    '--reserve_decades',str(reserve),'--no_amp']
                r=subprocess.run(cmd,cwd=follow.ROOT,capture_output=True,text=True,timeout=90,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                metrics=json.loads((out/'metrics.json').read_text())
                self.assertEqual(metrics['iters'],1);self.assertEqual(metrics['args']['certify_workers'],2)
                self.assertEqual(metrics['args']['certify'],2)
                self.assertGreater(json.loads((out/'certify_profile.json').read_text())['parallel_calls'],0)

    def test_spawn_matches_serial_with_repeated_indices(self):
        torch.set_num_threads(1)
        cfg=PDNConfig(n_freq=12,n_check=700,seed_peaks=False)
        batch=sample_batch(2,4,4,cfg,np.random.default_rng(13),n_probe=(1,1))
        indices=torch.tensor([0,1,0]);occ=torch.full((3,16),-1,dtype=torch.long)
        occ[2,3]=0
        original=batch.Zt.clone()
        expected=certify_parallel(batch,indices,occ,cfg,workers=1)
        with ExactVerifier(2) as verifier:
            actual=verifier(batch,indices,occ,cfg)
            self.assertEqual(verifier.profile['parallel_calls'],1)
            for x,y in zip(expected[:2],actual[:2]):np.testing.assert_array_equal(x,y)
            for x,y in zip(expected[2],actual[2]):np.testing.assert_array_equal(x,y)
            empty=verifier(batch,indices[:0],occ[:0],cfg)
            self.assertEqual(empty[0].size,0)
        torch.testing.assert_close(original,batch.Zt,rtol=0,atol=0)

    def test_existing_runner_lock_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'R2E-pkg-fix';lock=root/'runs/pkg_guard_v1/runner.lock'
            lock.parent.mkdir(parents=True)
            with lock.open('w') as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertTrue(follow.busy(root))
            proc=Path(tmp)/'proc';proc.mkdir()
            self.assertFalse(follow.busy(root,proc))

    def test_finish_mode_resumes_expired_clock_without_rewriting_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'results'
            cfg=dict(n=1,large_n=0,hours=24.,inbox=str(root/'inbox'),transport='manual',git='original',protocol_sha256='original')
            report.save(out/'config.json',cfg);report.save(out/'clock.json',dict(deadline=1.))
            report.save(out/'artifacts/manifest.json',dict(models=[dict(id=f'{role}_s{s}',role=role,seed=s,path='unused.pt') for role in report.ROLES for s in range(3)]))
            a=SimpleNamespace(out=str(out),inbox=str(root/'inbox'),transport='manual',n=1,large_n=0,hours=24.,finish_all=True)
            with patch.object(report,'cases',return_value=[]):report.run(a)
            self.assertEqual(json.loads((out/'status.json').read_text())['stage'],'complete')
            self.assertEqual(json.loads((out/'config.json').read_text()),cfg)
            self.assertEqual(json.loads((out/'clock.json').read_text())['deadline'],1.)
            self.assertIsNone(json.loads((out/'status.json').read_text())['deadline'])
            self.assertTrue((out/'completion_policy.json').exists())


if __name__=='__main__':unittest.main()
