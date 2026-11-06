"""Working-target consistency, original-target certification, and paired reporting."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from decap_loop.pdn import PDNConfig, sample_batch
from decap_loop.peak_env import PkEnv
from decap_loop.pkg_guard_env import GuardedPkEnv
from decap_loop.sim import _probe_diag, action_effects
from train import make_policy

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pkg_guard_study',ROOT/'scripts/pkg_guard_study.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def batch():
    cfg=PDNConfig(n_freq=12,n_check=24,seed_peaks=False)
    return sample_batch(1,4,4,cfg,np.random.default_rng(13),n_probe=(1,1),device='cpu')


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_zero_reserve_exactly_matches_original_environment(self):
        b=batch();idx=torch.tensor([0])
        original=PkEnv(b,idx,b.cfg,feat_mode='mott_pk');guard=GuardedPkEnv(b,idx,b.cfg,feat_mode='mott_pk')
        for key,value in original.observe().items():
            torch.testing.assert_close(value,guard.observe()[key],rtol=0,atol=0)

    def test_target_features_effects_and_mask_use_same_reserve(self):
        b=batch();idx=torch.tensor([0]);base=PkEnv(b,idx,b.cfg,feat_mode='mott_pk')
        # Original goal is just feasible on this working grid; reserved goal is not.
        b.Zt=_probe_diag(base.z_ip,b.probes).abs().amax(-1)*1.05
        targets=b.Zt.clone();parameters=b.zt_par.copy()
        base=PkEnv(b,idx,b.cfg,feat_mode='mott_pk')
        guard=GuardedPkEnv(b,idx,b.cfg,feat_mode='mott_pk',reserve_decades=.1)
        before=base.observe();after=guard.observe()
        self.assertTrue(before['feas'][0]);self.assertFalse(after['feas'][0])
        self.assertTrue(before['mask'][0,base.stop_idx]);self.assertFalse(after['mask'][0,guard.stop_idx])
        self.assertTrue(after['mask'][0,:guard.stop_idx].any())
        torch.testing.assert_close(after['m'],before['m']+.1)
        self.assertFalse(torch.equal(before['x'],after['x']))
        torch.testing.assert_close(guard.logZt,torch.log10(targets)-.1)
        expected=action_effects(guard.z_ip,guard.z_ii,_probe_diag(guard.z_ip,b.probes),guard.occ,guard.zd,guard.logZt)
        torch.testing.assert_close(after['Vn'],expected[0]);torch.testing.assert_close(after['mn'],expected[1])
        torch.testing.assert_close(b.Zt,targets,rtol=0,atol=0);np.testing.assert_array_equal(b.zt_par,parameters)

    def test_refresh_recomputes_without_accumulating_reserve(self):
        b=batch();guard=GuardedPkEnv(b,torch.tensor([0]),b.cfg,reserve_decades=.1,feat_mode='mott_pk')
        original=b.Zt.clone();guard.refresh();guard.refresh()
        torch.testing.assert_close(guard.logZt,torch.log10(original)-.1)
        b.add_freqs([[float(np.sqrt(b.freqs[0,2]*b.freqs[0,3]))]])
        guard.refresh();torch.testing.assert_close(guard.logZt,torch.log10(b.Zt)-.1)
        self.assertEqual(guard.logZt.shape,b.Zt.shape)

    def test_original_target_certificate_is_independent_of_working_reserve(self):
        b=batch();idx=torch.tensor([0]);targets=b.Zt.clone();parameters=b.zt_par.copy()
        args=dict(model='mott',peak_refine=True,d=16,layers=1,heads=2)
        policy=make_policy(b.cfg,args).eval()
        row=r.evaluate_case(policy,b,.1,max_steps=4,rounds=0)
        from decap_loop.verify import certify
        exact,_,_=certify(b,idx,torch.tensor([row['occ']]),b.cfg)
        self.assertAlmostEqual(row['worst'],float(exact[0]),places=10)
        self.assertEqual(row['success'],bool(exact[0]<=np.log10(1.01)))
        torch.testing.assert_close(b.Zt,targets);np.testing.assert_array_equal(b.zt_par,parameters)

    def test_paired_report_uses_common_boards_and_joint_success_count(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);a=SimpleNamespace(out=folder,seeds='0,1,2',reserve=.1)
            for seed in range(3):
                for arm,g,ok,count in [('baseline',0.,[True,False],[10,90]),('robust',.1,[True,True],[8,12])]:
                    rows=[dict(instance=i,success=ok[i],count=count[i],seconds=1.) for i in range(2)]
                    r.save(out/'eval'/f'{arm}_{seed}.json',dict(pool='eval_pkg16_1',pool_sha256='same',arm=arm,
                        seed=seed,eval_reserve=g,rows=rows,complete=True))
            r.report(a);contrasts=json.loads((out/'paired.json').read_text())
            c=next(c for c in contrasts if c['treated']==['robust',.1] and c['control']==['baseline',0.])
            self.assertEqual(len(c['per_seed']),3)
            self.assertEqual(c['per_seed'][0]['rescued'],1)
            self.assertEqual(c['per_seed'][0]['joint_success_n'],1)
            self.assertEqual(c['per_seed'][0]['joint_count_difference'],-2.)
            self.assertTrue((out.parent/(out.name+'_results.tgz')).is_file())
            (out.parent/(out.name+'_results.tgz')).unlink()

    def test_original_pending_queue_respects_after_and_done(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder);r.save(source/'agent/day2_queue.json',dict(train=[dict(name='ready'),dict(name='blocked',after='not_yet')]))
            self.assertEqual(r.old_pending(source),['ready'])
            r.save(source/'runs/2026-10-29/ready_rl/metrics.json',{})
            self.assertEqual(r.old_pending(source),[])

    def test_stopping_new_child_leaves_unrelated_process_alive(self):
        cmd=[sys.executable,'-c','import time; time.sleep(20)']
        unrelated=subprocess.Popen(cmd,start_new_session=True)
        owned=subprocess.Popen(cmd,start_new_session=True)
        try:
            r.stop_own_child(owned)
            self.assertIsNotNone(owned.poll());self.assertIsNone(unrelated.poll())
        finally:
            r.stop_own_child(owned);r.stop_own_child(unrelated)


if __name__=='__main__':unittest.main()
