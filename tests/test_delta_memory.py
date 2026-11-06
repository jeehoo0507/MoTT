"""Memory semantics, permutation invariance, chronological gradient replay, and report pairing."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from decap_loop.delta_memory import MemoryMoTT, ARMS
from decap_loop.model import MoTT
from decap_loop.peak_env import mott_pk_dims

spec=importlib.util.spec_from_file_location('experiment',Path(__file__).resolve().parents[1]/'scripts/server2_delta.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def inputs(n=9):
    dn,dm,de,dc=mott_pk_dims(3)
    x=torch.randn(1,n,dn+8*de+8*dm+3*dc)*.1
    start=dn+8*de
    modes=x[:,0,start:start+8*dm].reshape(1,8,dm).clone();modes[:,:,3]=1
    x[:,:,start:start+8*dm]=modes.reshape(1,1,-1)
    x[:,:,start+8*dm:]=x[:,0:1,start+8*dm:]
    return x,torch.ones(1,n,dtype=torch.bool)


class DeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_original_mott_output_is_preserved(self):
        torch.manual_seed(7)
        model=MemoryMoTT('mott',d=16,layers=1,heads=2)
        original=MoTT(3,8,mott_pk_dims(3),d=16,layers=1,heads=2)
        original.load_state_dict(model.state_dict())
        x,valid=inputs();node,stop=original(x,valid,3,3)
        logits,_=model.step(x,valid)
        torch.testing.assert_close(logits,torch.cat((node.reshape(1,-1),stop[:,None]),1))

    def test_candidate_permutation_equivariance_across_actions(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(7);model=MemoryMoTT(arm,d=16,layers=1,heads=2).eval()
                x,valid=inputs();perm=torch.randperm(9);inverse=torch.argsort(perm)
                l,s=model.step(x,valid);lp,sp=model.step(x[:,perm],valid[:,perm])
                torch.testing.assert_close(l[:,:-1].reshape(1,9,4),lp[:,:-1].reshape(1,9,4)[:,inverse],atol=1e-5,rtol=1e-5)
                a=torch.tensor([2*4+1]);ap=inverse[torch.tensor([2])]*4+1
                s=model.advance(s,a);sp=model.advance(sp,ap)
                x2=x.clone();x2[:,2,0]+=.3
                l,s=model.step(x2,valid,s);lp,sp=model.step(x2[:,perm],valid[:,perm],sp)
                torch.testing.assert_close(l[:,:-1].reshape(1,9,4),lp[:,:-1].reshape(1,9,4)[:,inverse],atol=2e-5,rtol=1e-5)
                torch.testing.assert_close(l[:,-1],lp[:,-1],atol=2e-5,rtol=1e-5)

    def test_history_and_reset_are_explicit(self):
        x,valid=inputs();model=MemoryMoTT('delta',d=16,layers=1,heads=2)
        first,state=model.step(x,valid);second,state=model.step(x,valid,model.advance(state,torch.tensor([1])))
        reset,_=model.step(x,valid)
        torch.testing.assert_close(first,reset)
        self.assertGreater(float((second-first).abs().max().detach()),1e-7)
        self.assertEqual(state['memory'].shape,(1,16,16))

    def test_tbptt_replay_updates_all_six_models(self):
        x,valid=inputs();mask=torch.ones(1,37,dtype=torch.bool)
        seq=[dict(x=x.clone(),valid=valid,mask=mask,a=torch.tensor([i]),active=torch.ones(1,dtype=torch.bool),physics=torch.linspace(-.1,.2,16)[None]) for i in range(4)]
        a=SimpleNamespace(device='cpu',tbptt=2,aux_weight=.1)
        for arm in ARMS:
            with self.subTest(arm=arm):
                model=MemoryMoTT(arm,d=16,layers=1,heads=2)
                opt=torch.optim.AdamW(model.parameters(),lr=.001)
                before=model.add_head[-1].weight.detach().clone()
                loss=r.fit_sequence(model,seq,opt,a)
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                self.assertFalse(torch.equal(before,model.add_head[-1].weight))
                if model.kind in ('gru','delta'):
                    self.assertIsNotNone(model.write.weight.grad)
                    self.assertGreater(float(model.write.weight.grad.abs().sum()),0)
                if model.use_aux:self.assertGreater(float(model.phys_head[-1].weight.grad.abs().sum()),0)

    def test_report_compares_joint_success_cases_and_pairs_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder); a=SimpleNamespace(out=folder)
            for arm,success,count in [('mott',[True,False],[10,99]),('delta_aux',[True,True],[8,12])]:
                for seed in range(3):
                    rows=[dict(instance=i,success=success[i],count=count[i],placement_seconds=1.,policy_seconds=.1) for i in range(2)]
                    r.save_json(out/'eval'/f'eval_pkg16_0__{arm}_s{seed}.json',dict(pool='eval_pkg16_0',pool_sha256='same',arm=arm,seed=seed,rows=rows))
            r.report(a)
            import json
            d=json.loads((out/'paired_comparisons.json').read_text())
            c=next(c for c in d['contrasts'] if c['treated']=='delta_aux' and c['base']=='mott')
            self.assertEqual(len(c['per_seed']),3)
            self.assertEqual(c['per_seed'][0]['rescued'],1)
            self.assertEqual(c['per_seed'][0]['joint_success_n'],1)
            self.assertEqual(c['per_seed'][0]['count_difference'],-2.)

    def test_gpu_gate_requires_all_six_valid_training_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            names=('r2pk','r2pk_s1','r2pk_s2','r3pk','r3pk_s1','r3pk_s2')
            for name in names:
                path=root/'runs/2026-11-05fresh'/f'{name}_rl'
                path.mkdir(parents=True)
                (path/'metrics.json').write_text('{}')
                (path/'best.pt').touch()
            self.assertTrue(r.old_training_complete(root))
            (path/'metrics.json').write_text('partial')
            self.assertFalse(r.old_training_complete(root))
            (path/'metrics.json').write_text('{}')
            (path/'best.pt').unlink()
            self.assertFalse(r.old_training_complete(root))


if __name__=='__main__':unittest.main()
