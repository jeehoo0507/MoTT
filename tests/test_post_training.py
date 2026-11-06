import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from decap_loop.bank import save_bank
from decap_loop.pdn import PDNConfig,sample_batch
from decap_loop.safe_verify import certify_parallel
from decap_loop.post_training import PostTraining,ARMS
from scripts.train_post_training import action_counts
from train import make_policy
from scripts import posttrain_campaign as post
from scripts import curriculum_campaign as campaign

ROOT=Path(__file__).resolve().parents[1]


class PostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def batch(self,seed=13):return sample_batch(2,4,4,PDNConfig(n_freq=12,n_check=700,seed_peaks=False),np.random.default_rng(seed),n_probe=(1,1))

    def test_phases_have_distinct_objectives_and_only_stage3_refines(self):
        full=[PostTraining('pt_full',i,.25) for i in (1,2,3)]
        self.assertEqual([x.stage for x in full],[1,2,3]);self.assertEqual([x.count_weight for x in full],[.1,.3,1.])
        b=self.batch();idx=torch.tensor([0,0,1,1]);occ=torch.full((2,16),-1);occ[:,0]=0
        for c in full:c.add(b,torch.arange(2),occ,np.array([0.,.1]))
        self.assertTrue((full[0].initial(b,idx)==-1).all())
        start=full[2].initial(b,idx)
        torch.testing.assert_close(start[0],start[1]);self.assertEqual(int(start[0,0]),0)
        self.assertTrue((start[2:]==-1).all())  # exact failure never enters archive
        self.assertTrue((full[2].initial(self.batch(14),idx)==-1).all()) # different boards never share starts

    def test_transfer_and_reward_gain_keep_original_failure_penalty(self):
        c=PostTraining('pt_full',2,.25);c.feedback.gaps['test']=.03
        d=PostTraining('pt_full',3,.25);d.restore(c.state(),transfer=True)
        self.assertEqual(d.feedback.gaps['test'],.03)
        reward=d.reward(torch.tensor([[3.,4.]]),torch.tensor([[False,True]]),
                        dict(start_count=torch.tensor([5.,5.]),V=torch.zeros(2)),torch.tensor([2.]),20.)
        torch.testing.assert_close(reward,torch.tensor([[1.,-19.5]]))
        with self.assertRaises(ValueError):PostTraining('pt_robust',3,.25).restore(c.state(),transfer=True)

    def test_action_diagnostics_distinguish_removal_and_replacement(self):
        self.assertEqual(action_counts(torch.tensor([[-1,-1]]),torch.tensor([[1,2,3,8,-1]]),3),
                         [dict(add=1,remove=1,replace=1,stop=1)])

    def test_real_three_phase_chains_all_four_arms_and_checkpoint_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pool=root/'pool';pool.mkdir();teachers=root/'teacher';teachers.mkdir();b=self.batch()
            placements=[]
            for i in range(2):
                found=None
                for kind in range(b.cfg.n_types):
                    occ=torch.full((1,16),-1,dtype=torch.long);occ[0,b.placeable()[i]]=kind
                    exact,_,_=certify_parallel(b,torch.tensor([i]),occ,b.cfg,workers=1)
                    if exact[0]<=1e-3:found=occ[0].numpy();break
                self.assertIsNotNone(found);placements.append(found)
            placements=np.array(placements);counts=(placements>=0).sum(1)
            ref={'greedy+prune':dict(count=counts,ok=np.ones(2,dtype=bool))}
            save_bank(pool/'train_pkg4_0.pkl',b,ref);save_bank(pool/'val_pkg10_0.pkl',b,ref)
            np.savez(teachers/'train_pkg4_0.npz',occ=placements,count=counts,ok=np.ones(2,dtype=bool),gp_count=counts)
            args=dict(model='mott',feat='mott_pk',peak_refine=True,d=16,layers=1,heads=2)
            init=root/'root.pt';torch.save(dict(model=make_policy(b.cfg,args).state_dict(),args=args),init)
            out=root/'study';(out/'inputs').mkdir(parents=True)
            (out/'inputs/pool').symlink_to(pool);(out/'inputs/teacher').symlink_to(teachers)
            def run_small(cmd,tag,gpu=False):
                for flag,value in (('--iters','1'),('--B','2'),('--M','2'),('--max_steps','4')):cmd[cmd.index(flag)+1]=value
                cmd+=['--device','cpu','--val_pools','val_pkg10_0','--no_amp']
                r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=90,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            with patch.object(post,'OUT',out),patch.object(campaign,'owned',side_effect=run_small):
                for arm in ARMS:
                    folder=out/f'models/{arm}_s0'
                    post.train_arm(0,arm,folder,dict(workers=2,train_hours=.75),init,args)
                    metrics=json.loads((folder/'metrics.json').read_text())
                    self.assertEqual(len(metrics['phase_metrics']),3);self.assertEqual(len(metrics['chain_sha256']),4)
                    for phase in (1,2,3):
                        stage=folder/f'phase{phase}'
                        self.assertTrue((stage/'phase_complete.json').exists())
                        events=[json.loads(l) for l in (stage/'trace.jsonl').read_text().splitlines()]
                        self.assertTrue(any(e['kind']=='phase_end' for e in events))
                        val=next(e for e in events if e['kind']=='validation')
                        self.assertTrue((stage/val['checkpoint']).exists())
                        self.assertTrue((stage/val['checkpoint']).with_suffix('.json').exists())
                        self.assertIn('gap',val['rows']['val_pkg10_0'][0])
                    if arm=='pt_full':
                        state=json.loads((folder/'phase3/post_state.json').read_text())
                        self.assertGreater(state['stats']['refinement_episodes'],0)
                self.assertTrue((out/'learning_curves.csv').exists());self.assertTrue((out/'REPORT.md').exists())


if __name__=='__main__':unittest.main()
