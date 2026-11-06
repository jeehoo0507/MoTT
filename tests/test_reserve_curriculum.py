import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from decap_loop.pdn import PDNConfig,sample_batch
from decap_loop.bank import save_bank
from decap_loop.reserve_curriculum import ReserveCurriculum
from decap_loop.pkg_guard_env import GuardedPkEnv
from decap_loop.sim import _probe_diag
from scripts.train_reserve_curriculum import original_goal_outcome
from scripts import curriculum_campaign as campaign
from scripts import switch_curriculum as switch
from scripts import report_validation as report
from scripts import report_model_bundle as bridge
from train import make_policy


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def batch(self):return sample_batch(2,4,4,PDNConfig(n_freq=12,n_check=700,seed_peaks=False),np.random.default_rng(13),n_probe=(1,1))

    def test_gap_feedback_annealing_and_resume(self):
        b=self.batch();c=ReserveCurriculum('adaptive',1.)
        self.assertAlmostEqual(c.reserve(b),.1)
        c.update(b,[0,0],[.02,.02]);self.assertAlmostEqual(c.reserve(b),.084)
        c.elapsed=1800;self.assertAlmostEqual(c.reserve(b),.042)
        copy=ReserveCurriculum('adaptive',1.);copy.restore(c.state());self.assertEqual(copy.state(),c.state())
        c.elapsed=2400;self.assertEqual(c.reserve(b),0.)
        self.assertEqual(ReserveCurriculum('control',1.).reserve(b),0.)
        with self.assertRaises(ValueError):ReserveCurriculum('annealed',1.).restore(copy.state())

    def test_reward_uses_original_target_even_when_reserved_goal_fails(self):
        b=self.batch();idx=torch.arange(2);env=GuardedPkEnv(b,idx,b.cfg,feat_mode='mott_pk')
        b.Zt=_probe_diag(env.z_ip,b.probes).abs().amax(-1)*1.05
        targets=b.Zt.clone();env=GuardedPkEnv(b,idx,b.cfg,feat_mode='mott_pk',reserve_decades=.1)
        obs=env.observe();self.assertFalse(obs['feas'].any())
        out=original_goal_outcome(env,{})
        self.assertTrue(out['feasible'].all());self.assertTrue((out['V']==0).all())
        torch.testing.assert_close(b.Zt,targets,rtol=0,atol=0)
        torch.testing.assert_close(env.logZt,torch.log10(targets)-.1)

    def test_validation_gate_rejects_regressions_and_excess_components(self):
        a={k:[dict(instance=i,success=True,count=10) for i in range(2)] for k in ('pcb','pkg','ip')}
        b=json.loads(json.dumps(a));b['pkg'][0]['count']=8
        self.assertTrue(campaign.expansion_gate(a,b)['expand'])
        b['pcb'][0]['success']=False;self.assertFalse(campaign.expansion_gate(a,b)['expand'])
        b=json.loads(json.dumps(a));b['pkg'][0]['count']=50
        self.assertFalse(campaign.expansion_gate(a,b)['expand'])

    def test_handover_does_not_terminate_an_active_case(self):
        with patch.object(switch,'alive',side_effect=[True,True,True]),patch.object(switch,'send') as send,\
             patch.object(Path,'read_text',side_effect=['456','']),patch.object(switch,'stop_manager') as stop,\
             patch.object(switch.time,'sleep'):
            switch.stop_between_cases(123)
        self.assertEqual(send.call_args_list[0].args,(123,signal.SIGSTOP))
        self.assertEqual(send.call_args_list[1].args,(123,signal.SIGCONT))
        stop.assert_called_once_with(123)

    def test_transition_requires_all_seed0_package_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for arm,g in (('frozen',0.),('baseline',0.),('robust',.1)):
                for size in (10,16):report.save(out/f'eval/{arm}_s0_pkg{size}_g{g}.json',dict(complete=True))
            self.assertFalse(switch.seed_zero_complete(out))
            for arm in ('baseline','robust'):report.save(out/f'models/{arm}_s0/metrics.json',{})
            self.assertTrue(switch.seed_zero_complete(out))
            report.save(out/'eval/robust_s0_pkg16_g0.1.json',dict(complete=False))
            self.assertFalse(switch.seed_zero_complete(out))

    def test_new_report_seed_plan_and_dynamic_inventory(self):
        self.assertTrue(set(c['problem_seed'] for c in report.cases(2,1)).isdisjoint(c['problem_seed'] for c in report.cases(2,1,3000000)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'out';plan=root/'plan.json'
            report.save(plan,dict(from_manifest=True,seed_offset=3000000))
            models=[dict(id=f'{r}_s0',role=r,seed=0,path='unused.pt') for r in campaign.ARMS]
            report.save(out/'artifacts/manifest.json',dict(models=models,expected=[m['id'] for m in models],campaign_final=True))
            a=SimpleNamespace(out=str(out),inbox=str(root/'inbox'),transport='manual',n=1,large_n=0,hours=24.,finish_all=True,plan=str(plan))
            with patch.object(report,'cases',return_value=[]):report.run(a)
            self.assertEqual(json.loads((out/'status.json').read_text())['stage'],'complete')

    def test_campaign_weights_roundtrip_on_separate_git_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();home=root/'home';out=root/'study';args=dict(model='mott',feat='mott_pk',peak_refine=True,d=16,layers=1,heads=2)
            model=make_policy(PDNConfig(),args)
            for seed in range(3):
                name=f'r2pk{"" if seed==0 else "_s"+str(seed)}_rl'
                folder=home/'R2E/runs/2026-10-29'/name;folder.mkdir(parents=True)
                torch.save(dict(model=model.state_dict(),args=dict(args,seed=seed)),folder/'best.pt')
                (folder/'metrics.json').write_text('{}')
                target=out/'inputs'/name;target.mkdir(parents=True);shutil.copy2(folder/'best.pt',target/'best.pt')
            src=home/'R2E/runs/2026-10-29/r2pk_rl/best.pt'
            for role in ('baseline','robust'):
                folder=home/f'R2E-pkg-fix/runs/pkg_guard_v1/models/{role}_s0';folder.mkdir(parents=True)
                torch.save(dict(model=model.state_dict(),args=dict(args,seed=0,reserve_decades=.1 if role=='robust' else 0.,init=str(src))),folder/'best.pt')
                (folder/'metrics.json').write_text('{}')
            for role in campaign.ARMS:
                folder=out/f'models/{role}_s0';folder.mkdir(parents=True)
                torch.save(dict(model=model.state_dict(),args=dict(args,seed=0,reserve_decades=0.,init=str(src))),folder/'best.pt')
                (folder/'metrics.json').write_text('{}')
            bare=root/'remote.git';subprocess.run(['git','init','--bare',str(bare)],capture_output=True,check=True)
            with patch.object(campaign,'OUT',out),patch.object(Path,'home',return_value=home),patch.object(bridge,'REPO',str(bare)):
                campaign.publish(final=True,seeds=(0,),decision=dict(expand=False))
                bridge.fetch_bundle(root/'cache',root/'models.tgz',branch=campaign.BRANCH)
            manifest=bridge.receive(root/'models.tgz',root/'received')
            self.assertTrue(manifest['campaign_final'])
            self.assertTrue({'control_s0','annealed_s0','adaptive_s0'}<={m['id'] for m in manifest['models']})
            self.assertNotIn('adaptive_s1',manifest['expected'])
            for m in manifest['models']:
                self.assertNotIn('init',torch.load(root/'received'/m['path'],weights_only=True)['args'])

    def test_all_new_roles_decode_with_original_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pool=root/'pool.pkl';b=self.batch();save_bank(pool,b)
            args=dict(model='mott',feat='mott_pk',peak_refine=True,d=16,layers=1,heads=2)
            model=make_policy(b.cfg,args);ck=root/'model.pt';torch.save(dict(model=model.state_dict(),args=args),ck)
            rows=[]
            for role in campaign.ARMS:
                a=SimpleNamespace(pool=str(pool),ckpt=str(ck),role=role,seed=0,result=str(root/f'{role}.json'),
                    case=json.dumps(dict(id='pkg4_0',pool='pkg4',tech='pkg',size=4,index=0,problem_seed=1)))
                report.evaluate(a);rows.append(json.loads(Path(a.result).read_text()))
            self.assertEqual(len({r['count'] for r in rows}),1)
            np.testing.assert_array_equal([r['worst'] for r in rows],[rows[0]['worst']]*3)
            self.assertTrue(all(r['checker_points']==700 for r in rows))

    def test_all_three_arms_complete_real_rl_and_certification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pool=root/'pool';pool.mkdir();teacher=root/'teacher';teacher.mkdir();b=self.batch()
            ref={'greedy+prune':dict(count=np.ones(2),ok=np.ones(2,dtype=bool))}
            save_bank(pool/'train_pkg4_0.pkl',b,ref);save_bank(pool/'val_pkg10_0.pkl',b,ref)
            np.savez(teacher/'train_pkg4_0.npz',occ=np.full((2,16),-1),count=np.ones(2),ok=np.ones(2,dtype=bool),gp_count=np.ones(2))
            for arm in campaign.ARMS:
                folder=root/arm
                cmd=[sys.executable,'scripts/train_reserve_curriculum.py','--curriculum',arm,'--certify_workers','2',
                    '--stage','rl','--model','mott','--peak_refine','--device','cpu','--pool',str(pool),
                    '--teacher_dir',str(teacher),'--val_pools','val_pkg10_0','--out',str(folder),
                    '--iters','1','--B','1','--M','2','--chunk','1','--max_steps','2','--d','16','--layers','1','--heads','2',
                    '--imit','0','--certify','2','--no_amp']
                r=subprocess.run(cmd,cwd=campaign.ROOT,capture_output=True,text=True,timeout=90,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                metrics=json.loads((folder/'metrics.json').read_text())
                self.assertEqual(metrics['iters'],1);self.assertEqual(metrics['args']['certify'],2)
                self.assertTrue((folder/'validation_best.json').exists());self.assertTrue((folder/'curriculum_state.json').exists())


if __name__=='__main__':unittest.main()
