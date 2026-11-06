import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from decap_loop.pdn import PDNConfig,sample_batch
from decap_loop.bank import save_bank
from train import make_policy
from scripts import report_model_bundle as bridge
from scripts import report_validation as v


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def make_sources(self,root):
        source=root/'source';study=root/'study';args=dict(model='mott',feat='mott_pk',peak_refine=True,d=16,layers=1,heads=2)
        model=make_policy(PDNConfig(),args)
        for seed in range(3):
            suffix='' if not seed else f'_s{seed}'
            folder=source/f'runs/2026-10-29/r2pk{suffix}_rl';folder.mkdir(parents=True)
            ck=folder/'best.pt';torch.save(dict(model=model.state_dict(),args=dict(args,seed=seed)),ck)
            (folder/'metrics.json').write_text('{}')
            for role,g in [('baseline',0.),('robust',.1)]:
                dst=study/'models'/f'{role}_s{seed}';dst.mkdir(parents=True)
                torch.save(dict(model=model.state_dict(),args=dict(args,seed=seed,reserve_decades=g,init=str(ck))),dst/'best.pt')
                (dst/'metrics.json').write_text('{}')
        return source,study

    def test_export_import_pairing_and_source_files_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,study=self.make_sources(root)
            before={str(p):bridge.sha(p) for d in (source,study) for p in d.rglob('*') if p.is_file()}
            self.assertTrue(bridge.export(source,study,root/'export'))
            manifest=bridge.receive(root/'export/models.tgz',root/'received')
            self.assertEqual(len(manifest['models']),9)
            self.assertEqual(before,{str(p):bridge.sha(p) for d in (source,study) for p in d.rglob('*') if p.is_file()})
            self.assertTrue(all('init' not in torch.load(root/'received'/m['path'],weights_only=False)['args'] for m in manifest['models']))
            self.assertEqual(len(bridge.receive(root/'export/models.tgz',root/'received')['models']),9)

    def test_import_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bad=root/'bad.tgz'
            with tarfile.open(bad,'w:gz') as tar:
                item=tarfile.TarInfo('../escape');item.size=1;tar.addfile(item,io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError,'Unexpected'):bridge.receive(bad,root/'received')
            self.assertFalse((root/'escape').exists())

    def test_export_rejects_unmatched_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,study=self.make_sources(root)
            p=study/'models/robust_s0/best.pt';ck=torch.load(p,weights_only=False);ck['args']['init']='nonexistent';torch.save(ck,p)
            with self.assertRaisesRegex(ValueError,'initialization'):bridge.export(source,study,root/'export')

    def test_seed_zero_representation_models_use_original_day(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,study=self.make_sources(root)
            original=source/'runs/2026-10-29/r2pk_rl/best.pt'
            for role in ('coord','modal'):
                dst=source/f'runs/2026-10-28/{bridge.BASES[role]}_rl';dst.mkdir(parents=True)
                ck=torch.load(original,weights_only=False);ck['args'].update(model='grid',feat=role)
                torch.save(ck,dst/'best.pt');(dst/'metrics.json').write_text('{}')
            bridge.export(source,study,root/'export')
            manifest=bridge.receive(root/'export/models.tgz',root/'received')
            self.assertTrue({'coord_s0','modal_s0'}<={m['id'] for m in manifest['models']})

    def test_seeds_are_frozen_unique_and_cases_not_outcome_selected(self):
        plan=v.cases(20,8)
        self.assertEqual(len(plan),144);self.assertEqual(len({c['problem_seed'] for c in plan}),144)
        self.assertEqual(plan[:20],[c for c in plan if c['pool']=='pkg16'])
        self.assertEqual(plan,v.cases(20,8))

    def test_git_transfer_publishes_only_artifacts_and_never_main(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,study=self.make_sources(root)
            bare=root/'remote.git';subprocess.run(['git','init','--bare',str(bare)],check=True,capture_output=True)
            bridge.export(source,study,root/'export')
            with patch.object(bridge,'REPO',str(bare)):
                bridge.publish(root/'export');bridge.fetch_bundle(root/'download',root/'inbox/models.tgz')
            received=bridge.receive(root/'inbox/models.tgz',root/'received')
            self.assertEqual(len(received['models']),9)
            refs=subprocess.check_output(['git','--git-dir',str(bare),'for-each-ref','--format=%(refname)'],text=True)
            self.assertNotIn('refs/heads/main',refs)
            self.assertIn(bridge.BRANCH,refs)

    def test_all_policy_families_use_same_exact_checker(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=PDNConfig(n_freq=12,n_check=24,seed_peaks=False)
            b=sample_batch(1,4,4,cfg,np.random.default_rng(11),n_probe=(1,1));pool=root/'pool.pkl';save_bank(pool,b)
            digest=bridge.sha(pool)
            configurations=[('frozen','mott','mott_pk'),('robust','mott','mott_pk'),('park','park_am','coord'),
                            ('manoharan','manoharan','manoharan'),('coord','grid','coord'),('modal','grid','modal')]
            for role,model,feat in configurations:
                args=dict(model=model,feat=feat,peak_refine=model=='mott',d=16,layers=1,heads=2,glob=False)
                policy=make_policy(cfg,args);ck=root/f'{role}.pt';torch.save(dict(model=policy.state_dict(),args=args),ck)
                result=root/f'{role}.json'
                a=SimpleNamespace(pool=str(pool),ckpt=str(ck),role=role,seed=0,result=str(result),
                    case=json.dumps(dict(id='pkg4_000',pool='pkg4',tech='pkg',size=4,index=0,problem_seed=123)))
                v.evaluate(a);r=json.loads(result.read_text())
                self.assertEqual(r['checker_points'],24);self.assertTrue(np.isfinite(r['worst']))
                self.assertGreaterEqual(r['total_seconds'],r['placement_seconds'])
            self.assertEqual(bridge.sha(pool),digest)


if __name__=='__main__':unittest.main()
