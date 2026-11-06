"""No API calls: verify isolated edits, publication scope, and protected-file guard."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('manager',Path(__file__).resolve().parents[1]/'scripts/codex_pkg_round.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def call(*args,cwd):
    return subprocess.run(args,cwd=cwd,check=True,capture_output=True,text=True).stdout


class ManagerTests(unittest.TestCase):
    def run_round(self,modify_protected=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'live';work=root/'isolated';bare=root/'remote.git'
            original.mkdir();work.mkdir();(original/'agent/logs').mkdir(parents=True)
            (original/'train.py').write_text('original live training code\n')
            (work/'scripts').mkdir();(work/'agent').mkdir();(work/'agent/CODEX_PKG_TASK.md').write_text('test task')
            (work/'train.py').write_text('protected copy\n')
            call('git','init','--bare',str(bare),cwd=root)
            call('git','init',cwd=work)
            call('git','-c','user.name=Test','-c','user.email=test@localhost','add','.',cwd=work)
            call('git','-c','user.name=Test','-c','user.email=test@localhost','commit','-m','fixture',cwd=work)
            call('git','remote','add','origin',str(bare),cwd=work)
            fake=root/'fake_codex'
            fake.write_text('#!'+sys.executable+'\nimport sys\nfrom pathlib import Path\nsys.stdin.read()\nw=Path(sys.argv[sys.argv.index("--cd")+1])\n(w/"PROGRESS.md").write_text("reported status\\n")\n(w/"scripts/pkg_generated.py").write_text("# isolated new module\\n")\n(w/"scripts/unrelated.py").write_text("# excluded from publication\\n")\n'+('(w/"train.py").write_text("changed\\n")\n' if modify_protected else ''))
            fake.chmod(0o755)
            with patch.object(m,'ROOT',original),patch.object(m,'ORIGINAL',original),patch.object(m,'WORK',work),patch.object(m,'snapshot',return_value={'processes':[]}),patch.object(sys,'argv',['manager','--codex',str(fake)]):
                if modify_protected:
                    with self.assertRaisesRegex(RuntimeError,'Protected'):m.main()
                else:m.main()
            self.assertEqual((original/'train.py').read_text(),'original live training code\n')
            if not modify_protected:
                published=call('git','show','--name-only','--format=',cwd=work)
                self.assertIn('scripts/pkg_generated.py',published)
                self.assertNotIn('scripts/unrelated.py',published)
                self.assertEqual(json.loads((original/'agent/CODEX_STATUS.json').read_text())['stage'],'round_done')
                self.assertTrue((original/'agent/CODEX_PROGRESS.md').exists())

    def test_isolated_round_publishes_only_allowed_paths(self):self.run_round()
    def test_protected_copy_change_blocks_publication(self):self.run_round(True)


if __name__=='__main__':unittest.main()
