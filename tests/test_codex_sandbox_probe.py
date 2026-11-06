"""Verify fail-closed acceptance checks without requiring Linux user namespaces on this host."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'scripts/codex_sandbox_probe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ProbeTests(unittest.TestCase):
    def test_accepts_only_verified_inside_write_and_outside_denial(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)/'work';outside=Path(d)/'outside';work.mkdir();outside.mkdir()
            def guarded(cmd,**kw):
                inner,outer,nonce=map(str,cmd[-3:])
                Path(inner).write_text(nonce)
                return SimpleNamespace(returncode=0,stdout='R2E_SANDBOX_OK:'+nonce,stderr='')
            with patch.object(m.subprocess,'run',side_effect=guarded):r=m.probe('/fake/codex',[],work,outside)
            self.assertTrue(r['success']);self.assertEqual(list(work.iterdir()),[])

    def test_rejects_execution_that_can_write_outside_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)/'work';outside=Path(d)/'outside';work.mkdir();outside.mkdir()
            def unguarded(cmd,**kw):
                inner,outer,nonce=map(str,cmd[-3:]);Path(inner).write_text(nonce);Path(outer).write_text(nonce)
                return SimpleNamespace(returncode=0,stdout='R2E_SANDBOX_OK:'+nonce,stderr='')
            with patch.object(m.subprocess,'run',side_effect=unguarded):r=m.probe('/fake/codex',[],work,outside)
            self.assertFalse(r['success']);self.assertEqual(list(outside.iterdir()),[])

    def test_failed_namespace_command_does_not_count_as_success(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='Operation not permitted')):
                r=m.probe('/fake/codex',[],Path(d),Path(d))
            self.assertFalse(r['success'])


if __name__=='__main__':unittest.main()
