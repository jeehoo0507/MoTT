import tempfile
from pathlib import Path
import unittest
from scripts.report_memory import snapshot, blocked, GIB

class MemoryTests(unittest.TestCase):
    def test_observed_cache_does_not_block(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'memory.current').write_text('33810120704')
            (p/'memory.max').write_text('64424509440')
            (p/'memory.stat').write_text('inactive_file 23234138112\nfile_dirty 28672\nfile_writeback 0\n')
            info=snapshot(p)
            self.assertAlmostEqual(info['working_gib'],9.85,places=1)
            self.assertFalse(blocked(info))
            self.assertFalse(blocked(info,True))
    def test_real_pressure_and_raw_headroom(self):
        self.assertTrue(blocked(dict(working_gib=29,raw_gib=30,limit_gib=60)))
        self.assertTrue(blocked(dict(working_gib=33,raw_gib=34,limit_gib=60),True))
        self.assertTrue(blocked(dict(working_gib=10,raw_gib=57,limit_gib=60)))
        self.assertTrue(blocked(dict(working_gib=10,raw_gib=59,limit_gib=60),True))
    def test_missing_stats_is_conservative(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'memory.current').write_text(str(31*GIB));(p/'memory.max').write_text('max')
            self.assertTrue(blocked(snapshot(p)))
    def test_dirty_pages_not_discounted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'memory.current').write_text(str(31*GIB));(p/'memory.max').write_text(str(60*GIB))
            (p/'memory.stat').write_text(f'inactive_file {10*GIB}\nfile_dirty {8*GIB}\n')
            self.assertEqual(snapshot(p)['working_gib'],29)
            self.assertTrue(blocked(snapshot(p)))
