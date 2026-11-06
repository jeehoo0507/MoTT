"""Cache-aware cgroup-v2 guard, scoped to report evaluation only."""
from pathlib import Path

GIB = 2**30


def snapshot(root=Path('/sys/fs/cgroup')):
    root = Path(root)
    if not (root/'memory.current').exists():
        return dict(raw_gib=0., working_gib=0., limit_gib=None, inactive_file_gib=0.)
    raw = int((root/'memory.current').read_text())
    try:
        stats = dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
        # Do not discount anonymous memory, active file cache, dirty or writeback pages.
        reclaimable = max(0, int(stats.get('inactive_file', 0))
                          - int(stats.get('file_dirty', 0)) - int(stats.get('file_writeback', 0)))
    except (OSError, ValueError):
        reclaimable = 0
    try:
        value = (root/'memory.max').read_text().strip()
        limit = None if value == 'max' else int(value)/GIB
    except (OSError, ValueError):
        limit = None
    return dict(raw_gib=raw/GIB, working_gib=max(0, raw-reclaimable)/GIB,
                limit_gib=limit, inactive_file_gib=reclaimable/GIB)


def blocked(info, running=False):
    if info['working_gib'] > (32 if running else 28):
        return True
    limit = info['limit_gib']
    # Keep raw headroom as well: cache reclaim is not instantaneous or guaranteed.
    return limit is not None and info['raw_gib'] > limit - (2 if running else 4)
