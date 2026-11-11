"""Versioned evidence-age semantics; actual completion is always wall-clock guarded."""
def protocol(mode):
    if mode not in ('availability','capture'):raise ValueError('Unknown evidence age clock')
    return dict(version='dual_clock_v1',age_clock=mode,availability_guard='actual_wall_clock_completion')
