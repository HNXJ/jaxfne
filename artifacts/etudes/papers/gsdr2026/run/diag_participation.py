"""Per-cycle participation diagnostic for the GSDR lane (pure NumPy, CPU).

Given a binary onset matrix (time, cells) at ``dt_ms`` and a gamma period,
partitions ``[t_l, t_r)`` into consecutive cycles of ``period_ms`` and
reports, per cycle, the fraction of cells with >= 1 onset, plus per-cell
window rates. No simulation here; the caller supplies onsets.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def cycle_participation(onsets, *, dt_ms, t_l_ms, t_r_ms, period_ms):
    """Return participation vector, cycle count and per-cell rates (Hz)."""
    import numpy as np

    b = np.asarray(onsets, dtype=bool)
    if b.ndim != 2:
        raise ValueError("onsets must be (time, cells), got %s" % (b.shape,))
    n_steps = b.shape[0]
    i0 = max(0, min(int(t_l_ms / float(dt_ms)), n_steps - 2))
    i1 = max(i0 + 2, min(int(t_r_ms / float(dt_ms)), n_steps - 1))
    steps_per_cycle = max(1, int(round(float(period_ms) / float(dt_ms))))
    win = b[i0:i1, :]
    n_cycles = win.shape[0] // steps_per_cycle
    if n_cycles == 0:
        raise ValueError("window holds no full cycle at period %s" % (period_ms,))
    trimmed = win[: n_cycles * steps_per_cycle, :]
    per_cycle = trimmed.reshape(n_cycles, steps_per_cycle,
                                trimmed.shape[1]).max(axis=1)
    participation = per_cycle.mean(axis=1)
    dur_s = (i1 - i0) * float(dt_ms) / 1000.0
    cell_rates = win.sum(axis=0) / dur_s
    return {"participation": np.asarray(participation, dtype=np.float64),
            "n_cycles": int(n_cycles),
            "cell_rates_hz": np.asarray(cell_rates, dtype=np.float64)}
