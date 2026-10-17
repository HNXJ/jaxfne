"""Per-cycle participation helper for the Step-6 regime diagnostic (CPU)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "artifacts" / "etudes" / "papers" / "gsdr2026"
                       / "run"))

from diag_participation import cycle_participation


def test_full_participation_all_cells_every_cycle():
    # 4 E cells, dt 0.1 ms, window [0, 100) ms, period 25 ms -> 4 cycles.
    # Every cell spikes once per cycle.
    dt = 0.1
    n_steps = int(110.0 / dt)
    onsets = np.zeros((n_steps, 4), dtype=bool)
    for c in range(4):
        for k in range(4):
            onsets[int((k * 25.0 + 1.0) / dt), c] = True
    out = cycle_participation(onsets, dt_ms=dt, t_l_ms=0.0, t_r_ms=100.0,
                              period_ms=25.0)
    assert out["n_cycles"] == 4
    assert float(np.mean(out["participation"])) == 1.0


def test_half_participation_two_of_four_cells():
    dt = 0.1
    n_steps = int(110.0 / dt)
    onsets = np.zeros((n_steps, 4), dtype=bool)
    for c in (0, 1):
        for k in range(4):
            onsets[int((k * 25.0 + 1.0) / dt), c] = True
    out = cycle_participation(onsets, dt_ms=dt, t_l_ms=0.0, t_r_ms=100.0,
                              period_ms=25.0)
    assert float(np.mean(out["participation"])) == 0.5
    # Silent cells carry zero window rate.
    assert out["cell_rates_hz"][2] == 0.0
    assert out["cell_rates_hz"][3] == 0.0
