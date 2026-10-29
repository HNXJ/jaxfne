"""T56B: ``binned_population_rate_hz`` refuses an off-grid ``bin_ms``.

Ruling 2026-10-28: ``bin_ms`` must be a positive multiple of ``dt_ms``
(the ``jaxfne.analysis.metrics`` rule), not silently rounded.
"""

from __future__ import annotations

import numpy as np
import pytest

DT_ON = 0.5
DT_OFF = 0.3
BIN_MS = 10.0


def _fixture():
    """Small fixed spike array (n_steps=40, n_units=3) with real spikes."""
    arr = np.zeros((40, 3))
    arr[0:4, 0] = 1.0
    arr[10:12, 1] = 1.0
    arr[20, 2] = 1.0
    arr[35, 0] = 1.0
    assert arr.sum() > 0
    return arr


def test_offgrid_bin_refused():
    from jaxfne.vis.core import binned_population_rate_hz

    with pytest.raises(ValueError, match="bin_ms"):
        binned_population_rate_hz(_fixture(), DT_OFF, BIN_MS)


def test_ongrid_bin_matches_hand_reshape_mean():
    from jaxfne.vis.core import binned_population_rate_hz

    arr = _fixture()
    centers, rate = binned_population_rate_hz(arr, DT_ON, BIN_MS)
    bin_steps = int(BIN_MS / DT_ON)  # 20
    n_bins = arr.shape[0] // bin_steps  # 2
    hand = arr[: n_bins * bin_steps].reshape(n_bins, bin_steps, -1).mean(axis=(1, 2)) * (1000.0 / DT_ON)
    np.testing.assert_allclose(rate, hand)
    np.testing.assert_allclose(centers, (np.arange(n_bins) + 0.5) * bin_steps * DT_ON)