"""Hand-computed pairwise-Fleiss kappa check for ``data_labels.window_kappa``.

No author code is imported or run here: the expected value below is worked by
hand from the pairwise-Fleiss definition (per-bin observed pair agreement
``P_i`` averaged to ``P_o``, chance agreement ``P_e`` from the pooled spike
rate, ``(P_o - P_e) / max(1e-6, 1 - P_e)``), on a tiny spike matrix written
out in full. Replaces the deleted ``scratch/_temp_crosscheck_kappa.py``,
which violated the read-as-data rule by importing ``compute_kappa``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_data_labels():
    path = (Path(__file__).resolve().parents[1] / "artifacts" / "etudes"
            / "papers" / "gsdr2026" / "run" / "data_labels.py")
    spec = importlib.util.spec_from_file_location("gsdr2026_data_labels_hand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dl = _load_data_labels()


def test_window_kappa_matches_hand_computed_pairwise_fleiss():
    """Tiny 3-unit x 11-sample traces give kappa exactly -1/3 by hand."""
    # Traces (convolved-spike scale: 0 baseline, 250 onset peaks, thr 20.0).
    # Onsets (upward crossings) land at t=0,5 (unit 0), t=0 (unit 1), none
    # (unit 2); with bin 5 ms / fs 1000 (5 samples/bin) over 10 onset
    # columns there are 2 bins: binned spike counts [2, 1], silences [1, 2].
    u0 = [0.0, 250.0, 0.0, 0.0, 0.0, 0.0, 250.0, 0.0, 0.0, 0.0, 0.0]
    u1 = [0.0, 250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u2 = [0.0] * 11
    mat = np.array([u0, u1, u2], dtype=np.float64)
    # Fixture check: the onset matrix is exactly what is written above.
    onsets = (mat[:, :-1] < 20.0) & (mat[:, 1:] >= 20.0)
    assert onsets.tolist() == [
        [True, False, False, False, False, True, False, False, False, False],
        [True, False, False, False, False, False, False, False, False, False],
        [False] * 10,
    ]
    # Hand computation: den = 3*2 = 6; P_0 = (2*1 + 1*0)/6 = 1/3,
    # P_1 = (1*0 + 2*1)/6 = 1/3; P_o = 1/3; p_spike = 3/6 = 1/2;
    # P_e = 1/4 + 1/4 = 1/2; kappa = (1/3 - 1/2)/(1 - 1/2) = -1/3.
    got = dl.window_kappa(mat, 0, 11, bin_ms=5.0, fs_hz=1000.0, threshold=20.0)
    assert got == pytest.approx(-1.0 / 3.0, abs=1e-12), \
        "hand kappa -1/3, got %r" % got
