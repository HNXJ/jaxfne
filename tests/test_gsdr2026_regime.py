"""GSDR regime-at-gstar tests: §6.2 quantities on a tiny synthetic raster (CPU, NumPy only).

Fail-first: on a tree where ``window_indices`` runs one sample long
(``i1 + 1``, the broken-window mutant), the window test and the rate test
fail (the boundary spike at t = 100 ms leaks in); where participation is
computed over all cells instead of the E slice, the half-participation test
fails.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

DT_MS = 1.0
T_L_MS, T_R_MS = 0.0, 100.0
PERIOD_MS = 25.0
N_E = 4
N_CELLS = 6


def _load_regime():
    path = (Path(__file__).resolve().parents[1] / "artifacts" / "etudes"
            / "papers" / "gsdr2026" / "diag" / "regime_at_gstar.py")
    spec = importlib.util.spec_from_file_location("gsdr2026_regime_at_gstar",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


regime_mod = _load_regime()


def _full_sync_onsets():
    # 4 E cells spike once per 25 ms cycle (indices 1, 26, 51, 76);
    # 2 extra cells silent. Window [0, 100) ms holds 4 full cycles.
    onsets = np.zeros((110, N_CELLS), dtype=bool)
    for c in range(N_E):
        for k in range(4):
            onsets[int(k * 25.0 + 1.0), c] = True
    return onsets


def _half_sync_onsets():
    # E cells 0-1 spike per cycle, E cells 2-3 silent; cell 0 also spikes
    # at index 100 (t = 100 ms), outside the [0, 100) ms window.
    onsets = np.zeros((110, N_CELLS), dtype=bool)
    for c in (0, 1):
        for k in range(4):
            onsets[int(k * 25.0 + 1.0), c] = True
    onsets[100, 0] = True
    return onsets


def test_window_indices_match_run_gsdr_convention():
    assert regime_mod.window_indices(110, 0.0, 100.0, 1.0) == (0, 100)
    assert regime_mod.window_indices(15001, 500.0, 1000.0, 0.1) == (5000, 10000)


def test_full_sync_participation_and_rates():
    rec = regime_mod.regime_from_onsets(
        _full_sync_onsets(), dt_ms=DT_MS, t_l_ms=T_L_MS, t_r_ms=T_R_MS,
        period_ms=PERIOD_MS, n_e=N_E)
    assert rec["window_ms"] == [0.0, 100.0]
    assert rec["gamma_period_ms"] == PERIOD_MS
    assert rec["n_gamma_cycles"] == 4
    assert rec["participation_mean"] == 1.0
    assert rec["participation_std"] == 0.0
    assert rec["participation_min"] == 1.0
    assert rec["participation_max"] == 1.0
    assert rec["e_rates_hz"] == {"min": 40.0, "p10": 40.0, "p25": 40.0,
                                "median": 40.0, "p75": 40.0, "p90": 40.0,
                                "max": 40.0}
    assert rec["e_frac_silent"] == 0.0
    assert rec["e_frac_above_10hz"] == 1.0
    assert rec["e_stim_window_mean_hz"] == 40.0
    assert rec["k_fit"] == rec["kappa_500_1000"]
    assert abs(rec["k_fit"] - 0.5384615384615382) < 1e-12


def test_half_sync_rates_and_silence():
    rec = regime_mod.regime_from_onsets(
        _half_sync_onsets(), dt_ms=DT_MS, t_l_ms=T_L_MS, t_r_ms=T_R_MS,
        period_ms=PERIOD_MS, n_e=N_E)
    assert rec["n_gamma_cycles"] == 4
    assert rec["participation_mean"] == 0.5
    assert rec["participation_min"] == 0.5
    assert rec["participation_max"] == 0.5
    # The index-100 boundary spike stays outside [0, 100): cell 0 is 40 Hz.
    assert rec["e_rates_hz"] == {"min": 0.0, "p10": 0.0, "p25": 0.0,
                                "median": 20.0, "p75": 40.0, "p90": 40.0,
                                "max": 40.0}
    assert rec["e_frac_silent"] == 0.5
    assert rec["e_frac_above_10hz"] == 0.5
    assert rec["e_stim_window_mean_hz"] == 20.0
    assert rec["k_fit"] == rec["kappa_500_1000"]


def test_thirds_counts_split_full_trial():
    onsets = np.zeros((1501, N_CELLS), dtype=bool)
    onsets[[5, 6], 0] = True
    onsets[[500, 501, 502], 1] = True
    onsets[1499, 2] = True
    assert regime_mod.thirds_counts(onsets, 1502, DT_MS) == [2, 3, 1]
