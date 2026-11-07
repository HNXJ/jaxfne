"""Tests for the Jansen-Rit Fig. 3 driver: drive hold, spectrum metrics on known signals, argument refusal."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_jr_fig3 as jr  # noqa: E402


def test_drive_holds_each_draw_and_seed_changes_it():
    d0 = jr.drive(0, 100, 1e-4, 2e-3)
    assert d0.shape == (100,)
    assert np.all(d0[:20] == d0[0]) and d0[20] != d0[0]
    assert np.all((d0 >= jr.P_LOW) & (d0 <= jr.P_HIGH))
    assert not np.array_equal(d0, jr.drive(1, 100, 1e-4, 2e-3))


def test_spectrum_metrics_sine_is_oscillatory_at_its_frequency():
    dt = 1e-3
    t = np.arange(int(9.0 / dt)) * dt
    m = jr.spectrum_metrics(np.sin(2 * np.pi * 10.0 * t), dt)
    assert abs(m["f_peak_hz"] - 10.0) <= 0.25
    assert m["osc_index"] > 0.9


def test_spectrum_metrics_white_noise_is_below_noise_threshold():
    rng = np.random.default_rng(0)
    m = jr.spectrum_metrics(rng.standard_normal(9000), 1e-3)
    assert m["osc_index"] < 0.4


def test_run_c_standard_set_is_alpha_and_c_is_consumed():
    jr.jaxfne.enable_x64()
    alpha = jr.run_c(135.0, 0, 5.0, 1e-4, 2e-3, 1.0)
    assert alpha["finite"] and 8.0 <= alpha["f_peak_hz"] <= 13.0 and alpha["osc_index"] >= 0.4
    assert 5.0 <= alpha["eeg_mean_mv"] <= 10.0
    spike = jr.run_c(270.0, 0, 5.0, 1e-4, 2e-3, 1.0)
    assert spike["eeg_std_mv"] > 5.0 * alpha["eeg_std_mv"], "C must change the output"


def test_main_refuses_run_shorter_than_discard_plus_segment(tmp_path):
    assert jr.main(["--seeds", "0", "--out", str(tmp_path), "--t-s", "4.5"]) == 2
    assert not any(tmp_path.iterdir())
