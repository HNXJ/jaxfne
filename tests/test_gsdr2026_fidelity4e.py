"""Lane 4e estimand fidelity: model z through the paper's own recipes.

Paper cites are the GSDR Fig. 6 run read as text (never imported):
``main_fig6.py:109-126`` label PSD (smoothed, min-subtracted, max-normalized),
``:129`` grid, ``:135``/``:139`` label windows, ``:306-344`` whole-trial rates,
``:407-421`` model PSD (max-only), ``:427`` psd_loss, ``:429-440`` band
penalty, ``analysis.py:109`` kappa (pairwise Fleiss, 5 ms, no clamp).

Fail-first: on the pre-4e tree ``run_gsdr`` has no ``spike_trace_psd`` /
``window_rates_hz``, so the import below fails.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_ETUDE = Path(__file__).resolve().parents[1] / "artifacts" / "etudes" \
    / "papers" / "gsdr2026" / "run"


def _load(name):
    path = _ETUDE / (name + ".py")
    spec = importlib.util.spec_from_file_location("gsdr2026_4e_" + name,
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rg = _load("run_gsdr")
dl = _load("data_labels")

DT_MS = 0.1
N_STEPS = 15001
N_CELLS = 50


def _spike_train_v(period_steps, width=1, amp=-10.0, rest=-70.0,
                   n_steps=N_STEPS, n_cells=N_CELLS):
    """Voltage traces with exact periodic upward -20 mV crossings."""
    v = np.full((n_steps, n_cells), np.float32(rest))
    for t in range(period_steps // 2, n_steps, period_steps):
        v[t:t + width, :] = np.float32(amp)
    return v


def test_spike_trace_psd_peaks_at_drive():
    """40 Hz synthetic spikes give a ~40 Hz like-for-like PSD peak."""
    v = _spike_train_v(250)  # 25 ms period at dt 0.1 -> 40 Hz
    psd = np.asarray(rg.spike_trace_psd(v))
    assert psd.shape == (40,)
    assert abs(psd.max() - 1.0) < 0.01  # max-normalized (label recipe)
    grid = np.asarray(rg.target_grid(), dtype=np.float64)
    assert abs(float(grid[int(np.argmax(psd))]) - 40.0) < 2.0


def test_window_rates_hz_counts_exact_spikes():
    """100 Hz synthetic crossings in [500, 1000) read exactly 100 Hz."""
    v = _spike_train_v(100)  # 10 ms period -> 100 Hz
    rates = np.asarray(rg.window_rates_hz(v, 500.0, 1000.0))
    assert rates.shape == (N_CELLS,)
    assert float(rates.mean()) == 100.0
    # Silence outside the window reads 0 inside it.
    v2 = np.full((N_STEPS, N_CELLS), np.float32(-70.0))
    v2[0:2000, :] = _spike_train_v(100, n_steps=2000)
    assert float(np.asarray(rg.window_rates_hz(v2, 500.0, 1000.0)).mean()) \
        == 0.0


def test_model_psd_is_max_only_while_label_psd_smooths():
    """Paper asymmetry pinned: model max-only, labels smoothed+min-subtracted."""
    t = np.arange(13000, dtype=np.float64) * 0.1
    sig = np.sin(2.0 * np.pi * 40.0 * (t / 1000.0))
    v = np.tile(sig[:, None], (1, N_CELLS))
    v = np.vstack([np.full((1000, N_CELLS), -70.0), v,
                   np.full((1001, N_CELLS), -70.0)])[:N_STEPS, :]
    got = np.asarray(rg.psd_mean_voltage(v))
    # Independent max-only recomputation (no smoothing anywhere).
    w = v[1000:14000, :].mean(axis=1).astype(np.float64)
    n = w.shape[0]
    fs = 1000.0 / DT_MS
    power = np.abs(np.fft.rfft(w)) ** 2 / (n * fs)
    freqs = np.fft.rfftfreq(n, d=DT_MS / 1000.0)
    interp = np.interp(np.asarray(rg.target_grid(), dtype=np.float64),
                       freqs, power)
    assert np.allclose(got, interp / (interp.max() + 1e-6), rtol=1e-9)
    # The label recipe on the same mean trace differs (smoothing+minsub).
    lab = np.asarray(dl.psd_of_mean_trace(w))
    assert float(np.abs(got - lab).max()) > 1e-6
