"""Known-answer tests for artifacts/etudes/izhikevich2003/izh_analysis.py.

Spec: PREREG_P3.md "Detector calibration" (each bullet is a test here), plus one
peak_excess test on a constructed signal. Every fixture asserts its own
precondition. The module is loaded by path, as in test_benchmark_protocol_v1.py.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "artifacts" / "etudes" / "izhikevich2003" / "izh_analysis.py"
DT_MS = 0.1  # kernel dt of the Fig. 2 runs; 10 ms = 100 rows


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("izh_analysis", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_burst_doublet_train_gives_one_burst_per_doublet():
    izh = _load_module()
    starts = [1000 * k for k in range(5)]  # doublet onsets 100 ms apart
    rows = []
    for s in starts:
        rows += [s, s + 50]  # inner ISI 5 ms
    assert len(rows) == 10  # fixture: the doublet train has spikes
    assert izh.bursts(rows, DT_MS) == [[s, s + 50] for s in starts]


def test_burst_regular_50hz_train_gives_none():
    izh = _load_module()
    rows = list(range(0, 20000, 200))  # ISI 20 ms, 2 s
    assert len(rows) >= 2 and np.all(np.diff(rows) == 200)  # fixture: regular 50 Hz with ISIs
    assert izh.bursts(rows, DT_MS) == []


def test_burst_isi_exactly_10ms_gives_none():
    izh = _load_module()
    rows = list(range(0, 10000, 100))  # ISI exactly 10.0 ms
    assert len(rows) >= 2 and np.all(np.diff(rows) == 100)  # fixture: every ISI is 10.0 ms
    assert izh.bursts(rows, DT_MS) == []


def test_burst_three_spikes_at_4ms_is_one_burst():
    izh = _load_module()
    rows = [500, 540, 580]  # ISIs 4 ms
    assert len(rows) == 3 and np.all(np.diff(rows) == 40)  # fixture: a 3-spike run at 4 ms
    assert izh.bursts(rows, DT_MS) == [[500, 540, 580]]


def test_rate_and_isi_ratio_known_answer():
    izh = _load_module()
    # ISIs 10, 20, 30 ms; the first spike is 5 ms into the window. Row k is stamped (k + 1) dt,
    # so row 49 is at 5.0 ms and the event is at row 0.
    rows = [49, 149, 349, 649]
    assert np.allclose(np.diff([(r + 1) * DT_MS for r in rows]), [10.0, 20.0, 30.0])  # fixture
    win = izh.window_rows(rows, 0, DT_MS)
    assert win == rows  # fixture: all four spikes are inside the 300 ms window
    assert izh.rate_hz(win) == pytest.approx(4 / 0.3, rel=1e-12)
    assert izh.isi_ratio(win) == pytest.approx(3.0, rel=1e-12)


def test_c8_maxima_of_decaying_20hz_oscillation():
    izh = _load_module()
    t = np.arange(1000) * 1e-4  # 0.1 s at 0.1 ms samples
    v = -62.5 + 2.0 * np.exp(-t / 0.05) * np.sin(2 * np.pi * 20.0 * t)
    assert v.max() > -61.0  # fixture: the oscillation rises above baseline
    m = izh.c8_maxima(v, 0)
    assert len(m) >= 2
    assert np.all(np.diff(v[m]) < 0)  # decreasing maxima
    assert abs(izh.c8_frequency_hz(m, DT_MS) - 20.0) < 0.1


def test_window_keeps_rows_e_through_e_plus_n_minus_1():
    izh = _load_module()
    e, n = 100, 3000  # event row 100, 300 ms = 3000 rows
    rows = [e - 1, e, e + n - 1, e + n]
    assert izh.window_rows(rows, e, DT_MS) == [e, e + n - 1]


def test_c8_prominence_sides_stop_at_the_window():
    izh = _load_module()
    # Left side with no higher sample in the window runs to the window start (row 2, 0.97):
    # prominence 1 - max(0.97, 0) = 0.03. Reaching past the start (to -3) would give 1.0.
    v = np.array([5.0, -3.0, 0.97, 1.0, 0.0, 0.0])
    assert v[:2].min() < v[2]  # fixture: lower samples sit before the window
    assert izh.c8_maxima(v, 2) == []
    # Left side stops at the nearest higher sample (row 1): prominence 0.9375 - 0.875 = 0.0625.
    v = np.array([0.0, 1.0, 0.875, 0.9375, 0.0, 0.0])
    assert izh.c8_maxima(v, 0) == [1, 3]
    v[2] = 0.9  # prominence 0.0375
    assert izh.c8_maxima(v, 0) == [1]
    # Right side runs to the window end: a peak with no higher sample after it and a shallow tail.
    v3 = np.array([0.0, 1.0, 0.98, 0.97])
    assert izh.c8_maxima(v3, 0) == []  # right min 0.97 > left min 0 -> prominence 0.03
    v3[3] = 0.9
    assert izh.c8_maxima(v3, 0) == [1]  # right min 0.9 -> prominence 0.1


def test_c2_initial_burst_then_regular_tail():
    izh = _load_module()
    burst = [0, 40, 80]  # 4 ms ISIs
    tail = [300, 600, 900]  # 30 ms ISIs, first gap 22 ms
    assert izh.c2_burst_then_tail(burst + tail, DT_MS)
    assert not izh.c2_burst_then_tail(burst + tail[:2], DT_MS)  # 2 tail spikes
    assert not izh.c2_burst_then_tail([0] + tail, DT_MS)  # no initial burst
    assert not izh.c2_burst_then_tail([0, 300, 340, 380, 600, 900], DT_MS)  # single, then a burst
    assert not izh.c2_burst_then_tail(burst + [300, 399, 700], DT_MS)  # tail ISI 9.9 ms
    assert izh.c2_burst_then_tail(burst + [300, 400, 700], DT_MS)  # tail ISI exactly 10 ms


def test_c3_burst_onset_frequency():
    izh = _load_module()
    rows = [s + d for s in (0, 400, 800) for d in (0, 30)]  # bursts 40 ms apart
    assert len(izh.bursts(rows, DT_MS)) == 3  # fixture
    assert izh.burst_onset_frequency_hz(rows, DT_MS) == pytest.approx(25.0, rel=1e-12)


def test_c8_monotone_relaxation_with_ripple_has_no_maximum():
    izh = _load_module()
    idx = np.arange(1000)
    t = idx * 1e-4
    ripple = 0.01 * np.where(idx % 2 == 0, 1.0, -1.0)  # +-0.01 mV alternating
    v = -62.5 + 2.0 * np.exp(-t / 0.05) + ripple
    raw = (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])
    assert raw.sum() > 0  # fixture: the ripple creates raw maxima before the prominence rule
    assert izh.c8_maxima(v, 0) == []


def test_n2_median_cv_poisson_8hz_in_band():
    izh = _load_module()
    rng = np.random.default_rng(0)
    p_row = 8.0 * DT_MS / 1000.0  # spike probability per 0.1 ms row at 8 Hz
    n_rows = 100_000  # 10 s
    cvs = []
    counts = []
    for _ in range(1000):
        rows = izh.spike_rows(rng.random(n_rows) < p_row)
        counts.append(len(rows))
        cvs.append(izh.isi_cv(rows))
    counts = np.asarray(counts)
    assert counts.min() >= 10 and abs(counts.mean() - 80) < 2  # fixture: ~8 Hz over 10 s, all qualify
    assert 0.6 <= float(np.median(cvs)) <= 1.5


def test_n2_median_cv_regular_8hz_below_0p6():
    izh = _load_module()
    rng = np.random.default_rng(0)
    n_rows = 100_000  # 10 s
    cvs = []
    for _ in range(1000):
        phase = int(rng.integers(0, 1250))  # random phase, period 125 ms = 1250 rows
        rows = list(range(phase, n_rows, 1250))
        assert len(rows) >= 10 and np.all(np.diff(rows) == 1250)  # fixture: regular 8 Hz
        cvs.append(izh.isi_cv(rows))
    assert float(np.median(cvs)) < 0.6


def test_peak_excess_finds_10hz_sinusoid_in_white_poisson():
    izh = _load_module()
    rng = np.random.default_rng(0)
    n_bins = 10_000  # 10 s at 1 ms
    counts = rng.poisson(lam=1000 * 8.0 * 1e-3, size=n_bins).astype(float)  # N=1000 at 8 Hz, 1 ms bins
    assert abs(counts.mean() - 8.0) < 0.5 and counts.var() > 4  # fixture: Poisson(8) counts
    t = np.arange(n_bins) / 1000.0
    tone = 1.0 * np.sin(2 * np.pi * 10.0 * t)  # 10 Hz, amplitude 1 count
    assert np.max(np.abs(tone)) == pytest.approx(1.0)  # fixture: the tone is present
    res = izh.peak_excess(counts + tone, 1000.0, (8.0, 13.0))
    assert res["pe"] is not None
    assert abs(res["f_pk"] - 10.0) <= 0.5
    assert res["pe"] > 6
    assert res["local_max"] and res["edge_ok"]
    assert izh.holds(res, 6.0)


def _constructed_psd(monkeypatch, izh, bump):
    """Replace Welch with an exact aperiodic PSD (a=0, k=1, chi=1.5) times ``bump(freqs)``."""
    freqs = np.arange(0, 500.5, 0.5)
    pxx = 10.0 ** izh.aperiodic_log10(freqs, 0.0, 1.0, 1.5) * bump(freqs)
    monkeypatch.setattr(izh, "welch", lambda *a, **k: (freqs, pxx))


def test_peak_excess_exclusion_and_background_on_an_exact_psd(monkeypatch):
    izh = _load_module()
    # Bump x5 at 10 Hz, x2 on 8.5-11.5 Hz: only a full +-2 Hz exclusion recovers the exact
    # background, so PE is 5 exactly; a narrower exclusion biases the refit.
    _constructed_psd(monkeypatch, izh, lambda f: np.where(f == 10.0, 5.0,
                                                          np.where(np.abs(f - 10.0) <= 1.5, 2.0, 1.0)))
    res = izh.peak_excess(np.zeros(10), 1000.0, (8.0, 13.0))
    assert res["f_pk"] == 10.0 and res["local_max"] and res["edge_ok"]
    assert res["pe"] == pytest.approx(5.0, rel=1e-4)


def test_peak_excess_edge_and_local_max_rules(monkeypatch):
    izh = _load_module()
    _constructed_psd(monkeypatch, izh, lambda f: np.where(f == 9.0, 5.0, 1.0))
    assert izh.peak_excess(np.zeros(10), 1000.0, (8.0, 13.0))["edge_ok"]  # exactly 1 Hz: allowed
    _constructed_psd(monkeypatch, izh, lambda f: np.where(f == 8.5, 5.0, 1.0))
    res = izh.peak_excess(np.zeros(10), 1000.0, (8.0, 13.0))
    assert res["f_pk"] == 8.5 and not res["edge_ok"] and not izh.holds(res, 2.0)
    p = 10.0 ** izh.aperiodic_log10(np.array([10.0, 10.5]), 0.0, 1.0, 1.5)
    # Exactly equal PSD at 10.0 and 10.5 Hz (a flat top): neither is a strict local maximum.
    _constructed_psd(monkeypatch, izh,
                     lambda f: np.where(f == 10.0, 5.0, np.where(f == 10.5, 5.0 * p[0] / p[1], 1.0)))
    res = izh.peak_excess(np.zeros(10), 1000.0, (8.0, 13.0))
    assert res["pe"] is None and not izh.holds(res, 2.0)  # no strict PSD local maximum in band


def test_peak_excess_picks_the_largest_residual_among_psd_local_maxima(monkeypatch):
    izh = _load_module()
    # Residual is largest at 10.5 Hz (x4.2), but on the falling background the PSD at 10.0 Hz
    # (x4.0) is higher, so 10.5 Hz is a flank bin; f_pk must be the local maximum at 10.0 Hz.
    _constructed_psd(monkeypatch, izh, lambda f: np.where(f == 10.0, 4.0, np.where(f == 10.5, 4.2, 1.0)))
    p = 10.0 ** izh.aperiodic_log10(np.array([10.0, 10.5]), 0.0, 1.0, 1.5)
    assert p[0] * 4.0 > p[1] * 4.2  # fixture: 10.5 Hz sits on the flank
    res = izh.peak_excess(np.zeros(10), 1000.0, (8.0, 13.0))
    assert res["f_pk"] == 10.0 and res["local_max"] and izh.holds(res, 2.0)


def test_peak_excess_second_fit_failure_is_no_peak(monkeypatch):
    izh = _load_module()
    real_fit = izh.fit_aperiodic
    calls = []

    def fails_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("Optimal parameters not found")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(izh, "fit_aperiodic", fails_second)
    counts = np.random.default_rng(1).poisson(8.0, size=10_000).astype(float)
    res = izh.peak_excess(counts, 1000.0, (8.0, 13.0))
    assert len(calls) == 2 and res["pe"] is None and res["f_pk"] is not None


def test_peak_excess_without_a_fit_is_no_peak(monkeypatch):
    izh = _load_module()
    zero = izh.peak_excess(np.zeros(10_000), 1000.0, (8.0, 13.0))
    assert zero["pe"] is None and not izh.holds(zero, 2.0)

    def no_convergence(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found")

    monkeypatch.setattr(izh, "fit_aperiodic", no_convergence)
    counts = np.random.default_rng(1).poisson(8.0, size=10_000).astype(float)
    assert counts.sum() > 0  # fixture: a nonzero PSD reaches the fit
    res = izh.peak_excess(counts, 1000.0, (8.0, 13.0))
    assert res["pe"] is None and not izh.holds(res, 2.0)
