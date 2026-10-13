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
    # Exactly equal PSD at 10.0 and 10.5 Hz (a flat top): neither is a strict local maximum.
    freqs = np.arange(0, 500.5, 0.5)
    pxx = 10.0 ** izh.aperiodic_log10(freqs, 0.0, 1.0, 1.5)
    pxx[20] *= 5.0
    pxx[21] = pxx[20]
    assert freqs[20] == 10.0 and freqs[21] == 10.5 and pxx[20] == pxx[21]  # fixture: flat top
    monkeypatch.setattr(izh, "welch", lambda *a, **k: (freqs, pxx))
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


# --- P3 runner/scorer tests (run_izh.py, net_reference.py) --------------------------------------
#
# Spec: PREREG_P3.md (Fig. 2 C1-C8, Fig. 3 N1-N4, completeness, S1). Each test
# loads its module by path inside the function, so on the baseline (no
# run_izh.py / net_reference.py) every test below fails while the tests above
# still run.

RUN_IZH_PATH = REPO_ROOT / "artifacts" / "etudes" / "izhikevich2003" / "run_izh.py"
NETREF_PATH = REPO_ROOT / "artifacts" / "etudes" / "izhikevich2003" / "net_reference.py"


def _load_run_izh() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_izh", RUN_IZH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_netref() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("net_reference", NETREF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p3_c1_pass_and_fail():
    r = _load_run_izh()
    rows = [0, 100, 300, 600]  # ISIs 10, 20, 30 ms in the window
    got = r.eval_c1(rows, 0, DT_MS)
    assert got["verdict"] == "pass" and got["numbers"]["isi_ratio"] == pytest.approx(3.0)
    assert r.eval_c1([0, 100], 0, DT_MS)["verdict"] == "fail"  # 2 spikes


def test_p3_c2_pass_and_fail():
    r = _load_run_izh()
    good = [0, 40, 80, 300, 600, 900]  # burst then a 30 ms tail
    assert r.eval_c2(good, 0, DT_MS)["verdict"] == "pass"
    assert r.eval_c2([0, 300, 600, 900], 0, DT_MS)["verdict"] == "fail"  # no burst
    assert r.eval_c2([0, 40, 80, 300, 600], 0, DT_MS)["verdict"] == "fail"  # 2 tail spikes


def test_p3_c3_pass_and_fail():
    r = _load_run_izh()
    rows = [s + d for s in (0, 400, 800) for d in (0, 30)]  # 3 bursts, 40 ms apart
    got = r.eval_c3(rows, 0, DT_MS)
    assert got["verdict"] == "pass" and got["numbers"]["onset_freq_hz"] == pytest.approx(25.0)
    assert r.eval_c3(rows[:4], 0, DT_MS)["verdict"] == "fail"  # 2 bursts


def test_p3_c4_pass_and_fail():
    r = _load_run_izh()
    fast = list(range(0, 3000, 50))  # 200 Hz, ISI ratio 1
    got = r.eval_c4(fast, 0, DT_MS)
    assert got["verdict"] == "pass" and got["numbers"]["rate_hz"] == pytest.approx(200.0)
    adapting = [0, 100, 300, 600]  # ISI ratio 3
    assert r.eval_c4(adapting, 0, DT_MS)["verdict"] == "fail"


def test_p3_c5_is_relational_to_c1():
    r = _load_run_izh()
    rows = [0, 100, 250, 450, 700, 1000, 1350, 1750, 2200, 2700]  # ratio 5, 33.3 Hz
    got = r.eval_c5(rows, 0, DT_MS, 26.0)
    assert got["verdict"] == "pass" and got["numbers"]["rate_hz"] == pytest.approx(100 / 3)
    assert r.eval_c5(rows, 0, DT_MS, 40.0)["verdict"] == "fail"  # not above RS
    assert r.eval_c5([0, 100], 0, DT_MS, 1.0)["verdict"] == "fail"  # too few spikes


def test_p3_c6_ascending_search():
    r = _load_run_izh()
    regular = [0, 100, 200, 300]  # 4 spikes, no burst, CV 0
    assert r.eval_c6_search([(1.0, [0, 10], 0), (2.0, regular, 0)], DT_MS)["numbers"][
        "selected_I_plus"] == 2.0
    good = r.eval_c6_search([(1.0, [0, 10], 0), (2.0, regular, 0)], DT_MS)
    assert good["verdict"] == "pass"
    assert r.eval_c6_search([(1.0, [0, 10], 0)], DT_MS)["verdict"] == "fail"
    assert r.eval_c6_search([], DT_MS)["verdict"] == "fail"  # empty search fails


def test_p3_c7_burst_within_50ms_of_release():
    r = _load_run_izh()
    assert r.eval_c7([100, 150, 2000, 2300, 2600], 0, DT_MS)["verdict"] == "pass"
    late = r.eval_c7([1000, 1050, 2000, 2300, 2600], 0, DT_MS)
    assert late["verdict"] == "fail"  # burst onset 100 ms after release
    assert r.eval_c7([100, 400, 700, 1000], 0, DT_MS)["verdict"] == "fail"  # no burst


def test_p3_c8_pass_and_fail():
    r = _load_run_izh()
    t = np.arange(1000) * 1e-4
    v = -62.5 + 2.0 * np.exp(-t / 0.05) * np.sin(2 * np.pi * 20.0 * t)
    got = r.eval_c8(v, 0, DT_MS, [])
    assert got["verdict"] == "pass"
    assert abs(got["numbers"]["freq_hz"] - 20.0) < 0.1
    assert r.eval_c8(v, 0, DT_MS, [500])["verdict"] == "fail"  # a spike fails C8
    idx = np.arange(1000)
    flat = -62.5 + 2.0 * np.exp(-idx * 1e-4 / 0.05) + 0.01 * np.where(idx % 2 == 0, 1.0, -1.0)
    assert r.eval_c8(flat, 0, DT_MS, [])["verdict"] == "fail"  # no maxima


def test_p3_fig2_missing_or_misconfigured_run_is_incomplete(tmp_path):
    import json as _json

    r = _load_run_izh()
    (tmp_path / "manifest.json").write_text(_json.dumps({"sensitivity": None, "runs": {}}))
    got = r.score_fig2_dir(tmp_path, 4.0, (8.0, 13.0))
    assert got["overall"] == "incomplete"  # every run missing
    assert all(v == "incomplete" for v in got["verdicts"].values())
    bad = dict(r.expected_fig2_settings()["C1"], file="C1.npz", dt_ms=0.5)
    (tmp_path / "manifest.json").write_text(
        _json.dumps({"sensitivity": None, "runs": {"C1": bad}}))
    got = r.score_fig2_dir(tmp_path, 4.0, (8.0, 13.0))
    assert got["criteria"]["C1"]["verdict"] == "incomplete"  # misconfigured run


def test_p3_zero_spike_seed_fails_n1_to_n4():
    r = _load_run_izh()
    got = r.score_network_seed([[]] * 10, 2000, 0.5, 4.0, (8.0, 13.0))
    assert got["N1"]["numbers"]["mean_rate_hz"] == 0.0
    assert all(got[n]["verdict"] == "fail" for n in ("N1", "N2", "N3", "N4"))
    assert got["N3"]["numbers"]["pe"] is None and got["N4"]["numbers"]["pe"] is None


def test_p3_fig3_keys_split_in_prereg_order():
    import jax

    r = _load_run_izh()
    k_r, k_w, k_noise = r.fig3_keys(1)
    for got, want in zip((k_r, k_w, k_noise), jax.random.split(jax.random.PRNGKey(1), 3)):
        assert np.array_equal(np.asarray(got), np.asarray(want))
    assert not np.array_equal(np.asarray(k_r), np.asarray(k_w))  # order matters
    p1, _, _ = r.build_fig3_network(1)
    p2, _, _ = r.build_fig3_network(1)
    assert np.array_equal(np.asarray(p1.W), np.asarray(p2.W))  # fixture: same seed, same W
    p3, _, _ = r.build_fig3_network(2)
    assert not np.array_equal(np.asarray(p1.W), np.asarray(p3.W))  # fixture: seed changes W


def test_p3_out_dir_with_scored_needs_signed():
    r = _load_run_izh()
    with pytest.raises(SystemExit):
        r.check_out("results/scored_run", False)
    r.check_out("results/scored_run", True)
    r.check_out("results/smoke_run", False)


def test_p3_review_completeness_rules(tmp_path, monkeypatch):
    """Review notes: C8 counts every spike, check_out ignores case, C6 needs amplitudes 1..n,
    and fig2 and fig3 must share one code SHA."""
    import json as _json

    r = _load_run_izh()
    t = np.arange(1000) * 1e-4
    v = -62.5 + 2.0 * np.exp(-t / 0.05) * np.sin(2 * np.pi * 50.0 * t)
    assert r.eval_c8(v, 200, DT_MS, [])["verdict"] == "pass", "fixture must pass without spikes"
    assert r.eval_c8(v, 200, DT_MS, [100])["verdict"] == "fail"  # spike before start_row
    with pytest.raises(SystemExit):
        r.check_out("results/Scored_run", False)

    expected = r.expected_fig2_settings()
    runs = {}
    for rid, rows in (("C6_I1", [0, 10]), ("C6_I3", [0, 100, 200, 300])):
        np.savez(tmp_path / f"{rid}.npz", spike_rows=np.asarray(rows, dtype=np.int64))
        runs[rid] = dict(expected[rid], file=f"{rid}.npz")
    (tmp_path / "manifest.json").write_text(_json.dumps({"sensitivity": None, "runs": runs}))
    got = r.score_fig2_dir(tmp_path, 4.0, (8.0, 13.0))
    assert got["criteria"]["C6"]["verdict"] == "incomplete"  # amplitude 2 missing

    d2, d3 = tmp_path / "f2", tmp_path / "f3"
    for d, sha in ((d2, "a" * 40), (d3, "b" * 40)):
        d.mkdir()
        (d / "manifest.json").write_text(_json.dumps({"code_sha": sha}))
    monkeypatch.setattr(r, "score_fig2_dir", lambda *a: {"overall": "pass"})
    monkeypatch.setattr(r, "score_fig3_dir", lambda *a: {"arm": "pass"})
    rep = r.cmd_score(d2, d3, tmp_path / "score.json")
    assert rep["fig2"]["overall"] == "incomplete" and rep["fig3"]["arm"] == "incomplete"
    (d3 / "manifest.json").write_text(_json.dumps({"code_sha": "a" * 40}))
    rep = r.cmd_score(d2, d3, tmp_path / "score.json")
    assert rep["fig2"]["overall"] == "pass" and rep["fig3"]["arm"] == "pass"


def test_p3_s1_reference_writes_and_reports(tmp_path, monkeypatch):
    """net_reference.write_seed writes paired rows and cols; cmd_s1 scores both arms."""
    import json as _json

    nref = _load_netref()
    r = _load_run_izh()
    entry = nref.write_seed(tmp_path, 1, 0.5)  # longer than the 200 ms discard
    z = np.load(tmp_path / "1.npz")
    assert entry["n_spikes"] > 0, "fixture must spike"
    assert z["spike_rows"].shape == z["spike_cols"].shape == (entry["n_spikes"],)
    assert int(z["spike_cols"].max()) < 1000 and int(z["spike_rows"].max()) < entry["n_steps"]
    (tmp_path / "manifest.json").write_text(_json.dumps(
        {"sensitivity": "S1", "code_sha": "x", "runs": [entry]}))
    rep = r.cmd_s1(tmp_path, tmp_path, tmp_path / "s1.json")
    arms = rep["seeds"]["1"]
    assert arms["reference"]["N1"] == arms["kernel"]["N1"]  # same files, same numbers
    assert arms["reference"]["N1"]["numbers"]["mean_rate_hz"] > 0


def test_p3_sensitivity_runs_never_enter_score(tmp_path):
    import json as _json

    r = _load_run_izh()
    (tmp_path / "manifest.json").write_text(_json.dumps({"sensitivity": "S1", "runs": {}}))
    with pytest.raises(SystemExit):
        r.score_fig2_dir(tmp_path, 4.0, (8.0, 13.0))
    with pytest.raises(SystemExit):
        r.score_fig3_dir(tmp_path, 4.0, (8.0, 13.0))


def test_p3_reference_arm_matches_kernel_first_spikes():
    import jax
    import jax.numpy as jnp

    from jaxfne.emitters import IzhikevichParams, simulate_eig_izhikevich

    nref = _load_netref()
    rng = np.random.default_rng(0)
    net = nref.build_network(rng, n_e=40, n_i=10)
    ref = nref.simulate(net, 200, np.random.default_rng(1), noise_on=False, drive=8.0)
    ref_spikes = ref["spikes"]
    assert ref_spikes.sum() > 0  # fixture: the driven reference spikes
    assert ref["mean_extra_u_per_spike"] > 0  # fixture: the overshoot adds u
    n = 50
    params = IzhikevichParams(
        a=jnp.asarray(net["a"], dtype=jnp.float32),
        b=jnp.asarray(net["b"], dtype=jnp.float32),
        c=jnp.asarray(net["c"], dtype=jnp.float32),
        d=jnp.asarray(net["d"], dtype=jnp.float32),
        drive=jnp.full((n,), 8.0, dtype=jnp.float32),
        sign=jnp.concatenate([jnp.ones((40,)), -jnp.ones((10,))]),
        W=jnp.asarray(net["S"], dtype=jnp.float32),
        v0=jnp.full((n,), -65.0, dtype=jnp.float32),
        u0=(jnp.asarray(net["b"], dtype=jnp.float32) * -65.0).astype(jnp.float32),
        source_scale=jnp.ones((n,), dtype=jnp.float32),
        labels=("E",) * 40 + ("I",) * 10)
    _, ksp, _ = simulate_eig_izhikevich(params, 400, 0.5, jax.random.PRNGKey(0),
                                        dtype="float32", noise_scale=0.0)
    ksp = np.asarray(ksp)
    assert ksp.sum() > 0  # fixture: the driven kernel spikes
    for cell in range(n):
        rt = np.flatnonzero(ref_spikes[:, cell])
        kt = np.flatnonzero(ksp[:, cell])
        assert len(rt) > 0 and len(kt) > 0  # fixture: every cell spikes on both arms
        # First spikes agree within one kernel half-step (PREREG_P3: the kernel
        # tests the threshold every 0.5 ms, the reference every 1 ms).
        assert abs(float(rt[0]) - (float(kt[0]) + 1.0) * 0.5) <= 0.5


# --- P3 sensitivity S2/S3/S4 (run_izh.py) ----------------------------------------------------------
#
# Spec: PREREG_P3.md "Sensitivity" (S2 line 223; S3 lines 224-225; S4 line 226).
# Each S-run's realized settings are read back from the manifest the run wrote;
# the label makes score refuse. Simulated T <= 0.5 s; no scored run.


def test_p3_sens_builder_matches_scored_at_scored_settings():
    r = _load_run_izh()
    p0, n0, _ = r.build_fig3_network(1)
    p1, n1, _ = r.build_sens_fig3_network(1)
    for name in ("a", "b", "c", "d", "W"):
        assert np.array_equal(np.asarray(getattr(p1, name)), np.asarray(getattr(p0, name)))
    assert np.array_equal(np.asarray(n1), np.asarray(n0))
    ps, _, _ = r.build_sens_fig3_network(2)
    assert not np.array_equal(np.asarray(ps.W), np.asarray(p1.W))  # fixture: seed changes W


def test_p3_sens_s2_float64_short(tmp_path):
    import jax

    r = _load_run_izh()
    was = bool(jax.config.read("jax_enable_x64"))
    try:
        man = r.cmd_sens("S2", tmp_path / "s2", t_s=0.3)
        e = man["runs"]["1"]
        assert man["sensitivity"] == "S2"  # the label keeps it out of score
        assert e["dtype"] == "float64" and e["output_dtype"] == "float64"  # run asserted it
        assert e["dt_ms"] == 0.5 and e["W"] == "2S" and e["T_s"] == 0.3
        assert bool(jax.config.read("jax_enable_x64"))  # set before any array was built
        z = np.load(tmp_path / "s2" / "1.npz")
        assert set(z.files) == {"spike_rows", "spike_cols", "n_steps", "n_cells", "dt_ms"}
        assert int(z["n_steps"]) == e["n_steps"] == 600 and float(z["dt_ms"]) == 0.5
        with pytest.raises(SystemExit):
            r.score_fig3_dir(tmp_path / "s2", 4.0, (8.0, 13.0))
    finally:
        if not was:
            jax.config.update("jax_enable_x64", False)
    with pytest.raises(SystemExit):
        r.cmd_sens("S9", tmp_path / "bad")


def test_p3_sens_s3_settings_short(tmp_path):
    r = _load_run_izh()
    man = r.cmd_sens("S3", tmp_path / "s3", t_s=0.3)
    e = man["runs"]["1"]
    assert man["sensitivity"] == "S3"
    assert e["dt_ms"] == 0.25 and e["W"] == "4S" and e["dtype"] == "float32"
    # The kernel realizes 2 x the paper's sigma at dt 0.25 (its own _noise_coef, D1 scaling).
    import jax.numpy as jnp
    from jaxfne.emitters import _noise_coef

    for ref, want in ((e["noise_ref_E"], 10.0), (e["noise_ref_I"], 4.0)):
        got = float(_noise_coef(jnp.asarray(ref), 0.25, jnp.float32))
        np.testing.assert_allclose(got, want, rtol=1e-6)
    assert e["noise_realized_E"] == pytest.approx(10.0) and e["noise_realized_I"] == pytest.approx(4.0)
    assert e["n_steps"] == 1200
    p0, n0, _ = r.build_fig3_network(1)
    p3, n3, _ = r.build_sens_fig3_network(1, w_multiple=4.0, noise_e=r.S3_NOISE_E,
                                          noise_i=r.S3_NOISE_I)
    assert np.array_equal(np.asarray(p3.W), 2.0 * np.asarray(p0.W))  # 4S is twice scored 2S
    assert np.array_equal(np.asarray(n3), np.asarray(n0))  # same parameter; dt sets the realized value
    with pytest.raises(SystemExit):
        r.score_fig3_dir(tmp_path / "s3", 4.0, (8.0, 13.0))


def test_p3_sens_s4_c7_current_and_report(tmp_path):
    r = _load_run_izh()
    man = r.cmd_sens("S4", tmp_path / "s4")
    assert man["sensitivity"] == "S4"
    e = man["runs"]["C7"]
    assert e["segments"] == [[10.0, 0.0], [100.0, -10.0], [410.0, 0.0]]  # only the step differs
    assert e["event_row"] == 1100 and e["n_steps"] == 5200 and e["kind"] == "release"
    z = np.load(tmp_path / "s4" / "C7.npz")
    assert set(z.files) == {"spike_rows", "v"}  # same format as fig2
    assert e["n_spikes"] == len(z["spike_rows"])
    with pytest.raises(SystemExit):
        r.score_fig2_dir(tmp_path / "s4", 4.0, (8.0, 13.0))
    kd = tmp_path / "k"
    kd.mkdir()
    np.savez(kd / "C7.npz", spike_rows=np.asarray([1100, 1150], dtype=np.int64),
             v=np.zeros(5200))
    rep = r.cmd_sens_report("S4", tmp_path / "s4", kd, tmp_path / "s4.json")
    assert rep["reported_only"] and rep["sensitivity"] == "S4"
    assert set(rep["C7"]) == {"sensitivity", "scored"}


def test_p3_sens_report_scores_both_arms_short(tmp_path):
    import json as _json

    r = _load_run_izh()
    kd, sd = tmp_path / "k", tmp_path / "s3"
    r.cmd_fig3(kd, [1], 0.3)
    kman = _json.loads((kd / "manifest.json").read_text(encoding="utf-8"))
    assert kman["sensitivity"] is None and kman["dt_ms"] == 0.5  # scored path unchanged
    rows, cols, _, _ = r.run_fig3_seed(1, 0.3)  # short scored seed is deterministic
    z = np.load(kd / "1.npz")
    assert np.array_equal(z["spike_rows"], rows) and np.array_equal(z["spike_cols"], cols)
    r.cmd_sens("S3", sd, t_s=0.3)
    rep = r.cmd_sens_report("S3", sd, kd, tmp_path / "rep.json")
    assert rep["reported_only"] and rep["sensitivity"] == "S3" and rep["seed"] == 1
    assert set(rep["arms"]) == {"sensitivity", "scored"}
    assert rep["arm_settings"]["sensitivity"]["W"] == "4S"
    cells, n_steps = r._seed_cells(kd, 1)
    direct = r.score_network_seed(cells, n_steps, 0.5, 4.0, (8.0, 13.0))
    assert rep["arms"]["scored"]["N1"] == direct["N1"]  # the report leaves scored numbers alone
    with pytest.raises(SystemExit):
        r.cmd_sens_report("S2", sd, kd, tmp_path / "bad.json")  # label mismatch

