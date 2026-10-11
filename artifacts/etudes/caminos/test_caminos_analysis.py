"""Synthetic tests with known answers for ``caminos_analysis`` (no net sim).

Covers PREREG_P3.md R1-R4 estimators: E-rate Welch peak/ratio, PV->SOM
lag, rate recovery, undefined rules, the run-driver signature gate and
pilot smoke, plus one failing mutant per estimator (restored byte-exact).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import caminos_analysis as ca

DT = 1e-4  # s, prereg Runs Euler step
T = 5.0  # s, prereg Runs duration
STEPS = int(round(T / DT))
KEPT_S = T - ca.DISCARD_S
SIZES = {"E": 1000, "PV": 50, "SOM": 50}
F_GAMMA = 35.0


def _mod_counts(seed, rates, sizes=SIZES, depth=0.0, freq=F_GAMMA, delays=None):
    """Poisson counts [steps, 3]; optionally rate-modulated per population.

    ``delays`` maps a population to its modulation delay in seconds
    (SOM delayed = SOM after PV).
    """
    rng = np.random.default_rng(seed)
    delays = delays or {}
    t = np.arange(STEPS) * DT
    counts = np.zeros((STEPS, 3), dtype=np.int64)
    for i, pop in enumerate(("E", "PV", "SOM")):
        lam = np.full(STEPS, rates[pop] * DT)
        if depth:
            d = delays.get(pop, 0.0)
            lam = lam * (1.0 + depth * np.sin(2.0 * np.pi * freq * (t - d)))
        counts[:, i] = rng.poisson(sizes[pop] * lam)
    return counts


def test_score_scopes_r1_r3_to_reference_runs(tmp_path):
    """R1-R3 use p = 0.07, seeds 1-5 only; a seed-0 pilot and the p = 0.02 grid in the same
    directory must not enter them (prereg Runs, R4)."""
    rates = {"E": 2.0, "PV": 12.0, "SOM": 7.0}
    runs = []

    def put(name, counts, seed, p):
        np.save(tmp_path / name, counts.astype(np.int32))
        runs.append({"file": name, "seed": seed, "p": p})

    for s in (1, 2, 3, 4, 5):
        put(f"ref{s}.npy", _mod_counts(100 + s, rates, depth=0.9, delays={"SOM": 0.0085}), s, 0.07)
    put("pilot0.npy", _mod_counts(200, rates), 0, 0.07)  # unmodulated pilot
    for s in (1, 2, 3):
        put(f"low{s}.npy", _mod_counts(300 + s, rates), s, 0.02)  # asynchronous grid point
    (tmp_path / "manifest.json").write_text(json.dumps({"dt": DT, "sizes": SIZES, "runs": runs}))
    res = ca.score(tmp_path)
    assert res["r1"]["n_runs"] == 5
    assert res["r1"]["pass"], res["r1"]
    assert res["r3"]["pass"], res["r3"]
    assert res["r4"]["pass"], res["r4"]


def test_welch_numpy_sinusoid():
    t = np.arange(int(KEPT_S / DT)) * DT
    x = np.sin(2.0 * np.pi * F_GAMMA * t)
    freqs, pxx = ca.welch_numpy(x, 1.0 / DT, int(1.0 / DT), int(0.5 / DT))
    peak = float(freqs[int(np.argmax(pxx))])
    assert abs(peak - F_GAMMA) <= 1.0, peak
    fs, ps = ca.welch_spectrum(x, 1.0 / DT, int(1.0 / DT), int(0.5 / DT))
    assert abs(float(fs[int(np.argmax(ps))]) - F_GAMMA) <= 1.0


def test_modulated_poisson_oscillates():
    counts = _mod_counts(11, {"E": 2.0, "PV": 12.0, "SOM": 7.0}, SIZES, depth=0.9)
    res = ca.analyze_run(counts, DT, SIZES)
    assert abs(res["peak_freq_E_Hz"] - F_GAMMA) <= 1.0, res["peak_freq_E_Hz"]
    assert res["peak_ratio_E"] >= 5.0, res["peak_ratio_E"]
    assert res["oscillates"]


def test_unmodulated_poisson_no_oscillation():
    counts = _mod_counts(11, {"E": 2.0, "PV": 12.0, "SOM": 7.0}, SIZES)
    res = ca.analyze_run(counts, DT, SIZES)
    print("unmodulated ratio=%.2f (expected about 2.6)" % res["peak_ratio_E"])
    assert res["peak_ratio_E"] < 5.0, res["peak_ratio_E"]
    assert not res["oscillates"]


def test_lag_positive():
    counts = _mod_counts(21, {"E": 2.0, "PV": 12.0, "SOM": 7.0}, SIZES,
                         depth=0.9, delays={"SOM": 0.0085})
    lag = ca.pv_som_lag(counts, DT, SIZES)
    assert abs(lag - 8.5) <= 0.5, lag


def test_lag_negative():
    counts = _mod_counts(21, {"E": 2.0, "PV": 12.0, "SOM": 7.0}, SIZES,
                         depth=0.9, delays={"PV": 0.0085})
    lag = ca.pv_som_lag(counts, DT, SIZES)
    assert lag < 0.0, lag
    assert abs(lag + 8.5) <= 0.5, lag


def test_lag_noisy_dataset_passes_true_window():
    sizes = {"E": 1000, "PV": 10, "SOM": 10}
    counts = _mod_counts(21, {"E": 2.0, "PV": 6.0, "SOM": 5.0}, sizes,
                         depth=0.7, delays={"SOM": 0.0085})
    assert abs(ca.pv_som_lag(counts, DT, sizes) - 8.5) <= 0.5


def test_rates_within_3se():
    true = {"E": 2.0, "PV": 12.0, "SOM": 7.0}
    counts = _mod_counts(31, true, SIZES)
    res = ca.analyze_run(counts, DT, SIZES)
    for i, pop in enumerate(("E", "PV", "SOM")):
        n = int(counts[int(round(ca.DISCARD_S / DT)):, i].sum())
        se = np.sqrt(n) / KEPT_S / SIZES[pop]
        assert abs(res["rates_Hz"][pop] - true[pop]) <= 3 * se, (pop, res["rates_Hz"][pop], se)


def test_zero_spike_undefined():
    counts = np.zeros((STEPS, 3), dtype=np.int64)
    res = ca.analyze_run(counts, DT, SIZES)
    assert res["peak_ratio_E"] is None
    assert res["lag_PV_SOM_ms"] is None
    assert res["oscillates"] is False
    assert "ratio" in res["undefined"] and "lag" in res["undefined"]


def test_run_refuses_scored_without_signed(tmp_path, capsys):
    import run_caminos
    with pytest.raises(SystemExit) as ei:
        run_caminos.main(["--seeds", "1", "--p", "0.07", "--T", "5.0",
                          "--out", str(tmp_path / "runs")])
    assert ei.value.code != 0
    out = capsys.readouterr().out
    assert "PREREG_P3.md" in out and "unsigned" in out
    assert list(tmp_path.iterdir()) == []


def test_pilot_writes_counts_and_manifest(tmp_path):
    import run_caminos
    out = tmp_path / "pilot"
    man = run_caminos.main(["--seeds", "0", "--p", "0.07", "--T", "0.05",
                            "--out", str(out), "--pilot"])
    files = sorted(out.glob("*.npy"))
    assert len(files) == 1
    counts = np.load(files[0])
    assert counts.dtype == np.int32
    assert counts.shape == (int(round(0.05 / DT)), 3)
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["git_sha"] == man["git_sha"] and len(saved["git_sha"]) == 40
    assert "jax_version" in saved and "jax_devices" in saved
    assert saved["runs"][0]["wall_s"] >= 0.0


_ANALYSIS = Path(__file__).resolve().parent / "caminos_analysis.py"


def _run_mutant(old: bytes, new: bytes, check):
    """Apply a one-anchor mutant, run ``check`` on it, restore byte-exact."""
    raw = _ANALYSIS.read_bytes()
    sha_before = hashlib.sha256(raw).hexdigest()
    assert raw.count(old) == 1, "mutant anchor not unique"
    try:
        _ANALYSIS.write_bytes(raw.replace(old, new))
        importlib.reload(ca)
        check()
    finally:
        _ANALYSIS.write_bytes(raw)
        importlib.reload(ca)
        sha_after = hashlib.sha256(_ANALYSIS.read_bytes()).hexdigest()
        assert sha_after == sha_before, "restore not byte-exact"


def test_mutant_lag_sign_flipped_fails():
    def check():
        counts = _mod_counts(21, {"E": 2.0, "PV": 12.0, "SOM": 7.0}, SIZES,
                             depth=0.9, delays={"SOM": 0.0085})
        lag = ca.pv_som_lag(counts, DT, SIZES)
        assert lag < 0.0, lag  # sign flipped: acceptance (8.5 +- 0.5) fails
        assert not abs(lag - 8.5) <= 0.5
    _run_mutant(b"lag_ms = lag_steps * dt_ms  # positive: SOM after PV",
                b"lag_ms = -lag_steps * dt_ms  # positive: SOM after PV", check)


def test_mutant_wide_window_fails():
    # Noisier lag data (seed 21): unmutated window gives 8.3 ms; the
    # 25 ms window admits the one-period alias at about -19 ms.
    sizes = {"E": 1000, "PV": 10, "SOM": 10}

    def check():
        counts = _mod_counts(21, {"E": 2.0, "PV": 6.0, "SOM": 5.0}, sizes,
                             depth=0.7, delays={"SOM": 0.0085})
        lag = ca.pv_som_lag(counts, DT, sizes)
        assert not abs(lag - 8.5) <= 0.5, lag  # 25 ms window admits the alias
    _run_mutant(b"LAG_WINDOW_MS = 14.0", b"LAG_WINDOW_MS = 25.0", check)


def test_mutant_low_threshold_fails():
    def check():
        counts = _mod_counts(11, {"E": 2.0, "PV": 12.0, "SOM": 7.0}, SIZES)
        res = ca.analyze_run(counts, DT, SIZES)
        assert res["oscillates"], res["peak_ratio_E"]  # async now "oscillates"
    _run_mutant(b"RATIO_THRESHOLD = 5.0", b"RATIO_THRESHOLD = 2.5", check)
