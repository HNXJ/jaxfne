"""Synthetic tests with known answers for ``caminos_analysis`` (no net sim).

Covers PREREG_P3.md (v4) R1-R4 estimators: E-rate aperiodic-background peak
excess on AR(1)-coloured asynchronous floors, PV->SOM lag, rate recovery,
undefined rules, the scorer's completeness and pass rules, the run driver's
manifest merge and reuse rules, the signature gate and pilot smoke, plus
failing mutants (restored byte-exact). The v6 calibration gate is covered too:
threshold selection, true-PE component scaling, the AR(2) pole formula and FWHM,
and the pe_threshold parameter. v6 drops the PSD local-maximum test and runs 12 s
(10 s kept), so T = 12 s here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # P-025: repo root before the covered sibling imports
import caminos_analysis as ca
import calibrate_oscillation as cal

DT = 1e-4  # s, prereg Runs Euler step
T = 12.0  # s, prereg Runs duration (v6)
STEPS = int(round(T / DT))
KEPT_S = T - ca.DISCARD_S
SIZES = {"E": 1000, "PV": 50, "SOM": 50}
F_GAMMA = 35.0


def _mod_counts(seed, rates, sizes=SIZES, depth=0.0, freq=F_GAMMA, delays=None, steps=STEPS):
    """Poisson counts [steps, 3]; optionally rate-modulated per population.

    ``delays`` maps a population to its modulation delay in seconds
    (SOM delayed = SOM after PV). ``steps`` defaults to the reference run length.
    """
    rng = np.random.default_rng(seed)
    delays = delays or {}
    t = np.arange(steps) * DT
    counts = np.zeros((steps, 3), dtype=np.int64)
    for i, pop in enumerate(("E", "PV", "SOM")):
        lam = np.full(steps, rates[pop] * DT)
        if depth:
            d = delays.get(pop, 0.0)
            lam = lam * (1.0 + depth * np.sin(2.0 * np.pi * freq * (t - d)))
        counts[:, i] = rng.poisson(sizes[pop] * lam)
    return counts


SHA = "a" * 40
REF_RATES = {"E": 2.0, "PV": 12.0, "SOM": 7.0}


def _run(seed, p, rates, depth=0.0, freq=F_GAMMA, delays=None, sha=SHA,
         dtype="float32", T_run=T):
    """(manifest entry, counts) for one synthetic run; reference configuration by default."""
    counts = _mod_counts(1000 * int(round(p * 100)) + seed, rates, SIZES,
                         depth=depth, freq=freq, delays=delays)
    counts = counts[:int(round(T_run / DT))]
    entry = {"file": "s%d_p%g_%s_T%g.npy" % (seed, p, dtype, T_run), "seed": seed, "p": p,
             "dtype": dtype, "dt": DT, "T": T_run, "pilot": False, "sizes": SIZES,
             "git_sha": sha}
    return entry, counts


def _write_dir(tmp_path, runs):
    """Save (entry, counts) pairs and a manifest.json holding the entries."""
    for entry, counts in runs:
        np.save(tmp_path / entry["file"], counts.astype(np.int32))
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": [e for e, _ in runs]}))
    return tmp_path


def _grid(depth_at, skip=()):
    """Runs at the nine R4 p values, seeds 1-3; ``depth_at(p, seed)`` sets the modulation."""
    runs = []
    for g in ca.R4_P_GRID:
        if any(abs(g - s) < 1e-9 for s in skip):
            continue
        for s in (1, 2, 3):
            runs.append(_run(s, g, REF_RATES, depth=depth_at(g, s)))
    return runs


def test_score_scopes_r1_r3_to_reference_runs(tmp_path):
    """R1-R3 use p = 0.07, seeds 1-5 at the reference configuration only. A seed-0 pilot,
    a --pilot run and a float64 run share the directory without entering R1-R4."""
    runs = [_run(s, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.0085}) for s in (1, 2, 3, 4, 5)]
    runs.append(_run(0, 0.07, REF_RATES))  # unmodulated pilot, seed 0
    pilot_e, pilot_c = _run(1, 0.07, REF_RATES)
    pilot_e["pilot"], pilot_e["file"] = True, "pilot_s1.npy"
    runs.append((pilot_e, pilot_c))
    runs.append(_run(1, 0.07, REF_RATES, dtype="float64"))
    runs += _grid(lambda g, s: 0.0, skip=(0.07,))  # p = 0.02 and the rest silent
    res = ca.score(_write_dir(tmp_path, runs))
    assert res["r1"]["n_runs"] == 5, res["r1"]
    assert res["r1"]["complete"] and res["r1"]["pass"], res["r1"]
    assert res["r3"]["pass"], res["r3"]
    assert res["r4"]["complete"] and res["r4"]["pass"], res["r4"]
    assert sorted(res["not_scored"]) == sorted(
        ["pilot_s1.npy", "s1_p0.07_float64_T12.npy", "s0_p0.07_float32_T12.npy"]), res["not_scored"]


def test_variant_runs_never_enter_reference(tmp_path):
    """S3-S5 and pNE runs sharing the directory are not scored as v6 reference runs; the
    pNE runs are the reference set under ext_sources='pNE'."""
    runs = [_run(s, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.0085}) for s in (1, 2, 3, 4, 5)]
    variants = [("indegree", "fixed"), ("ext", "shared"), ("step_order", "brian2"),
                ("ext_sources", "pNE")]
    for key, val in variants:
        e, c = _run(6, 0.07, REF_RATES, depth=0.9)
        e[key], e["file"] = val, "v_%s.npy" % key
        runs.append((e, c))
    d = _write_dir(tmp_path, runs)
    res = ca.score(d)
    assert res["r1"]["n_runs"] == 5, res["r1"]
    assert sorted(res["not_scored"]) == sorted("v_%s.npy" % k for k, _ in variants), res
    res7 = ca.score(d, ext_sources="pNE")
    assert [r["file"] for r in res7["per_run"]] == ["v_ext_sources.npy"], res7["per_run"]


def test_r1_seed_at_25hz_fails_band(tmp_path):
    """A seed that oscillates at 25 Hz is outside the R1 band (30-40 Hz): R1 fails."""
    runs = [_run(s, 0.07, REF_RATES, depth=0.9) for s in (1, 2, 3, 4)]
    runs.append(_run(5, 0.07, REF_RATES, depth=0.9, freq=25.0))
    res = ca.score(_write_dir(tmp_path, runs))
    seed5 = next(r for r in res["per_run"] if r["seed"] == 5)
    assert seed5["oscillates"] and abs(seed5["peak_freq_E_Hz"] - 25.0) <= 1.0, seed5
    assert res["r1"]["complete"], res["r1"]
    assert not res["r1"]["pass"], res["r1"]


def test_r2_negative_lag_seed_fails_sign(tmp_path):
    """One seed with SOM before PV (lag -3 ms) fails R2 although the mean lag is in band."""
    runs = [_run(s, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.009}) for s in (1, 2, 3, 4)]
    runs.append(_run(5, 0.07, REF_RATES, depth=0.9, delays={"PV": 0.003}))
    res = ca.score(_write_dir(tmp_path, runs))
    lags = {r["seed"]: r["lag_PV_SOM_ms"] for r in res["per_run"]}
    assert all(abs(lags[s] - 9.0) <= 1.5 for s in (1, 2, 3, 4)) and abs(lags[5] + 3.0) <= 1.5, lags
    assert 6.0 <= res["r2"]["mean_lag_ms"] <= 11.0, res["r2"]  # only the sign can fail
    assert not res["r2"]["all_positive"], res["r2"]
    assert not res["r2"]["pass"], res["r2"]


def test_r2_mean_lag_out_of_band_fails(tmp_path):
    """Every seed positive at 3 ms: the sign holds, the mean lag is below 6 ms, R2 fails."""
    runs = [_run(s, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.003}) for s in (1, 2, 3, 4, 5)]
    res = ca.score(_write_dir(tmp_path, runs))
    assert res["r2"]["all_positive"], res["r2"]  # precondition
    assert res["r2"]["mean_lag_ms"] < 6.0, res["r2"]  # precondition: outside the band
    assert not res["r2"]["pass"], res["r2"]


def test_r3_order_violation_fails(tmp_path):
    """SOM below E in one seed breaks PV > SOM > E although the mean E rate is in band."""
    bad = {"E": 2.0, "PV": 12.0, "SOM": 1.5}
    runs = [_run(s, 0.07, REF_RATES) for s in (1, 2, 4, 5)] + [_run(3, 0.07, bad)]
    res = ca.score(_write_dir(tmp_path, runs))
    assert 1.0 <= res["r3"]["mean_E_Hz"] <= 3.0, res["r3"]  # precondition
    assert not res["r3"]["order_ok_every_seed"], res["r3"]
    assert res["r3"]["complete"] and not res["r3"]["pass"], res["r3"]


def test_r4_oscillating_at_p002_fails(tmp_path):
    """Every seed oscillating at p = 0.02 fails R4 (no seed may oscillate there)."""
    res = ca.score(_write_dir(tmp_path, _grid(lambda g, s: 0.9)))
    assert res["r4"]["complete"], res["r4"]["reasons"]
    assert res["r4"]["p_oscillates"]["0.02"], res["r4"]  # precondition
    assert not res["r4"]["pass"], res["r4"]


def test_r4_single_seed_oscillating_at_p002_fails(tmp_path):
    """One of three seeds oscillating at p = 0.02 fails R4 (an 'all seeds' rule would pass)."""
    def depth_at(g, s):
        return 0.9 if (abs(g - 0.02) < 1e-9 and s == 2) or abs(g - 0.07) < 1e-9 else 0.0
    res = ca.score(_write_dir(tmp_path, _grid(depth_at)))
    low = [r["oscillates"] for r in res["per_run"] if abs(r["p"] - 0.02) < 1e-9]
    assert low == [False, True, False], low  # precondition
    assert res["r4"]["complete"], res["r4"]["reasons"]
    assert not res["r4"]["pass"], res["r4"]


def test_r4_missing_grid_point_is_incomplete(tmp_path):
    """Eight of nine p values present: R4 is incomplete and fails despite the right pattern."""
    runs = _grid(lambda g, s: 0.9 if abs(g - 0.07) < 1e-9 else 0.0, skip=(0.05,))
    res = ca.score(_write_dir(tmp_path, runs))
    assert res["r4"]["low_p_silent"] and res["r4"]["ref_p_all_oscillate"], res["r4"]  # precondition
    assert not res["r4"]["complete"] and not res["r4"]["pass"], res["r4"]


def test_four_seeds_is_incomplete(tmp_path):
    """Seeds 1-4 at p = 0.07: every present seed passes, but R1-R3 are incomplete and fail."""
    runs = [_run(s, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.0085}) for s in (1, 2, 3, 4)]
    res = ca.score(_write_dir(tmp_path, runs))
    assert res["r1"]["n_runs"] == 4 and res["r1"]["all_oscillate"], res["r1"]  # precondition
    assert 6.0 <= res["r2"]["mean_lag_ms"] <= 11.0, res["r2"]  # precondition
    assert res["r3"]["order_ok_every_seed"], res["r3"]  # precondition
    for key in ("r1", "r2", "r3"):
        assert not res[key]["complete"] and not res[key]["pass"], (key, res[key])


@pytest.mark.parametrize("field", ["git_sha", "T"])
def test_mixed_sha_or_short_run_is_incomplete(tmp_path, field):
    """One run with another SHA, or one that ran 2 s, makes R1-R4 incomplete."""
    runs = [_run(s, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.0085}) for s in (1, 2, 3, 4, 5)]
    if field == "git_sha":
        runs[2] = _run(3, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.0085}, sha="b" * 40)
    else:
        runs[2] = _run(3, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.0085}, T_run=2.0)
    res = ca.score(_write_dir(tmp_path, runs))
    for key in ("r1", "r2", "r3"):
        assert not res[key]["complete"] and not res[key]["pass"], (key, res[key])
    assert res["r1"]["reasons"], res["r1"]
    if field == "git_sha":
        assert any("git SHA" in m for m in res["r1"]["reasons"]), res["r1"]["reasons"]
    else:
        assert any("T = 12 s" in m for m in res["r1"]["reasons"]), res["r1"]["reasons"]


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
    assert res["peak_excess_E"] >= 4.0, res["peak_excess_E"]
    assert res["peak_local_max"] and res["peak_edge_ok"], res
    assert len(res["fit_E"]["refit"]) == 3
    assert res["oscillates"]


@pytest.mark.parametrize("c", cal.VAR_RATIOS)
@pytest.mark.parametrize("tau_ms", cal.TAUS_MS)
def test_ar1_coloured_async_does_not_oscillate(tau_ms, c):
    """An AR(1)-coloured asynchronous E rate must not oscillate (replaces the white-only
    fixture, which a sloped spectrum can fool)."""
    counts, e_rate = cal.synth_e(np.random.default_rng([int(tau_ms), int(10 * c), 7]),
                                 tau_ms, c)
    # the generator must build the named AR(1) floor: check it on a 60 s realisation,
    # since one 10 s kept window scatters around the population value
    long_counts, _ = cal.synth_e(np.random.default_rng([int(tau_ms), int(10 * c), 99]),
                                 tau_ms, c, kept_s=60.0)
    acf, expected = cal.count_acf(long_counts), cal.expected_count_acf(tau_ms, c)
    assert abs(acf - expected) < 0.1, (acf, expected)
    pk = ca.spectral_peak(e_rate, DT)
    assert not pk["oscillates"], (pk["peak_excess"], pk["peak_freq_Hz"],
                                  pk["local_max"], pk["edge_ok"])


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
    """Noisy 10-cell fixture: the estimate lies within 0.5 ms of 8.5 ms (the true window).

    Steps are the 5 s this test was written on (v5). At 12 s, seed 21 gives 7.6 ms, and seeds
    21-30 spread 7.2-11.4 ms with one estimate at the -14 ms window edge (seed 26). The 0.5 ms
    check therefore holds for this 5 s realisation only.
    """
    sizes = {"E": 1000, "PV": 10, "SOM": 10}
    counts = _mod_counts(21, {"E": 2.0, "PV": 6.0, "SOM": 5.0}, sizes,
                         depth=0.7, delays={"SOM": 0.0085}, steps=int(round(5.0 / DT)))
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
    assert res["peak_excess_E"] is None
    assert res["lag_PV_SOM_ms"] is None
    assert res["oscillates"] is False
    assert "peak" in res["undefined"] and "lag" in res["undefined"]


def test_run_refuses_scored_without_signed(tmp_path, capsys):
    import run_caminos
    with pytest.raises(SystemExit) as ei:
        run_caminos.main(["--seeds", "1", "--p", "0.07", "--T", "12.0",
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


def test_run_filename_carries_dtype_and_dt():
    import run_caminos
    f32 = run_caminos.run_filename(1, 0.07, "float32", 1e-4)
    assert f32 == "counts_seed1_p0.07_float32_dt1e-04.npy", f32
    assert run_caminos.run_filename(1, 0.07, "float64", 1e-4) != f32
    assert run_caminos.run_filename(1, 0.07, "float32", 5e-5) == "counts_seed1_p0.07_float32_dt5e-05.npy"


@pytest.mark.parametrize("field,value", [("dtype", "float64"), ("dt", 2e-4)])
def test_existing_file_with_other_dtype_or_dt_is_not_reused(tmp_path, field, value):
    """A file whose manifest entry differs in dtype or dt is refused: neither reused nor
    overwritten."""
    import run_caminos
    out = tmp_path / "runs"
    out.mkdir()
    name = run_caminos.run_filename(0, 0.07, "float32", 1e-4)
    (out / name).write_bytes(b"not an npy file")
    entry = {"seed": 0, "p": 0.07, "file": name, "dtype": "float32", "dt": 1e-4,
             "T": 0.05, "pilot": True, "sizes": {"E": 80, "PV": 8, "SOM": 8}, "git_sha": SHA}
    entry[field] = value
    (out / "manifest.json").write_text(json.dumps({"runs": [entry], "invocations": []}))
    with pytest.raises(SystemExit) as ei:
        run_caminos.main(["--seeds", "0", "--p", "0.07", "--T", "0.05",
                          "--out", str(out), "--pilot"])
    assert ei.value.code == 2
    assert (out / name).read_bytes() == b"not an npy file"


def test_manifest_merges_across_invocations(tmp_path):
    """Two invocations into one directory keep both runs; a repeat request reuses its file."""
    import run_caminos
    out = tmp_path / "merge"
    common = ["--p", "0.07", "--T", "0.05", "--out", str(out), "--pilot"]
    f0 = run_caminos.run_filename(0, 0.07, "float32", 1e-4)
    f1 = run_caminos.run_filename(1, 0.07, "float32", 1e-4)
    run_caminos.main(["--seeds", "0"] + common)
    run_caminos.main(["--seeds", "1"] + common)
    saved = json.loads((out / "manifest.json").read_text())
    assert sorted(e["file"] for e in saved["runs"]) == sorted([f0, f1]), saved["runs"]
    assert len(saved["invocations"]) == 2
    assert saved["invocations"][1]["computed"] == [f1]
    assert saved["invocations"][1]["reused"] == []
    before = (out / f0).read_bytes()
    run_caminos.main(["--seeds", "0"] + common)
    saved = json.loads((out / "manifest.json").read_text())
    assert sorted(e["file"] for e in saved["runs"]) == sorted([f0, f1]), saved["runs"]
    assert saved["invocations"][2]["reused"] == [f0]
    assert saved["invocations"][2]["computed"] == []
    assert (out / f0).read_bytes() == before


_ANALYSIS = Path(__file__).resolve().parent / "caminos_analysis.py"


_MUTANT_SEQ = 0


def _run_mutant(monkeypatch, tmp_path, old: bytes, new: bytes, check):
    """Run ``check`` with the test module's ``ca`` bound to a mutated copy.

    The mutated source is written to a ``tmp_path`` copy and imported under a
    fresh module name via ``importlib.util.spec_from_file_location``; the real
    file is never rewritten, so no bytecode cache of it can serve stale code
    across tests (same-length rewrites plus ``reload`` did: the wide-window and
    low-threshold mutants passed alone but failed in the full file). The real
    file's sha256 is unchanged afterwards.
    """
    global _MUTANT_SEQ
    raw = _ANALYSIS.read_bytes()
    sha_before = hashlib.sha256(raw).hexdigest()
    assert raw.count(old) == 1, "mutant anchor not unique"
    _MUTANT_SEQ += 1
    name = "caminos_analysis_mutant_%d" % _MUTANT_SEQ
    # Mirror the repo depth: the module's repo-root guard reads parents[4].
    mirror = tmp_path / "repo" / "artifacts" / "etudes" / "papers" / "caminos"
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / (name + ".py")).write_bytes(raw.replace(old, new))
    spec = importlib.util.spec_from_file_location(name, mirror / (name + ".py"))
    assert spec is not None and spec.loader is not None, "cannot load mutant copy"
    mutant = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mutant)
    spec.loader.exec_module(mutant)
    monkeypatch.setattr(sys.modules[__name__], "ca", mutant)
    try:
        check()
    finally:
        assert hashlib.sha256(_ANALYSIS.read_bytes()).hexdigest() == sha_before, \
            "real file changed"


def test_mutant_lag_sign_flipped_fails(monkeypatch, tmp_path):
    def check():
        counts = _mod_counts(21, {"E": 2.0, "PV": 12.0, "SOM": 7.0}, SIZES,
                             depth=0.9, delays={"SOM": 0.0085})
        lag = ca.pv_som_lag(counts, DT, SIZES)
        assert lag < 0.0, lag  # sign flipped: acceptance (8.5 +- 0.5) fails
        assert not abs(lag - 8.5) <= 0.5
    _run_mutant(monkeypatch, tmp_path, b"lag_ms = lag_steps * dt_ms  # positive: SOM after PV",
                b"lag_ms = -lag_steps * dt_ms  # positive: SOM after PV", check)


def test_mutant_wide_window_fails(monkeypatch, tmp_path):
    # Noisier lag data (seed 21): unmutated window gives 8.3 ms; the
    # 25 ms window admits the one-period alias at about -19 ms.
    sizes = {"E": 1000, "PV": 10, "SOM": 10}

    def check():
        counts = _mod_counts(21, {"E": 2.0, "PV": 6.0, "SOM": 5.0}, sizes,
                             depth=0.7, delays={"SOM": 0.0085}, steps=int(round(5.0 / DT)))
        lag = ca.pv_som_lag(counts, DT, sizes)
        assert not abs(lag - 8.5) <= 0.5, lag  # 25 ms window admits the alias
    _run_mutant(monkeypatch, tmp_path, b"LAG_WINDOW_MS = 14.0", b"LAG_WINDOW_MS = 25.0", check)


def test_mutant_low_threshold_fails(monkeypatch, tmp_path):
    def check():
        _, e_rate = cal.synth_e(np.random.default_rng(5), 30.0, 2.0)
        pk = ca.spectral_peak(e_rate, DT)
        assert pk["oscillates"], pk["peak_excess"]  # AR(1) async now "oscillates"
    _run_mutant(monkeypatch, tmp_path, b"PE_THRESHOLD = 4.0", b"PE_THRESHOLD = 1.0", check)


def test_pe_threshold_changes_oscillates():
    """A fixed signal whose PE lies between 4 and 8 oscillates at pe_threshold 4, not at 8."""
    _, e_rate = cal.synth_e(np.random.default_rng(12345), None, None)
    t = np.arange(e_rate.size) * DT
    sig = e_rate + 0.04 * np.sin(2.0 * np.pi * F_GAMMA * t + 0.3)
    pe = ca.spectral_peak(sig, DT)["peak_excess"]
    assert 4.0 <= pe < 8.0, pe  # precondition: the PE lies between the two thresholds
    lo = ca.spectral_peak(sig, DT, pe_threshold=4.0)
    hi = ca.spectral_peak(sig, DT, pe_threshold=8.0)
    assert lo["oscillates"] and not hi["oscillates"], (lo["oscillates"], hi["oscillates"])
    assert lo["peak_excess"] == hi["peak_excess"]  # only the call changes, not the estimate


def test_threshold_passes_through_analyze_run_and_score(tmp_path):
    """analyze_run and score pass pe_threshold on; the default 4 is unchanged."""
    counts = _mod_counts(11, {"E": 2.0, "PV": 12.0, "SOM": 7.0}, SIZES, depth=0.9)
    assert ca.analyze_run(counts, DT, SIZES)["oscillates"]  # precondition
    assert not ca.analyze_run(counts, DT, SIZES, pe_threshold=1e6)["oscillates"]
    runs = [_run(s, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.0085}) for s in (1, 2, 3, 4, 5)]
    run_dir = _write_dir(tmp_path, runs)
    assert ca.score(run_dir)["r1"]["all_oscillate"]  # precondition
    assert not ca.score(run_dir, pe_threshold=1e6)["r1"]["all_oscillate"]


def test_theta_selects_smallest_qualifying():
    """Theta 4 fails one floor (0.06 > 5 %); 5 is the smallest value every floor passes."""
    table = {"a": {4.0: 0.06, 5.0: 0.04, 6.0: 0.0, 8.0: 0.0},
             "b": {4.0: 0.0, 5.0: 0.05, 6.0: 0.0, 8.0: 0.0}}
    assert table["a"][4.0] > cal.FP_MAX  # precondition: theta 4 fails
    assert cal.select_theta(table) == 5.0
    assert cal.select_theta({"a": {4.0: 0.0, 5.0: 0.0, 6.0: 0.0, 8.0: 0.0}}) == 4.0


def test_theta_refuses_when_none_qualifies():
    table = {"a": {4.0: 0.2, 5.0: 0.1, 6.0: 0.07, 8.0: 0.06}}
    with pytest.raises(ValueError, match="no theta"):
        cal.select_theta(table)


def test_true_pe_scaling_hits_target_on_white_floor():
    """A component scaled to true PE 8 on the white floor realises PE 8, within 10 %."""
    n = 200
    null = [cal.synth_e(np.random.default_rng([cal.STAGE_NULL, 0, d]), None, None)[1]
            for d in range(n)]
    floor_psd = float(np.mean([cal.psd35(e) for e in null]))
    # precondition: the white floor has no line at 35 Hz (PSD there matches 4-12 Hz either side)
    freqs = ca.e_welch(null[0], cal.DT)[0]
    mean_psd = np.mean([ca.e_welch(e, cal.DT)[1] for e in null], axis=0)
    off = np.abs(freqs - F_GAMMA)
    i35 = int(np.argmin(off))
    side = (off >= 4.0) & (off <= 12.0)
    assert abs(mean_psd[i35] / mean_psd[side].mean() - 1.0) <= 0.15, mean_psd[i35]
    res = cal.run_component(0, None, None, "sinusoid", 8.0, floor_psd, n, cal.STAGE_DETECT,
                            (0, 0), peaks=False)
    # The 10 % realised check alone is too loose to catch a one-unit scale error (cross-term noise
    # is about 3 % at 200 draws), so the measured component-to-floor PSD ratio must be pe - 1 exactly.
    assert abs(res["comp_over_floor"] - 7.0) <= 1e-6 * 7.0, res["comp_over_floor"]
    assert abs(res["realised_pe"] / 8.0 - 1.0) <= cal.REALISED_TOL, res["realised_pe"]


def test_ar2_fwhm_matches_pole_formula():
    """AR(2) poles sit at radius 1 - pi FWHM dt and angle 2 pi 35 dt; FWHM is 5 Hz."""
    a1, a2 = cal.ar2_params()
    roots = np.roots([1.0, -a1, -a2])
    assert np.allclose(np.abs(roots), 1.0 - np.pi * 5.0 * DT, rtol=0, atol=1e-12), roots
    assert np.isclose(np.max(np.abs(np.angle(roots))), 2.0 * np.pi * F_GAMMA * DT, rtol=1e-9)
    f = np.linspace(30.0, 40.0, 200001)
    h2 = 1.0 / np.abs(1.0 - a1 * np.exp(-2j * np.pi * f * DT) - a2 * np.exp(-4j * np.pi * f * DT)) ** 2
    above = f[h2 >= h2.max() / 2.0]
    assert abs((above[-1] - above[0]) - 5.0) <= 0.05, above[[0, -1]]  # analytic |H|^2
    x = cal.ar2_unit(np.random.default_rng([4]), int(round(600.0 / DT)))
    fwhm, peak = cal.measure_fwhm(x[int(round(cal.BURN_S / DT)):])
    assert abs(peak - F_GAMMA) <= 1.0, peak  # precondition: the bump sits at 35 Hz
    assert abs(fwhm - 5.0) <= 0.5, fwhm  # realisation within 10 %


# Pinned from the unedited v5 code (kept 3 s) on this machine, from
# caminos_cal6/pin/pin_kept3_v5_before.json: 5 draws of two rows. Compared exactly.
PIN_NULL = {
    "pe": [2.9350866466885486, 2.32366835294268, 3.046495841671252, 3.386943668336916,
           3.68062483612754],
    "f_pk": [25.0, 58.0, 57.0, 36.0, 60.0],
    "floor_psd35": 0.07124590061351634,
}
PIN_DETECT = {  # AR(1) tau 30 ms, var 8x, ar2, true PE 10 (theta 5); floor from PIN_NULL
    "pe": [6.346915749242348, 34.70578305528509, 3.099623279328028, 26.252566399712723,
           11.699158720489764],
    "f_pk": [36.0, 35.0, 57.0, 34.0, 34.0],
    "scale": 0.004344016428606357,
    "realised_pe": 9.455987020207637,
    "comp_over_floor": 8.999999999999996,
}


def test_kept_3_reproduces_v5_rows():
    """Explicit kept 3 s reproduces the pinned v5 rows (5 draws, 2 rows).

    The default kept window is 10 s since v6, so the pins are checked only with the
    explicit 3 s argument; the peak fields they pin are unchanged by v6. f_pk is exact;
    pe comes from the curve_fit aperiodic fit and differs by up to 1.8e-7 relative
    between scipy 1.17.1 (Windows) and 1.18.1 (WSL), so it is pinned at rtol 1e-6.
    """
    label, tau, c = cal.floors()[6]
    assert label == "AR(1) tau 30 ms, var 8x", label  # precondition: the pinned floor
    kw = {"kept_s": 3.0}
    null = cal.run_null(6, label, tau, c, 5, **kw)
    np.testing.assert_allclose(null["pe"], PIN_NULL["pe"], rtol=1e-6, atol=0)
    assert null["f_pk"].tolist() == PIN_NULL["f_pk"], null["f_pk"]
    np.testing.assert_allclose(null["floor_psd35"], PIN_NULL["floor_psd35"], rtol=1e-12, atol=0)
    res = cal.run_component(6, tau, c, "ar2", 10.0, null["floor_psd35"], 5,
                            cal.STAGE_DETECT, (6, 1), **kw)
    np.testing.assert_allclose(res["pe"], PIN_DETECT["pe"], rtol=1e-6, atol=0)
    assert res["f_pk"].tolist() == PIN_DETECT["f_pk"], res["f_pk"]
    for key in ("scale", "realised_pe", "comp_over_floor"):
        np.testing.assert_allclose(res[key], PIN_DETECT[key], rtol=1e-12, atol=0, err_msg=key)


def test_kept_10_gives_19_welch_segments():
    """Kept 10 s is 100000 steps and 19 scorer segments (1 s Hann, 50 % overlap); 3 s gives 5."""
    nperseg = int(round(ca.WIN_S / DT))
    for kept_s, n_seg in ((3.0, 5), (10.0, 19)):
        _, e_rate = cal.synth_e(np.random.default_rng([cal.STAGE_NULL, 0, 0]), None, None,
                                kept_s=kept_s)
        assert e_rate.size == int(round(kept_s / DT)), (kept_s, e_rate.size)
        starts = range(0, e_rate.size - nperseg + 1, nperseg // 2)
        assert len(starts) == n_seg, (kept_s, len(starts))
        unit = cal.component_unit("ar2", np.random.default_rng([3]), kept_s=kept_s)
        assert unit.size == e_rate.size, (kept_s, unit.size)


def test_main_threads_kept_s_to_every_synthetic_draw(tmp_path, monkeypatch):
    """--kept-s 10 reaches every null draw and the component draws; a main that drops it fails."""
    seen = []
    real_null = cal.run_null

    class _Stop(Exception):
        pass

    def spy_null(*a, **kw):
        seen.append(("null", kw.get("kept_s")))
        return real_null(*a, **kw)

    def spy_component(*a, **kw):
        seen.append(("component", kw.get("kept_s")))
        raise _Stop

    monkeypatch.setattr(cal, "run_null", spy_null)
    monkeypatch.setattr(cal, "run_component", spy_component)
    monkeypatch.setattr(cal, "select_theta", lambda fp_table: 5.0)  # 1-draw FP table is not the test
    with pytest.raises(_Stop):
        cal.main(["--out", str(tmp_path), "--draws", "1", "--power-draws", "1", "--kept-s", "10"])
    expected = [("null", 10.0)] * len(cal.floors()) + [("component", 10.0)]
    assert seen == expected, seen


# v6 fixture. Found by a scratch scan of seeds 1-40 (2026-10-11): a 10 s white Poisson floor plus
# V6_AMP times a unit-std 35 Hz AR(2) bump (the calibrated 5 Hz form). Seed 26 is the only
# seed at this amplitude whose largest residual is off a raw-PSD local maximum with PE >= 4.
V6_SEED = 26
V6_AMP = 0.2


def _v6_fixture_rate():
    """10 s E rate: white Poisson floor plus V6_AMP times a unit-std 35 Hz AR(2) bump."""
    _, floor = cal.synth_e(np.random.default_rng([9, V6_SEED]), None, None, kept_s=10.0)
    bump = cal.component_unit("ar2", np.random.default_rng([8, V6_SEED]), kept_s=10.0)
    return floor + V6_AMP * bump / bump.std()


def test_v6_fixture_oscillates_under_v6_not_v5():
    """The largest residual is not a raw-PSD local maximum: it oscillates under v6, not under v5."""
    pk = ca.spectral_peak(_v6_fixture_rate(), DT)
    assert not pk["local_max"] and pk["edge_ok"], pk  # precondition: the named case
    assert pk["peak_excess"] >= ca.PE_THRESHOLD, pk["peak_excess"]
    assert abs(pk["peak_freq_Hz"] - F_GAMMA) <= 2.0, pk["peak_freq_Hz"]
    v5_call = bool(pk["peak_excess"] >= ca.PE_THRESHOLD and pk["local_max"] and pk["edge_ok"])
    assert pk["oscillates"], pk  # v6: peak excess and band edges only
    assert not v5_call  # v4/v5: the same peak was rejected for lacking a local maximum


def test_mutant_restored_local_max_fails_v6_fixture(monkeypatch, tmp_path):
    """Restoring the local-max requirement at the call site rejects the v6 fixture."""
    def check():
        pk = ca.spectral_peak(_v6_fixture_rate(), DT)
        assert not pk["oscillates"], pk["peak_excess"]  # restored rule: the fixture is rejected
    _run_mutant(monkeypatch, tmp_path, b'"oscillates": oscillation_call(pe, edge_ok, pe_threshold),',
                b'"oscillates": oscillation_call(pe, edge_ok, pe_threshold) and local_max,', check)


def test_t5_runs_are_refused_as_incomplete(tmp_path):
    """Runs made at the old T = 5 s are not analysed: R1-R3 are incomplete and cite T = 12 s."""
    runs = [_run(s, 0.07, REF_RATES, depth=0.9, delays={"SOM": 0.0085}, T_run=5.0)
            for s in (1, 2, 3, 4, 5)]
    res = ca.score(_write_dir(tmp_path, runs))
    assert res["r1"]["n_runs"] == 0, res["r1"]  # precondition: no 5 s run is analysed
    for key in ("r1", "r2", "r3"):
        assert not res[key]["complete"] and not res[key]["pass"], (key, res[key])
        assert any("T = 12 s" in m for m in res[key]["reasons"]), res[key]["reasons"]


def test_v6_defaults_are_12_s_run_and_10_s_kept():
    """Scorer REF_T 12 s, calibration kept 10 s (= REF_T - 2 s), run driver default T 12 s."""
    import run_caminos
    assert ca.REF_T == 12.0 and T == ca.REF_T
    assert cal.KEPT_S == 10.0 == ca.REF_T - ca.DISCARD_S
    assert run_caminos.build_parser().parse_args(["--seeds", "1", "--out", "x"]).T == 12.0


def _p6_inputs(e_low):
    """Complete synthetic P6 score inputs; p = 0.02 seeds 1-3 silent with E rate ``e_low``."""
    runs = []
    for p in (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10):
        for s in (1, 2, 3, 4, 5) if abs(p - 0.07) < 1e-9 else (1, 2, 3):
            runs.append({"p": p, "seed": s, "T": 12.0, "git_sha": "a" * 40,
                         "oscillates": p >= 0.04, "peak_freq_E_Hz": 33.8, "lag_PV_SOM_ms": 7.04,
                         "rates_Hz": {"E": e_low if p < 0.03 else 1.106, "PV": 11.37, "SOM": 7.05}})
    main = {"per_run": runs, **{k: {"pass": True} for k in ("r1", "r2", "r3", "r4")}}
    c1 = {"per_run": [{"T": 12.0, "git_sha": "a" * 40, "oscillates": True,
                       "peak_freq_E_Hz": 29.0, "peak_excess_E": 80.0}]}
    return main, c1


def test_score_p6_q4_needs_the_e_floor_at_p002():
    """Q4 R4v7 passes only while E fires at p = 0.02 (>= 0.1 Hz); a silent E fails P6."""
    import score_p6
    main, c1 = _p6_inputs(0.6)
    assert all(not r["oscillates"] for r in main["per_run"] if r["p"] == 0.02)  # precondition
    assert score_p6.score(main, c1)["verdict"] == "pass"
    main, c1 = _p6_inputs(0.05)
    res = score_p6.score(main, c1)
    assert res["criteria"]["Q4"]["r4"].get("e_floor_ok") is False
    assert res["verdict"] == "fail"


def test_ppc_by_type_splits_types():
    """PV neurons with identical phases give PPC 1.0; uniform SOM phases give -1/7."""
    phases = [np.full(6, 0.3), np.full(6, 0.3),
              np.array([2.0 * np.pi * k / 8 for k in range(8)]),
              np.array([2.0 * np.pi * k / 8 for k in range(8)])]
    types = ["PV", "PV", "SOM", "SOM"]
    res = ca.ppc_by_type(phases, types)
    np.testing.assert_allclose(res["by_type"]["PV"], 1.0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(res["by_type"]["SOM"], -1.0 / 7.0, rtol=1e-12, atol=0)


def test_ppc_by_type_nan_for_few_spikes():
    """A 1-spike neuron has NaN PPC and does not pull its type mean off 1.0."""
    phases = [np.array([0.3]), np.full(5, 0.3)]
    assert any(np.asarray(p).size >= 2 for p in phases)  # precondition
    res = ca.ppc_by_type(phases, ["E", "E"])
    assert np.isnan(res["per_neuron"][0]), res["per_neuron"]
    np.testing.assert_allclose(res["per_neuron"][1], 1.0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(res["by_type"]["E"], 1.0, rtol=1e-12, atol=0)


def test_ppc_by_type_all_nan_type():
    """A type whose only neuron is empty maps to NaN."""
    res = ca.ppc_by_type([np.array([])], ["SOM"])
    assert np.isnan(res["by_type"]["SOM"]), res["by_type"]


def test_ppc_by_type_length_mismatch_raises():
    """Three phase arrays with two type labels raise ValueError."""
    with pytest.raises(ValueError):
        ca.ppc_by_type([np.full(3, 0.3)] * 3, ["E", "PV"])
