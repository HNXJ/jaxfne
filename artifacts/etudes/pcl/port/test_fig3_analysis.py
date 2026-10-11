"""Synthetic tests for fig3_analysis.py (P3_fig3.md T1-T4). No scored data is read."""

import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fig3_analysis as fa

W0 = 1.99  # synthetic reversal frequency (Hz)
TWIN = 1.78  # synthetic stimulus window (s)
BASE, AMP = 0.6, 24.0
SIG_O, SIG_S = 1.3, 2.2


def _stars(n):
    i = np.arange(n)
    return i % 8, 9 + ((i * 7) % 5) - 2, i % 10


def _rates(n, kind, rng):
    """Mean spikes per (ori, sf, phase, trial): Gaussian ori/sf tuning (P3 C1).

    kinds: "simple" (phase-modulated), "complex" (phase-flat), "untuned".
    """
    os_, ss_, p0 = _stars(n)
    o = np.arange(8)[:, None]
    d = np.abs(o - os_[None, :])
    go = np.exp(-np.minimum(d, 8 - d) ** 2 / (2 * SIG_O**2))
    f = np.arange(18)[:, None]
    gf = np.exp(-(f - ss_[None, :]) ** 2 / (2 * SIG_S**2))
    if kind == "simple":
        p = np.arange(10)[:, None]
        ph = 0.05 + 0.95 * (0.5 + 0.5 * np.cos(2 * np.pi * (p - p0[None, :]) / 10))
    elif kind == "complex":
        ph = np.ones((10, n))
    elif kind == "untuned":
        return np.full((8, 18, 10, n), 1.2)
    else:
        raise ValueError(kind)
    return BASE + AMP * go[:, None, None, :] * gf[None, :, None, :] * ph[None, None, :, :]


def _counts(rates, rng):
    n = rates.shape[3]
    lam = np.broadcast_to(rates[:, :, :, None, :], (8, 18, 10, 2, n))
    return rng.poisson(lam)


def _locked(n, freq, phi, sigma, rng):
    if n == 0:
        return np.zeros(0)
    k = rng.integers(0, max(1, int(TWIN * freq)), size=n).astype(float)
    return np.clip(k / freq + phi + rng.normal(0.0, sigma, size=n), 0.0, TWIN - 1e-9)


def _times(counts, lock, sigma, seed):
    """Spike-time dict in load_run layout; locked cells fire at ``lock`` Hz at (o*, f*)."""
    rng = np.random.default_rng(seed)
    n = counts.shape[4]
    os_, ss_, _ = _stars(n)
    times = {}
    idx = np.arange(n)
    for o in range(8):
        for f in range(18):
            for p in range(10):
                for s in range(2):
                    m = counts[o, f, p, s]
                    tot = int(m.sum())
                    times[(o, f, p, s)] = (
                        np.repeat(idx, m).astype(np.int32),
                        rng.uniform(0.0, TWIN, size=tot),
                    )
    for i, freq in lock.items():
        phi = rng.uniform(0.0, 1.0 / freq)
        for p in range(10):
            for s in range(2):
                key = (int(os_[i]), int(ss_[i]), p, s)
                pos, t = times[key]
                keep = pos != i
                m = int(counts[key[0], key[1], p, s, i])
                times[key] = (
                    np.concatenate([pos[keep], np.full(m, i, dtype=np.int32)]),
                    np.concatenate([t[keep], _locked(m, freq, phi, sigma, rng)]),
                )
    return times


def _prefs(counts):
    return fa.fit_prefs_all(counts)


def test_cell_sets_geometry():
    sets = fa.cell_sets()
    assert sets["simple"].tolist() == (2560 + np.arange(64)).tolist()
    exp = np.concatenate([(cx * 6 + cy) * 32 + np.arange(32) for cx, cy in ((2, 2), (2, 3), (3, 2), (3, 3))])
    assert sets["complex"].tolist() == exp.tolist()
    assert sets["simple"].max() < 9 * 9 * 64 and sets["complex"].max() < 6 * 6 * 32


def test_t1t2_simple_passes():
    rng = np.random.default_rng(11)
    counts = _counts(_rates(24, "simple", rng), rng)
    res = fa.t1_t2(counts, np.random.default_rng(11), 1000)
    assert res["t1_pass"] and res["t2_pass"], res
    assert res["r_ori"] >= 2.0 and res["r_ori"] > res["r_ori_null99"]
    assert res["r_sf"] >= 2.0 and res["r_sf"] > res["r_sf_null99"]


def test_t1t2_complex_passes():
    rng = np.random.default_rng(12)
    counts = _counts(_rates(32, "complex", rng), rng)
    res = fa.t1_t2(counts, np.random.default_rng(12), 1000)
    assert res["t1_pass"] and res["t2_pass"], res


def test_t1t2_untuned_fails():
    rng = np.random.default_rng(13)
    counts = _counts(_rates(48, "untuned", rng), rng)
    res = fa.t1_t2(counts, np.random.default_rng(13), 1000)
    assert not res["t1_pass"] and not res["t2_pass"], res
    assert res["r_ori"] < res["r_ori_null99"], res  # the null catches it
    assert res["r_sf"] < res["r_sf_null99"], res


def test_t3_simple_and_flat_complex_pass():
    rng = np.random.default_rng(14)
    cs = _counts(_rates(24, "simple", rng), rng)
    cc = _counts(_rates(32, "complex", rng), rng)
    res = fa.t3(cs, cc)
    assert res["t3_simple_pass"], res
    assert res["t3_complex_pass"], res
    assert res["median_pmi_simple"] >= 0.3
    assert res["median_pmi_complex"] <= 0.5 * res["median_pmi_simple"]


def test_t3_modulated_complex_fails():
    rng = np.random.default_rng(15)
    cs = _counts(_rates(24, "simple", rng), rng)
    cc = _counts(_rates(32, "simple", rng), rng)  # phase-modulated "complex"
    res = fa.t3(cs, cc)
    assert res["t3_simple_pass"], res
    assert not res["t3_complex_pass"], res
    assert res["median_pmi_complex"] > 0.5 * res["median_pmi_simple"]


def test_t4_simple_vs_complex_pass():
    rng = np.random.default_rng(16)
    cs = _counts(_rates(24, "simple", rng), rng)
    cc = _counts(_rates(32, "complex", rng), rng)
    ts = _times(cs, {i: W0 for i in range(24)}, 0.05, 161)
    tc = _times(cc, {i: 2 * W0 for i in range(32)}, 0.03, 162)
    res = fa.t4((ts, tc), W0, (_prefs(cs), _prefs(cc)), np.random.default_rng(16), 1000)
    assert res["t4_pass"], res
    assert res["median_L_simple"] < 0.0 < res["median_L_complex"]


def test_t4_complex_at_w_fails():
    rng = np.random.default_rng(17)
    cs = _counts(_rates(24, "simple", rng), rng)
    cc = _counts(_rates(32, "complex", rng), rng)
    ts = _times(cs, {i: W0 for i in range(24)}, 0.05, 171)
    tc = _times(cc, {i: W0 for i in range(32)}, 0.05, 172)  # complex firing at w
    res = fa.t4((ts, tc), W0, (_prefs(cs), _prefs(cc)), np.random.default_rng(17), 1000)
    assert not res["t4_pass"], res
    assert res["median_L_complex"] < 0.0, res


def test_t4_mostly_excluded_set_fails():
    """P3 Inclusion: more than half of a set below the floor fails T4 for that set."""
    rng = np.random.default_rng(18)
    cs = _counts(_rates(24, "simple", rng), rng)
    cc = _counts(_rates(32, "complex", rng), rng)
    cc[..., :20] = 0  # 20 of 32 complex cells silent
    ts = _times(cs, {i: W0 for i in range(24)}, 0.05, 181)
    tc = _times(cc, {i: 2 * W0 for i in range(32)}, 0.03, 182)
    res = fa.t4((ts, tc), W0, (_prefs(cs), _prefs(cc)), np.random.default_rng(18), 200)
    assert res["excluded_frac_complex"] > 0.5, res["excluded_frac_complex"]
    assert not res["t4_pass"]


def _write_input_file(path, seed):
    rng = np.random.default_rng(seed)
    t0 = 48000
    hot = (t0 + (np.arange(4) / W0 * 1e6 + rng.normal(0, 2000, size=4))).astype(np.int64)
    t = np.concatenate([hot, [t0 + 100000, t0 + 900000]])
    x = np.concatenate([np.full(4, 10), [30, 30]]).astype(np.int64)
    y = np.concatenate([np.full(4, 20), [40, 40]]).astype(np.int64)
    p = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
    c = np.zeros(6, dtype=np.int64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, arr_0=t, arr_1=x, arr_2=y, arr_3=p, arr_4=c)


def test_omega_from_input(tmp_path):
    p = tmp_path / "7.npz"
    _write_input_file(p, 18)
    assert abs(fa.omega_from_input(p) - 1.99) <= 0.01


def _write_run_grid(root, rng):
    from concurrent.futures import ThreadPoolExecutor

    sets = fa.cell_sets()
    grids = {}
    for name, n, kind, lock, sig in (("simple", 64, "simple", W0, 0.05),
                                      ("complex", 128, "complex", 2 * W0, 0.03)):
        rates = _rates(n, kind, rng)
        counts = _counts(rates, rng)
        times = _times(counts, {i: lock for i in range(n)}, sig, 1000 + n)
        grids[name] = (sets[name], counts, times)
    sizes = {"simple": 9 * 9 * 64, "complex": 6 * 6 * 32}
    payloads = []
    for oi, ori in enumerate(fa.ORIS):
        for f in range(18):
            for p in range(10):
                for s in range(2):
                    arr = {}
                    for name in ("simple", "complex"):
                        cells, counts, times = grids[name]
                        full = np.zeros(sizes[name], dtype=np.int32)
                        full[cells] = counts[oi, f, p, s]
                        pos, t = times[(oi, f, p, s)]
                        flat = cells[pos]
                        order = np.lexsort((flat, t))
                        arr[f"{name}_counts"] = full
                        arr[f"{name}_times_us"] = (t[order] * 1e6).astype(np.int64)
                        arr[f"{name}_times_cell"] = flat[order].astype(np.int32)
                    d = root / ori / str(f) / str(p)
                    payloads.append((d / f"{s}.npz", arr))
    seen = set()
    for d, _ in payloads:
        if d.parent not in seen:
            seen.add(d.parent)
            d.parent.mkdir(parents=True, exist_ok=True)

    def _write_one(job):
        path, arr = job
        np.savez(path, **arr)

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(_write_one, payloads))
    for f in (0, 9, 17):
        _write_input_file(root.parent / "inputs" / fa.ORIS[0] / str(f) / "0" / "0.npz", 18)
    return grids


def test_score_endtoend(tmp_path):
    run = tmp_path / "run"
    rng = np.random.default_rng(21)
    grids = _write_run_grid(run, rng)
    res = fa.score(run, run.parent / "inputs", seed=21, n_null=1000)
    assert res["all_pass"], res
    for key in ("t1_simple_pass", "t1_complex_pass", "t2_simple_pass", "t2_complex_pass",
                "t3_simple_pass", "t3_complex_pass", "t4_pass"):
        assert res[key], (key, res)
    assert abs(res["omega"] - 1.99) <= 0.05, res
    sets = fa.cell_sets()
    counts_s, _ = fa.load_run(run, sets["simple"], "simple")
    assert counts_s.shape == (8, 18, 10, 2, 64)
    assert np.array_equal(counts_s, grids["simple"][1]), "counts roundtrip"


def test_cli_refuses_runs_path(capsys):
    rc = fa.main(["F:/warehouse/pcl/runs/fig3_cSTDP3", "dummy_in", "--out", "dummy_out.json"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unsigned" in err and "--signed" in err


def _run_mutant(old, new, check):
    """Swap one bytes span of the implementation, show ``check`` failing, restore by sha256."""
    path = Path(fa.__file__)
    raw = path.read_bytes()
    before = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    assert text.count(old) == 1, f"anchor {old!r} found {text.count(old)}x"
    path.write_bytes(text.replace(old, new).encode("utf-8"))
    importlib.reload(fa)
    try:
        check()
    finally:
        path.write_bytes(raw)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before, "restore mismatch"
        importlib.reload(fa)


def test_mutant_inverted_t1_ratio_fails():
    def check():
        rng = np.random.default_rng(11)
        res = fa.t1_t2(_counts(_rates(24, "simple", rng), rng), np.random.default_rng(11), 200)
        assert not res["t1_pass"], f"mutant should fail T1: {res['r_ori']}"

    _run_mutant(
        "r_ori = _ratio(curve_ori[0], curve_ori[N_ORI // 2])",
        "r_ori = _ratio(curve_ori[N_ORI // 2], curve_ori[0])",
        check,
    )


def test_mutant_dropped_guards_passes_untuned():
    def check():
        rng = np.random.default_rng(13)
        res = fa.t1_t2(_counts(_rates(48, "untuned", rng), rng), np.random.default_rng(13), 200)
        assert res["t1_pass"], "mutant without guards should pass untuned cells"

    _run_mutant("    return bool(r >= thr and r > p99)", "    return True", check)


def test_mutant_weak_t3_criterion_passes_modulated():
    def check():
        rng = np.random.default_rng(15)
        res = fa.t3(_counts(_rates(24, "simple", rng), rng), _counts(_rates(32, "simple", rng), rng))
        assert res["t3_complex_pass"], "weakened criterion should pass modulated complex"

    _run_mutant("med_c <= T3_COMPLEX_FRAC * med_s", "med_c >= 0.0", check)


def test_mutant_swapped_frequencies_fails_t4():
    def check():
        rng = np.random.default_rng(16)
        cs = _counts(_rates(24, "simple", rng), rng)
        cc = _counts(_rates(32, "complex", rng), rng)
        ts = _times(cs, {i: W0 for i in range(24)}, 0.05, 161)
        tc = _times(cc, {i: 2 * W0 for i in range(32)}, 0.03, 162)
        res = fa.t4((ts, tc), W0, (_prefs(cs), _prefs(cc)), np.random.default_rng(16), 200)
        assert not res["t4_pass"], f"swapped w/2w should fail T4: {res['median_L_simple'], res['median_L_complex']}"

    _run_mutant("return float(math.log(a_hi / a_lo))", "return float(math.log(a_lo / a_hi))", check)


def test_mutant_wrong_band_misses_omega(tmp_path):
    p = tmp_path / "7.npz"
    _write_input_file(p, 18)

    def check():
        assert fa.omega_from_input(p) > 4.0, "mutant band should miss 1.99 Hz"

    _run_mutant(
        "freqs = np.arange(_OM_FMIN, _OM_FMAX + step / 2, step)",
        "freqs = np.arange(5.0, 7.0 + step / 2, step)",
        check,
    )


def test_mutant_flipped_refusal_code_proceeds():
    def check():
        rc = fa.main(["F:/warehouse/pcl/runs/fig3_cSTDP3", "dummy_in", "--out", "dummy_out.json"])
        assert rc == 0, "mutant refusal flip should let the scored path through"

    _run_mutant("_REFUSAL_CODE = 2", "_REFUSAL_CODE = 0", check)
