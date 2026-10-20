"""Synthetic tests for fig3_analysis.py (P3_fig3.md v4 T1-T4). No scored data is read."""

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

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


def _drive_counts(n, axis):
    """Identical cells under a stimulus-wide gain (P3 v4 N-pref null case).

    Peaked on one sf ("sf", flat over ori) or one ori ("ori", flat over sf),
    with no cell-specific tuning. Exact integers: every cell fits the same
    preference, so permuting preferences across cells changes nothing.
    """
    if axis == "sf":
        prof = np.round(30 * np.exp(-(np.arange(18) - 9) ** 2 / (2 * 2.2**2))).astype(int)
        gain = np.broadcast_to(prof[None, :], (8, 18))
    elif axis == "ori":
        d = np.minimum(np.abs(np.arange(8) - 3), 8 - np.abs(np.arange(8) - 3))
        prof = np.round(30 * np.exp(-d**2 / (2 * 1.3**2))).astype(int)
        gain = np.broadcast_to(prof[:, None], (8, 18))
    else:
        raise ValueError(axis)
    return np.broadcast_to((1 + gain)[:, :, None, None, None], (8, 18, 10, 2, n)).copy()


def _tuned_exact_counts(n, base=6, amp=3):
    """Cell-specific Gaussian ori/sf tuning, exact integers, phase-flat.

    Peak/flank ~= (base+amp)/base; every cell fits its own preference, so
    both nulls sit near 1 while R is set by base/amp alone.
    """
    os_ = np.arange(n) % 8
    ss_ = 9 + ((np.arange(n) * 7) % 5) - 2
    o = np.arange(8)[:, None]
    d = np.abs(o - os_[None, :])
    go = np.exp(-np.minimum(d, 8 - d) ** 2 / (2 * SIG_O**2))
    f = np.arange(18)[:, None]
    gf = np.exp(-(f - ss_[None, :]) ** 2 / (2 * SIG_S**2))
    bump = np.round(amp * go[:, None, None, :] * gf[None, :, None, :]).astype(int)
    return np.broadcast_to(base + bump[:, :, None, :, :], (8, 18, 10, 2, n)).copy()


def _lowcount_rates(n, kind):
    """About 6 spikes per stimulus at the preference (P3 v4 T3 low-count regime)."""
    if kind == "simple":
        p = np.arange(10)[:, None]
        ph = 0.05 + 0.95 * (0.5 + 0.5 * np.cos(2 * np.pi * p / 10))
        base = np.broadcast_to(0.5 + 11.5 * ph[:, :1], (10, n))
    elif kind == "complex":
        base = np.full((10, n), 3.0)
    else:
        raise ValueError(kind)
    os_ = np.arange(n) % 8
    ss_ = np.full(n, 9)
    o = np.arange(8)[:, None]
    d = np.abs(o - os_[None, :])
    go = np.exp(-np.minimum(d, 8 - d) ** 2 / (2 * SIG_O**2))
    f = np.arange(18)[:, None]
    gf = np.exp(-(f - ss_[None, :]) ** 2 / (2 * SIG_S**2))
    return go[:, None, None, :] * gf[None, :, None, :] * base[None, None, :, :] + 0.05


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
    assert res["r_ori"] >= 2.0 and res["r_ori"] > res["r_ori_null99"] and res["r_ori"] > res["r_ori_npref99"]
    assert res["r_sf"] >= 2.0 and res["r_sf"] > res["r_sf_null99"] and res["r_sf"] > res["r_sf_npref99"]
    assert sum(res["pref_ori_counts"]) == 24 - res["n_excluded"]
    assert sum(res["pref_sf_counts"]) == 24 - res["n_excluded"]
    assert res["n_excluded"] == 0


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
    assert res["r_ori"] < res["r_ori_null99"], res  # the label null catches it
    assert res["r_sf"] < res["r_sf_null99"], res


def test_t2_drive_effect_fails_npref_only():
    """P3 v4 N-pref: a stimulus-wide sf gain passes N-label but not N-pref."""
    counts = _drive_counts(32, "sf")
    res = fa.t1_t2(counts, np.random.default_rng(201), 1000)
    assert res["r_sf"] >= 2.0, res
    assert res["r_sf"] > res["r_sf_null99"], res  # structure in the scored fold
    assert res["r_sf"] <= res["r_sf_npref99"], res  # another cell's pref does as well
    assert not res["t2_pass"], res
    assert res["pref_sf_counts"][9] == 32 - res["n_excluded"], res["pref_sf_counts"]


def test_t1_drive_effect_fails_npref_only():
    """P3 v4 N-pref on orientation: strong R, above N-label, not above N-pref."""
    counts = _drive_counts(32, "ori")
    res = fa.t1_t2(counts, np.random.default_rng(202), 1000)
    assert res["r_ori"] >= 2.0, res
    assert res["r_ori"] > res["r_ori_null99"], res
    assert res["r_ori"] <= res["r_ori_npref99"], res
    assert not res["t1_pass"], res


def test_t1t2_weak_tuning_fails_floor_only():
    """R in [1.2, 2) above both nulls fails on the R >= 2 floor alone."""
    counts = _tuned_exact_counts(32)
    res = fa.t1_t2(counts, np.random.default_rng(203), 1000)
    for key, n99, np99 in (("r_ori", "r_ori_null99", "r_ori_npref99"),
                           ("r_sf", "r_sf_null99", "r_sf_npref99")):
        assert 1.2 <= res[key] < 2.0, (key, res[key])
        assert res[key] > res[n99] and res[key] > res[np99], (key, res)
    assert not res["t1_pass"] and not res["t2_pass"], res


def test_t3_simple_and_flat_complex_pass():
    rng = np.random.default_rng(14)
    cs = _counts(_rates(24, "simple", rng), rng)
    cc = _counts(_rates(32, "complex", rng), rng)
    res = fa.t3(cs, cc, np.random.default_rng(14))
    assert res["t3_simple_pass"], res
    assert res["t3_complex_pass"], res
    assert res["median_X_simple"] >= 0.15
    assert res["median_X_complex"] <= 0.5 * res["median_X_simple"]
    assert res["n_excluded_simple"] == 0 and res["n_excluded_complex"] == 0


def test_t3_modulated_complex_fails():
    rng = np.random.default_rng(15)
    cs = _counts(_rates(24, "simple", rng), rng)
    cc = _counts(_rates(32, "simple", rng), rng)  # phase-modulated "complex"
    res = fa.t3(cs, cc, np.random.default_rng(15))
    assert res["t3_simple_pass"], res
    assert not res["t3_complex_pass"], res
    assert res["median_X_complex"] > 0.5 * res["median_X_simple"]


def test_t3_lowcount_flat_complex_passes_only_via_poisson_correction():
    """P3 v4 T3: sampling noise inflates raw PMI at ~6 spikes/stimulus; X removes it."""
    rng = np.random.default_rng(104)
    lam_s = np.broadcast_to(_lowcount_rates(24, "simple")[:, :, :, None, :], (8, 18, 10, 2, 24))
    lam_c = np.broadcast_to(_lowcount_rates(32, "complex")[:, :, :, None, :], (8, 18, 10, 2, 32))
    cs = rng.poisson(lam_s)
    cc = rng.poisson(lam_c)
    res = fa.t3(cs, cc, np.random.default_rng(104))
    assert res["t3_simple_pass"], res
    assert res["t3_complex_pass"], res
    raw_would_pass = res["median_pmi_complex"] <= 0.5 * res["median_pmi_simple"]
    assert not raw_would_pass, res  # without PMI0 the flat set looks modulated


def test_t3_floor_boundary_49_vs_50():
    """P3 v4 Inclusion: 50 spikes at (ori*, sf*) included, 49 excluded."""
    counts_s = np.zeros((8, 18, 10, 2, 2), dtype=np.int64)
    counts_s[2, 9, :8, :, 0] = 3  # 3*8*2 = 48 ...
    counts_s[2, 9, 8, 0, 0] = 2  # ... + 2 = 50
    counts_s[2, 9, :8, :, 1] = 3  # 48 ...
    counts_s[2, 9, 8, 0, 1] = 1  # ... + 1 = 49
    rng = np.random.default_rng(105)
    counts_c = _counts(_rates(8, "complex", rng), rng)
    res = fa.t3(counts_s, counts_c, np.random.default_rng(105))
    assert np.isfinite(res["X_simple"][0]), res
    assert np.isnan(res["X_simple"][1]), res
    assert res["n_excluded_simple"] == 1, res


def test_t3_majority_excluded_simple_fails():
    """P3 Inclusion: more than half of a set below the floor fails that set."""
    rng = np.random.default_rng(106)
    cs = _counts(_rates(24, "simple", rng), rng)
    cs[..., 7:] = 0  # 17 of 24 simple cells silent
    cc = _counts(_rates(32, "complex", rng), rng)
    res = fa.t3(cs, cc, np.random.default_rng(106))
    assert res["n_excluded_simple"] == 17, res
    assert res["excluded_frac_simple"] > 0.5
    assert not res["t3_simple_pass"], res


def test_t4_simple_vs_complex_pass():
    rng = np.random.default_rng(16)
    cs = _counts(_rates(24, "simple", rng), rng)
    cc = _counts(_rates(32, "complex", rng), rng)
    ts = _times(cs, {i: W0 for i in range(24)}, 0.05, 161)
    tc = _times(cc, {i: 2 * W0 for i in range(32)}, 0.03, 162)
    res = fa.t4((ts, tc), W0, (_prefs(cs), _prefs(cc)), np.random.default_rng(16), 1000)
    assert res["t4_pass"], res
    assert res["median_L_simple"] < 0.0 < res["median_L_complex"]
    assert res["omega_bad_files"] == [] and res["omega_mismatches"] == []


def test_t4_complex_at_w_fails():
    rng = np.random.default_rng(17)
    cs = _counts(_rates(24, "simple", rng), rng)
    cc = _counts(_rates(32, "complex", rng), rng)
    ts = _times(cs, {i: W0 for i in range(24)}, 0.05, 171)
    tc = _times(cc, {i: W0 for i in range(32)}, 0.05, 172)  # complex firing at w
    res = fa.t4((ts, tc), W0, (_prefs(cs), _prefs(cc)), np.random.default_rng(17), 1000)
    assert not res["t4_pass"], res
    assert res["median_L_complex"] < 0.0, res


def test_t4_below_floor_cell_excluded():
    """P3 v4 Inclusion: fewer than 10 spikes at (ori*, sf*, phase*) excluded."""
    rng = np.random.default_rng(107)
    cs = _counts(_rates(4, "simple", rng), rng)
    cc = _counts(_rates(4, "complex", rng), rng)
    cs[..., 0] = 0
    cs[2, 9, 4, 0, 0] = 5
    cs[2, 9, 4, 1, 0] = 4  # cell 0: 9 spikes at its only nonzero triple
    ts = _times(cs, {i: W0 for i in range(1, 4)}, 0.05, 1710)
    tc = _times(cc, {i: 2 * W0 for i in range(4)}, 0.03, 1711)
    res = fa.t4((ts, tc), W0, (_prefs(cs), _prefs(cc)), np.random.default_rng(107), 200)
    assert res["n_low_count_simple"] == 1, res
    assert np.isnan(res["L_simple"][0]), res
    assert res["n_excluded_simple"] == 1, res


def test_t4_small_diff_below_null_fails():
    """Signs right but median diff below the permutation p99 fails on the null alone."""
    rng = np.random.default_rng(108)
    cs = _counts(_rates(8, "simple", rng), rng)
    cc = _counts(_rates(8, "complex", rng), rng)
    ts = _times(cs, {i: W0 for i in range(8)}, 0.35, 1810)
    tc = _times(cc, {i: 2 * W0 for i in range(8)}, 0.35, 1811)
    res = fa.t4((ts, tc), W0, (_prefs(cs), _prefs(cc)), np.random.default_rng(108), 1000)
    assert res["median_L_simple"] < 0.0 < res["median_L_complex"], res
    assert res["diff"] <= res["diff_null99"], res
    assert not res["t4_pass"], res


def test_t4_per_file_omega_mismatch_uses_mean():
    """P3 v4 T4: trial omegas differing by > 0.01 Hz are reported, mean is used."""
    rng = np.random.default_rng(109)
    cs = _counts(_rates(2, "simple", rng), rng)
    cc = _counts(_rates(2, "complex", rng), rng)
    ts = _times(cs, {0: W0, 1: W0}, 0.05, 1910)
    tc = _times(cc, {0: 2 * W0, 1: 2 * W0}, 0.03, 1911)
    ps, pc = _prefs(cs), _prefs(cc)
    omegas = {}
    for pref, times in ((ps, ts), (pc, tc)):
        for i in range(2):
            key = (int(pref[0][i]), int(pref[1][i]), int(pref[2][i]))
            omegas[key + (0,)] = W0
            omegas[key + (1,)] = W0
    key0 = (int(ps[0][0]), int(ps[1][0]), int(ps[2][0]))
    omegas[key0 + (0,)] = 1.98
    omegas[key0 + (1,)] = 2.00
    res = fa.t4((ts, tc), omegas, (ps, pc), np.random.default_rng(109), 200)
    assert [m["key"] for m in res["omega_mismatches"]] == [list(key0)], res
    assert res["used_omega_simple"][0] == 1.99, res
    pos, t = ts[(key0[0], key0[1], key0[2], 0)]
    t0 = np.asarray(t)[pos == 0]
    pos, t = ts[(key0[0], key0[1], key0[2], 1)]
    t1 = np.asarray(t)[pos == 0]
    expect = fa._log_ratio(np.concatenate([t0, t1]), 1.99)
    assert res["L_simple"][0] == expect, res


def test_t4_bad_omega_file_excluded():
    """P3 v4 T4: a file with omega outside 1.9-2.1 Hz is reported, its cell excluded."""
    rng = np.random.default_rng(110)
    cs = _counts(_rates(2, "simple", rng), rng)
    cc = _counts(_rates(2, "complex", rng), rng)
    ts = _times(cs, {0: W0, 1: W0}, 0.05, 1920)
    tc = _times(cc, {0: 2 * W0, 1: 2 * W0}, 0.03, 1921)
    ps, pc = _prefs(cs), _prefs(cc)
    omegas = {}
    for pref in (ps, pc):
        for i in range(2):
            key = (int(pref[0][i]), int(pref[1][i]), int(pref[2][i]))
            omegas[key + (0,)] = W0
            omegas[key + (1,)] = W0
    key0 = (int(ps[0][0]), int(ps[1][0]), int(ps[2][0]))
    omegas[key0 + (0,)] = 2.3
    omegas[key0 + (1,)] = 2.3
    res = fa.t4((ts, tc), omegas, (ps, pc), np.random.default_rng(110), 200)
    assert res["n_bad_omega_simple"] == 1, res
    assert [tuple(k) for k in res["omega_bad_files"]] == [key0 + (0,), key0 + (1,)], res
    assert np.isnan(res["L_simple"][0]) or res["n_excluded_simple"] == 1, res


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


def test_per_file_omega_map_reads_files(tmp_path):
    prefs = (np.array([0, 1]), np.array([9, 9]), np.array([4, 5]))
    for o, f, p, s in ((0, 9, 4, 0), (0, 9, 4, 1), (1, 9, 5, 0), (1, 9, 5, 1)):
        _write_input_file(tmp_path / fa.ORIS[o] / str(f) / str(p) / f"{s}.npz", 18)
    omap = fa._omega_map(tmp_path, prefs, prefs)
    assert len(omap) == 4, omap
    assert all(abs(w - 1.99) <= 0.01 for w in omap.values()), omap


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
    return grids


def test_score_endtoend(tmp_path):
    run = tmp_path / "run"
    rng = np.random.default_rng(21)
    grids = _write_run_grid(run, rng)
    res = fa.score(run, tmp_path / "inputs", seed=21, n_null=500, omegas=W0)
    assert res["all_pass"], res
    for key in ("t1_simple_pass", "t1_complex_pass", "t2_simple_pass", "t2_complex_pass",
                "t3_simple_pass", "t3_complex_pass", "t4_pass"):
        assert res[key], (key, res)
    assert abs(res["omega"] - 1.99) <= 0.05, res
    assert res["omega_bad_files"] == [] and res["omega_mismatches"] == []
    assert res["n_excluded_t12_simple"] == 0 and res["n_excluded_t12_complex"] == 0
    assert res["n_excluded_t3_simple"] == 0 and res["n_excluded_t3_complex"] == 0
    assert res["n_excluded_t4_simple"] == 0 and res["n_excluded_t4_complex"] == 0
    sets = fa.cell_sets()
    counts_s, _ = fa.load_run(run, sets["simple"], "simple")
    assert counts_s.shape == (8, 18, 10, 2, 64)
    assert np.array_equal(counts_s, grids["simple"][1]), "counts roundtrip"


def test_cli_refuses_runs_path(capsys):
    rc = fa.main(["F:/warehouse/pcl/runs/fig3_cSTDP3", "dummy_in", "--out", "dummy_out.json"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unsigned" in err and "--signed" in err


def _drop_bytecode(path):
    """Delete the cached .pyc so ``reload`` compiles the bytes on disk: a same-size rewrite inside
    the cache's mtime resolution otherwise reloads stale code (seen under WSL on /mnt/c: the
    mutant never applied)."""
    Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
    importlib.invalidate_caches()


def _run_mutant(old, new, check):
    """Swap one bytes span of the implementation, show ``check`` failing, restore by sha256."""
    path = Path(fa.__file__)
    raw = path.read_bytes()
    before = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    assert text.count(old) == 1, f"anchor {old!r} found {text.count(old)}x"
    path.write_bytes(text.replace(old, new).encode("utf-8"))
    _drop_bytecode(path)
    importlib.reload(fa)
    try:
        check()
    finally:
        path.write_bytes(raw)
        _drop_bytecode(path)
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

    _run_mutant("    return bool(r >= thr and r > p99_label and r > p99_pref)", "    return True", check)


def test_mutant_no_npref_passes_drive_effect():
    def check():
        res = fa.t1_t2(_drive_counts(32, "sf"), np.random.default_rng(201), 200)
        assert res["t2_pass"], "mutant without N-pref should pass the drive effect"

    _run_mutant("r > p99_label and r > p99_pref", "r > p99_label", check)


def test_mutant_lowered_floor_passes_weak_tuning():
    def check():
        res = fa.t1_t2(_tuned_exact_counts(32), np.random.default_rng(203), 200)
        assert res["t1_pass"] and res["t2_pass"], "mutant floor 1.0 should pass weak tuning"

    _run_mutant("return bool(r >= thr", "return bool(r >= 1.0", check)


def test_mutant_weak_t3_criterion_passes_modulated():
    def check():
        rng = np.random.default_rng(15)
        res = fa.t3(_counts(_rates(24, "simple", rng), rng), _counts(_rates(32, "simple", rng), rng),
                    np.random.default_rng(15), 200)
        assert res["t3_complex_pass"], "weakened criterion should pass modulated complex"

    _run_mutant("med_xc <= T3_COMPLEX_FRAC * med_xs", "med_xc >= 0.0", check)


def test_mutant_no_poisson_baseline_fails_lowcount_complex():
    def check():
        rng = np.random.default_rng(104)
        lam_c = np.broadcast_to(_lowcount_rates(32, "complex")[:, :, :, None, :], (8, 18, 10, 2, 32))
        cc = rng.poisson(lam_c)
        rng = np.random.default_rng(104)
        lam_s = np.broadcast_to(_lowcount_rates(24, "simple")[:, :, :, None, :], (8, 18, 10, 2, 24))
        cs = rng.poisson(lam_s)
        res = fa.t3(cs, cc, np.random.default_rng(104), 300)
        assert not res["t3_complex_pass"], "mutant without PMI0 should fail flat low-count complex"

    _run_mutant("pmi0[i] = float(pmi0_t.mean())", "pmi0[i] = 0.0", check)


def _duplicated_untuned(seed, n=24, lam=5.0):
    """Untuned cells (Poisson lam per phase per trial) whose two trials are identical."""
    one = np.random.default_rng(seed).poisson(lam, size=(8, 18, 10, 1, n))
    return np.concatenate([one, one], axis=3)


def _pooled_x_median(counts, seed):
    """Median X under the v4 draft's pooled null (pooled PMI vs Poisson at the pooled mean)."""
    o, f, _ = fa.fit_prefs_all(counts)
    cell = np.arange(counts.shape[4])
    v = counts[o, f, :, :, cell].sum(axis=2).astype(np.float64)  # [cell, phase]
    rng = np.random.default_rng(seed)
    pmi0 = [fa._pmi_of_rows(rng.poisson(row.mean(), size=(300, 10)).astype(np.float64)).mean()
            for row in v]
    return float(np.median(fa._pmi_of_rows(v) - np.array(pmi0)))


def test_t3_duplicated_trials_untuned_is_not_modulated():
    """P3 v4 T3 (second review): near-identical trials must not inflate X on untuned cells."""
    counts = _duplicated_untuned(106)
    assert _pooled_x_median(counts, 106) > 0.1  # precondition: the pooled null is biased here
    res = fa.t3(counts, counts, np.random.default_rng(106), 300)
    assert abs(res["median_X_simple"]) < 0.05, res["median_X_simple"]
    assert not res["t3_simple_pass"], res


def test_mutant_pooled_trials_inflates_duplicated_untuned():
    counts = _duplicated_untuned(106)

    def check():
        res = fa.t3(counts, counts, np.random.default_rng(106), 300)
        assert res["median_X_simple"] > 0.1, "pooling both trials should inflate X"

    _run_mutant("        per_trial = v[i].T  # [trial, phase]",
                "        per_trial = v[i].sum(axis=1)[None, :]  # mutant: trials pooled", check)


def test_mutant_floor49_includes_boundary_cell():
    counts_s = np.zeros((8, 18, 10, 2, 1), dtype=np.int64)
    counts_s[2, 9, :8, :, 0] = 3
    counts_s[2, 9, 8, 0, 0] = 1  # 49 spikes
    rng = np.random.default_rng(105)
    counts_c = _counts(_rates(8, "complex", rng), rng)

    def check():
        res = fa.t3(counts_s, counts_c, np.random.default_rng(105), 100)
        assert np.isfinite(res["X_simple"][0]), "mutant floor should include the 49-spike cell"

    _run_mutant("included = total >= T3_MIN_SPIKES", "included = total >= T3_MIN_SPIKES - 1", check)


def test_mutant_no_exclusion_cap_passes_empty_simple():
    def check():
        rng = np.random.default_rng(106)
        cs = _counts(_rates(24, "simple", rng), rng)
        cs[..., 7:] = 0
        cc = _counts(_rates(32, "complex", rng), rng)
        res = fa.t3(cs, cc, np.random.default_rng(106), 100)
        assert res["t3_simple_pass"], "mutant without the exclusion cap should pass"

    _run_mutant("excl_s <= MAX_EXCLUDE_FRAC and med_xs >= T3_X_MIN_SIMPLE", "med_xs >= T3_X_MIN_SIMPLE", check)


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


def test_mutant_no_band_includes_bad_omega():
    rng = np.random.default_rng(110)
    cs = _counts(_rates(2, "simple", rng), rng)
    cc = _counts(_rates(2, "complex", rng), rng)
    ts = _times(cs, {0: W0, 1: W0}, 0.05, 1920)
    tc = _times(cc, {0: 2 * W0, 1: 2 * W0}, 0.03, 1921)
    ps, pc = _prefs(cs), _prefs(cc)
    omegas = {}
    for pref in (ps, pc):
        for i in range(2):
            key = (int(pref[0][i]), int(pref[1][i]), int(pref[2][i]))
            omegas[key + (0,)] = W0
            omegas[key + (1,)] = W0
    key0 = (int(ps[0][0]), int(ps[1][0]), int(ps[2][0]))
    omegas[key0 + (0,)] = 2.3
    omegas[key0 + (1,)] = 2.3

    def check():
        res = fa.t4((ts, tc), omegas, (ps, pc), np.random.default_rng(110), 100)
        assert res["n_bad_omega_simple"] == 0, "mutant without the band keeps the 2.3 Hz cell"
        assert res["n_excluded_simple"] == 0, res

    _run_mutant("band_ok[i] = bool(ok0 and ok1)", "band_ok[i] = True", check)


def test_mutant_first_trial_omega_ignores_mean():
    rng = np.random.default_rng(109)
    cs = _counts(_rates(2, "simple", rng), rng)
    cc = _counts(_rates(2, "complex", rng), rng)
    ts = _times(cs, {0: W0, 1: W0}, 0.05, 1910)
    tc = _times(cc, {0: 2 * W0, 1: 2 * W0}, 0.03, 1911)
    ps, pc = _prefs(cs), _prefs(cc)
    omegas = {}
    for pref in (ps, pc):
        for i in range(2):
            key = (int(pref[0][i]), int(pref[1][i]), int(pref[2][i]))
            omegas[key + (0,)] = W0
            omegas[key + (1,)] = W0
    key0 = (int(ps[0][0]), int(ps[1][0]), int(ps[2][0]))
    omegas[key0 + (0,)] = 1.98
    omegas[key0 + (1,)] = 2.00

    def check():
        res = fa.t4((ts, tc), omegas, (ps, pc), np.random.default_rng(109), 100)
        assert res["used_omega_simple"][0] != 1.99, "mutant uses the first trial only"

    _run_mutant("w = (w0 + w1) / 2.0", "w = w0", check)


def test_mutant_no_t4_null_passes_small_diff():
    def check():
        rng = np.random.default_rng(108)
        cs = _counts(_rates(8, "simple", rng), rng)
        cc = _counts(_rates(8, "complex", rng), rng)
        ts = _times(cs, {i: W0 for i in range(8)}, 0.35, 1810)
        tc = _times(cc, {i: 2 * W0 for i in range(8)}, 0.35, 1811)
        res = fa.t4((ts, tc), W0, (_prefs(cs), _prefs(cc)), np.random.default_rng(108), 200)
        assert res["median_L_simple"] < 0.0 < res["median_L_complex"], res
        assert res["t4_pass"], "mutant without the null should pass the small diff"

    _run_mutant("and med_s < 0.0 and 0.0 < med_c and diff > p99", "and med_s < 0.0 and 0.0 < med_c", check)


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


def test_mutant_zero_refusal_code_proceeds_on_scored_path():
    def check():
        rc = fa.main(["F:/warehouse/pcl/runs/fig3_cSTDP3", "dummy_in", "--out", "dummy_out.json"])
        assert rc == 0, "mutant refusal flip should let the scored path through"

    _run_mutant("_REFUSAL_CODE = 2", "_REFUSAL_CODE = 0", check)


def test_default_score_unchanged_by_new_options(tmp_path):
    """New S2/S3 options default to the preregistered behaviour (full dict)."""
    run = tmp_path / "run"
    _write_run_grid(run, np.random.default_rng(21))
    kw = dict(seed=21, n_null=100, omegas=W0)
    res_default = fa.score(run, tmp_path / "inputs", **kw)
    res_explicit = fa.score(run, tmp_path / "inputs", s2_all_phases=False, simple_col=None, **kw)
    assert res_default == res_explicit  # the full dict, keys and values
    assert "simple_col" not in res_default
    assert not any(k.startswith("s2_") for k in res_default)


def _fold_vs_all_counts():
    """One cell peaking at ori 0 on even phases, at ori 4 on odd phases.

    Even-fit o* is 0, all-phase-fit o* is 4, so the folded and S2 R_ori
    differ. Background 1 spike per (phase, trial); bumps of 50 (even) and
    60 (odd) per (phase, trial) at (ori 0 / 4, sf 9).
    """
    counts = np.ones((8, 18, 10, 2, 1), dtype=np.int64)
    counts[0, 9, 0:10:2, :, 0] += 50
    counts[4, 9, 1:10:2, :, 0] += 60
    return counts


def test_s2_all_phases_differs_when_prefs_differ(monkeypatch):
    counts_s = _fold_vs_all_counts()
    counts_c = np.full((8, 18, 10, 2, 2), 3, dtype=np.int64)
    r_fold = fa.t1_t2(counts_s, np.random.default_rng(0), 50)
    r_all = fa.t1_t2(counts_s, np.random.default_rng(0), 50,
                     fit_phases=list(range(10)), score_phases=list(range(10)))
    assert r_fold["o_star"] != r_all["o_star"], (r_fold["o_star"], r_all["o_star"])
    empty = {(o, f, p, s): (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64))
             for o in range(8) for f in range(18) for p in range(10) for s in range(2)}
    monkeypatch.setattr(fa, "_load_grid",
                        lambda run_dir, cs, cc=None: (counts_s, empty, counts_c, empty))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # T4 null on empty pools
        res = fa.score("dummy_run", "dummy_in", seed=0, n_null=50, omegas=W0, s2_all_phases=True)
    assert res["t1_r_ori_simple"] == r_fold["r_ori"], res
    assert res["s2_t1_r_ori_simple"] == r_all["r_ori"], res
    assert res["s2_t1_r_ori_simple"] != res["t1_r_ori_simple"], res


def test_cell_sets_simple_col():
    sets = fa.cell_sets(simple_col=(3, 3))
    assert sets["simple"].tolist() == (((3 * fa.NY + 3) * fa.ND + np.arange(fa.ND)).tolist())
    assert sets["complex"].tolist() == fa.cell_sets()["complex"].tolist()
    with pytest.raises(ValueError):
        fa.cell_sets(simple_col=(9, 0))
