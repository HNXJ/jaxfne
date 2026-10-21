"""Tests for fig3_v5_t2.py (P3_fig3_v5.md T2). No scored data is read."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fig3_analysis as fa
import fig3_v5_t2 as t2

SF_STAR = 10
DEN = 128
# Chosen function of x, sampled at the bins (dyadic values, peak 1.0 at x=1).
G = {"0.5": 0.25, "0.6": 0.5, "0.7": 0.5, "0.8": 0.75, "0.9": 0.75,
     "1.0": 1.0, "1.1": 0.75, "1.2": 0.5, "1.4": 0.5, "1.7": 0.25, "2.0": 0.25}


def _sf10_profile():
    """Ori/sf count profile fitting ori*=0, sf*=10 with g(i) = G(x(i))."""
    keys = [f"{float(np.round((20 - SF_STAR) / (20 - i), 1)):.1f}" for i in range(18)]
    assert set(keys[:16]) == set(G), keys  # i=16,17 fall outside 0.4..2.0
    prof = np.full((8, 18), 8, dtype=np.int64)
    prof[0, :16] = np.array([int(DEN * G[k]) for k in keys[:16]], dtype=np.int64)
    return prof


def _spread(prof, n):
    """Counts [8,18,10,2,n] constant over phases, trials and cells."""
    return np.broadcast_to(prof[:, :, None, None, None], (8, 18, 10, 2, n)).copy()


def _write_counts_run(root, counts_s, counts_c):
    """Grid files in run_tuning.py layout from counts arrays (empty times)."""
    root = Path(root)
    sets = fa.cell_sets()
    sizes = {"simple": 9 * 9 * 64, "complex": 6 * 6 * 32}

    def _one(job):
        path, arr = job
        np.savez(path, **arr)

    jobs = []
    for oi, ori in enumerate(fa.ORIS):
        for f in range(fa.N_SF):
            for p in range(fa.N_PHASE):
                for s in range(fa.N_TRIAL):
                    arr = {}
                    for name, cells, counts in (("simple", sets["simple"], counts_s),
                                                ("complex", sets["complex"], counts_c)):
                        full = np.zeros(sizes[name], dtype=np.int32)
                        full[cells] = np.asarray(counts[oi, f, p, s], dtype=np.int32)
                        arr[f"{name}_counts"] = full
                        arr[f"{name}_times_us"] = np.zeros(0, dtype=np.int64)
                        arr[f"{name}_times_cell"] = np.zeros(0, dtype=np.int32)
                    d = root / ori / str(f) / str(p)
                    d.mkdir(parents=True, exist_ok=True)
                    jobs.append((d / f"{s}.npz", arr))
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(_one, jobs))


def test_curve_matches_chosen_function_endtoend(tmp_path):
    run = tmp_path / "run"
    _write_counts_run(run, _spread(_sf10_profile(), 64),
                      np.zeros((8, 18, 10, 2, 128), dtype=np.int64))
    res = t2.score(run)
    assert res["phases"] == [0, 2, 4, 6, 8]
    assert res["reference_sha256"] == json.loads(
        t2.reference_path().read_text(encoding="utf-8"))["sha256"]
    s = res["simple"]
    assert s["n_cells"] == 64 and s["n_skipped"] == 0
    assert s["curve"]["1.0"] == {"mean": 1.0, "se": 0.0, "n": 64}
    assert s["curve"]["0.5"]["mean"] == 0.25
    assert s["curve"]["0.5"]["se"] == 0.0
    assert s["curve"]["0.5"]["n"] == 128
    assert s["curve"]["2.0"]["mean"] == 0.25
    for key, g in G.items():
        assert s["curve"][key]["mean"] == g, key
        assert s["curve"][key]["se"] == 0.0, key
    assert res["complex"]["n_skipped"] == 128
    assert res["complex"]["curve"] == {}
    assert len(res["complex_by_column"]) == 4
    for col, (cx, cy) in zip(res["complex_by_column"], fa.COMPLEX_COLS):
        assert col["scored"] is False
        assert col["column"] == [cx, cy]
        assert col["n_cells"] == 32 and col["n_skipped"] == 32


def test_zero_preference_cell_skipped():
    counts = _spread(_sf10_profile(), 2)
    counts = np.concatenate([counts, np.zeros((8, 18, 10, 2, 1), dtype=np.int64)], axis=4)
    res = t2.score_set(counts, "simple", t2.PANEL_SIMPLE, t2.EVEN_PHASES, t2.load_reference())
    assert res["n_cells"] == 3
    assert res["n_skipped"] == 1
    assert res["curve"]["1.0"] == {"mean": 1.0, "se": 0.0, "n": 2}


def test_simple_uses_phase_star_complex_averages():
    """One peaked phase: simple reads phase* only, complex means the five."""
    n = 2
    counts = np.zeros((8, 18, 10, 2, n), dtype=np.int64)
    counts[:, :, :, :, :] = 5
    counts[0, :, :, :, :] = 10
    counts[0, 9, 0, :, :] = 100  # phase 0 peaks at the (ori*, sf*) preference
    counts[0, 9, [2, 4, 6, 8], :, :] = 0
    cs, _ = t2.tuning_curve(counts, "simple", t2.EVEN_PHASES)
    cc, _ = t2.tuning_curve(counts, "complex", t2.EVEN_PHASES)
    k0 = f"{float(np.round(11 / 20, 1)):.1f}"  # x(i=0) at sf*=9
    assert k0 != "1.0"
    assert cs["1.0"]["mean"] == 1.0 and cc["1.0"]["mean"] == 1.0
    assert cs[k0]["mean"] == 10 / 100  # r[ori*, i, phase*] / 100
    assert cc[k0]["mean"] == 10 / 20  # mean over phases / 20
    assert cs[k0]["mean"] != cc[k0]["mean"]


def _fig_curve(ref, panel, transform):
    pts = t2._scored_points(ref, panel)
    return ({f"{k:.1f}": {"mean": transform(k, fm), "se": 0.0, "n": 64}
             for k, fm, _ in pts}, pts)


def test_criterion_exact_figure_passes():
    ref = t2.load_reference()
    curve, pts = _fig_curve(ref, "c_simple", lambda k, fm: fm)
    res = t2.apply_criterion(curve, "c_simple", ref)
    assert res["frac"] == 1.0  # precondition: every point passes
    k0, fm0, fs0 = pts[0]
    assert res["points"][0]["tol"] == t2.EPS + t2.Z * fs0  # se_ours = 0
    assert res["pass_frac"] is True and res["guard"] is True
    assert res["t2_pass"] is True


def test_criterion_flat_at_mean_fails_guard():
    ref = t2.load_reference()
    for panel, expect_frac in (("c_simple", 10 / 16), ("d_complex", 9 / 15)):
        pts = t2._scored_points(ref, panel)
        m = sum(fm for _, fm, _ in pts) / len(pts)
        curve = {f"{k:.1f}": {"mean": m, "se": 0.0, "n": 64} for k, _, _ in pts}
        res = t2.apply_criterion(curve, panel, ref)
        assert res["frac"] == expect_frac, panel  # precondition: prereg 10/16, 9/15
        assert res["sse"] > t2.GUARD * res["sse_flat"], panel
        assert res["guard"] is False, panel
        assert res["t2_pass"] is False, panel


def test_criterion_offset_curve_fails():
    ref = t2.load_reference()
    curve, _ = _fig_curve(ref, "c_simple", lambda k, fm: fm + 0.3)
    res = t2.apply_criterion(curve, "c_simple", ref)
    assert res["frac"] == 1 / 16  # precondition: only the widest band survives
    assert res["pass_frac"] is False
    assert res["t2_pass"] is False


def test_criterion_missing_key_fails_point():
    ref = t2.load_reference()
    curve, _ = _fig_curve(ref, "c_simple", lambda k, fm: fm)
    del curve["0.4"]
    res = t2.apply_criterion(curve, "c_simple", ref)
    pt = [p for p in res["points"] if p["j"] == 0.4][0]
    assert pt["ours"] is None and pt["ok"] is False
    assert res["frac"] == 15 / 16
    assert res["t2_pass"] is True  # panel c still passes with one key missing


def test_cli_refuses_scored_tree_without_signed(capsys, tmp_path):
    # A path string under the scored tree that this test never creates.
    target = "F:/warehouse/pcl/runs/__synthetic_no_such_run__"
    assert not Path(target).exists()
    rc = t2.main([target, "--out", str(tmp_path / "o.json")])
    assert rc == fa._REFUSAL_CODE
    err = capsys.readouterr().err
    assert "unsigned" in err and "--signed" in err


def test_tie_rules_first_index():
    counts = np.full((8, 18, 10, 2, 2), 7, dtype=np.int64)
    o_star, f_star = t2.fit_prefs(counts, t2.EVEN_PHASES)
    assert o_star.tolist() == [0, 0]
    assert f_star.tolist() == [0, 0]
    # Two-step preference is not a flattened global argmax: peaks at
    # (ori 0, sf 5) and (ori 3, sf 2) resolve to sf*=2 first, then ori*=3.
    prof = np.ones((8, 18), dtype=np.int64)
    prof[0, 5] = 10
    prof[3, 2] = 10
    o2, f2 = t2.fit_prefs(_spread(prof, 1), t2.EVEN_PHASES)
    assert (int(o2[0]), int(f2[0])) == (3, 2)


def test_phases_flags():
    assert t2._parse_phases("even") == (0, 2, 4, 6, 8)
    assert t2._parse_phases("all") == tuple(range(10))
    assert t2._parse_phases("0-4") == (0, 1, 2, 3, 4)
