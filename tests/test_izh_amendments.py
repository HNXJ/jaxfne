"""Izhikevich amendment tests: v8.2 (PREREG_P3_v8.md) and the P3b arm and gate (PREREG_P3b.md).

Synthetic and fast: no scored runs, no F: paths. Each test uses a symbol added for its
amendment, so on the baseline implementation it fails.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import pathlib
import types

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_IZH_PATH = REPO_ROOT / "artifacts" / "etudes" / "izhikevich2003" / "run_izh.py"
DT_MS = 0.1  # kernel dt of the Fig. 2 runs; 10 ms = 100 rows
TEST_SHA = "a" * 40
TEST_SHA_8 = "b" * 40


def _load_run_izh() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_izh_v8", RUN_IZH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows_from_gaps(gaps, start=0):
    rows = [start]
    for g in gaps:
        rows.append(rows[-1] + int(g))
    return rows


def test_v8_c8_protocol_amplitude_2_while_v7_stays_5():
    r = _load_run_izh()
    v8 = r.expected_fig2_v8_settings()["C8"]
    assert v8["segments"] == [[10.0, 0.0], [1.0, 2.0], [311.0, 0.0]]
    assert v8["event_row"] == 110 and v8["n_steps"] == 3220
    assert r.V8_C8_AMPLITUDE == 2.0
    v7 = r.FIG2_RUNS["C8"]
    assert v7[1][1][1] == 5.0 and v7[2] == 110 and v7[3] == 3220


def test_v8_c4_relative_pass():
    r = _load_run_izh()
    c4 = _rows_from_gaps([100] + [150] * 13)  # 15 spikes, 50 Hz, ratio 1.5
    c1 = _rows_from_gaps([100, 200, 300])  # ratio 3.0
    c5 = _rows_from_gaps([100, 200, 400])  # ratio 4.0
    got = r.eval_c4_v8(c4, c1, c5, DT_MS, 0, 0, 0)
    assert got["numbers"]["rate_hz"] == pytest.approx(50.0)
    assert got["numbers"]["isi_ratio"] == pytest.approx(1.5)
    assert got["numbers"]["c1_isi_ratio"] == pytest.approx(3.0)
    assert got["numbers"]["c5_isi_ratio"] == pytest.approx(4.0)
    assert got["verdict"] == "pass"


def test_v8_c4_strict_equality_fails():
    r = _load_run_izh()
    c4 = _rows_from_gaps([100] * 7 + [300] * 7)  # 15 spikes, 50 Hz, ratio 3.0
    c1 = _rows_from_gaps([100, 200, 300])  # ratio 3.0
    c5 = _rows_from_gaps([100, 200, 400])  # ratio 4.0
    got = r.eval_c4_v8(c4, c1, c5, DT_MS, 0, 0, 0)
    assert got["numbers"]["isi_ratio"] == pytest.approx(got["numbers"]["c1_isi_ratio"])
    assert got["verdict"] == "fail"  # strictly below is required


def test_v8_c4_none_comparator_fails():
    r = _load_run_izh()
    c4 = _rows_from_gaps([100] + [150] * 13)  # valid: 50 Hz, ratio 1.5
    c1 = [0, 100]  # 2 spikes: no ratio
    c5 = _rows_from_gaps([100, 200, 400])
    got = r.eval_c4_v8(c4, c1, c5, DT_MS, 0, 0, 0)
    assert got["numbers"]["c1_isi_ratio"] is None
    assert got["verdict"] == "fail"


def _write_v7_fig2(r, d, sha, with_c5=True):
    runs = {}
    rows_by_id = {
        "C1": _rows_from_gaps([100, 200, 300], start=100),
        "C4": _rows_from_gaps([100] + [150] * 13, start=100),
        "C5": _rows_from_gaps([100, 200, 400], start=100),
    }
    expected = r.expected_fig2_settings()
    for rid, rows in rows_by_id.items():
        if rid == "C5" and not with_c5:
            continue
        np.savez(d / f"{rid}.npz", spike_rows=np.asarray(rows, dtype=np.int64))
        runs[rid] = dict(expected[rid], file=f"{rid}.npz", n_spikes=len(rows))
    (d / "manifest.json").write_text(
        json.dumps({"code_sha": sha, "sensitivity": None, "runs": runs}))


def _write_v8_fig2(r, d, sha):
    expected = r.expected_fig2_v8_settings()["C8"]
    # Damped 20 Hz oscillation starting at the pulse end (row 110), flat before.
    idx = np.arange(1110) - 110
    t = np.where(idx >= 0, idx, 0) * 1e-4
    osc = 2.0 * np.exp(-t / 0.05) * np.sin(2 * np.pi * 20.0 * t)
    v = -62.5 + np.where(idx >= 0, osc, 0.0)
    np.savez(d / "C8.npz", spike_rows=np.asarray([], dtype=np.int64), v=v)
    runs = {"C8": dict(expected, file="C8.npz", n_spikes=0)}
    (d / "manifest.json").write_text(
        json.dumps({"code_sha": sha, "amendment": "v8.2",
                    "sensitivity": None, "runs": runs}))


def _write_v7_score(d, sha):
    criteria = {rid: {"verdict": "pass", "numbers": {"n": 1}}
                for rid in ("C1", "C2", "C3", "C5", "C6", "C7")}
    score = {"prereg": "PREREG_P3.md signed v7", "run_code_shas": [sha],
             "fig2": {"criteria": criteria},
             "fig3": {"overall": {"N1": "pass", "N2": "fail",
                                  "N3": "pass", "N4": "pass"}}}
    (d / "v7_score.json").write_text(json.dumps(score))
    return d / "v7_score.json"


def test_v8_missing_c5_run_makes_c4_incomplete(tmp_path):
    r = _load_run_izh()
    v7d, v8d = tmp_path / "v7", tmp_path / "v8"
    v7d.mkdir()
    v8d.mkdir()
    _write_v7_fig2(r, v7d, TEST_SHA, with_c5=False)
    _write_v8_fig2(r, v8d, TEST_SHA_8)
    score = _write_v7_score(tmp_path, TEST_SHA)
    rep = r.cmd_v8(score, v7d, v8d, tmp_path / "v8.json")
    assert rep["v7_sha"] == TEST_SHA and rep["v8_sha"] == TEST_SHA_8
    assert rep["fig2"]["criteria"]["C4"]["verdict"] == "incomplete"
    assert rep["fig2"]["criteria"]["C1"] == {"verdict": "pass", "numbers": {"n": 1}}
    assert rep["fig2"]["overall"] == "incomplete"
    assert rep["fig3"]["overall"]["N2"] == "fail"  # Fig. 3 copied unchanged


def test_v8_scores_c4_and_c8_on_synthetic_runs(tmp_path):
    r = _load_run_izh()
    v7d, v8d = tmp_path / "v7", tmp_path / "v8"
    v7d.mkdir()
    v8d.mkdir()
    _write_v7_fig2(r, v7d, TEST_SHA)
    _write_v8_fig2(r, v8d, TEST_SHA_8)
    score = _write_v7_score(tmp_path, TEST_SHA)
    rep = r.cmd_v8(score, v7d, v8d, tmp_path / "v8.json")
    assert rep["fig2"]["criteria"]["C4"]["verdict"] == "pass"
    assert rep["fig2"]["criteria"]["C8"]["verdict"] == "pass"
    assert rep["amendment"] == "v8.2"


def test_v8_one_sha_mismatch_refuses(tmp_path):
    r = _load_run_izh()
    v7d, v8d = tmp_path / "v7", tmp_path / "v8"
    v7d.mkdir()
    v8d.mkdir()
    _write_v7_fig2(r, v7d, TEST_SHA)
    _write_v8_fig2(r, v8d, TEST_SHA_8)
    score = _write_v7_score(tmp_path, "c" * 40)  # not the v7 dir's SHA
    with pytest.raises(SystemExit):
        r.cmd_v8(score, v7d, v8d, tmp_path / "v8.json")


def test_v8_missing_v8_sha_refuses(tmp_path):
    r = _load_run_izh()
    v7d, v8d = tmp_path / "v7", tmp_path / "v8"
    v7d.mkdir()
    v8d.mkdir()
    _write_v7_fig2(r, v7d, TEST_SHA)
    _write_v8_fig2(r, v8d, TEST_SHA_8)
    (v8d / "manifest.json").write_text(json.dumps({"sensitivity": None, "runs": {}}))
    score = _write_v7_score(tmp_path, TEST_SHA)
    with pytest.raises(SystemExit):
        r.cmd_v8(score, v7d, v8d, tmp_path / "v8.json")


def test_v8_tree_dirty_refuses_fig2_v8(tmp_path, monkeypatch):
    r = _load_run_izh()
    assert r._v8_tree_dirty(" M artifacts/etudes/izhikevich2003/run_izh.py")
    assert not r._v8_tree_dirty("?? scratch/new-file.txt\n")
    assert not r._v8_tree_dirty("")
    monkeypatch.setattr(r, "_git_status_porcelain",
                        lambda: " M artifacts/etudes/izhikevich2003/run_izh.py")
    with pytest.raises(SystemExit) as exc:
        r.cmd_fig2_v8(tmp_path / "v8")
    assert exc.value.code == 2


def test_v8_failed_git_status_reads_dirty(monkeypatch):
    r = _load_run_izh()
    failed = types.SimpleNamespace(returncode=128, stdout="")
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **k: failed)
    assert r._v8_tree_dirty()

    def _raise(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(r.subprocess, "run", _raise)
    assert r._v8_tree_dirty()


# --- P3b (PREREG_P3b.md signed v4, gate amendment v5) ---

ETUDE = REPO_ROOT / "artifacts" / "etudes" / "izhikevich2003"
NETREF_PATH = ETUDE / "net_reference.py"
KERNEL_PATH = ETUDE / "kernel_net_m.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_netref() -> types.ModuleType:
    return _load(NETREF_PATH, "p3b_net_reference")


def _load_kernel() -> types.ModuleType:
    return _load(KERNEL_PATH, "p3b_kernel_net_m")


class _ExplodingRng:
    def __getattr__(self, name):
        raise AssertionError(f"rng must not be read when thal is given ({name})")


def _small_net(nref, seed=20261013, n_e=8, n_i=2):
    return nref.build_network(np.random.default_rng(seed), n_e=n_e, n_i=n_i)


def _kernel_run(km, net, thal, n, dtype):
    import jax.numpy as jnp

    params = {k: jnp.asarray(net[k], dtype=dtype) for k in ("a", "b", "c", "d")}
    v0 = jnp.full((n,), -65.0, dtype=dtype)
    return km.simulate_net_m(params, jnp.asarray(net["S"], dtype=dtype),
                             jnp.asarray(thal, dtype=dtype), v0,
                             params["b"] * -65.0, return_v=True)


def test_p3b_variant_matches_reference_10cell_200ms():
    import jax

    nref, km = _load_netref(), _load_kernel()
    was = bool(jax.config.read("jax_enable_x64"))
    try:
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        net = _small_net(nref)
        n, steps = 10, 200
        thal = np.full((steps, n), 6.0, dtype=np.float64)
        ref = nref.simulate(net, steps, None, thal=thal, return_v=True)
        ker = _kernel_run(km, net, thal, n, jnp.float64)
        assert int(np.asarray(ref["spikes"]).sum()) > 0, "fixture must spike"
        assert np.array_equal(np.asarray(ker["spikes"]), np.asarray(ref["spikes"]))
        assert np.asarray(ker["spikes"]).dtype == bool
        assert np.asarray(ker["spikes"]).shape == (steps, n)
    finally:
        if not was:
            jax.config.update("jax_enable_x64", False)


def test_p3b_thal_injection_leaves_rng_unread():
    nref = _load_netref()
    net = _small_net(nref)
    n, steps = 10, 60
    thal = np.full((steps, n), 6.0, dtype=np.float64)
    got = nref.simulate(net, steps, _ExplodingRng(), thal=thal)
    null = nref.simulate(net, steps, None, thal=thal)
    assert np.array_equal(got["spikes"], null["spikes"])
    assert int(got["spikes"].sum()) > 0, "fixture must spike"


def test_p3b_default_path_unchanged_vs_baseline():
    # Pins computed from net_reference.py at 9ff85e7f (before thal/return_v), same call.
    nref = _load_netref()
    assert "thal" in inspect.signature(nref.simulate).parameters
    assert "return_v" in inspect.signature(nref.simulate).parameters
    net = _small_net(nref, seed=7, n_e=10, n_i=3)
    new = nref.simulate(net, 60, np.random.default_rng(3), noise_on=True)
    assert set(new) == {"spikes", "extra_u", "mean_extra_u_per_spike"}
    assert int(new["spikes"].sum()) == 6, "fixture must spike"
    digest = hashlib.sha256(np.packbits(new["spikes"]).tobytes()).hexdigest()
    assert digest == "dc076c73b5a2c79509250a1ac3c55ae2ff67cfa3b70716e6d7a335b6971c466f"
    np.testing.assert_allclose(float(new["extra_u"].sum()), 6.471066277787681, rtol=1e-12)
    np.testing.assert_allclose(new["mean_extra_u_per_spike"], 2.1570220925958936, rtol=1e-12)


def test_p3b_return_v_shape_and_first_row():
    nref = _load_netref()
    net = _small_net(nref)
    n, steps = 10, 60
    thal = np.full((steps, n), 6.0, dtype=np.float64)
    got = nref.simulate(net, steps, None, thal=thal, return_v=True)
    assert got["v_test"].shape == (steps, n)
    assert np.array_equal(got["v_test"][0], np.full(n, -65.0))  # v before any update
    assert "v_test" not in nref.simulate(net, steps, None, thal=thal)


def _arm(spikes, extra, v):
    return {"spikes": spikes, "extra_u": extra, "v_test": v,
            "mean_extra_u_per_spike": float(extra[spikes].mean()) if spikes.any() else 0.0}


def _v5_case(r_row, kernel_flip_row, extra_rel=0.0, kernel_extra_spikes=0):
    """10 rows x 4 cells; control v departs by 1e-3 at r_row; kernel flips a spike at kernel_flip_row."""
    steps, n = 10, 4
    sp = np.zeros((steps, n), dtype=bool)
    sp[::2, :] = True  # 20 spikes
    extra = np.where(sp, 2.0, 0.0)
    v = np.zeros((steps, n))
    ctrl_v = v.copy()
    if r_row is not None:
        ctrl_v[r_row:, 0] += 1e-3
    ker_sp = sp.copy()
    if kernel_flip_row is not None:
        ker_sp[kernel_flip_row, 1] = not ker_sp[kernel_flip_row, 1]
    for k in range(kernel_extra_spikes):
        ker_sp[1 + 2 * k, 2] = True
    ker_extra = extra * (1.0 + extra_rel)
    return _arm(sp, extra, v), _arm(ker_sp, ker_extra, v), _arm(sp, extra, ctrl_v)


def test_p3b_v5_bar_ignores_flips_after_r_and_counts_them():
    gate = _load(ETUDE / "gate_p3b.py", "p3b_gate")
    ref, ker, ctrl = _v5_case(r_row=5, kernel_flip_row=7)
    got = gate.v5_step(ref, ker, ctrl)
    assert got["R"] == 5 and got["kernel_first_divergent_row"] == 7
    assert got["rasters_identical_before_R"] and got["verdict"] == "FAIL"  # 21 vs 20: 5 % off
    ref, ker, ctrl = _v5_case(r_row=5, kernel_flip_row=None)
    assert gate.v5_step(ref, ker, ctrl)["verdict"] == "PASS"


def test_p3b_v5_bar_fails_a_flip_before_r_and_extra_u_off():
    gate = _load(ETUDE / "gate_p3b.py", "p3b_gate")
    ref, ker, ctrl = _v5_case(r_row=5, kernel_flip_row=2)
    got = gate.v5_step(ref, ker, ctrl)
    assert not got["rasters_identical_before_R"] and got["verdict"] == "FAIL"
    ref, ker, ctrl = _v5_case(r_row=None, kernel_flip_row=None, extra_rel=1e-5)
    got = gate.v5_step(ref, ker, ctrl)
    assert got["R"] == 10 and not got["extra_u_rtol1e-6_before_R"]
    ref, ker, ctrl = _v5_case(r_row=None, kernel_flip_row=None, extra_rel=1e-7)
    assert gate.v5_step(ref, ker, ctrl)["verdict"] == "PASS"
