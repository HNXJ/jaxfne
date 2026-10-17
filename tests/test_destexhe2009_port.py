"""Port tests for the Destexhe 2009 replication (S1 single cells, wiring, jumps, metrics).

Fail-first: these tests import ``artifacts/etudes/papers/destexhe2009/destexhe_api.py``
and fail until it exists. Single-cell patterns run at the paper's parameters
through the core (``jaxfne.adex_destexhe`` table + ``simulate_adex`` kernel).
"""

from __future__ import annotations

import numpy as np

from artifacts.etudes.papers.destexhe2009 import destexhe_api as D


def _spike_times(spikes, dt_ms=0.1):
    idx = np.flatnonzero(np.asarray(spikes, dtype=bool))
    return idx.astype(float) * dt_ms


def test_single_cell_rs_adapts():
    spikes = D.single_cell("RS", protocol="depol")
    t = _spike_times(spikes)
    assert t.size >= 6, "fixture must spike"
    isis = np.diff(t)
    assert isis[-1] > 1.5 * isis[0], f"RS must adapt: {isis}"


def test_single_cell_fs_regular():
    spikes = D.single_cell("FS", protocol="depol")
    t = _spike_times(spikes)
    assert t.size >= 6, "fixture must spike"
    isis = np.diff(t)
    cv = float(isis.std() / isis.mean())
    assert cv < 0.3, f"FS must fire regularly: cv={cv}"


def test_single_cell_lts_rebounds():
    lts = D.single_cell("LTS", protocol="hyper")
    rs = D.single_cell("RS", protocol="hyper")
    assert _spike_times(lts).size >= 2, "LTS must rebound-burst after release"
    assert _spike_times(lts).size > _spike_times(rs).size, "LTS rebound exceeds RS"


def test_wiring_counts_structure_and_no_autapses():
    edges = D.build_edges(n=2000, seed=1)
    pre, post = edges["pre"], edges["post"]
    assert pre.shape == post.shape
    assert np.all(pre != post), "autapses excluded (gap G10)"
    assert edges["n"] == 2000
    counts = edges["block_counts"]
    # 1600 RS (E) + 400 FS (PV); every block at p = 0.02; autapses excluded.
    assert counts["E", "E"] == 1600 * 1599
    assert counts["E", "PV"] == 1600 * 400
    # Realized density near 0.02 in every block (Bernoulli scatter).
    for key, realized in edges["block_density"].items():
        assert abs(realized - 0.02) < 0.004, (key, realized)
    # Deterministic in the seed, and the seed matters.
    again = D.build_edges(n=2000, seed=1)
    other = D.build_edges(n=2000, seed=2)
    assert np.array_equal(again["pre"], pre)
    assert not np.array_equal(other["pre"], pre)


def test_conductance_jump_scales_with_weight():
    # Subthreshold probe (2/4 nS): the jump scales with weight. At the S2
    # quantum (6 nS) the resting post cell already spikes under the authors'
    # detection (Vpeak = VT = -50 mV, F6), so 6/12 nS no longer scales —
    # see test_s2_quantum_spikes_resting_post below.
    low = D.epsp_peak(weight_nS=2.0)
    high = D.epsp_peak(weight_nS=4.0)
    assert low > 0.5, f"2 nS must depolarize the post cell: {low}"
    ratio = high / low
    assert abs(ratio - 2.0) < 0.1, f"EPSP scales with weight: {ratio}"


def test_s2_quantum_spikes_resting_post():
    """F6 consequence: one 6 nS S2 quantum spikes a resting RSweak cell.

    Under the authors' detection (Vpeak_mV = VT = -50 mV) a single
    ge-sized EPSP from rest crosses threshold; under the old core
    default (VT + 5*Delta = -37.5 mV) it stayed subthreshold. The
    port therefore over-spikes relative to the old default at the
    same conductance drive — the wrong direction to explain S2 death
    by under-drive (CALIBRATION.md step 5).
    """
    import jax.numpy as jnp

    from jaxfne.adex_destexhe import DESTEXHE_CELL_DEFAULTS
    from jaxfne.emitters_adex import AdExParams, simulate_adex

    fields = (
        "C_pF",
        "gL_nS",
        "EL_mV",
        "VT_mV",
        "Delta_mV",
        "Vreset_mV",
        "tauw_ms",
        "a_nS",
        "b_pA",
        "Tref_ms",
        "Vpeak_mV",
    )
    rows = [dict(DESTEXHE_CELL_DEFAULTS["RSweak"])] * 2
    cols = {k: jnp.asarray([r[k] for r in rows], dtype=jnp.float32) for k in fields}
    params = AdExParams(labels=("E", "E"), **cols)
    stim = np.zeros((400, 2), dtype=np.float32)
    stim[50:350, 0] = 400.0
    out = simulate_adex(
        params,
        n_steps=400,
        dt_ms=0.1,
        edge_pre=np.array([0]),
        edge_post=np.array([1]),
        edge_weight_nS=np.array([6.0]),
        edge_receptor=np.array([0]),
        edge_delay_steps=np.array([1]),
        receptor_reversal_mV=(0.0,),
        receptor_tau_ms=(5.0,),
        decay="exp",
        I_ext_pA=stim,
        record_v=False,
    )
    post = np.asarray(out["spikes"])[:, 1]
    assert int(post.sum()) >= 1, "one 6 nS quantum must spike the resting post cell"


def test_cv_isi_known_answers():
    regular = np.zeros((1000, 1), bool)
    regular[::100, 0] = True  # one spike every 10 ms
    assert D.cv_isi(regular, 0.1) < 0.05
    rng = np.random.default_rng(0)
    isi = rng.exponential(20.0, size=200)
    edges = np.cumsum(isi) / 0.1
    poisson = np.zeros((int(edges.max()) + 1, 1), bool)
    poisson[edges.astype(int), 0] = True
    assert abs(D.cv_isi(poisson, 0.1) - 1.0) < 0.2


def test_pairwise_cc_known_answers():
    rng = np.random.default_rng(1)
    base = rng.random((2000, 1)) < 0.02
    synced = np.repeat(base, 4, axis=1)
    assert D.pairwise_cc(synced, 0.1, seed=0) > 0.95
    indep = rng.random((2000, 4)) < 0.02
    assert abs(D.pairwise_cc(indep, 0.1, seed=0)) < 0.2


def _score_raster(tmp_path, spikes, dt_ms=0.1):
    """Score a synthetic raster through the CLI scorer; return its report."""
    import argparse
    import io
    import json
    from contextlib import redirect_stdout

    from artifacts.etudes.papers.destexhe2009 import run_destexhe as R

    with open(tmp_path / "spikes_seed1_t.npz", "wb") as f:
        np.savez(f, spikes=np.asarray(spikes, dtype=bool), dt_ms=np.float64(dt_ms))
    args = argparse.Namespace(seed=1, tag="t", out=str(tmp_path), report=None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        R.cmd_score(args)
    return json.loads(buf.getvalue())


def test_score_alive_reads_last_second_of_time(tmp_path):
    steps, n = 100000, 4  # 10 s free run at dt = 0.1 ms
    early = np.zeros((steps, n), bool)
    early[:5000] = np.arange(5000)[:, None] % n == np.arange(n)
    rep = _score_raster(tmp_path, early)
    assert rep["alive_last_s"] is False, "spikes only in first 0.5 s: dead"
    late = np.zeros((steps, n), bool)
    late[-5000:] = np.arange(5000)[:, None] % n == np.arange(n)
    rep = _score_raster(tmp_path, late)
    assert rep["alive_last_s"] is True, "spikes in last 0.5 s: alive"


def test_run_continuous_writes_free_window_only(tmp_path):
    """Smoke the F2 continuous run at tiny scale through the core (CPU)."""
    import json

    import numpy as np

    out = D.run_continuous(1, "fig6b_cont", str(tmp_path), free_t_s=0.05, n_rs=16, n_fs=4)
    assert out["spikes"].shape == (500, 20), out["spikes"].shape
    assert out["spikes"].dtype == bool
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["runs"][0]["continuous"] is True
    assert manifest["runs"][0]["steps"] == 500
    het = D.run_continuous(1, "fig6b_v0", str(tmp_path), free_t_s=0.05, n_rs=16, n_fs=4, v0_seed=1)
    assert het["spikes"].shape == (500, 20)
    assert np.any(het["spikes"] != out["spikes"]), "V0 must change the run"
    nokick = D.run_continuous(
        1,
        "fig6b_nokick",
        str(tmp_path),
        free_t_s=0.05,
        n_rs=16,
        n_fs=4,
        kick_fraction=0.0,
        v0_seed=1,
    )
    assert nokick["spikes"].shape == (500, 20)
    assert nokick["spikes"].dtype == bool


def test_run_continuous_forwards_probability(tmp_path):
    """The Fig. 9 N-trend needs rescaled p through the continuous runner."""
    import json

    lo = D.run_continuous(
        1, "fig6b_p_lo", str(tmp_path), free_t_s=0.02, n_rs=16, n_fs=4, probability=0.02
    )
    hi = D.run_continuous(
        1, "fig6b_p_hi", str(tmp_path), free_t_s=0.02, n_rs=16, n_fs=4, probability=0.5
    )
    assert hi["n_edges"] > lo["n_edges"], (hi["n_edges"], lo["n_edges"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    probs = {e["file"]: e["probability"] for e in manifest["runs"]}
    assert probs["spikes_seed1_fig6b_p_lo.npz"] == 0.02
    assert probs["spikes_seed1_fig6b_p_hi.npz"] == 0.5
