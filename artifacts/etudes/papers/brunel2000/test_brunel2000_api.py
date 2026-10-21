"""Tests for the brunel2000 lane API (P2: PORT_SPEC rows, G1-G3 settled).

Passing tests pin exact port behaviour; gap tests fail on the baseline
implementation and carry ``xfail(strict=True)`` so a core change that closes
a gap turns them green loudly.
"""

import math
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jaxfne as jtfne
assert "site-packages" not in jtfne.__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)
from artifacts.etudes.papers.brunel2000 import brunel2000_api as L

N_SMALL = 200
DT = 0.5
D_STEPS = int(round(1.5 / DT))


def _lif_model(n=N_SMALL, seed=0, point="D"):
    from jaxfne import neuronal_tensor as nt

    return jtfne.construct(
        L.build_spec(point=point, n=n, seed=seed),
        nt.RuntimeConfiguration(seed=seed, duration_ms=20.0, dt_ms=DT,
                                emitter="lif"),
    )


def _edges(model):
    el = model.params["edge_list"]
    rows = model.neuron_table()
    lab = np.array([r["cell_type"] for r in rows])
    return np.asarray(el.pre), np.asarray(el.post), np.asarray(el.weight), \
        np.asarray(el.delay_steps), lab


def test_build_spec_smoke():
    """The lane spec builds (fixture must build)."""
    spec = L.build_spec(n=N_SMALL, seed=0)
    assert spec is not None, "fixture must build the lane spec"


def test_lif_extras_zero_closed_form():
    """Zeroed PCL extras equal the closed-form Brunel LIF step (PORT_SPEC 6)."""
    jtfne.enable_x64()
    from jaxfne.emitters_lif import lif_step, make_lif_params

    lif = dict(L._LIF_BRNEL, t_ref_ms=0.0)
    params = make_lif_params(["simple"], {"simple": lif})
    dt, tau, thr, rst, iamp = 0.5, 20.0, 20.0, 10.0, 5.0
    decay = math.exp(-dt / tau)
    n_steps = 6
    v = np.zeros(1)
    a = np.zeros(1)
    tl = np.full(1, -1.0e30)
    spikes, trace = [], []
    import jax.numpy as jnp

    for t in range(n_steps):
        (v, a, tl), spk = lif_step(
            (jnp.asarray(v), jnp.asarray(a), jnp.asarray(tl)),
            jnp.asarray([iamp]), jnp.asarray([0]),
            float(t * dt), params, dt)
        v, a, tl = (np.asarray(x, dtype=np.float64) for x in (v, a, tl))
        spikes.append(bool(spk[0]))
        trace.append(float(v[0]))
    # Closed form: V_k = I (1-decay^k)/(1-decay); first crossing at step 4.
    expect = [iamp * (1.0 - decay ** k) / (1.0 - decay) for k in range(1, 5)]
    assert expect[3] < thr < iamp + expect[3] * decay, "fixture must cross threshold on step 4"
    assert spikes == [False, False, False, False, True, False], \
        "first spike must land on step 4, then reset below threshold"
    assert all(np.isfinite(trace)), "fixture must stay finite"
    for got, want in zip(trace[:4], expect):
        assert want == pytest.approx(got, rel=1e-12), \
            "zeroed kernel must match the closed form"
    assert trace[4] == pytest.approx(rst), "spike must reset to Vr exactly"


def test_hard_refractory():
    """Brunel cells cannot re-fire within taurp=2 ms (PORT_SPEC 3)."""
    from jaxfne import neuronal_tensor as nt

    lif = dict(L._LIF_BRNEL)
    tensor = nt.NeuronalTensor(
        name="brunel-g1",
        areas=[nt.Area(name="net", layers=[nt.Layer(
            name="L0", n_neurons=1,
            neuron_types=[nt.NeuronType.make("E", fraction=1.0, emitter="lif",
                                             emitter_params=dict(lif),
                                             drive=25.0)])])],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=DT, seed=0,
                           runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    spikes = np.asarray(model.simulate(sim).spikes).reshape(-1)
    idx = np.where(spikes)[0]
    assert len(idx) > 1, "fixture must spike repeatedly"
    assert int(np.diff(idx).min()) >= 4, "no re-fire within taurp=2 ms"


def test_fixed_indegree_exact_totals():
    """Fixed-indegree rules realize exactly in_degree*n_post edges per rule."""
    pre, post, _, _, lab = _edges(_lif_model())
    spec = L.build_spec(n=N_SMALL, seed=0)
    want = {}
    for ac in spec.area_connections:
        npost = int((lab == ac.target_neuron_type).sum())
        want[(ac.source_neuron_type, ac.target_neuron_type)] = \
            int(ac.in_degree) * npost
    for src in ("E", "PV"):
        for tgt in ("E", "PV"):
            got = int(((lab[pre] == src) & (lab[post] == tgt)).sum())
            assert got == want[(src, tgt)], \
                "rule %s->%s must realize its exact fixed total" % (src, tgt)
    rules = {(a.source_neuron_type, a.target_neuron_type): a
             for a in spec.area_connections}
    assert all(a.sampling == "fixed_indegree" for a in rules.values()), \
        "lane rules must sample fixed_indegree"
    assert all(a.probability is None for a in rules.values()), \
        "fixed-indegree rules carry no probability"


def test_exact_indegree():
    """Every cell holds exactly CE/CI inputs (PORT_SPEC 8)."""
    pre, post, _, _, lab = _edges(_lif_model())
    ce = round(0.1 * int((lab == "E").sum()))
    ci = round(0.1 * int((lab == "PV").sum()))
    for i in range(len(lab)):
        fan = pre[post == i]
        assert int((lab[fan] == "E").sum()) == ce, "exact CE per cell"
        assert int((lab[fan] == "PV").sum()) == ci, "exact CI per cell"


def test_uniform_delay_steps():
    """Every recurrent edge realizes D=1.5 ms as exactly 3 steps (PORT_SPEC 9)."""
    _, _, _, delay, _ = _edges(_lif_model())
    assert len(delay) > 0, "fixture must wire edges"
    assert set(delay.tolist()) == {D_STEPS}, "uniform-D delays"


def test_external_drive_counts():
    """The lane drive declares Cext trains at nuext each (PORT_SPEC 11)."""
    ne, _ = L._realized_counts(N_SMALL)
    cext = int(round(L.CONNECTIVITY_P * ne))
    for point in ("A", "B", "C", "D"):
        sim = L.build_sim(point=point, n=N_SMALL, seed=0)
        pd = dict(sim.poisson_drive)
        _, eta = L.FIG8_POINTS[point]
        assert int(pd["counts"]) == cext, "lane declares Cext trains per cell"
        assert float(pd["rate_hz"]) == pytest.approx(float(eta) * L.NUTHR_HZ), \
            "per-train rate is nuext"
        assert float(pd["rate_hz"]) * cext == pytest.approx(
            L.external_rate_hz(point=point, n=N_SMALL)), \
            "counts x per-train rate is the paper total"
        assert float(pd["amplitude"]) == pytest.approx(L.J_MV), \
            "each train arrives at amplitude J"
        model = _lif_model(n=N_SMALL, seed=1, point=point)
        signals = model.simulate(L.build_sim(point=point, n=N_SMALL, T_s=0.02,
                                             seed=1))
        assert int(signals.metadata["poisson_drive"]["counts"]) == cext, \
            "declared counts must reach the run receipt"


def test_zero_drive_silent():
    """Declared drive=0.0 holds the network silent with no input (PORT_SPEC 13)."""
    model = _lif_model()
    drv = np.asarray(model.params["emitter"].drive)
    assert set(np.round(drv, 6).tolist()) == {0.0}, "lane drive must be 0.0"
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=DT, seed=0,
                           runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    spikes = np.asarray(model.simulate(sim).spikes)
    assert int(spikes.sum()) == 0, "no-drive network must stay silent"


def test_realized_weights():
    """Realized edges carry J (E) and -gJ (I) in mV (PORT_SPEC 7)."""
    g, _ = L.FIG8_POINTS["D"]
    pre, _, weight, _, lab = _edges(_lif_model(point="D"))
    assert np.allclose(np.abs(weight[lab[pre] == "E"]), 0.1, rtol=1e-6), \
        "E edges must realize J in mV"
    assert np.allclose(np.abs(weight[lab[pre] == "PV"]), g * 0.1, rtol=1e-6), \
        "PV edges must realize gJ in mV"
    assert bool((weight[lab[pre] == "E"] > 0).all()), "E sources excite"
    assert bool((weight[lab[pre] == "PV"] < 0).all()), "PV sources inhibit"


def test_score_keys():
    """The lane scorer returns the analysis estimands (checklist E3)."""
    regime = L.score(np.zeros((100, 10), dtype=bool), DT)
    assert set(regime) == {"rate_hz", "cv_isi", "kappa", "active_fraction",
                           "spectral_peak_excess", "spectral_peak_hz", "label"}, "scorer keys"
    assert regime["label"] == "silent", "empty raster must read silent"


def test_score_drops_burn_in():
    """``burn_in_ms`` removes the transient before every estimand."""
    spikes = np.zeros((200, 10), dtype=bool)
    spikes[:100:5] = True
    assert L.score(spikes, DT)["rate_hz"] > 0, "fixture must fire in the transient"
    assert L.score(spikes, DT, burn_in_ms=100 * DT)["rate_hz"] == 0.0, \
        "a burn-in covering the transient leaves a silent window"


def test_score_freq_res_sets_peak_grid():
    """``freq_res_hz=1`` reads a 22 Hz rhythm at 22 Hz; the default 7.8 Hz grid reads 23.4."""
    rng = np.random.default_rng(5)
    t_s = np.arange(4000) * DT * 1e-3
    p = 0.01 * (1.0 + np.sin(2.0 * np.pi * 22.0 * t_s))
    spikes = rng.random((4000, 400)) < p[:, None]
    assert spikes.sum() > 0, "fixture must spike"
    coarse = L.score(spikes, DT)
    fine = L.score(spikes, DT, freq_res_hz=1.0)
    assert coarse["spectral_peak_hz"] == pytest.approx(23.4375)
    assert fine["spectral_peak_hz"] == pytest.approx(22.0)
    assert fine["label"] == coarse["label"], "labels stay on the default grid"
    with pytest.raises(ValueError, match="freq_res_hz"):
        L.score(spikes, DT, freq_res_hz=0.0)


def test_score_label_grid_fixed_in_time():
    """A 22 Hz rhythm reads oscillatory at dt=0.1 ms as at dt=0.5 ms (128 ms segments)."""
    rng = np.random.default_rng(6)
    labels = {}
    for dt in (0.5, 0.1):
        n_steps = int(round(2000 / dt))
        t_s = np.arange(n_steps) * dt * 1e-3
        p = 0.002 * dt / 0.1 * (1.0 + 0.3 * np.sin(2.0 * np.pi * 22.0 * t_s))
        spikes = rng.random((n_steps, 300)) < p[:, None]
        assert spikes.sum() > 0, "fixture must spike"
        labels[dt] = L.score(spikes, dt)["label"]
    assert labels == {0.5: "oscillatory", 0.1: "oscillatory"}, labels


def test_run_consumes_dt(tmp_path):
    """``run(dt_ms=...)`` sets the step count and the manifest dt (G4 convergence runs)."""
    import json

    out = L.run(0, "dt025", tmp_path, point="D", n=50, T_s=0.02, dt_ms=0.25)
    assert out["steps"] == 80, "20 ms at dt=0.25 is 80 steps"
    entry = json.loads((tmp_path / "manifest.json").read_text())["runs"][0]
    assert entry["dt_ms"] == 0.25 and entry["steps"] == 80


def test_run_consumes_dtype(tmp_path):
    """``run(dtype="float64")`` runs in float64 and records it (prereg P3 float64 check)."""
    import json

    import jax

    prev = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        out = L.run(0, "f64", tmp_path, point="D", n=50, T_s=0.02, dtype="float64")
    finally:
        jax.config.update("jax_enable_x64", prev)
    assert out["steps"] == 40
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["dtype"] == "float64" and m["runs"][0]["dtype"] == "float64"


def test_run_refuses_float64_without_x64(tmp_path):
    """float64 without JAX x64 is refused before any run."""
    import jax

    prev = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(Exception, match="(?i)x64|float64"):
            L.run(0, "f64", tmp_path, point="D", n=50, T_s=0.02, dtype="float64")
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("bad", [-5.0, float("nan")])
def test_score_refuses_bad_burn_in(bad):
    """A negative or NaN burn-in is refused; slicing would score the tail."""
    spikes = np.zeros((200, 10), dtype=bool)
    spikes[:100:5] = True
    with pytest.raises(ValueError, match="burn_in_ms"):
        L.score(spikes, DT, burn_in_ms=bad)
