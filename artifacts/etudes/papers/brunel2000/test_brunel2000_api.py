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
    spec = L.build_spec(n=20, seed=0)
    assert spec is not None, "fixture must build the lane spec"


def test_lif_extras_zero_closed_form():
    """Zeroed PCL extras equal the closed-form Brunel LIF step (PORT_SPEC 3, 6)."""
    jtfne.enable_x64()
    from jaxfne.emitters_lif import lif_step, make_lif_params

    params = make_lif_params(["simple"], {"simple": dict(L._LIF_BRNEL)})
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


@pytest.mark.xfail(strict=True)
def test_hard_refractory_gap():
    """Brunel cells cannot re-fire within taurp=2 ms (gap G1: no such term)."""
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


def test_density_exact_totals():
    """Density sampling realizes exactly round(p*n_pre*n_post) per rule."""
    pre, post, _, _, lab = _edges(_lif_model())
    ne = int((lab == "E").sum())
    ni = int((lab == "PV").sum())
    for src, npre in (("E", ne), ("PV", ni)):
        for tgt in ("E", "PV"):
            npost = ne if tgt == "E" else ni
            got = int(((lab[pre] == src) & (lab[post] == tgt)).sum())
            assert got == round(0.1 * npre * npost), \
                "rule %s->%s must realize its exact density total" % (src, tgt)


@pytest.mark.xfail(strict=True)
def test_exact_indegree_gap():
    """Every cell holds exactly CE/CI inputs (gap G2: binomial indegree)."""
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


@pytest.mark.xfail(strict=True)
def test_external_drive_same_step_gap():
    """External drive arrives delayed by D (gap G3: same-step arrival)."""
    from jaxfne import neuronal_tensor as nt

    lif = dict(L._LIF_BRNEL)
    tensor = nt.NeuronalTensor(
        name="brunel-g3",
        areas=[nt.Area(name="net", layers=[nt.Layer(
            name="L0", n_neurons=1,
            neuron_types=[nt.NeuronType.make("E", fraction=1.0, emitter="lif",
                                             emitter_params=dict(lif),
                                             drive=0.0)])])],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    sim = jtfne.Simulation(
        duration_ms=20.0, dt_ms=DT, seed=0,
        poisson_drive={"rate_hz": 1000.0 / DT, "amplitude": 25.0, "seed": 0},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    spikes = np.asarray(model.simulate(sim).spikes).reshape(-1)
    assert int(spikes.sum()) > 0, "fixture must spike"
    assert int(np.where(spikes)[0][0]) == D_STEPS, "first arrival after D"


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
                           "spectral_peak_excess", "label"}, "scorer keys"
    assert regime["label"] == "silent", "empty raster must read silent"
