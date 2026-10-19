"""Tests for the mejias2016 lane API (P2: PORT_SPEC rows, G1/G5 closed).

Passing tests pin exact port behaviour; the Wilson-Cowan rate pair (G1)
builds and runs through the core ``rate`` route and the Granger/DAI
estimators (G5) ship in ``jaxfne.analysis``.
"""

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
from artifacts.etudes.papers.mejias2016 import mejias2016_api as L

N_SMALL = 10
DT = 0.5
D_STEPS = int(round(1.0 / DT))


def _model(tensor, seed=0, T_s=0.2):
    from jaxfne import neuronal_tensor as nt

    return jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=seed, duration_ms=T_s * 1000.0,
                                        dt_ms=DT, emitter="lif"),
    )


def _edges(model):
    el = model.params["edge_list"]
    rows = model.neuron_table()
    area = np.array([r["area"] for r in rows])
    layer = np.array([r["layer"] for r in rows])
    lab = np.array([r["cell_type"] for r in rows])
    return (np.asarray(el.pre), np.asarray(el.post), np.asarray(el.weight),
            np.asarray(el.delay_steps), area, layer, lab)


def test_build_spec_smoke():
    """The lane spec builds (fixture must build)."""
    spec = L.build_spec(n=20, seed=0)
    assert spec is not None, "fixture must build the lane spec"


def test_wilson_rate_emitter():
    """A Wilson-Cowan rate pair builds and runs (gap G1 closed by the core route)."""
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="mejias2016-g1",
        areas=[nt.Area(name="V1", layers=[nt.Layer(
            name="L23", n_neurons=2,
            neuron_types=[nt.NeuronType.make("E", fraction=0.5, emitter="rate"),
                          nt.NeuronType.make("I", fraction=0.5, emitter="rate")])])],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="rate"))
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=DT, seed=0,
                           runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    signals = model.simulate(sim)
    rates = np.asarray(signals.V_m)
    assert bool(np.isfinite(rates).all()), "rate state must stay finite"
    assert int(np.sum(np.asarray(signals.spikes))) == 0, "rate route emits no spikes"
    assert signals.metadata["emitter_family"] == "rate"


def test_interlaminar_motif_exact():
    """Fig. 3A motif: L23E->L56E and L56E->L23PV exist; nothing else crosses."""
    pre, post, _, _, _, layer, lab = _edges(_model(L.build_spec(n=N_SMALL)))
    is_e = lab == "E"
    xlay = layer[pre] != layer[post]
    assert int((xlay & is_e[pre]).sum()) > 0, "fixture must wire cross-layer E edges"
    # L23E may only target L56E across layers.
    m = xlay & is_e[pre] & (layer[pre] == "L23")
    assert bool(((layer[post][m] == "L56") & is_e[post][m]).all()), \
        "L23E crosses only to L56E"
    # L56E may only target L23PV across layers (net downward inhibition).
    m = xlay & is_e[pre] & (layer[pre] == "L56")
    assert int(m.sum()) > 0, "fixture must wire the downward L56E projection"
    assert bool(((layer[post][m] == "L23") & (lab[post][m] == "PV")).all()), \
        "L56E crosses only to L23PV"
    # No inhibitory source crosses layers.
    assert int((xlay & ~is_e[pre]).sum()) == 0, "no cross-layer inhibition"


def test_ff_fb_motif_exact():
    """FF onto L23E, FB onto all four pops infra-weighted (PORT_SPEC 5-6)."""
    pre, post, weight, _, area, layer, lab = _edges(
        _model(L.build_two_area(n=N_SMALL)))
    xarea = area[pre] != area[post]
    assert int(xarea.sum()) > 0, "fixture must wire cross-area edges"
    # FF: V1 L23E -> V4 L23E only.
    ff = xarea & (area[pre] == "V1")
    assert int(ff.sum()) > 0, "fixture must wire feedforward"
    assert bool(((layer[pre][ff] == "L23") & (lab[pre][ff] == "E")).all()), \
        "FF sources are V1 L23E"
    assert bool(((layer[post][ff] == "L23") & (lab[post][ff] == "E")).all()), \
        "FF targets are V4 L23E"
    # FB: V4 L56E -> all four V1 pops.
    fb = xarea & (area[pre] == "V4")
    assert bool(((layer[pre][fb] == "L56") & (lab[pre][fb] == "E")).all()), \
        "FB sources are V4 L56E"
    targets = {(str(a), str(b)) for a, b in
               zip(layer[post][fb].tolist(), lab[post][fb].tolist())}
    assert targets == {("L23", "E"), ("L23", "PV"), ("L56", "E"), ("L56", "PV")}, \
        "FB must reach all four V1 populations"
    w_infra = np.abs(weight[fb & (layer[post] == "L56")])
    w_supra = np.abs(weight[fb & (layer[post] == "L23")])
    assert float(w_infra.mean() / w_supra.mean()) == pytest.approx(2.0), \
        "FB infra:supra weight ratio is the declared 2:1"


def test_uniform_delay_steps():
    """Every edge realizes D=1.0 ms as exactly 2 steps (PORT_SPEC 10)."""
    _, _, _, delay, _, _, _ = _edges(_model(L.build_spec(n=N_SMALL)))
    assert len(delay) > 0, "fixture must wire edges"
    assert set(delay.tolist()) == {D_STEPS}, "uniform-D delays"


def test_density_exact_totals():
    """Density sampling realizes exactly round(p*n_pre*n_post) per rule."""
    pre, _, _, _, _, _, _ = _edges(_model(L.build_spec(n=N_SMALL)))
    # 8 intra-layer rules (2 layers x E/PV x E/PV) + 2 motif rules, each
    # exact at p=0.5 with ne=8 E and ni=2 PV per layer at n=10.
    ne, ni, p = 8, 2, 0.5
    intra = (round(p * ne * ne) + round(p * ne * ni)
             + round(p * ni * ne) + round(p * ni * ni))
    motif = round(p * ne * ne) + round(p * ne * ni)
    assert int(len(pre)) == 2 * intra + motif, \
        "wired edges must equal the exact per-rule density totals"


def test_dai_granger():
    """Spectral Granger causality / DAI ships in jaxfne.analysis (gap G5 closed)."""
    from jaxfne.analysis import directionality_index, granger_spectral

    rng = np.random.default_rng(0)
    n = 2000
    x = np.zeros(n)
    y = np.zeros(n)
    ex = rng.standard_normal(n)
    ey = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.5 * y[t - 1] + 0.6 * x[t - 1] + ey[t]
    out = granger_spectral(x, y, DT, 2, np.array([10.0, 40.0]))
    assert bool((out["gc_xy"] > out["gc_yx"]).all()), "x -> y must dominate"
    dai = directionality_index(out["gc_xy"], out["gc_yx"])
    assert bool((dai > 0.0).all()), "DAI must read x -> y as positive"


def test_score_keys():
    """The lane scorer returns the analysis estimands (checklist E3)."""
    regime = L.score(np.zeros((100, 10), dtype=bool), DT)
    assert set(regime) == {"rate_hz", "cv_isi", "kappa", "active_fraction",
                           "spectral_peak_excess", "label"}, "scorer keys"
    assert regime["label"] == "silent", "empty raster must read silent"


def test_stand_in_runs_end_to_end():
    """Small S1 module simulates: finite V, spikes present, estimators return."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _model(L.build_spec(n=N_SMALL), T_s=0.2)
        sim = L.build_sim(T_s=0.2, seed=0)
        signals = model.simulate(sim)
    spikes = np.asarray(signals.spikes)
    assert spikes.shape == (sim.n_steps, 20), "raster must cover every neuron"
    assert int(spikes.sum()) > 0, "fixture must spike"
    assert bool(np.isfinite(np.asarray(signals.V_m)).all()), "V must stay finite"
    regime = L.score(spikes, DT)
    assert regime["label"] != "silent", "driven module must not read silent"
    assert np.isfinite(regime["spectral_peak_excess"]), "Welch peak must return"
