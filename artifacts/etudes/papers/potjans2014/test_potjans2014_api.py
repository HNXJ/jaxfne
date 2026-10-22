"""Tests for the potjans2014 lane API (P2: PORT_SPEC rows, G-MULT/G-TAUSYN/
G-POPDRIVE/G-WDIST/G-DDIST settled).

Passing tests pin exact port behaviour; gap tests fail on the baseline
implementation and carry ``xfail(strict=True)`` so a core change that closes
a gap turns them green loudly.
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
from artifacts.etudes.papers.potjans2014 import potjans2014_api as L

N_SMALL = 800
DT = 0.5


def _lif_model(n=N_SMALL, seed=0):
    from jaxfne import neuronal_tensor as nt

    return jtfne.construct(
        L.build_spec(n=n, seed=seed),
        nt.RuntimeConfiguration(seed=seed, duration_ms=20.0, dt_ms=DT,
                                emitter="lif"),
    )


def _edges(model):
    el = model.params["edge_list"]
    rows = model.neuron_table()
    lab = [(str(r["layer"]), str(r["cell_type"])) for r in rows]
    return (np.asarray(el.pre), np.asarray(el.post), np.asarray(el.weight),
            np.asarray(el.delay_steps), lab)


def test_build_spec_smoke():
    """The lane spec builds (fixture must build)."""
    spec = L.build_spec(n=N_SMALL, seed=0)
    assert spec is not None, "fixture must build the lane spec"
    assert len(spec.area_connections) == 55, "55 nonzero paper projections"


def test_eight_population_counts():
    """Realized counts follow the paper Table 5 fractions (PORT_SPEC 16)."""
    model = _lif_model()
    rows = model.neuron_table()
    assert len(rows) == N_SMALL, "pilot scale realizes all cells"
    lab = [(str(r["layer"]), str(r["cell_type"])) for r in rows]
    want = L._pop_counts(N_SMALL)
    assert sum(want.values()) == N_SMALL, "fixture counts must tile n"
    for pop, n_want in sorted(want.items()):
        got = sum(1 for q in lab if q == pop)
        assert got == n_want, "population %r realizes its share" % (pop,)
        assert got > 0, "every paper population is present at pilot scale"


def test_lif_params_exact():
    """Realized LIF columns carry the paper Table 5 values (PORT_SPEC 1-3)."""
    em = _lif_model().params["emitter"]
    n = len(np.asarray(em.drive))
    assert n == N_SMALL, "emitter covers every neuron"
    assert np.allclose(np.asarray(em.tau_m_ms), 10.0), "tau_m = 10 ms"
    assert np.allclose(np.asarray(em.t_ref_ms), 2.0), "tau_ref = 2 ms"
    v_th = np.asarray(em.v_thresh_mV)
    v_re = np.asarray(em.v_reset_mV)
    assert np.allclose(v_th - v_re, 15.0), "paper theta-V_reset gap exact"
    assert np.allclose(v_re, 0.0), "origin at V_reset (PORT_SPEC D-VSHIFT)"
    assert np.allclose(np.asarray(em.eta_rp_mV), 0.0), "no PCL refractory tail"
    assert np.allclose(np.asarray(em.delta_sra_mV), 0.0), "no adaptation"
    assert np.allclose(np.asarray(em.drive), 0.0), "lane drive is 0.0"


def test_density_exact_totals():
    """Density rules realize exactly round(p*n_pre*n_post) edges (PORT_SPEC 8)."""
    model = _lif_model()
    pre, post, _, _, lab = _edges(model)
    lab_arr = np.array(lab)
    spec = L.build_spec(n=N_SMALL, seed=0)
    for ac in spec.area_connections:
        src_q = (ac.source_layer, ac.source_neuron_type)
        tgt_q = (ac.target_layer, ac.target_neuron_type)
        n_pre = sum(1 for q in lab if q == src_q)
        n_post = sum(1 for q in lab if q == tgt_q)
        want = int(round(float(ac.probability) * n_pre * n_post))
        got = int(((lab_arr == np.array(src_q)).all(axis=1)[pre]
                   & (lab_arr == np.array(tgt_q)).all(axis=1)[post]).sum())
        assert got == want, \
            "rule %s%s->%s%s realizes its exact density total" % (src_q + tgt_q)
    assert all(a.sampling == "density" for a in spec.area_connections), \
        "lane rules sample density"


def test_realized_weights():
    """Edges carry +0.1 (E), -0.4 (I) and doubled L4e->L2/3e (PORT_SPEC 9)."""
    pre, post, weight, _, lab = _edges(_lif_model())
    lab_arr = np.array(lab)
    e_mask = np.array([q[1] == "E" for q in lab])[pre]
    assert bool((weight[e_mask] > 0).all()), "E sources excite"
    assert bool((weight[~e_mask] < 0).all()), "PV sources inhibit"
    assert np.allclose(np.abs(weight[e_mask]).max(), 0.2, rtol=1e-6), \
        "strongest E edge is the doubled L4e->L2/3e projection"
    l4e = (lab_arr == np.array(("L4", "E"))).all(axis=1)[pre]
    l23e = (lab_arr == np.array(("L23", "E"))).all(axis=1)[post]
    assert np.allclose(weight[l4e & l23e], 0.2, rtol=1e-6), \
        "every L4e->L2/3e edge carries the doubled weight"
    assert np.allclose(np.abs(weight[~e_mask]), 0.4, rtol=1e-6), \
        "I edges carry g x w (g=4)"


def test_delay_steps():
    """E rules realize 3 steps, I rules 2 steps at dt=0.5 (PORT_SPEC 10)."""
    pre, _, _, delay, lab = _edges(_lif_model())
    is_e = np.array([q[1] == "E" for q in lab])[pre]
    assert set(delay.tolist()) == {2, 3}, "only the two declared delays"
    assert set(delay[is_e].tolist()) == {3}, "E delay 1.5 ms is 3 steps"
    assert set(delay[~is_e].tolist()) == {2}, "I delay 1.0 ms is 2 steps"


def test_external_drive_uniform():
    """The lane declares one uniform drive: 8 Hz x 100 trains (PORT_SPEC 12)."""
    sim = L.build_sim(T_s=0.02, seed=0)
    pd = dict(sim.poisson_drive)
    assert float(pd["rate_hz"]) == pytest.approx(8.0), "nu_bg per train"
    assert int(pd["counts"]) == 2000, "uniform stand-in train count"
    assert float(pd["amplitude"]) == pytest.approx(0.1), "stand-in amplitude"
    model = _lif_model()
    signals = model.simulate(L.build_sim(T_s=0.02, seed=1))
    assert int(signals.metadata["poisson_drive"]["counts"]) == 2000, \
        "declared counts must reach the run receipt"


def test_zero_drive_silent():
    """Declared drive=0.0 holds the network silent with no input (PORT_SPEC 13)."""
    model = _lif_model()
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=DT, seed=0,
                           runtime=jtfne.RuntimeConfig(
                               recurrent_backend="edge_list"))
    spikes = np.asarray(model.simulate(sim).spikes)
    assert int(spikes.sum()) == 0, "no-drive network must stay silent"


def test_score_keys():
    """The lane scorer returns the analysis estimands (checklist E3)."""
    regime = L.score(np.zeros((100, 10), dtype=bool), DT)
    assert set(regime) == {"rate_hz", "cv_isi", "kappa", "active_fraction",
                           "spectral_peak_excess", "label"}, "scorer keys"
    assert regime["label"] == "silent", "empty raster must read silent"
    pops = L.score_by_population(np.zeros((100, 10), dtype=bool),
                                 [("L23", "E")] * 5 + [("L4", "E")] * 5, DT)
    assert set(pops) == {"L23E", "L4E"}, "per-population keys"
    assert pops["L23E"]["n"] == 5, "per-population sizes"


def test_gap_multapse_sampler():
    """Paper wiring draws K synapses per projection with replacement
    (multapses allowed, SOURCE_P1.md section 1); the core draws distinct
    pairs only. Proposed: sampling="multinomial" with n_synapses=K."""
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="potjans-gmult",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=10,
            neuron_types=[
                nt.NeuronType.make("E", fraction=0.8, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
                nt.NeuronType.make("PV", fraction=0.2, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
            ])])],
        area_connections=[nt.AreaConnection(
            source_area="mc", source_layer="L23", source_neuron_type="E",
            target_area="mc", target_layer="L23", target_neuron_type="E",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=1.0),
            delay_ms=1.5, sampling="multinomial", probability=0.5)],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    assert model is not None, "multapse sampler must construct"


@pytest.mark.xfail(strict=True, reason="G-TAUSYN: no synaptic current filter")
def test_gap_tau_syn_filter():
    """Paper PSCs decay with tau_syn = 0.5 ms (SOURCE_P1.md section 1); the
    lif route sums delta arrivals with no filter. Proposed: tau_syn_ms LIF
    param consumed by the kernel."""
    from jaxfne import neuronal_tensor as nt

    lif = dict(L._LIF_POTJANS, tau_syn_ms=0.5)
    tensor = nt.NeuronalTensor(
        name="potjans-gtausyn",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=2,
            neuron_types=[nt.NeuronType.make(
                "E", fraction=1.0, emitter="lif", emitter_params=dict(lif),
                drive=0.0)])])],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    assert model is not None, "tau_syn_ms must construct and filter arrivals"


@pytest.mark.xfail(strict=True, reason="G-POPDRIVE: poisson target is all-only")
def test_gap_per_population_drive():
    """Paper external input is per-population k_ext trains (SOURCE_P1.md
    section 2: 1600-2900 per cell); poisson_drive target="all" is the only
    path (jaxfne/_signals.py:768-772). Proposed: a triple target."""
    model = _lif_model()
    sim = jtfne.Simulation(
        duration_ms=20.0, dt_ms=DT, seed=0,
        poisson_drive={"rate_hz": 8.0, "amplitude": 0.1, "counts": 1600,
                       "seed": 0, "target": ("mc", "L23", "E")},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    signals = model.simulate(sim)
    assert np.asarray(signals.spikes).shape[0] == sim.n_steps, \
        "targeted drive must simulate"


def test_gap_weight_distribution():
    """Paper weights are Gaussian w +- delta_w, sign-preserving
    (SOURCE_P1.md section 2); the core realizes one constant per rule.
    Proposed: PlasticParams(w_std=delta_w)."""
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="potjans-gwdist",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=10,
            neuron_types=[
                nt.NeuronType.make("E", fraction=0.8, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
                nt.NeuronType.make("PV", fraction=0.2, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
            ])])],
        area_connections=[nt.AreaConnection(
            source_area="mc", source_layer="L23", source_neuron_type="E",
            target_area="mc", target_layer="L23", target_neuron_type="E",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=1.0, w_std=0.1),
            delay_ms=1.5, sampling="density", probability=0.5)],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    assert model is not None, "weight spread must construct"


def test_gap_delay_distribution():
    """Paper delays are Gaussian d +- delta_d, sign-preserving and snapped
    to the grid (SOURCE_P1.md section 2); the core fans one constant per
    rule out. Proposed: AreaConnection delay_std_ms."""
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="potjans-gddist",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=10,
            neuron_types=[
                nt.NeuronType.make("E", fraction=0.8, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
                nt.NeuronType.make("PV", fraction=0.2, emitter="lif",
                                   emitter_params=dict(L._LIF_POTJANS),
                                   drive=0.0),
            ])])],
        area_connections=[nt.AreaConnection(
            source_area="mc", source_layer="L23", source_neuron_type="E",
            target_area="mc", target_layer="L23", target_neuron_type="E",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=1.0),
            delay_ms=1.5, delay_std_ms=0.75,
            sampling="density", probability=0.5)],
    )
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=DT,
                                        emitter="lif"))
    assert model is not None, "delay spread must construct"
