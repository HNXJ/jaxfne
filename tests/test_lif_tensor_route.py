"""Stage A: the ``lif`` tensor route on the jaxfne core (lane 2b).

Builds a PCL-LIF model from a NeuronalTensor resolving to ``"lif"`` and
checks the route contract: the bridge population and edge list are reused
bit-exactly (only the emitter object changes), per-triple LIF overrides are
consumed or refused by name, ``simulate()`` runs finite and deterministic
through the plain ``jtfne.simulate(model)`` idiom, HDP runs a registered
rule only, and every unconsumed input is refused naming itself.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.emitters_lif import LIFNetworkParams

DT_MS = 0.5
DUR_MS = 20.0


def _nt(*args, **kwargs):
    return nt.NeuronType.make(*args, **kwargs)


def _tensor(types=None, delay_ms=1.0):
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
        delay_ms=delay_ms,
    )
    return nt.NeuronalTensor(
        name="lif-route",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(
                name="L4", n_neurons=6,
                neuron_types=list(types) if types is not None else [_nt("E"), _nt("PV")])],
            inter_connections=[conn],
        )],
    )


def _lif_runtime(**kwargs):
    return nt.RuntimeConfiguration(
        seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter="lif", **kwargs)


def _lif_model(tensor=None):
    return jtfne.construct(_tensor() if tensor is None else tensor, _lif_runtime())


def _edge_arrays(model):
    edges = model.params["edge_list"]
    return (
        np.asarray(edges.pre), np.asarray(edges.post),
        np.asarray(edges.weight, dtype=float),
        np.asarray(edges.delay_steps),
    )


def test_lif_build_reuses_the_bridge_population_and_edges():
    """Same tensor, same seed: lif keeps the bridge neurons and signed edges."""
    tensor = _tensor()
    lif = _lif_model(tensor)
    izhi = jtfne.construct(
        tensor,
        nt.RuntimeConfiguration(
            seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter="izhikevich"))
    assert isinstance(lif.params["emitter"], LIFNetworkParams)
    assert list(lif.params["emitter"].labels) == [
        str(x) for x in izhi.params["emitter"].labels]
    for a, b in zip(_edge_arrays(lif)[:3], _edge_arrays(izhi)[:3]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(_edge_arrays(lif)[3], _edge_arrays(izhi)[3])
    assert lif.static["lif"]["emitter_family"] == "lif"
    assert lif.cfg.metadata.get("recurrent_backend") == "edge_list"


def test_lif_simulate_runs_finite_and_deterministic():
    """Plain jtfne.simulate(model) inherits the edge-list backend and repeats."""
    model = _lif_model()
    first = jtfne.simulate(model)
    second = jtfne.simulate(model)
    assert bool(np.isfinite(np.asarray(first.V_m)).all())
    assert np.asarray(first.spikes).shape == np.asarray(first.V_m).shape
    assert np.array_equal(np.asarray(first.spikes), np.asarray(second.spikes))
    assert np.array_equal(np.asarray(first.V_m), np.asarray(second.V_m))
    assert first.metadata["emitter_family"] == "lif"
    assert first.metadata["physical_amplitude_calibrated"] is False
    assert first.metadata["not_recorded"] == ["sources", "field"]
    assert first.field is None and first.sources is None


def test_lif_floor_receipt_counts_sub_one_step_edges():
    """Zero-step constructed edges floor to 1 at simulate time, counted."""
    model = _lif_model()
    signals = jtfne.simulate(model)
    _, _, _, delay = _edge_arrays(model)
    assert int((delay < 1).sum()) == int(signals.metadata["n_floored_to_one_step"])


def test_lif_declared_drive_is_the_executed_drive():
    """NeuronType.drive overrides the drive of exactly its triple."""
    model = _lif_model(_tensor(types=[_nt("E", drive=9.0), _nt("PV")]))
    params = model.params["emitter"]
    labels = [str(x) for x in params.labels]
    drives = np.asarray(params.drive, dtype=float)
    assert bool((drives[[x == "E" for x in labels]] == 9.0).all())
    assert bool((drives[[x == "PV" for x in labels]] == 3.0).all())
    assert bool(np.isfinite(np.asarray(jtfne.simulate(model).V_m)).all())


def test_lif_triple_param_overrides_are_consumed():
    """emitter_params on a lif NeuronType land on exactly that triple."""
    tensor = _tensor(types=[
        _nt("E", emitter="lif", emitter_params={"tau_m_ms": 25.0}),
        _nt("PV", emitter="lif"),
    ])
    model = jtfne.construct(tensor, _lif_runtime())
    params = model.params["emitter"]
    labels = [str(x) for x in params.labels]
    taus = np.asarray(params.tau_m_ms, dtype=float)
    assert bool((taus[[x == "E" for x in labels]] == 25.0).all())
    assert bool((taus[[x == "PV" for x in labels]] == 18.0).all())


def test_lif_triple_param_overrides_refused_by_name():
    """Unknown fields and non-finite values fail closed, naming the field."""
    with pytest.raises(ValueError, match="tau_m"):
        _lif_model(_tensor(types=[
            _nt("E", emitter="lif", emitter_params={"tau_m": 25.0}),
            _nt("PV", emitter="lif"),
        ]))
    with pytest.raises(ValueError, match="tau_m_ms"):
        _lif_model(_tensor(types=[
            _nt("E", emitter="lif", emitter_params={"tau_m_ms": float("inf")}),
            _nt("PV", emitter="lif"),
        ]))


def _pcl_rule_params(n_edges, eta):
    from jaxfne.hdp_rules_pcl import rule_params as pcl_params

    return pcl_params(
        tau=np.full(n_edges, 10.0), w_max=np.full(n_edges, 5.0),
        eta=np.full(n_edges, eta), f=np.full(n_edges, 1.0),
        lam=np.full(n_edges, 1.0), norm=np.zeros(n_edges),
        group=np.zeros(n_edges, dtype=np.int32), n_groups=1,
    )


def _lif_hdp_runtime(rule_params):
    return jtfne.RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True,
        hdp_params={"hdp_rule": "pcl_stdp", "hdp_rule_params": rule_params},
    )


def _spiking_lif_model():
    model = _lif_model()
    probe = model.simulate(jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
    assert int(np.sum(np.asarray(probe.spikes))) > 0, "fixture must spike"
    return model


def test_lif_hdp_rule_runs_and_seeds_from_H0():
    """pcl_stdp runs on the lif route; the construct-seeded H0 rides along."""
    model = _spiking_lif_model()
    n_edges = int(_edge_arrays(model)[0].shape[0])
    assert n_edges > 0
    assert model.params.get("hdp_initial_H") is not None
    sim = jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
        runtime=_lif_hdp_runtime(_pcl_rule_params(n_edges, eta=0.5)))
    signals = model.simulate(sim)
    assert bool(np.isfinite(np.asarray(signals.V_m)).all())
    assert signals.metadata["hdp_rule"] == "pcl_stdp"
    assert signals.metadata["hdp_rule_registered"] is True


def test_lif_hdp_rule_with_zero_eta_matches_the_rule_free_run():
    """An identity rule changes no weight: bit-identical spikes and V."""
    model = _spiking_lif_model()
    n_edges = int(_edge_arrays(model)[0].shape[0])
    sim_kw = dict(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0})
    plain = model.simulate(jtfne.Simulation(
        **sim_kw, runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
    ruled = model.simulate(jtfne.Simulation(
        **sim_kw, runtime=_lif_hdp_runtime(_pcl_rule_params(n_edges, eta=0.0))))
    assert np.array_equal(np.asarray(ruled.spikes), np.asarray(plain.spikes))
    assert np.array_equal(np.asarray(ruled.V_m), np.asarray(plain.V_m))


def test_lif_hdp_rule_with_positive_eta_moves_the_trajectory():
    """A live rule changes weights, so the driven trajectory differs."""
    model = _spiking_lif_model()
    n_edges = int(_edge_arrays(model)[0].shape[0])
    sim_kw = dict(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0})
    plain = model.simulate(jtfne.Simulation(
        **sim_kw, runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
    ruled = model.simulate(jtfne.Simulation(
        **sim_kw, runtime=_lif_hdp_runtime(_pcl_rule_params(n_edges, eta=0.5))))
    assert float(np.max(np.abs(
        np.asarray(ruled.V_m, dtype=float)
        - np.asarray(plain.V_m, dtype=float)))) > 0.0


def test_lif_hdp_requires_a_registered_rule():
    """enable_hdp without (or with a bad) hdp_rule is refused, naming keys."""
    model = _lif_model()
    sim_kw = dict(duration_ms=DUR_MS, dt_ms=DT_MS)
    with pytest.raises(ValueError, match="registered hdp_rule"):
        model.simulate(jtfne.Simulation(
            **sim_kw,
            runtime=jtfne.RuntimeConfig(
                recurrent_backend="edge_list", enable_hdp=True)))
    with pytest.raises(ValueError, match="registered hdp_rule"):
        model.simulate(jtfne.Simulation(
            **sim_kw,
            runtime=jtfne.RuntimeConfig(
                recurrent_backend="edge_list", enable_hdp=True,
                hdp_params={"hdp_rule": "signed_linear"})))


def test_lif_hdp_refuses_unconsumed_hdp_keys():
    """Trace/record keys have no lif consumer: refused by name."""
    model = _lif_model()
    n_edges = int(_edge_arrays(model)[0].shape[0])
    hp = {"hdp_rule": "pcl_stdp",
          "hdp_rule_params": _pcl_rule_params(n_edges, eta=0.0),
          "record_weight_trace": True}
    with pytest.raises(ValueError, match="record_weight_trace"):
        model.simulate(jtfne.Simulation(
            duration_ms=DUR_MS, dt_ms=DT_MS,
            runtime=jtfne.RuntimeConfig(
                recurrent_backend="edge_list", enable_hdp=True,
                hdp_params=hp)))


def test_lif_hdp_refuses_seeded_weights_under_a_rule():
    """hdp_initial_w has no kernel input on lif: refused when a rule runs."""
    model = _spiking_lif_model()
    n_edges = int(_edge_arrays(model)[0].shape[0])
    seeded = model.with_hdp_initial_state(w0=np.zeros(n_edges))
    with pytest.raises(ValueError, match="hdp_initial_w"):
        seeded.simulate(jtfne.Simulation(
            duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
            poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
            runtime=_lif_hdp_runtime(_pcl_rule_params(n_edges, eta=0.0))))


def test_lif_hdp_seeded_H_is_inert_without_a_rule():
    """The construct-seeded H0 changes nothing while HDP stays disabled."""
    model = _lif_model()
    assert model.params.get("hdp_initial_H") is not None
    assert bool(np.isfinite(np.asarray(jtfne.simulate(model).V_m)).all())


def _lif_sim(model, **kwargs):
    kwargs.setdefault(
        "runtime", jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    return model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, **kwargs))


def test_lif_refuses_paradigm_continuation_and_ablation():
    """No paradigm current, no continuation, no ablation mask on lif."""
    model = _lif_model()
    with pytest.raises(ValueError, match="paradigm"):
        model.simulate(
            jtfne.Simulation(
                duration_ms=DUR_MS, dt_ms=DT_MS,
                runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
            paradigm="no-drive")
    with pytest.raises(ValueError, match="continuation"):
        model.simulate(
            jtfne.Simulation(
                duration_ms=DUR_MS, dt_ms=DT_MS,
                runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
            continuation="no-state")
    with pytest.raises(ValueError, match="return_state"):
        model.simulate(
            jtfne.Simulation(
                duration_ms=DUR_MS, dt_ms=DT_MS,
                runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
            return_state=True)
    with pytest.raises(ValueError, match="ablation"):
        _lif_sim(model, ablation="E_silence")
    with pytest.raises(ValueError, match="simulate_condition"):
        model.simulate_condition(
            jtfne.Simulation(
                duration_ms=DUR_MS, dt_ms=DT_MS,
                runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
            condition="no-condition")


def test_lif_refuses_homeostasis_kernel_and_backend():
    """The homeostasis controller, the receptor kernel and dense edges."""
    model = _lif_model()
    with pytest.raises(ValueError, match="homeostasis"):
        _lif_sim(
            model,
            runtime=jtfne.RuntimeConfig(
                recurrent_backend="edge_list", enable_homeostasis=True))
    with pytest.raises(ValueError, match="synaptic_kernel"):
        _lif_sim(
            model,
            runtime=jtfne.RuntimeConfig(
                recurrent_backend="edge_list",
                synaptic_kernel="receptor_exponential"))
    with pytest.raises(ValueError, match="edge_list"):
        _lif_sim(model, runtime=jtfne.RuntimeConfig(recurrent_backend="dense"))


def test_lif_has_no_configuration_route():
    """The direct current-based bridge refuses lif, naming the tensor route."""
    with pytest.raises(ValueError, match="construct_neuronal_tensor"):
        nt.neuronal_tensor_to_configuration(_tensor(), emitter="lif")
    with pytest.raises(ValueError, match="not routed"):
        jtfne.construct(_tensor(), nt.RuntimeConfiguration(
            seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter="glif"))
    with pytest.raises(ValueError, match="n_contacts"):
        jtfne.construct(_tensor(), _lif_runtime(n_contacts=8))


def test_lif_simulate_honours_jit_and_backend():
    """jit=True compiles (bit-identical to eager); backend=cpu runs scoped."""
    model = _lif_model()
    eager = _lif_sim(model)
    compiled = _lif_sim(
        model, runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", jit=True))
    assert np.array_equal(
        np.asarray(compiled.spikes), np.asarray(eager.spikes))
    assert np.array_equal(np.asarray(compiled.V_m), np.asarray(eager.V_m))
    assert any(
        key[0] == "simulate_lif" for key in model._compiled_cache
    ), "jit=True must compile the lif kernel through the shared guard"
    scoped = _lif_sim(
        model, runtime=jtfne.RuntimeConfig(
            recurrent_backend="edge_list", backend="cpu"))
    assert np.array_equal(np.asarray(scoped.spikes), np.asarray(eager.spikes))


def test_lif_simulate_refuses_vmap():
    """Single-trial simulate() has no batch to vectorize: vmap refused."""
    with pytest.raises(ValueError, match="vmap"):
        _lif_sim(model=_lif_model(), runtime=jtfne.RuntimeConfig(
            recurrent_backend="edge_list", vmap=True))


def test_lif_refuses_foreign_hdp_params_without_a_rule():
    """noise_scale with enable_hdp=False is refused by name (no noise stream)."""
    with pytest.raises(ValueError, match="noise_scale"):
        _lif_sim(model=_lif_model(), runtime=jtfne.RuntimeConfig(
            recurrent_backend="edge_list", enable_hdp=False,
            hdp_params={"noise_scale": 3.0}))


def _sharing_tensor():
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
        footprint={"size": [10.0, 10.0], "offset": [-5.0, -5.0]},
        weight_sharing="patch", allow_self=True,
    )
    return nt.NeuronalTensor(
        name="lif-sharing",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=6,
                             neuron_types=[_nt("E")])],
            inter_connections=[conn])],
    )


def test_lif_weight_sharing_runs_one_weight_per_group():
    """Shared edges carry one executed weight before and after an HDP run."""
    from jaxfne._construct_lif import lif_edge_arrays

    model = _lif_model(_sharing_tensor())
    edges = lif_edge_arrays(model)
    assert bool(edges["weight_sharing"]) is True
    group = np.asarray(edges["group"])
    w_init = np.asarray(edges["weight"], dtype=float)
    members = np.flatnonzero(group == group[0])
    assert members.shape[0] >= 2, "fixture needs a shared pair"
    assert np.all(w_init[members] == w_init[members[0]])
    n_edges = int(group.shape[0])
    signals = _lif_sim(
        model, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
        runtime=_lif_hdp_runtime(_pcl_rule_params(n_edges, eta=0.5)))
    spikes = np.asarray(signals.spikes)
    assert int(spikes.sum()) > 0, "fixture must spike"
    w_final = np.asarray(
        model.last_hdp_diagnostics()["w_final"], dtype=float)
    assert np.all(w_final[members] == w_final[members[0]])
    assert bool((w_final[members] != w_init[members]).any())
    # Non-vacuous: a member pair with differing presynaptic trains, so
    # per-edge STDP updates differ and only the shared accumulation keeps
    # the pair equal.
    pre = np.asarray(edges["pre"])
    pair = next(
        ((a, b) for x, a in enumerate(members) for b in members[x + 1:]
         if bool((spikes[:, pre[a]] != spikes[:, pre[b]]).any())),
        None)
    assert pair is not None, "fixture needs a pair with differing drive"
    assert signals.metadata["weight_sharing"] == "patch"
    assert signals.metadata["n_sharing_groups"] == int(group.max()) + 1


def _inhibition_tensor(spec):
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
        local_inhibition=spec,
    )
    return nt.NeuronalTensor(
        name="lif-inhibition",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=6,
                             neuron_types=[_nt("E"), _nt("PV")])],
            inter_connections=[conn])],
    )


def _driven_spikes(tensor, **kwargs):
    return np.asarray(_lif_sim(
        model=_lif_model(tensor), seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
        **kwargs).spikes)


def test_lif_local_inhibition_changes_spikes():
    """The declared cap runs inside the step: capped spikes differ."""
    plain = _driven_spikes(_tensor())
    assert int(plain.sum()) > 0, "fixture must spike"
    capped = _driven_spikes(_inhibition_tensor({"rounds": 1}))
    assert int(capped.sum(axis=1).max()) <= 1
    assert not np.array_equal(capped, plain)


def test_lif_local_inhibition_rounds_and_order_matter():
    """R and the round order select different winners: spikes differ."""
    asc = _driven_spikes(_inhibition_tensor({"rounds": 1}))
    desc = _driven_spikes(
        _inhibition_tensor({"rounds": 1, "order": "descending"}))
    assert not np.array_equal(asc, desc), "order must select winners"
    wider = _driven_spikes(_inhibition_tensor({"rounds": 2}))
    assert not np.array_equal(asc, wider), "R must admit more winners"
