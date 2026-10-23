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

import jax
import jax.numpy as jnp
import jaxlib

import jaxfne as jtfne
from jaxfne._model_simulate import simulate_lif_seed_batch
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
    kwargs.setdefault("duration_ms", DUR_MS)
    return nt.RuntimeConfiguration(
        seed=0, dt_ms=DT_MS, emitter="lif", **kwargs)


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


def _solo_tensor():
    """Four independent lif cells with no baseline drive (no edges)."""
    return nt.NeuronalTensor(
        name="lif-solo",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=4,
                             neuron_types=[_nt("E", drive=0.0)])],
            inter_connections=[])],
    )


def test_lif_paradigm_schedule_drives_targeted_cells_only():
    """A StimulusSchedule rides the lif route as input current (part of x).

    Fail-first for Gap 1: the lif route refuses paradigm drive, so DVS
    clips cannot feed a core tensor.
    """
    model = _lif_model(_solo_tensor())
    off = np.asarray(_lif_sim(model, seed=1).spikes)
    assert int(off.sum()) == 0, "fixture must be silent without drive"
    schedule = jtfne.stimulus_schedule(
        [{"onset_ms": 0.0, "duration_ms": DUR_MS, "amplitude": 60.0,
          "target_indices": [0, 1]}],
        n_neurons=4)
    signals = model.simulate(
        jtfne.Simulation(
            duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
        paradigm=schedule)
    spikes = np.asarray(signals.spikes)
    assert int(spikes[:, [0, 1]].sum()) > 0, "targeted cells must spike"
    assert int(spikes[:, 2:].sum()) == 0, "untargeted cells must stay silent"
    assert signals.metadata["paradigm"] is not None


def test_lif_refuses_a_non_schedule_paradigm():
    """A paradigm of unknown type is refused, naming paradigm."""
    model = _lif_model()
    with pytest.raises((ValueError, TypeError), match="paradigm"):
        model.simulate(
            jtfne.Simulation(
                duration_ms=DUR_MS, dt_ms=DT_MS,
                runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
            paradigm="no-drive")


def test_lif_refuses_continuation_and_ablation():
    """No continuation, no ablation mask on lif (paradigm now drives)."""
    model = _lif_model()
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
    # One retinotopic position: the degenerate point co-locates all 6 cells,
    # so rounds compete across the whole layer (random positions would leave
    # every cell a singleton group and the cap inert).
    return nt.NeuronalTensor(
        name="lif-inhibition",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(
                name="L4", n_neurons=6,
                neuron_types=[_nt("E"), _nt("PV")],
                geometry=nt.Geometry3D(
                    x_range=(0.0, 0.0), y_range=(0.0, 0.0)))],
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


def _two_row_tensor():
    """One simple-row cell and one complex-row cell with an S->C edge."""
    conn = nt.InterConnection(
        source_layer="LS", source_neuron_type="E",
        target_layer="LC", target_neuron_type="complex",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
    )
    return nt.NeuronalTensor(
        name="lif-two-rows",
        areas=[nt.Area(
            name="V1",
            layers=[
                nt.Layer(name="LS", n_neurons=1,
                         neuron_types=[_nt("E", drive=0.0)],
                         geometry=nt.Geometry3D(
                             x_range=(0.0, 0.0), y_range=(0.0, 0.0))),
                nt.Layer(name="LC", n_neurons=1,
                         neuron_types=[_nt("complex", drive=0.0)],
                         geometry=nt.Geometry3D(
                             x_range=(5.0, 5.0), y_range=(0.0, 0.0))),
            ],
            inter_connections=[conn])],
    )


def test_lif_pcl_labels_route_to_pcl_rows():
    """Fail-first for Gap 3a: lif-only labels build through the lif bridge.

    ``"complex"`` names no Izhikevich row, so the bridge cannot sample it;
    the lif route aliases it for the bridge population only and restores
    the PCL row (tau 50 ms, threshold 3 mV) afterwards.
    """
    model = _lif_model(_two_row_tensor())
    params = model.params["emitter"]
    labels = [str(x) for x in params.labels]
    assert labels == ["E", "complex"]
    taus = np.asarray(params.tau_m_ms, dtype=float)
    assert taus.tolist() == [18.0, 50.0]
    assert np.asarray(params.v_thresh_mV, dtype=float).tolist() == [10.0, 3.0]
    assert np.asarray(params.drive, dtype=float).tolist() == [0.0, 0.0]
    assert np.asarray(params.sign, dtype=float).tolist() == [1.0, 1.0]
    assert bool(np.isfinite(np.asarray(jtfne.simulate(model).V_m)).all())


def _tiny_scan_case():
    """One covered pixel, 2 simples + 1 complex, 3 feed events."""
    from jaxfne.pcl_topology import (
        EventScanDrive,
        build_s1_scan_tensor,
        event_scan_events,
        event_scan_tables,
        local_inhibition_blocks,
        scan_inhibition_triplets,
    )

    nx = ny = cx = cy = cd = 1
    nd, ns, nc = 2, 2, 1
    wi_s = np.array([[0.0, 0.1], [0.1, 0.0]])
    wi_c = np.array([[0.0]])
    tables = event_scan_tables(
        nx, ny, nd, cx, cy, cd,
        origins_x=[0], origins_y=[0], size=(1, 1),
        weights=np.full((nd, 2, 1, 1), 4.0),
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0],
        c_size=(1, 1), c_weights=np.zeros((cd, 1, 1, nd)),
        vw=4, vh=4, inhib_range=(0, 0))
    groups = np.array([0, 0, 1], dtype=np.int64)
    pre, post, mag = scan_inhibition_triplets(
        nx, ny, nd, cx, cy, cd, ns, wi_s, wi_c)
    blocks = local_inhibition_blocks(groups, pre, post, mag)
    t = np.array([0, 500, 1000], dtype=np.int64)
    n_steps = int(t.max() // 500) + 1 + 3
    events = event_scan_events(
        t, np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64),
        np.zeros(3, dtype=np.int64), dt_ms=DT_MS, n_steps=n_steps,
        vw=4, vh=4)
    drive = EventScanDrive(
        events, tables, blocks, groups, dt_ms=DT_MS, n_steps=n_steps)
    return drive, build_s1_scan_tensor(ns=ns, nc=nc), n_steps


def test_lif_event_scan_tensor_route_matches_edge_kernel():
    """Fail-first for Step 9: the tensor route replays the scan exactly.

    A short event slice through ``construct -> simulate`` (paradigm is the
    scan x-source) must equal ``simulate_lif_edge`` spike-for-spike; the
    tensor path refuses ``event_scan`` by name on baseline.
    """
    from jaxfne.emitters_lif import make_lif_params, simulate_lif_edge

    drive, tensor, n_steps = _tiny_scan_case()
    model = jtfne.construct(
        tensor, _lif_runtime(duration_ms=n_steps * DT_MS))
    sim = jtfne.Simulation(
        duration_ms=n_steps * DT_MS, dt_ms=DT_MS, seed=1,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    signals = model.simulate(sim, paradigm=drive)
    ref = simulate_lif_edge(
        make_lif_params(["simple"] * 2 + ["complex"]),
        n_steps=n_steps, dt_ms=DT_MS,
        local_inhibition_groups=drive.groups,
        local_inhibition_resolution="event_scan",
        local_inhibition_blocks=drive.blocks,
        event_scan_events=drive.events,
        event_scan_tables=drive.tables)
    got = np.asarray(signals.spikes)
    assert int(got.sum()) > 0, "fixture must spike"
    assert np.array_equal(got, np.asarray(ref["spikes"]))
    assert np.array_equal(
        np.asarray(signals.metadata["scan_counts"]),
        np.asarray(ref["counts"]))
    assert np.array_equal(
        np.asarray(signals.metadata["scan_first_spike_us"]),
        np.asarray(ref["first_spike_us"]))
    assert bool(np.isnan(np.asarray(signals.V_m)).all())


def test_lif_event_scan_refuses_unmapped_inputs_by_name():
    """The mapped scan case refuses schedule/poisson/absent/HDP channels."""
    drive, tensor, n_steps = _tiny_scan_case()
    model = jtfne.construct(
        tensor, _lif_runtime(duration_ms=n_steps * DT_MS))

    def _sim(**kwargs):
        kwargs.setdefault(
            "runtime", jtfne.RuntimeConfig(recurrent_backend="edge_list"))
        return jtfne.Simulation(
            duration_ms=n_steps * DT_MS, dt_ms=DT_MS, **kwargs)

    with pytest.raises(ValueError, match="StimulusSchedule"):
        model.simulate(
            _sim(), paradigm=jtfne.stimulus_schedule(
                [{"onset_ms": 0.0, "duration_ms": 1.0, "amplitude": 1.0}],
                n_neurons=3))
    with pytest.raises(ValueError, match="EventScanDrive"):
        model.simulate(_sim(), paradigm=None)
    with pytest.raises(ValueError, match="poisson_drive"):
        model.simulate(
            _sim(poisson_drive={"rate_hz": 50.0, "amplitude": 100.0}),
            paradigm=drive)
    with pytest.raises(ValueError, match="enable_hdp"):
        model.simulate(
            _sim(runtime=jtfne.RuntimeConfig(
                recurrent_backend="edge_list", enable_hdp=True)),
            paradigm=drive)
    with pytest.raises((ValueError, TypeError), match="paradigm"):
        _lif_sim(model=_lif_model(), paradigm=drive)


# T-32c seed batching: one compiled call over N seeds, each member keyed
# PRNGKey(seed_i), bit-equal to N serial runs.


def _lif_seed_batch_sim(**kwargs):
    base = dict(
        duration_ms=DUR_MS, dt_ms=DT_MS,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    base.update(kwargs)
    return jtfne.Simulation(**base)


# T-32c numerical contract (Hamm ruling 2026-10-23, option a): spikes bitwise
# exact; V_m allclose rtol=0, atol=1e-4 mV (provisional, ~2x the largest WSL
# discrepancy observed; do NOT raise on a future failure). H_final/w_final use
# the same atol: they share the float32 codegen origin as V_m and were exact
# where observed. Bitwise tests stay as platform diagnostics (bitwise_diagnostic).
V_RTOL = 0.0
V_ATOL_MV = 1e-4
HW_ATOL = 1e-4
EVIDENCE = {
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "backend": jax.default_backend(),
    "device": str(jax.devices()[0]),
    "dtype": "float32",
    "x64": bool(jax.config.read("jax_enable_x64")),
}


@pytest.mark.bitwise_diagnostic
@pytest.mark.xfail(strict=False, reason="bitwise V_m is platform/run-dependent; diagnostic evidence only")
def test_lif_seed_batch_matches_serial_per_member():
    """DIAGNOSTIC (bitwise_diagnostic): bitwise V_m holds on Windows CPU and some
    WSL runs; other WSL runs show ulp-scale V_m differences (max ~5e-5 mV).
    Spike assertions are contract; V_m bitwise here is evidence, not gate."""
    model = _spiking_lif_model()
    seeds = [11, 12, 13]
    out = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), seeds)
    assert [int(s) for s in out["metadata"]["seeds"]] == seeds
    for k, s in enumerate(seeds):
        ref = model.simulate(_lif_seed_batch_sim(seed=s))
        assert np.array_equal(np.asarray(out["spikes"][k]), np.asarray(ref.spikes))
        assert np.array_equal(np.asarray(out["V_m"][k]), np.asarray(ref.V_m))


def test_lif_seed_batch_members_are_independent():
    model = _spiking_lif_model()
    out = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), [11, 12, 13])
    spk = np.asarray(out["spikes"])
    assert int(spk.sum()) > 0, "fixture must spike"
    assert not (np.array_equal(spk[0], spk[1]) and np.array_equal(spk[1], spk[2]))
    same = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), [7, 7])
    assert np.array_equal(np.asarray(same["spikes"][0]), np.asarray(same["spikes"][1]))


@pytest.mark.bitwise_diagnostic
@pytest.mark.xfail(strict=False, reason="bitwise V_m is platform/run-dependent; diagnostic evidence only")
def test_lif_seed_batch_matches_serial_under_hdp_rule():
    """DIAGNOSTIC (bitwise_diagnostic): same platform note as the serial-parity
    diagnostic above, under the pcl_stdp rule."""
    model = _spiking_lif_model()
    n_edges = int(_edge_arrays(model)[0].shape[0])
    runtime = _lif_hdp_runtime(_pcl_rule_params(n_edges, eta=0.5))
    seeds = [11, 12, 13]
    out = simulate_lif_seed_batch(model, _lif_seed_batch_sim(runtime=runtime), seeds)
    for k, s in enumerate(seeds):
        ref = model.simulate(_lif_seed_batch_sim(seed=s, runtime=runtime))
        assert np.array_equal(np.asarray(out["spikes"][k]), np.asarray(ref.spikes))
        assert np.array_equal(np.asarray(out["V_m"][k]), np.asarray(ref.V_m))


def test_lif_seed_batch_states_the_continuation_refusal():
    assert "continuation" in simulate_lif_seed_batch.__doc__


def test_lif_seed_batch_is_permutation_and_batch_size_invariant():
    model = _spiking_lif_model()
    fwd = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), [11, 12, 13])
    rev = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), [13, 11, 12])
    for k, s in enumerate([13, 11, 12]):
        j = [11, 12, 13].index(s)
        assert np.array_equal(np.asarray(rev["spikes"][k]), np.asarray(fwd["spikes"][j]))
        assert np.array_equal(np.asarray(rev["V_m"][k]), np.asarray(fwd["V_m"][j]))
    wide = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), [11, 12, 13, 14])
    slim = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), [11, 12])
    for k in range(2):
        assert np.array_equal(np.asarray(wide["spikes"][k]), np.asarray(slim["spikes"][k]))
        assert np.array_equal(np.asarray(wide["V_m"][k]), np.asarray(slim["V_m"][k]))


def test_lif_seed_batch_final_weights_match_the_edge_list():
    from jaxfne._construct_lif import lif_edge_arrays
    model = _spiking_lif_model()
    out = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), [11, 12])
    want = np.asarray(lif_edge_arrays(model)["weight"], dtype=float)
    assert want.shape[0] > 0, "fixture must carry executed edges"
    for k in range(2):
        assert np.allclose(np.asarray(out["w_final"][k], dtype=float), want)


@pytest.mark.bitwise_diagnostic
@pytest.mark.xfail(strict=False, reason="bitwise H_final/w_final is platform/run-dependent; diagnostic evidence only")
def test_lif_seed_batch_hdp_final_state_matches_serial_diagnostics():
    """DIAGNOSTIC (bitwise_diagnostic): bitwise H_final/w_final holds on Windows
    CPU; cross-form float32 codegen may differ at ulp scale elsewhere."""
    model = _spiking_lif_model()
    n_edges = int(_edge_arrays(model)[0].shape[0])
    runtime = _lif_hdp_runtime(_pcl_rule_params(n_edges, eta=0.5))
    seeds = [11, 12, 13]
    out = simulate_lif_seed_batch(model, _lif_seed_batch_sim(runtime=runtime), seeds)
    assert "H_final" in out
    for k, s in enumerate(seeds):
        model.simulate(_lif_seed_batch_sim(seed=s, runtime=runtime))
        diag = model.last_hdp_diagnostics()
        assert np.array_equal(np.asarray(out["H_final"][k]), np.asarray(diag["H_final"]))
        assert np.array_equal(np.asarray(out["w_final"][k]), np.asarray(diag["w_final"]))


def test_lif_seed_batch_contract_spikes_exact_voltage_close():
    """CONTRACT: spikes bitwise at every timestep; V_m allclose rtol=0, atol=1e-4."""
    model = _spiking_lif_model()
    seeds = [11, 12, 13]
    out = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), seeds)
    worst = 0.0
    for k, s in enumerate(seeds):
        ref = model.simulate(_lif_seed_batch_sim(seed=s))
        b_spk = np.asarray(out["spikes"][k])
        r_spk = np.asarray(ref.spikes)
        assert b_spk.shape == r_spk.shape
        for t in range(b_spk.shape[0]):
            assert np.array_equal(b_spk[t], r_spk[t]), f"spike mismatch seed {s} step {t}"
        np.testing.assert_allclose(np.asarray(out["V_m"][k]), np.asarray(ref.V_m),
                                   rtol=V_RTOL, atol=V_ATOL_MV)
        worst = max(worst, float(np.max(np.abs(
            np.asarray(out["V_m"][k], dtype=float) - np.asarray(ref.V_m, dtype=float)))))
    print(f"lif contract max |dV| mV: {worst:.3e} evidence: {EVIDENCE}")
    assert worst <= V_ATOL_MV


def test_lif_seed_batch_hdp_contract_states_close():
    """CONTRACT under pcl_stdp: spikes exact; V/H/w allclose with stated atols."""
    model = _spiking_lif_model()
    n_edges = int(_edge_arrays(model)[0].shape[0])
    runtime = _lif_hdp_runtime(_pcl_rule_params(n_edges, eta=0.5))
    seeds = [11, 12, 13]
    out = simulate_lif_seed_batch(model, _lif_seed_batch_sim(runtime=runtime), seeds)
    worst_v = worst_h = worst_w = 0.0
    for k, s in enumerate(seeds):
        ref = model.simulate(_lif_seed_batch_sim(seed=s, runtime=runtime))
        assert np.array_equal(np.asarray(out["spikes"][k]), np.asarray(ref.spikes))
        np.testing.assert_allclose(np.asarray(out["V_m"][k]), np.asarray(ref.V_m),
                                   rtol=V_RTOL, atol=V_ATOL_MV)
        diag = model.last_hdp_diagnostics()
        np.testing.assert_allclose(np.asarray(out["H_final"][k]), np.asarray(diag["H_final"]),
                                   rtol=0.0, atol=HW_ATOL)
        np.testing.assert_allclose(np.asarray(out["w_final"][k]), np.asarray(diag["w_final"]),
                                   rtol=0.0, atol=HW_ATOL)
        worst_v = max(worst_v, float(np.max(np.abs(
            np.asarray(out["V_m"][k], dtype=float) - np.asarray(ref.V_m, dtype=float)))))
        worst_h = max(worst_h, float(np.max(np.abs(
            np.asarray(out["H_final"][k], dtype=float)
            - np.asarray(diag["H_final"], dtype=float)))))
        worst_w = max(worst_w, float(np.max(np.abs(
            np.asarray(out["w_final"][k], dtype=float)
            - np.asarray(diag["w_final"], dtype=float)))))
    print(f"lif hdp max |dV|={worst_v:.3e} |dH|={worst_h:.3e} |dw|={worst_w:.3e} {EVIDENCE}")
    assert worst_v <= V_ATOL_MV and worst_h <= HW_ATOL and worst_w <= HW_ATOL


def test_lif_seed_batch_repeated_compiled_runs_are_bit_identical():
    """Batched side self-deterministic: two compiled calls agree bitwise."""
    model = _spiking_lif_model()
    a = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), [11, 12, 13])
    b = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), [11, 12, 13])
    for key in ("spikes", "V_m", "w_final"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]))


def test_lif_seed_batch_vmap_form_matches_contract():
    """Same seeds across scalar, vmap and compiled forms: spikes exact, V close."""
    from jaxfne._construct_lif import lif_edge_arrays
    from jaxfne._signals import (
        _make_poisson_drive,
        poisson_chunk_seed,
        resolve_poisson_drive_target,
    )
    from jaxfne.emitters_lif import simulate_lif_edge

    model = _spiking_lif_model()
    seeds = [11, 12, 13]
    sim0 = _lif_seed_batch_sim()
    dtype = sim0.resolved_runtime.jnp_dtype
    emitter = model.params["emitter"]
    edges = lif_edge_arrays(model)
    params = emitter.to_lif_params()
    n = emitter.n_neurons
    steps = sim0.n_steps
    pd = sim0.poisson_drive
    drive = jnp.asarray(emitter.drive, dtype=dtype)
    drive_2d = jnp.broadcast_to(drive, (steps, n))
    rows = jnp.stack([
        drive_2d + jnp.asarray(_make_poisson_drive(
            steps, n, float(pd.get("rate_hz", 2.0)), float(pd.get("amplitude", 0.5)),
            sim0.dt_ms, poisson_chunk_seed(s + 7919, 0), "all",
            int(pd.get("counts", 1)),
            target_indices=resolve_poisson_drive_target(model, "all")), dtype=dtype)
        for s in seeds])
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    vmapped = jax.jit(jax.vmap(lambda dk: simulate_lif_edge(
        params, n_steps=steps, dt_ms=sim0.dt_ms,
        edge_pre=edges["pre"], edge_post=edges["post"], edge_weight=edges["weight"],
        edge_delay_steps=edges["delay_steps"], edge_group=edges.get("group"),
        I_ext_pA=dk[0], record_v=True, key=dk[1])))((rows, keys))
    batched = simulate_lif_seed_batch(model, sim0, seeds)
    worst = 0.0
    for k, s in enumerate(seeds):
        ref = model.simulate(_lif_seed_batch_sim(seed=s))
        assert np.array_equal(np.asarray(vmapped["spikes"][k]), np.asarray(ref.spikes))
        np.testing.assert_allclose(np.asarray(vmapped["V_mV"][k]), np.asarray(ref.V_m),
                                   rtol=V_RTOL, atol=V_ATOL_MV)
        assert np.array_equal(np.asarray(vmapped["spikes"][k]),
                              np.asarray(batched["spikes"][k]))
        worst = max(worst, float(np.max(np.abs(
            np.asarray(vmapped["V_mV"][k], dtype=float)
            - np.asarray(batched["V_m"][k], dtype=float)))))
    print(f"lif vmap-vs-batch max |dV| mV: {worst:.3e} evidence: {EVIDENCE}")
    assert worst <= V_ATOL_MV


def test_lif_seed_batch_near_threshold_stays_spike_exact():
    """Sensitive case: a member passing within 0.5 mV of threshold stays exact."""
    model = _spiking_lif_model()
    vth = np.asarray(model.params["emitter"].to_lif_params().v_thresh_mV, dtype=float)
    pick = None
    for amp in (60.0, 100.0, 140.0, 200.0):
        sim = _lif_seed_batch_sim(
            poisson_drive={"rate_hz": 50.0, "amplitude": amp},
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
        probe = simulate_lif_seed_batch(model, sim, [11])
        gap = float(np.min(np.abs(np.asarray(probe["V_m"][0], dtype=float) - vth[None, :])))
        if gap < 0.5:
            pick = (amp, gap)
            break
    assert pick is not None, "fixture has no near-threshold pass"
    amp, gap = pick
    sim = _lif_seed_batch_sim(
        poisson_drive={"rate_hz": 50.0, "amplitude": amp},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    out = simulate_lif_seed_batch(model, sim, [11, 12])
    for k, s in enumerate([11, 12]):
        ref = model.simulate(_lif_seed_batch_sim(
            seed=s, poisson_drive={"rate_hz": 50.0, "amplitude": amp},
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
        assert np.array_equal(np.asarray(out["spikes"][k]), np.asarray(ref.spikes))
        np.testing.assert_allclose(np.asarray(out["V_m"][k]), np.asarray(ref.V_m),
                                   rtol=V_RTOL, atol=V_ATOL_MV)
    print(f"lif near-threshold min |V-vth| mV: {gap:.3e} at amplitude {amp}")


def test_lif_seed_batch_benchmark_compile_steady_and_memory():
    """Benchmark: compile vs steady-state vs warm serial; output bytes pinned."""
    import time
    import tracemalloc

    model = _spiking_lif_model()
    seeds = [11, 12, 13]
    t0 = time.perf_counter()
    out = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), seeds)
    t_compile = time.perf_counter() - t0
    tracemalloc.start()
    t0 = time.perf_counter()
    out = simulate_lif_seed_batch(model, _lif_seed_batch_sim(), seeds)
    t_steady = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    for _ in range(2):
        for s in seeds:
            model.simulate(_lif_seed_batch_sim(seed=s))
    t0 = time.perf_counter()
    for s in seeds:
        model.simulate(_lif_seed_batch_sim(seed=s))
    t_serial = time.perf_counter() - t0
    n_b = len(seeds)
    n_t = int(DUR_MS / DT_MS)
    n_n = 6
    assert np.asarray(out["spikes"]).shape == (n_b, n_t, n_n)
    assert np.asarray(out["V_m"]).shape == (n_b, n_t, n_n)
    print(f"lif bench compile={t_compile:.3f}s steady={t_steady:.3f}s "
          f"serial3={t_serial:.3f}s host_peak_MiB={peak / 2 ** 20:.1f} "
          f"bytes={np.asarray(out['spikes']).nbytes + np.asarray(out['V_m']).nbytes}")
    assert t_steady <= t_serial
