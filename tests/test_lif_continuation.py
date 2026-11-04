"""Full-state continuation for the lif edge-list route.

Long Potjans runs execute in bounded-memory segments: ``simulate(...,
return_state=True)`` returns ``(Signals, LIFContinuationState)`` and
``simulate(..., continuation=state)`` starts the next segment from it.
Global step times and the rule-noise stream continue, so a split run
reproduces the whole run bit-exactly; each continued Poisson chunk derives
its seed from ``poisson_chunk_seed(seed, chunk_index)``. Uncarryable inputs
(event_scan/event_order resolutions) are refused, naming themselves.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne._pipeline import LIFContinuationState
from jaxfne._signals import poisson_chunk_seed

DT_MS = 0.5
DUR_MS = 20.0
SEG_MS = 7.5


def _nt(*args, **kwargs):
    return nt.NeuronType.make(*args, **kwargs)


def _tensor(n_neurons=6, **conn_kwargs):
    return nt.NeuronalTensor(
        name="lif-continuation",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(
                name="L4", n_neurons=n_neurons,
                neuron_types=[_nt("E"), _nt("PV")])],
            inter_connections=[nt.InterConnection(
                source_layer="L4", source_neuron_type="E",
                target_layer="L4", target_neuron_type="E",
                mechanism="AMPA",
                static=nt.StaticParams(dT_ms=5.0),
                **conn_kwargs,
            )],
        )],
    )


def _lif_model(tensor=None):
    return jtfne.construct(
        _tensor() if tensor is None else tensor,
        nt.RuntimeConfiguration(
            seed=0, dt_ms=DT_MS, emitter="lif", duration_ms=DUR_MS))


def _rt():
    return jtfne.RuntimeConfig(recurrent_backend="edge_list")


def _sched(events, n=6):
    return jtfne.stimulus_schedule(events, n)


def _segments():
    full = _sched([
        {"onset_ms": 1.0, "duration_ms": 1.0, "amplitude": 60.0,
         "target_indices": [0]},
        {"onset_ms": 11.0, "duration_ms": 1.0, "amplitude": 60.0,
         "target_indices": [1]},
    ])
    first = _sched([
        {"onset_ms": 1.0, "duration_ms": 1.0, "amplitude": 60.0,
         "target_indices": [0]},
    ])
    second = _sched([
        {"onset_ms": 11.0 - SEG_MS, "duration_ms": 1.0, "amplitude": 60.0,
         "target_indices": [1]},
    ])
    return full, first, second


def test_split_handoff_matches_the_whole_run_bit_exact():
    """A 15+25-step handoff reproduces the 40-step run: V and spikes."""
    model = _lif_model()
    full_sched, first_sched, second_sched = _segments()
    whole = model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
                         runtime=_rt()),
        paradigm=full_sched)
    assert int(np.sum(np.asarray(whole.spikes))) > 0, "fixture must spike"
    first, st0 = model.simulate(
        jtfne.Simulation(duration_ms=SEG_MS, dt_ms=DT_MS, seed=1,
                         runtime=_rt()),
        paradigm=first_sched, return_state=True)
    assert isinstance(st0, LIFContinuationState)
    assert (st0.step_index, st0.chunk_index) == (15, 1)
    assert float(st0.dt_ms) == DT_MS
    second = model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, dt_ms=DT_MS, seed=1,
                         runtime=_rt()),
        paradigm=second_sched, continuation=st0)
    assert np.array_equal(
        np.asarray(whole.spikes),
        np.concatenate([np.asarray(first.spikes),
                        np.asarray(second.spikes)], axis=0))
    assert np.array_equal(
        np.asarray(whole.V_m),
        np.concatenate([np.asarray(first.V_m),
                        np.asarray(second.V_m)], axis=0))


def test_split_handoff_matches_under_a_live_rule():
    """HDP state (w) threads the boundary: a split ruled run reruns exactly."""
    from jaxfne.hdp_rules_pcl import rule_params as pcl_params

    model = _lif_model()
    n_edges = int(np.asarray(model.params["edge_list"].pre).shape[0])
    rp = pcl_params(
        tau=np.full(n_edges, 10.0), w_max=np.full(n_edges, 5.0),
        eta=np.full(n_edges, 0.5), f=np.full(n_edges, 1.0),
        lam=np.full(n_edges, 1.0), norm=np.zeros(n_edges),
        group=np.zeros(n_edges, dtype=np.int32), n_groups=1)
    runtime = jtfne.RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True,
        hdp_params={"hdp_rule": "pcl_stdp", "hdp_rule_params": rp})
    kw = dict(dt_ms=DT_MS, seed=1,
              poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
              runtime=runtime)
    first, st0 = model.simulate(
        jtfne.Simulation(duration_ms=SEG_MS, **kw), return_state=True)
    assert st0.H is not None and st0.aux is not None
    second, st1 = model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, **kw),
        continuation=st0, return_state=True)
    assert (st1.step_index, st1.chunk_index) == (40, 2)
    assert second.metadata["hdp_rule"] == "pcl_stdp"
    rerun, st1b = model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, **kw),
        continuation=st0, return_state=True)
    assert np.array_equal(np.asarray(rerun.spikes), np.asarray(second.spikes))
    assert np.array_equal(np.asarray(rerun.V_m), np.asarray(second.V_m))
    assert np.array_equal(np.asarray(st1b.w), np.asarray(st1.w))


def _pcl_stdp_runtime(n_edges):
    from jaxfne.hdp_rules_pcl import rule_params as pcl_params

    rp = pcl_params(
        tau=np.full(n_edges, 10.0), w_max=np.full(n_edges, 5.0),
        eta=np.full(n_edges, 0.5), f=np.full(n_edges, 1.0),
        lam=np.full(n_edges, 1.0), norm=np.zeros(n_edges),
        group=np.zeros(n_edges, dtype=np.int32), n_groups=1)
    return jtfne.RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True,
        hdp_params={"hdp_rule": "pcl_stdp", "hdp_rule_params": rp})


def test_ruled_split_matches_the_whole_run_bit_exact():
    """A ruled split run with no Poisson drive equals the whole run exactly."""
    from jaxfne._construct_lif import lif_edge_arrays

    model = _lif_model()
    n_edges = int(np.asarray(model.params["edge_list"].pre).shape[0])
    w_init = np.asarray(lif_edge_arrays(model)["weight"], dtype=float)
    runtime = _pcl_stdp_runtime(n_edges)
    kw = dict(dt_ms=DT_MS, seed=1, runtime=runtime)
    whole, stw = model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS, **kw), return_state=True)
    assert int(np.sum(np.asarray(whole.spikes))) > 0, "fixture must spike"
    assert not np.array_equal(np.asarray(stw.w), w_init), "rule must move w"
    first, st0 = model.simulate(
        jtfne.Simulation(duration_ms=SEG_MS, **kw), return_state=True)
    second, st1 = model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, **kw),
        continuation=st0, return_state=True)
    assert np.array_equal(
        np.asarray(whole.spikes),
        np.concatenate([np.asarray(first.spikes),
                        np.asarray(second.spikes)], axis=0))
    assert np.array_equal(
        np.asarray(whole.V_m),
        np.concatenate([np.asarray(first.V_m),
                        np.asarray(second.V_m)], axis=0))
    assert np.array_equal(np.asarray(st1.w), np.asarray(stw.w))
    assert np.array_equal(np.asarray(st1.H), np.asarray(stw.H))


def test_return_state_receipt_reports_the_eager_decision():
    """A continued segment runs eager: the runtime receipt says jit False."""
    model = _lif_model()
    first, st0 = model.simulate(
        jtfne.Simulation(duration_ms=SEG_MS, dt_ms=DT_MS, seed=1,
                         runtime=_rt()),
        paradigm=_segments()[1], return_state=True)
    assert first.metadata["runtime"]["jit"] is False
    second = model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, dt_ms=DT_MS, seed=1,
                         runtime=_rt()),
        paradigm=_segments()[2], continuation=st0)
    assert second.metadata["runtime"]["jit"] is False


def test_ruled_state_with_a_dropped_aux_is_refused_by_name():
    """A ruled state missing its aux carry names 'aux'."""
    model = _lif_model()
    n_edges = int(np.asarray(model.params["edge_list"].pre).shape[0])
    kw = dict(dt_ms=DT_MS, seed=1, runtime=_pcl_stdp_runtime(n_edges))
    _, st0 = model.simulate(
        jtfne.Simulation(duration_ms=SEG_MS, **kw), return_state=True)
    assert st0.aux is not None
    with pytest.raises(ValueError, match="'aux'"):
        model.simulate(
            jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, **kw),
            continuation=st0._replace(aux=None))


def test_chunk0_matches_a_plain_call_bit_exact():
    """The first continued chunk draws the declared seed, like a plain call."""
    model = _lif_model()
    kw = dict(duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
              poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
              runtime=_rt())
    plain = model.simulate(jtfne.Simulation(**kw))
    assert int(np.sum(np.asarray(plain.spikes))) > 0, "fixture must spike"
    first, st0 = model.simulate(jtfne.Simulation(**kw), return_state=True)
    assert np.array_equal(np.asarray(first.spikes), np.asarray(plain.spikes))
    assert np.array_equal(np.asarray(first.V_m), np.asarray(plain.V_m))
    assert (st0.step_index, st0.chunk_index) == (40, 1)


def test_chunk_seeds_derive_deterministically_with_receipts():
    """Later chunks draw their own realization; receipts name seed and chunk."""
    from jaxfne._signals import poisson_chunk_seed as _seed

    assert _seed(11, 0) == 11
    assert _seed(11, -1) == 11
    s1, s2 = _seed(11, 1), _seed(11, 2)
    assert s1 != 11 and s2 != 11 and s1 != s2
    assert _seed(11, 1) == s1
    model = _lif_model()
    kw = dict(dt_ms=DT_MS, seed=1,
              poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
              runtime=_rt())
    first, st0 = model.simulate(
        jtfne.Simulation(duration_ms=SEG_MS, **kw), return_state=True)
    assert first.metadata["poisson_drive"]["chunk_index"] == 0
    assert first.metadata["poisson_drive"]["chunk_seed"] == 7920
    second, st1 = model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, **kw),
        continuation=st0, return_state=True)
    assert not np.array_equal(
        np.asarray(second.spikes), np.asarray(first.spikes))
    receipt = second.metadata["poisson_drive"]
    assert receipt["chunk_index"] == 1
    assert receipt["chunk_seed"] == poisson_chunk_seed(7920, 1)
    assert st1.chunk_index == 2
    rerun, _ = model.simulate(
        jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, **kw),
        continuation=st0, return_state=True)
    assert np.array_equal(np.asarray(rerun.spikes), np.asarray(second.spikes))


def test_wrong_state_type_is_refused_by_name():
    """A non-lif continuation (string, other object) names the lif state."""
    model = _lif_model()
    sim = jtfne.Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, runtime=_rt())
    with pytest.raises(ValueError, match="LIFContinuationState"):
        model.simulate(sim, continuation="no-state")
    with pytest.raises(ValueError, match="LIFContinuationState"):
        model.simulate(sim, continuation=object())
    izhi = jtfne.construct(
        _tensor(),
        nt.RuntimeConfiguration(
            seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter="izhikevich"))
    _, izhi_state = jtfne.simulate(
        izhi, jtfne.Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
                               runtime=_rt()),
        return_state=True)
    assert type(izhi_state).__name__ == "ContinuationState"
    with pytest.raises(ValueError, match="LIFContinuationState"):
        model.simulate(sim, continuation=izhi_state)


def test_wrong_model_state_is_refused_by_shape():
    """A state from another population disagrees with this model by shape."""
    model = _lif_model()
    other = _lif_model(_tensor(n_neurons=4))
    _, st0 = other.simulate(
        jtfne.Simulation(duration_ms=SEG_MS, dt_ms=DT_MS, seed=1,
                         runtime=_rt()),
        paradigm=_sched([
            {"onset_ms": 1.0, "duration_ms": 1.0, "amplitude": 60.0,
             "target_indices": [0]}], n=4),
        return_state=True)
    with pytest.raises(ValueError, match=r"\('V'|V.*shape"):
        model.simulate(
            jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, dt_ms=DT_MS, seed=1,
                             runtime=_rt()),
            continuation=st0)


def test_missing_field_and_grid_mismatch_are_refused():
    """A state with a dropped field, or another grid, names the gap."""
    model = _lif_model()
    _, st0 = model.simulate(
        jtfne.Simulation(duration_ms=SEG_MS, dt_ms=DT_MS, seed=1,
                         runtime=_rt()),
        paradigm=_segments()[1], return_state=True)
    dropped = st0._replace(ring=None)
    with pytest.raises(ValueError, match="missing 'ring'"):
        model.simulate(
            jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, dt_ms=DT_MS, seed=1,
                             runtime=_rt()),
            continuation=dropped)
    with pytest.raises(ValueError, match="dt_ms"):
        model.simulate(
            jtfne.Simulation(duration_ms=DUR_MS - SEG_MS, dt_ms=DT_MS / 2,
                             seed=1, runtime=_rt()),
            continuation=st0)


def test_uncarryable_resolutions_are_refused_by_name():
    """event_scan/event_order declare no handoff: continuation names them."""
    from jaxfne.pcl_topology import build_s1_scan_tensor

    ordered = _lif_model(
        _tensor(local_inhibition={"resolution": "event_order", "rounds": 1}))
    with pytest.raises(ValueError, match="event_order"):
        ordered.simulate(
            jtfne.Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
                             runtime=_rt()),
            return_state=True)
    scan = jtfne.construct(
        build_s1_scan_tensor(ns=2, nc=1),
        nt.RuntimeConfiguration(
            seed=0, dt_ms=DT_MS, emitter="lif", duration_ms=DUR_MS))
    with pytest.raises(ValueError, match="event_scan"):
        scan.simulate(
            jtfne.Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
                             runtime=_rt()),
            return_state=True)


def test_plain_calls_keep_the_baseline_receipt_exactly():
    """A default call records no chunk fields and repeats bit-exactly."""
    model = _lif_model()
    kw = dict(duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
              poisson_drive={"rate_hz": 50.0, "amplitude": 100.0,
                             "counts": 2},
              runtime=_rt())
    first = model.simulate(jtfne.Simulation(**kw))
    receipt = first.metadata["poisson_drive"]
    assert receipt == {"rate_hz": 50.0, "amplitude": 100.0, "seed": 7920,
                       "status": "stochastic_drive_applied", "counts": 2}
    second = model.simulate(jtfne.Simulation(**kw))
    assert np.array_equal(np.asarray(second.spikes), np.asarray(first.spikes))
    assert np.array_equal(np.asarray(second.V_m), np.asarray(first.V_m))
