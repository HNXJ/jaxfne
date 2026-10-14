"""The (h, y) = f(x, h') doctrine on the NeuronalTensor path.

Every model is written in the form (h, y) = f(x, h') on a neural tensor A
(x : A : y): declared dynamics equal executed dynamics, geometry is part of
A, and every space-dependent term reads it from A, so A -> A' changes the
dynamics exactly through those terms (RESIDUAL_PROTOCOL.md, Engine).
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt

DT_MS = 0.5
DUR_MS = 20.0


def _nt(*args, **kwargs):
    return nt.NeuronType.make(*args, **kwargs)


def _ampa_static(reversal=0.0):
    return nt.StaticParams(dT_ms=5.0, reversal_potentials_mV={"AMPA": reversal})


def _ee_conn(**kwargs):
    return nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(),
        delay_ms=1.0, **kwargs,
    )


def _single_area_tensor(types, connections, n_neurons=6, name="hyx"):
    return nt.NeuronalTensor(
        name=name,
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=n_neurons, neuron_types=types)],
            inter_connections=connections,
        )],
    )


def _construct(tensor, emitter="izhikevich"):
    runtime = nt.RuntimeConfiguration(
        seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter=emitter)
    return jtfne.construct(tensor, runtime)


def test_declared_but_unbuildable_dynamics_refused_at_bridge():
    """A lif declaration cannot execute: the bridge refuses it, never a late kernel."""
    tensor = _single_area_tensor(
        [_nt("E", emitter="lif"), _nt("PV", emitter="lif")], [_ee_conn()])
    with pytest.raises(ValueError, match="not routed"):
        _construct(tensor, emitter="lif")


def test_declared_drive_is_the_executed_drive():
    """NeuronType.drive overrides the emitter default for exactly its triple."""
    tensor = _single_area_tensor(
        [_nt("E", drive=9.0), _nt("PV")], [_ee_conn()])
    params = _construct(tensor).params["emitter"]
    labels = [str(x) for x in params.labels]
    drives = np.asarray(params.drive, dtype=float)
    assert bool((drives[[x == "E" for x in labels]] == 9.0).all())
    assert bool((drives[[x == "PV" for x in labels]] == 3.0).all())
    signals = jtfne.simulate(_construct(tensor))
    assert bool(np.isfinite(np.asarray(signals.V_m)).all())


def test_geometry_change_moves_dynamics_only_through_delays():
    """A -> A' (a pose translation) changes the dynamics through the delay term."""
    def tensor(offset):
        mk = lambda name, dx: nt.Area(
            name=name, layers=[nt.Layer(
                name="L4", n_neurons=4, neuron_types=[_nt("E")])],
            pose=nt.Pose3D(translation=(dx, 0.0, 0.0)))
        conn = nt.AreaConnection(
            source_area="V1", source_layer="L4", source_neuron_type="E",
            target_area="V4", target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(),
            conduction_velocity_m_per_s=1.0)
        return nt.NeuronalTensor(
            name="hyx-pose", areas=[mk("V1", 0.0), mk("V4", offset)],
            area_connections=[conn])
    near = _construct(tensor(10.0)).params["edge_list"]
    far = _construct(tensor(30.0)).params["edge_list"]
    assert np.array_equal(np.asarray(near.pre), np.asarray(far.pre))
    assert np.array_equal(np.asarray(near.post), np.asarray(far.post))
    assert int(np.max(np.asarray(far.delay_steps))) > int(np.max(np.asarray(near.delay_steps)))


def test_single_self_exciting_neuron_is_a_model():
    """N = 1: one area, one layer, one connection builds and simulates finite."""
    tensor = _single_area_tensor([_nt("E")], [_ee_conn()], n_neurons=1, name="hyx-n1")
    model = _construct(tensor)
    assert len(np.asarray(model.params["edge_list"].pre)) == 0
    signals = jtfne.simulate(model)
    assert bool(np.isfinite(np.asarray(signals.V_m)).all())
    driven = model.simulate(jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0}))
    assert int(np.sum(np.asarray(driven.spikes))) > 0


def test_same_declaration_same_seed_same_signals():
    """(h, y) = f(x, h'): one declaration plus one seed gives one trajectory."""
    tensor = _single_area_tensor([_nt("E"), _nt("PV")], [_ee_conn()], name="hyx-det")
    first = jtfne.simulate(_construct(tensor))
    second = jtfne.simulate(_construct(tensor))
    assert np.array_equal(np.asarray(first.spikes), np.asarray(second.spikes))
    assert bool(np.isfinite(np.asarray(first.V_m)).all())
    assert np.asarray(first.spikes).shape == np.asarray(second.spikes).shape
    assert np.asarray(first.spikes).size > 0, "fixture must emit a signal array"
