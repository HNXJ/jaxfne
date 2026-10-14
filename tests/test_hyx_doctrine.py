"""The (h, y) = f(x, h') doctrine on the NeuronalTensor path.

Every model is written in the form (h, y) = f(x, h') on a neural tensor A
(x : A : y): declared dynamics equal executed dynamics, geometry is part of
A, and every space-dependent term reads it from A, so A -> A' changes the
dynamics exactly through those terms (RESIDUAL_PROTOCOL.md, Engine).

Behavior claims below hold on every routed emitter family
(``sorted(ROUTABLE_TENSOR_EMITTER_FAMILIES)``); family-specific accessors
(edge delays, drive current, simulate runtime) branch per family, the
doctrine claim does not. The conductance route differs in one declared
place: reversal is a live dynamics term there, while the current-based
route carries it as metadata only (pinned in lane6b).
"""

import math

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt

DT_MS = 0.5
DUR_MS = 20.0
# The conductance (adex) fixture runs longer: its 5 ms refractory clamp and
# per-edge delays (1 ms self, 10/30 ms geometry) keep every recurrent arrival
# outside a 20 ms window or under the clamp, so a 20 ms run shows bit-identical
# V with and without edges; 60 ms puts the arrivals on non-refractory cells.
ADEX_DUR_MS = 60.0

ROUTABLE = sorted(nt.ROUTABLE_TENSOR_EMITTER_FAMILIES)


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
        seed=0, duration_ms=ADEX_DUR_MS if emitter == "adex" else DUR_MS,
        dt_ms=DT_MS, emitter=emitter)
    return jtfne.construct(tensor, runtime)


def _simulate(model, emitter):
    """Run a tensor-built model to signals on either routed family."""
    if emitter == "adex":
        return jtfne.simulate(
            model, duration_ms=ADEX_DUR_MS, dt_ms=DT_MS,
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    return jtfne.simulate(model)


def _adex_poisson_drive(model, rate_hz=200.0, weight_nS=2.0):
    """A tensor-path poisson drive against the model's own declared receptor."""
    (mech,) = model.static["adex"]["receptor_names"]
    return {
        "n_sources": 4, "rate_hz": rate_hz, "weight_nS": weight_nS,
        "mechanism": mech, "target_cell_types": ["E"],
        "delay_ms": 1.0, "seed": 0,
    }


def _drive_simulate(model, emitter):
    """A spiking run on either family (drives differ; the claim does not)."""
    if emitter == "adex":
        return model.simulate(jtfne.Simulation(
            duration_ms=ADEX_DUR_MS, dt_ms=DT_MS,
            poisson_drive=_adex_poisson_drive(model),
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
    return model.simulate(jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0}))


@pytest.mark.parametrize("runtime_emitter", [*ROUTABLE, "lif"])
def test_declared_but_unbuildable_dynamics_refused_at_bridge(runtime_emitter):
    """A lif declaration cannot execute under any runtime: refused at the bridge, never a late kernel."""
    tensor = _single_area_tensor(
        [_nt("E", emitter="lif"), _nt("PV", emitter="lif")], [_ee_conn()])
    with pytest.raises(ValueError, match="not routed"):
        _construct(tensor, emitter=runtime_emitter)


@pytest.mark.parametrize("emitter", ROUTABLE)
def test_declared_drive_is_the_executed_drive(emitter):
    """NeuronType.drive overrides the emitter default for exactly its triple."""
    tensor = _single_area_tensor(
        [_nt("E", drive=9.0), _nt("PV")], [_ee_conn()])
    model = _construct(tensor, emitter=emitter)
    if emitter == "adex":
        # One declared drive unit reads as one pA of constant external
        # current on every neuron of exactly that triple; undeclared
        # triples carry none.
        rows = model.static["neuron_metadata"]
        got = np.asarray(model.static["adex"]["triple_drive_pA"], dtype=float)
        assert got.shape == (len(rows),)
        for value, row in zip(got.tolist(), rows):
            assert value == pytest.approx(9.0 if row["cell_type"] == "E" else 0.0)
        from jaxfne._construct_adex import adex_kernel_args

        assert "I_ext_pA" in adex_kernel_args(model, DT_MS)
        plain = _construct(
            _single_area_tensor([_nt("E"), _nt("PV")], [_ee_conn()]),
            emitter="adex")
        assert "I_ext_pA" not in adex_kernel_args(plain, DT_MS)
        driven = _simulate(model, emitter)
        idle = _simulate(plain, emitter)
        both_finite = np.isfinite(np.asarray(driven.V_m)).all()
        assert bool(both_finite and np.isfinite(np.asarray(idle.V_m)).all())
        assert float(np.max(np.abs(
            np.asarray(driven.V_m, dtype=float)
            - np.asarray(idle.V_m, dtype=float)))) > 0.0
    else:
        params = model.params["emitter"]
        labels = [str(x) for x in params.labels]
        drives = np.asarray(params.drive, dtype=float)
        assert bool((drives[[x == "E" for x in labels]] == 9.0).all())
        assert bool((drives[[x == "PV" for x in labels]] == 3.0).all())
        signals = _simulate(model, emitter)
        assert bool(np.isfinite(np.asarray(signals.V_m)).all())


@pytest.mark.parametrize("emitter", ROUTABLE)
def test_geometry_change_moves_dynamics_only_through_delays(emitter):
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
    near = _construct(tensor(10.0), emitter=emitter)
    far = _construct(tensor(30.0), emitter=emitter)
    if emitter == "adex":
        near_pre = np.asarray(near.static["adex"]["pre"])
        far_pre = np.asarray(far.static["adex"]["pre"])
        near_post = np.asarray(near.static["adex"]["post"])
        far_post = np.asarray(far.static["adex"]["post"])
        assert np.array_equal(near_pre, far_pre)
        assert np.array_equal(near_post, far_post)
        near_ms = np.asarray(near.static["adex"]["delay_ms"], dtype=float)
        far_ms = np.asarray(far.static["adex"]["delay_ms"], dtype=float)
        assert float(np.max(far_ms)) > float(np.max(near_ms))
    else:
        near, far = near.params["edge_list"], far.params["edge_list"]
        assert np.array_equal(np.asarray(near.pre), np.asarray(far.pre))
        assert np.array_equal(np.asarray(near.post), np.asarray(far.post))
        assert int(np.max(np.asarray(far.delay_steps))) > int(np.max(np.asarray(near.delay_steps)))


@pytest.mark.parametrize("emitter", ROUTABLE)
def test_single_self_exciting_neuron_is_a_model(emitter):
    """N = 1: one area, one layer, one connection builds and simulates finite."""
    tensor = _single_area_tensor([_nt("E")], [_ee_conn()], n_neurons=1, name="hyx-n1")
    model = _construct(tensor, emitter=emitter)
    if emitter == "adex":
        assert len(np.asarray(model.static["adex"]["pre"])) == 0
    else:
        assert len(np.asarray(model.params["edge_list"].pre)) == 0
    signals = _simulate(model, emitter)
    assert bool(np.isfinite(np.asarray(signals.V_m)).all())
    if emitter == "adex":
        driven = model.simulate(jtfne.Simulation(
            duration_ms=ADEX_DUR_MS, dt_ms=DT_MS,
            poisson_drive=_adex_poisson_drive(model, rate_hz=500.0, weight_nS=5.0),
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
    else:
        driven = model.simulate(jtfne.Simulation(
            duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
            poisson_drive={"rate_hz": 50.0, "amplitude": 100.0}))
    assert int(np.sum(np.asarray(driven.spikes))) > 0


@pytest.mark.parametrize("emitter", ROUTABLE)
def test_same_declaration_same_seed_same_signals(emitter):
    """(h, y) = f(x, h'): one declaration plus one seed gives one trajectory."""
    tensor = _single_area_tensor([_nt("E"), _nt("PV")], [_ee_conn()], name="hyx-det")
    first = _simulate(_construct(tensor, emitter=emitter), emitter)
    second = _simulate(_construct(tensor, emitter=emitter), emitter)
    assert np.array_equal(np.asarray(first.spikes), np.asarray(second.spikes))
    assert bool(np.isfinite(np.asarray(first.V_m)).all())
    assert np.asarray(first.spikes).shape == np.asarray(second.spikes).shape
    assert np.asarray(first.spikes).size > 0, "fixture must emit a signal array"


def test_reversal_moves_dynamics_on_the_conductance_route():
    """Reversal is a live dynamics term on the conductance route.

    Adex-only by design: the current-based route carries reversal as
    metadata only (pinned in lane6b), so only the conductance leg can
    show a reversal-driven trajectory change.
    """
    def tensor(reversal):
        conn = nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(reversal=reversal),
            delay_ms=1.0)
        return _single_area_tensor(
            [_nt("E", emitter="adex")], [conn], n_neurons=4,
            name="hyx-reversal")
    lo, hi = (
        _construct(tensor(reversal), emitter="adex") for reversal in (0.0, 25.0))
    assert (lo.static["adex"]["receptor_reversal_mV"]
            != hi.static["adex"]["receptor_reversal_mV"])
    sig_lo = _drive_simulate(lo, "adex")
    sig_hi = _drive_simulate(hi, "adex")
    assert bool(np.isfinite(np.asarray(sig_lo.V_m)).all())
    assert bool(np.isfinite(np.asarray(sig_hi.V_m)).all())
    assert (int(np.sum(np.asarray(sig_lo.spikes)))
            + int(np.sum(np.asarray(sig_hi.spikes)))) > 0, "fixture must spike"
    assert float(np.max(np.abs(
        np.asarray(sig_lo.V_m, dtype=float)
        - np.asarray(sig_hi.V_m, dtype=float)))) > 0.0


def _rigid_pose(pose, theta_deg, tx, ty):
    """Rigid-motion image of a plane-xy pose: rotation grows, T' = Rz.@T + t."""
    theta = math.radians(theta_deg)
    ox, oy, oz = pose.translation
    return nt.Pose3D(
        plane="xy", rotation_deg=pose.rotation_deg + theta_deg,
        translation=(math.cos(theta) * ox - math.sin(theta) * oy + tx,
                     math.sin(theta) * ox + math.cos(theta) * oy + ty, oz))


def _two_area_velocity_tensor(xform=None):
    poses = [nt.Pose3D(translation=(0.0, 0.0, 0.0)),
             nt.Pose3D(translation=(10.0, 0.0, 0.0))]
    if xform is not None:
        poses = [_rigid_pose(pose, *xform) for pose in poses]
    areas = [
        nt.Area(name=name, layers=[nt.Layer(
            name="L4", n_neurons=4, neuron_types=[_nt("E")])], pose=pose)
        for name, pose in zip(("V1", "V4"), poses)]
    conn = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(),
        conduction_velocity_m_per_s=1.0)
    return nt.NeuronalTensor(
        name="hyx-rigid", areas=areas, area_connections=[conn])


@pytest.mark.parametrize("emitter", ROUTABLE)
def test_rigid_motion_preserves_dynamics(emitter):
    """A -> A' by one rigid motion changes no space-dependent term: delays, spikes and V agree."""
    base = _construct(_two_area_velocity_tensor(), emitter=emitter)
    moved = _construct(
        _two_area_velocity_tensor(xform=(90.0, 5.0, -3.0)), emitter=emitter)
    pos_base = np.asarray(base.params["positions"], dtype=float)
    pos_moved = np.asarray(moved.params["positions"], dtype=float)
    dist_base = np.linalg.norm(pos_base[:, None, :] - pos_base[None, :, :], axis=-1)
    dist_moved = np.linalg.norm(pos_moved[:, None, :] - pos_moved[None, :, :], axis=-1)
    # float32 pose rounding moves sub-mm pairs at the 1e-6 relative level.
    np.testing.assert_allclose(dist_base, dist_moved, rtol=1e-5, atol=1e-6)
    if emitter == "adex":
        from jaxfne._construct_adex import adex_kernel_args

        delay_base = np.asarray(base.static["adex"]["delay_ms"], dtype=float)
        delay_moved = np.asarray(moved.static["adex"]["delay_ms"], dtype=float)
        np.testing.assert_allclose(delay_base, delay_moved, rtol=1e-5, atol=1e-6)
        steps_base = adex_kernel_args(base, DT_MS)["edge_delay_steps"]
        steps_moved = adex_kernel_args(moved, DT_MS)["edge_delay_steps"]
        assert np.array_equal(np.asarray(steps_base), np.asarray(steps_moved))
    else:
        delay_base = np.asarray(base.params["edge_list"].delay_steps)
        delay_moved = np.asarray(moved.params["edge_list"].delay_steps)
        assert np.array_equal(delay_base, delay_moved)
    sig_base = _drive_simulate(base, emitter)
    sig_moved = _drive_simulate(moved, emitter)
    assert int(np.sum(np.asarray(sig_base.spikes))) > 0, "fixture must spike"
    assert np.array_equal(np.asarray(sig_base.spikes), np.asarray(sig_moved.spikes))
    assert np.array_equal(np.asarray(sig_base.V_m), np.asarray(sig_moved.V_m))


def test_unconsumed_space_term_refused_names_the_missing_consumer():
    """spatial_sigma has no localized sampler on the conductance route: refused, naming max_in_degree."""
    tensor = _single_area_tensor(
        [_nt("E", emitter="adex")], [_ee_conn(spatial_sigma=0.2)],
        n_neurons=2, name="hyx-sigma")
    with pytest.raises(ValueError, match="max_in_degree"):
        _construct(tensor, emitter="adex")


def test_unconsumed_space_term_refused_on_the_current_route():
    """A sigma without the fan-in cap is refused on the current route too: no silent carry."""
    tensor = _single_area_tensor(
        [_nt("E")], [_ee_conn(spatial_sigma=0.2)],
        n_neurons=2, name="hyx-sigma-izhi")
    with pytest.raises(ValueError, match="spatial_sigma needs max_in_degree"):
        _construct(tensor)


def _edge_pairs(model, emitter):
    if emitter == "adex":
        return (np.asarray(model.static["adex"]["pre"]),
                np.asarray(model.static["adex"]["post"]))
    edges = model.params["edge_list"]
    return np.asarray(edges.pre), np.asarray(edges.post)


@pytest.mark.parametrize("emitter", ROUTABLE)
def test_one_self_exciting_neuron_has_its_self_edge_and_feels_it(emitter):
    """N = 1 self-excitation: one i -> i edge, and that edge changes the trajectory."""
    with_self = _construct(_single_area_tensor(
        [_nt("E")], [_ee_conn(allow_self=True)], n_neurons=1, name="hyx-n1-self"),
        emitter=emitter)
    without = _construct(_single_area_tensor(
        [_nt("E")], [_ee_conn()], n_neurons=1, name="hyx-n1-self"), emitter=emitter)
    pre, post = _edge_pairs(with_self, emitter)
    assert pre.tolist() == [0] and post.tolist() == [0]
    assert len(_edge_pairs(without, emitter)[0]) == 0
    sig_self = _drive_simulate(with_self, emitter)
    sig_none = _drive_simulate(without, emitter)
    assert int(np.sum(np.asarray(sig_none.spikes))) > 0, "fixture must spike"
    assert float(np.max(np.abs(
        np.asarray(sig_self.V_m, dtype=float)
        - np.asarray(sig_none.V_m, dtype=float)))) > 0.0


@pytest.mark.parametrize("emitter", ROUTABLE)
def test_geometry_change_changes_the_trajectory(emitter):
    """A -> A': moving V4 from 10 to 30 mm (1 m/s) changes when V1 spikes reach V4.

    Over 60 ms the 10 ms delay delivers V1 spikes well before the 30 ms delay
    does, so V4's membrane trajectory differs; V1's does not.
    """
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
            name="hyx-pose-dyn", areas=[mk("V1", 0.0), mk("V4", offset)],
            area_connections=[conn])
    near = _drive_simulate(_construct(tensor(10.0), emitter=emitter), emitter)
    far = _drive_simulate(_construct(tensor(30.0), emitter=emitter), emitter)
    v_near = np.asarray(near.V_m, dtype=float)
    v_far = np.asarray(far.V_m, dtype=float)
    spikes = np.asarray(near.spikes)
    assert int(np.sum(spikes)) > 0, "fixture must spike"
    assert float(np.max(np.abs(v_near - v_far))) > 0.0
