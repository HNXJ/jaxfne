"""Lane 6: geometry drives dynamics on the NeuronalTensor path.

Covers: per-NeuronType emitter dispatch (uniform adex tensors route to the
conductance builder with harvested rows, posed positions and per-triple param
overrides), geometry-derived per-edge delays on both families (constant-delay
edges untouched), cross-family position equality, the N=1 self-edge drop on
both families, and refusal of unknown/partial/heterogeneous emitter
declarations, emitter_params without an emitter, unconsumed params, missing
adex delays/reversals, unspecified AdEx connectivity, and mixed
declared/undeclared area poses. Lane 6f: localized sampling reads the posed
layout (one area's translation rewires; a whole-tensor rigid motion preserves
the edge set, within-area included).
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.connectivity import delay_steps_from_ms

DT_MS = 0.5
DUR_MS = 20.0

DRIVE = {
    "n_sources": 200, "rate_hz": 8.0, "weight_nS": 4.0, "mechanism": None,
    "target_cell_types": ["E"], "delay_ms": 1.0, "seed": 7,
}


def _nt(*args, **kwargs):
    return nt.NeuronType.make(*args, **kwargs)


def _adex_types(names=("E", "PV")):
    return [_nt(name, emitter="adex") for name in names]


def _ampa_static(reversal=0.0):
    return nt.StaticParams(dT_ms=5.0, reversal_potentials_mV={"AMPA": reversal})


def _ee_conn(delay_ms=1.0, velocity=None, mechanism="AMPA"):
    return nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism=mechanism, static=_ampa_static(),
        delay_ms=delay_ms, conduction_velocity_m_per_s=velocity,
    )


def _single_area_tensor(types, connections, n_neurons=6, name="lane6"):
    return nt.NeuronalTensor(
        name=name,
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=n_neurons, neuron_types=types)],
            inter_connections=connections,
        )],
    )


def _adex_tensor(**kwargs):
    return _single_area_tensor(_adex_types(), [_ee_conn()], **kwargs)


def _construct(tensor, emitter="adex", **kwargs):
    runtime = nt.RuntimeConfiguration(
        seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter=emitter, **kwargs)
    return jtfne.construct(tensor, runtime)


def _drive_sim(model, **kwargs):
    drive = dict(DRIVE, mechanism=model.static["adex"]["receptor_names"][0])
    drive.update(kwargs)
    return jtfne.Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, poisson_drive=drive)


def _quiet_adex_sim():
    return jtfne.Simulation(duration_ms=DUR_MS, dt_ms=DT_MS)


# AdEx routing and harvest.
def test_adex_tensor_routes_to_conductance_with_harvested_rows():
    model = _construct(_adex_tensor())
    assert model.params["edge_list"] is None
    st = model.static["adex"]
    assert len(st["pre"]) == 9 - 3, "3 E cells fully wired minus dropped self-edges"
    summary = model.static["neuron_metadata_summary"]
    assert summary["areas"] == ["V1"]
    assert summary["layers"] == ["L4"]
    assert summary["cell_types"] == ["E", "PV"]
    rows = model.static["neuron_metadata"]
    assert [r["area"] for r in rows] == ["V1"] * 6
    positions = np.asarray(model.params["positions"])
    assert bool(np.isfinite(positions).all())
    assert float(np.abs(positions).sum()) > 0, "harvested positions, not zeros"
    assert st["decay"] == "exp"
    assert st["step_order"] == "default"


def test_som_harvest_keeps_adex_labels():
    types = [_nt("E", emitter="adex"), _nt("SOM", emitter="adex")]
    model = _construct(_single_area_tensor(types, [_ee_conn()]))
    rows = model.static["neuron_metadata"]
    assert sorted({r["cell_type"] for r in rows}) == ["E", "SOM"]
    assert len(model.static["adex"]["pre"]) == 6


def test_adex_tensor_drive_evokes_spikes():
    model = _construct(_adex_tensor())
    quiet = model.simulate(_quiet_adex_sim())
    assert int(np.sum(np.asarray(quiet.spikes))) == 0, "fixture runs quiet with no drive"
    signals = model.simulate(_drive_sim(model))
    assert int(np.sum(np.asarray(signals.spikes))) > 0, "the drive must spike the fixture"


def test_adex_per_triple_overrides_reach_the_kernel_params():
    types = [_nt("E", emitter="adex", emitter_params={"C_pF": 150.0}),
             _nt("PV", emitter="adex")]
    model = _construct(_single_area_tensor(types, [_ee_conn()]))
    params = model.params["emitter"]
    labels = np.array(list(params.labels))
    np.testing.assert_allclose(np.asarray(params.C_pF)[labels == "E"], 150.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(params.C_pF)[labels == "PV"], 200.0, rtol=1e-6)


def test_cross_family_position_equality():
    plain = _single_area_tensor([_nt("E"), _nt("PV")], [_ee_conn()])
    adex = _adex_tensor()
    pos_plain = np.asarray(_construct(plain, emitter="izhikevich").params["positions"])
    pos_adex = np.asarray(_construct(adex).params["positions"])
    assert bool((pos_plain == pos_adex).all()), "one harvest, one pose, both families"


# Geometry-derived delays.
def _two_area_velocity_tensor(emitter, velocity=1.0, delay_ms=None):
    def area(name, pose=None):
        types = [nt.NeuronType.make("E", **({"emitter": "adex"} if emitter == "adex" else {}))]
        return nt.Area(name=name, layers=[nt.Layer(
            name="L4", n_neurons=4, neuron_types=types)], **({"pose": pose} if pose else {}))
    conn = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(),
        delay_ms=delay_ms, conduction_velocity_m_per_s=velocity)
    return nt.NeuronalTensor(
        name="lane6-velocity",
        areas=[area("V1", pose=nt.Pose3D()),
               area("V4", pose=nt.Pose3D(translation=(10.0, 0.0, 0.0)))],
        area_connections=[conn])


def _expected_velocity_steps(model, velocity=1.0):
    pos = np.asarray(model.params["positions"])
    pre = np.asarray(model.params["edge_list"].pre)
    post = np.asarray(model.params["edge_list"].post)
    dist = np.linalg.norm(pos[pre] - pos[post], axis=1)
    return [delay_steps_from_ms(d / velocity, DT_MS) for d in dist.tolist()]


def test_velocity_delays_current_route_match_pairwise_distance():
    model = _construct(_two_area_velocity_tensor(emitter="izhikevich"),
                       emitter="izhikevich")
    steps = np.asarray(model.params["edge_list"].delay_steps).tolist()
    assert steps == _expected_velocity_steps(model)
    assert len(set(steps)) > 1, "geometry spreads the delays"


def test_velocity_delays_leave_constant_edges_untouched():
    tensor = _two_area_velocity_tensor(emitter="izhikevich")
    tensor.areas[0].inter_connections = [_ee_conn(delay_ms=2.0)]
    tensor.areas[0].connectivity_mode = "explicit"
    model = _construct(tensor, emitter="izhikevich")
    steps = np.asarray(model.params["edge_list"].delay_steps)
    rows = model.static["neuron_metadata"]
    areas = [r["area"] for r in rows]
    pre = np.asarray(model.params["edge_list"].pre).tolist()
    post = np.asarray(model.params["edge_list"].post).tolist()
    for idx, (a, b) in enumerate(zip(pre, post)):
        if areas[a] == "V1" and areas[b] == "V1":
            assert steps[idx] == delay_steps_from_ms(2.0, DT_MS)
        else:
            pos = np.asarray(model.params["positions"])
            d = float(np.linalg.norm(pos[a] - pos[b]))
            assert steps[idx] == delay_steps_from_ms(d / 1.0, DT_MS)


def test_velocity_delays_adex_route_per_edge_and_rule_median():
    model = _construct(_two_area_velocity_tensor(emitter="adex"))
    st = model.static["adex"]
    delay_ms = np.asarray(st["delay_ms"], dtype=float)
    pos = np.asarray(model.params["positions"])
    dist = np.linalg.norm(pos[st["pre"]] - pos[st["post"]], axis=1)
    np.testing.assert_allclose(delay_ms, dist / 1.0, rtol=1e-6)
    assert len(st["rules"]) == 1
    assert st["rules"][0][1] == pytest.approx(float(np.median(dist / 1.0)))


# N=1 self-edge drop on both families.
def test_n1_self_edge_dropped_current_route():
    tensor = _single_area_tensor([_nt("E")], [_ee_conn()], n_neurons=1, name="lane6-n1")
    model = _construct(tensor, emitter="izhikevich")
    assert len(np.asarray(model.params["edge_list"].pre)) == 0
    signals = jtfne.simulate(model)
    assert bool(np.isfinite(np.asarray(signals.V_m)).all())
    driven = model.simulate(jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0}))
    assert int(np.sum(np.asarray(driven.spikes))) > 0


def test_n1_self_edge_dropped_adex_route():
    tensor = _single_area_tensor([_nt("E", emitter="adex")], [_ee_conn()],
                                 n_neurons=1, name="lane6-n1-adex")
    model = _construct(tensor)
    assert len(model.static["adex"]["pre"]) == 0
    signals = model.simulate(_quiet_adex_sim())
    assert bool(np.isfinite(np.asarray(signals.spikes)).all())


# Refusals.
def test_unknown_emitter_name_refused():
    tensor = _single_area_tensor([_nt("E", emitter="morris_lecar")], [_ee_conn()])
    with pytest.raises(ValueError, match="unknown registered emitter"):
        _construct(tensor, emitter="morris_lecar")


def test_partial_emitter_declaration_refused():
    tensor = _single_area_tensor([_nt("E", emitter="adex"), _nt("PV")], [_ee_conn()])
    with pytest.raises(ValueError, match="partial per-NeuronType"):
        _construct(tensor)


def test_heterogeneous_emitter_declaration_refused():
    tensor = _single_area_tensor(
        [_nt("E", emitter="adex"), _nt("PV", emitter="lif")], [_ee_conn()])
    with pytest.raises(ValueError, match="heterogeneous"):
        _construct(tensor)


def test_runtime_emitter_must_agree_with_declaration():
    with pytest.raises(ValueError, match="runtime asks"):
        _construct(_adex_tensor(), emitter="izhikevich")


def test_unconsumed_emitter_params_refused_on_current_route():
    tensor = _single_area_tensor(
        [_nt("E", emitter="izhikevich", emitter_params={"C_pF": 1.0}),
         _nt("PV", emitter="izhikevich", emitter_params={"C_pF": 1.0})],
        [_ee_conn()])
    with pytest.raises(ValueError, match="does not consume"):
        _construct(tensor, emitter="izhikevich")


def test_adex_tensor_needs_delay_per_rule():
    conn = _ee_conn(delay_ms=None)
    with pytest.raises(ValueError, match="neither delay_ms nor"):
        _construct(_single_area_tensor(_adex_types(), [conn]))


def test_adex_tensor_needs_reversal_per_rule():
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0), delay_ms=1.0)
    with pytest.raises(ValueError, match="reversal"):
        _construct(_single_area_tensor(_adex_types(), [conn]))


def test_delay_and_velocity_together_refused():
    with pytest.raises(ValueError, match="declare one or neither"):
        _ee_conn(delay_ms=1.0, velocity=1.0)


def test_velocity_connection_selecting_no_edge_refused():
    tensor = _two_area_velocity_tensor(emitter="izhikevich")
    tensor.area_connections[0].target_layer = "L5"
    with pytest.raises(ValueError, match="selects no compiled edge"):
        _construct(tensor, emitter="izhikevich")


def test_overlapping_velocity_rules_refused():
    tensor = _two_area_velocity_tensor(emitter="izhikevich")
    extra = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
        conduction_velocity_m_per_s=2.0)
    tensor.area_connections = [tensor.area_connections[0], extra]
    with pytest.raises(ValueError, match="duplicate executable synaptic identities"):
        _construct(tensor, emitter="izhikevich")


def test_non_adex_cell_type_refused_on_adex_route():
    tensor = _single_area_tensor(
        [_nt("E", emitter="adex"), _nt("RS", emitter="adex")], [_ee_conn()])
    with pytest.raises(ValueError, match="unknown AdEx cell type"):
        _construct(tensor)


def test_nondefault_adex_knobs_refused_off_route():
    tensor = _single_area_tensor([_nt("E"), _nt("PV")], [_ee_conn()])
    with pytest.raises(ValueError, match="not consumed"):
        _construct(tensor, emitter="izhikevich", adex_decay="euler")


def test_unknown_adex_decay_refused_on_route():
    with pytest.raises(ValueError, match="adex_decay"):
        _construct(_adex_tensor(), adex_decay="rk4")


# Lane 6e: emitter_params need a declared emitter; overrides apply per triple.
def test_emitter_params_without_emitter_refused():
    with pytest.raises(ValueError, match="emitter_params"):
        _nt("E", emitter_params={"C_pF": 150.0})


def test_per_triple_overrides_do_not_leak_across_layers():
    """One type name in two layers carries two values, one per triple."""
    def conn(layer):
        return nt.InterConnection(
            source_layer=layer, source_neuron_type="E",
            target_layer=layer, target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(), delay_ms=1.0)
    tensor = nt.NeuronalTensor(
        name="lane6e-triple-leak",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(
                        name="L4", n_neurons=2,
                        neuron_types=[_nt("E", emitter="adex",
                                          emitter_params={"C_pF": 150.0})]),
                    nt.Layer(
                        name="L23", n_neurons=2,
                        neuron_types=[_nt("E", emitter="adex",
                                          emitter_params={"C_pF": 250.0})])],
            inter_connections=[conn("L4"), conn("L23")],
        )],
    )
    model = _construct(tensor)
    rows = model.static["neuron_metadata"]
    got = np.asarray(model.params["emitter"].C_pF, dtype=float).tolist()
    assert len(got) == 4
    for value, row in zip(got, rows):
        assert value == pytest.approx(150.0 if row["layer"] == "L4" else 250.0)


# Lane 6e: the conductance route wires only declared connections.
def _unspecified_tensor(types, n_neurons=4, name="lane6e-unspec"):
    return nt.NeuronalTensor(
        name=name,
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=n_neurons, neuron_types=types)],
        )],
    )


def test_adex_unspecified_connectivity_refused():
    tensor = _unspecified_tensor(_adex_types())
    assert tensor.connectivity_mode == "unspecified"
    with pytest.raises(ValueError, match="0 edges"):
        _construct(tensor)
    plain = _construct(_unspecified_tensor([_nt("E")]), emitter="izhikevich")
    assert len(np.asarray(plain.params["edge_list"].pre)) > 0


# Lane 6e: rigid motion holds only when every area pose is explicit.
def _two_area_pose_tensor(v1_pose, v4_pose):
    def area(name, pose):
        kw = {} if pose is None else {"pose": pose}
        return nt.Area(
            name=name,
            layers=[nt.Layer(name="L4", n_neurons=2, neuron_types=[_nt("E")])],
            **kw)
    return nt.NeuronalTensor(
        name="lane6e-pose",
        areas=[area("V1", v1_pose), area("V4", v4_pose)],
        area_connections=[nt.AreaConnection(
            source_area="V1", source_layer="L4", source_neuron_type="E",
            target_area="V4", target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(), delay_ms=1.0)],
    )


def test_mixed_pose_specification_refused():
    tensor = _two_area_pose_tensor(None, nt.Pose3D(translation=(10.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="mixes declared poses"):
        _construct(tensor, emitter="izhikevich")


def test_default_valued_pose_counts_as_declared():
    tensor = _two_area_pose_tensor(None, nt.Pose3D())
    with pytest.raises(ValueError, match="mixes declared poses"):
        _construct(tensor, emitter="izhikevich")


def test_uniform_poses_build_on_both_families():
    for v1, v4 in [(None, None),
                   (nt.Pose3D(), nt.Pose3D(translation=(10.0, 0.0, 0.0)))]:
        tensor = _two_area_pose_tensor(v1, v4)
        izhi = _construct(tensor, emitter="izhikevich")
        assert len(np.asarray(izhi.params["edge_list"].pre)) > 0
        adex_types = _two_area_pose_tensor(v1, v4)
        adex_types.areas[0].layers[0].neuron_types = [_nt("E", emitter="adex")]
        adex_types.areas[1].layers[0].neuron_types = [_nt("E", emitter="adex")]
        adex = _construct(adex_types)
        assert len(adex.static["adex"]["pre"]) > 0


def test_merged_tensors_keep_pose_uniformity():
    left = _single_area_tensor([_nt("E")], [_ee_conn()], n_neurons=2, name="l")
    right = _single_area_tensor([_nt("E")], [_ee_conn()], n_neurons=2, name="r")
    right.areas[0].name = "V2"
    merged = jtfne.merge_neuronal_tensors([left, right], name="lane6e-merged")
    assert len(np.asarray(
        _construct(merged, emitter="izhikevich").params["edge_list"].pre)) > 0


# Lane 6f: localized (max_in_degree) sampling reads the posed layout, so
# geometry moves the realized edges; a rigid motion of the whole tensor
# preserves them. sigma=1.0 over a 3.0 mm cell keeps the whole cloud in one
# sampler bucket, so candidate order is selector order in both layouts.
def _localized_two_area(v1_pose, v4_pose, name="lane6f-posed"):
    mk = lambda nm, pose: nt.Area(
        name=nm, layers=[nt.Layer(name="L4", n_neurons=8, neuron_types=[_nt("E")])],
        pose=pose)
    conn = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(),
        delay_ms=1.0, max_in_degree=3, spatial_sigma=1.0)
    return nt.NeuronalTensor(
        name=name, areas=[mk("V1", v1_pose), mk("V4", v4_pose)],
        area_connections=[conn])


def _localized_one_area(pose, name="lane6f-posed-one"):
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(),
        delay_ms=1.0, max_in_degree=3, spatial_sigma=1.0)
    return nt.NeuronalTensor(
        name=name,
        areas=[nt.Area(name="V1", layers=[nt.Layer(
            name="L4", n_neurons=8, neuron_types=[_nt("E")])],
            pose=pose, inter_connections=[conn])])


def _localized_edges(tensor, seed=0):
    runtime = nt.RuntimeConfiguration(
        seed=seed, duration_ms=DUR_MS, dt_ms=DT_MS, emitter="izhikevich")
    edge = jtfne.construct(tensor, runtime).params["edge_list"]
    return set(zip(np.asarray(edge.pre).tolist(), np.asarray(edge.post).tolist()))


def test_translating_one_area_changes_localized_edges():
    """(a) moving V4 20 mm out changes the realized localized edge set."""
    origin = nt.Pose3D(translation=(0.0, 0.0, 0.0))
    base = _localized_edges(_localized_two_area(origin, origin, name="lane6f-a0"))
    assert len(base) > 0, "fixture must wire edges before the move"
    moved = _localized_edges(_localized_two_area(
        origin, nt.Pose3D(translation=(20.0, 0.0, 0.0)), name="lane6f-a1"))
    assert moved != base, "posed layout feeds the sampler, so the move rewires"


def _rigid_pair():
    # Base: explicit non-default poses on both areas (all-default poses skip
    # origin subtraction, so the base must be explicit for one geometry).
    # Image: the same cloud under a 180-degree rigid motion plus (8, 1.5, 1).
    base = nt.Pose3D(translation=(4.0, 1.5, 1.0))
    moved = nt.Pose3D(rotation_deg=180.0, translation=(8.0, 1.5, 1.0))
    return base, moved


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_whole_tensor_rigid_motion_preserves_localized_edges(seed):
    """(b) one rigid motion of both areas preserves the edge set."""
    base_pose, moved_pose = _rigid_pair()
    base = _localized_edges(
        _localized_two_area(base_pose, base_pose, name="lane6f-b0"), seed=seed)
    moved = _localized_edges(
        _localized_two_area(moved_pose, moved_pose, name="lane6f-b1"), seed=seed)
    assert len(base) > 0, "fixture must wire edges"
    assert moved == base


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_area_rotation_preserves_within_area_localized_edges(seed):
    """(c) the same rigid motion preserves within-area (InterConnection) edges."""
    base_pose, moved_pose = _rigid_pair()
    base = _localized_edges(
        _localized_one_area(base_pose, name="lane6f-c0"), seed=seed)
    moved = _localized_edges(
        _localized_one_area(moved_pose, name="lane6f-c1"), seed=seed)
    assert len(base) > 0, "fixture must wire edges"
    assert moved == base


def _areaconn_delay_ms(cfg):
    rules = [r for r in cfg.metadata["circuit"]["connections"] if r["name"] == "areaconn_0"]
    assert len(rules) == 1, "fixture must bridge exactly one area connection"
    return rules[0]["delay_ms"]


def test_standalone_adex_bridge_reads_posed_distances():
    """The standalone bridge (no harvested pair) poses like construct: moving
    V4 from 10 to 40 mm lengthens the velocity delay."""
    near = _two_area_velocity_tensor(emitter="adex")
    far = _two_area_velocity_tensor(emitter="adex")
    far.areas[1].pose = nt.Pose3D(translation=(40.0, 0.0, 0.0))
    d_near = _areaconn_delay_ms(nt.neuronal_tensor_to_adex_configuration(near, dt_ms=DT_MS))
    d_far = _areaconn_delay_ms(nt.neuronal_tensor_to_adex_configuration(far, dt_ms=DT_MS))
    assert d_near > 0.0
    assert d_far > d_near + 20.0, (d_near, d_far)
