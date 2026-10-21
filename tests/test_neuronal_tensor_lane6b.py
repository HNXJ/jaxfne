"""Lane 6b: close the NeuronalTensor gaps G1/G2/G4/G5/G6/G7/G9.

Covers: lif/glif refused at bridge time as registered-but-unroutable (G1);
per-(area, layer, type) drive declared on NeuronType, bridged as
tensor_triple_drive metadata, applied after baseline drive and before
cell_params, consumed as constant external current on the adex route (G5);
spatial_sigma refused without max_in_degree at build time (with-degree
forwarding pinned), allow_self forwarding, the
with_rule_allow_self_connections helper, and the adex
spatial_sigma refusal (G7/G9); reversal metadata-only invariance, pose-driven
delay changes and velocity pins (G2/G4/G6). Lane 6e: localized sampling via
max_in_degree + spatial_sigma on the current route (both refused on AdEx),
emitter_params refused without an emitter, per-triple param overrides, AdEx
unspecified-connectivity and allow_self+velocity refusals, impossible
allow_self, mixed-pose and non-mm length_unit refusals, plus a canonical
bit-identity pin and a save/load round-trip for the new fields.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.emitters_lif import LIFNetworkParams
from jaxfne.connectivity import (
    delay_steps_from_ms,
    with_rule_allow_self_connections,
)
import jaxfne.connectivity as connectivity

DT_MS = 0.5
DUR_MS = 20.0


def _nt(*args, **kwargs):
    return nt.NeuronType.make(*args, **kwargs)


def _ampa_static(reversal=0.0):
    return nt.StaticParams(dT_ms=5.0, reversal_potentials_mV={"AMPA": reversal})


def _ee_conn(delay_ms=1.0, velocity=None, mechanism="AMPA", **kwargs):
    return nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism=mechanism, static=_ampa_static(),
        delay_ms=delay_ms, conduction_velocity_m_per_s=velocity,
        **kwargs,
    )


def _single_area_tensor(types, connections, n_neurons=6, name="lane6b"):
    return nt.NeuronalTensor(
        name=name,
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=n_neurons, neuron_types=types)],
            inter_connections=connections,
        )],
    )


def _construct(tensor, emitter="izhikevich", **kwargs):
    runtime = nt.RuntimeConfiguration(
        seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter=emitter, **kwargs)
    return jtfne.construct(tensor, runtime)


def _bridge(tensor, emitter="izhikevich"):
    return nt.neuronal_tensor_to_configuration(
        tensor, seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter=emitter)


def _rules(cfg):
    return list(cfg.metadata["circuit"]["connections"])


# G1: glif is registered but not routed by the tensor bridge; lif routes.
def test_lif_tensor_routes_through_the_bridge():
    tensor = _single_area_tensor(
        [_nt("E", emitter="lif"), _nt("PV", emitter="lif")], [_ee_conn()])
    model = _construct(tensor, emitter="lif")
    assert isinstance(model.params["emitter"], LIFNetworkParams)
    assert model.static["lif"]["emitter_family"] == "lif"


def test_glif_tensor_refused_at_bridge_as_unroutable():
    tensor = _single_area_tensor(
        [_nt("E", emitter="glif"), _nt("PV", emitter="glif")], [_ee_conn()])
    with pytest.raises(ValueError, match="not routed"):
        _construct(tensor, emitter="glif")


def test_undeclared_runtime_lif_routes_through_the_bridge():
    tensor = _single_area_tensor([_nt("E"), _nt("PV")], [_ee_conn()])
    model = _construct(tensor, emitter="lif")
    assert isinstance(model.params["emitter"], LIFNetworkParams)


# G5: per-triple drive.
def test_triple_drive_recorded_as_bridge_metadata():
    tensor = _single_area_tensor(
        [_nt("E", drive=9.0), _nt("PV")], [_ee_conn()])
    cfg = _bridge(tensor)
    assert cfg.metadata["tensor_triple_drive"] == [
        {"area": "V1", "layer": "L4", "cell_type": "E", "drive": 9.0}
    ]


def test_undeclared_drives_write_no_metadata():
    cfg = _bridge(_single_area_tensor([_nt("E"), _nt("PV")], [_ee_conn()]))
    assert "tensor_triple_drive" not in cfg.metadata


def test_triple_drive_overrides_the_built_neuron_drive():
    tensor = _single_area_tensor(
        [_nt("E", drive=9.0), _nt("PV")], [_ee_conn()])
    params = _construct(tensor).params["emitter"]
    labels = [str(x) for x in params.labels]
    drives = np.asarray(params.drive, dtype=float)
    assert len(labels) == 6 and set(labels) == {"E", "PV"}
    assert bool((drives[[x == "E" for x in labels]] == 9.0).all())
    assert bool((drives[[x == "PV" for x in labels]] == 3.0).all())


def test_triple_drive_applies_after_baseline_drive():
    tensor = _single_area_tensor([_nt("E", drive=7.0)], [_ee_conn()], n_neurons=4)
    cfg = _bridge(tensor).drive(baseline_drive_by_cell_type={"E": 1.0})
    params = jtfne.construct(cfg).params["emitter"]
    np.testing.assert_allclose(
        np.asarray(params.drive, dtype=float), 7.0, rtol=1e-6)


def test_cell_params_applies_after_triple_drive():
    tensor = _single_area_tensor([_nt("E", drive=7.0)], [_ee_conn()], n_neurons=4)
    cfg = _bridge(tensor).cell_params({"cell_type": "E"}, {"drive": 2.0})
    params = jtfne.construct(cfg).params["emitter"]
    np.testing.assert_allclose(
        np.asarray(params.drive, dtype=float), 2.0, rtol=1e-6)


def test_drive_consumed_on_the_adex_route():
    tensor = _single_area_tensor(
        [_nt("E", emitter="adex", drive=9.0),
         _nt("PV", emitter="adex")], [_ee_conn()])
    model = _construct(tensor, emitter="adex")
    rows = model.static["neuron_metadata"]
    got = np.asarray(model.static["adex"]["triple_drive_pA"], dtype=float)
    assert got.shape == (len(rows),)
    for value, row in zip(got.tolist(), rows):
        assert value == pytest.approx(9.0 if row["cell_type"] == "E" else 0.0)
    from jaxfne._construct_adex import adex_kernel_args

    assert "I_ext_pA" in adex_kernel_args(model, DT_MS)
    plain = _construct(
        _single_area_tensor(
            [_nt("E", emitter="adex"), _nt("PV", emitter="adex")], [_ee_conn()]),
        emitter="adex")
    assert plain.static["adex"]["triple_drive_pA"] is None
    assert "I_ext_pA" not in adex_kernel_args(plain, DT_MS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "5.0", "x"])
def test_drive_validation_refuses_non_finite_reals(bad):
    with pytest.raises(ValueError, match="drive"):
        _nt("E", drive=bad)


def test_drive_validation_refuses_bool_but_accepts_zero():
    with pytest.raises(ValueError, match="drive"):
        _nt("E", drive=True)
    assert _nt("E", drive=0.0).drive == 0.0


# G7/G9 + lane 6e item 1: spatial_sigma, allow_self, max_in_degree.
def test_spatial_sigma_without_degree_refused_at_build():
    """A declared sigma without the fan-in cap has no consumer: refused at build."""
    tensor = _single_area_tensor([_nt("E")], [_ee_conn(spatial_sigma=0.25)],
                                 n_neurons=4)
    with pytest.raises(ValueError, match="spatial_sigma needs max_in_degree"):
        _construct(tensor)
    with_degree = _single_area_tensor(
        [_nt("E")], [_ee_conn(spatial_sigma=0.25, max_in_degree=2)],
        n_neurons=4, name="lane6b-sigma-degree")
    (rule,) = _rules(_bridge(with_degree))
    assert rule["spatial_sigma"] == pytest.approx(0.25)
    assert rule["max_in_degree"] == 2


def _localized_tensor(sigma, degree=3, n_neurons=8, name="lane6e-localized"):
    return _single_area_tensor(
        [_nt("E")], [_ee_conn(max_in_degree=degree, spatial_sigma=sigma)],
        n_neurons=n_neurons, name=name)


def _edge_lengths(model):
    pos = np.asarray(model.params["positions"], dtype=float)
    pre = np.asarray(model.params["edge_list"].pre)
    post = np.asarray(model.params["edge_list"].post)
    return pre, post, np.linalg.norm(pos[pre] - pos[post], axis=1)


def test_localized_sigma_changes_edges_and_shortens_them():
    """A declared sigma is consumed: small vs large sigma differ, small is shorter."""
    pre_s, post_s, len_s = _edge_lengths(_construct(_localized_tensor(0.05)))
    pre_l, post_l, len_l = _edge_lengths(_construct(_localized_tensor(5.0)))
    assert not (np.array_equal(pre_s, pre_l) and np.array_equal(post_s, post_l)), \
        "sigma must change the realized edge set"
    assert float(np.mean(len_s)) < float(np.mean(len_l)), \
        "small-sigma edges are shorter on average"


def test_max_in_degree_routes_rule_off_the_density_path():
    (rule,) = _rules(_bridge(_localized_tensor(0.05, degree=2)))
    assert rule["max_in_degree"] == 2
    assert rule["probability"] is None


def test_max_in_degree_caps_fan_in():
    import collections
    model = _construct(_localized_tensor(None, degree=2, name="lane6e-degree"))
    post = np.asarray(model.params["edge_list"].post).tolist()
    assert max(collections.Counter(post).values()) <= 2
    assert len(post) == 16, "8 posts take the full cap of 2 at default sigma"


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_max_in_degree_validation(bad):
    with pytest.raises(ValueError, match="max_in_degree"):
        _ee_conn(max_in_degree=bad)


def test_area_connection_degree_and_probability_refused_jointly():
    with pytest.raises(ValueError, match="mutually exclusive"):
        nt.AreaConnection(
            source_area="V1", source_layer="L4", source_neuron_type="E",
            target_area="V4", target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(), probability=0.5,
            delay_ms=1.0, max_in_degree=2)


def test_spatial_sigma_area_connection_without_degree_refused_at_build():
    """Same refusal for between-area connections; with-degree still forwards."""
    conn = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(), probability=0.5,
        delay_ms=1.0, spatial_sigma=0.3)
    tensor = nt.NeuronalTensor(
        name="lane6b-sigma",
        areas=[nt.Area(name="V1", layers=[nt.Layer(
            name="L4", n_neurons=4, neuron_types=[_nt("E")])]),
               nt.Area(name="V4", layers=[nt.Layer(
                   name="L4", n_neurons=4, neuron_types=[_nt("E")])])],
        area_connections=[conn])
    with pytest.raises(ValueError, match="spatial_sigma needs max_in_degree"):
        _construct(tensor)
    with_degree = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(),
        delay_ms=1.0, spatial_sigma=0.3, max_in_degree=2)
    tensor.area_connections = [with_degree]
    (rule,) = _rules(_bridge(tensor))
    assert rule["spatial_sigma"] == pytest.approx(0.3)
    assert rule["max_in_degree"] == 2


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "0.2"])
def test_spatial_sigma_validation(bad):
    with pytest.raises(ValueError, match="spatial_sigma"):
        _ee_conn(spatial_sigma=bad)


@pytest.mark.parametrize("bad", ["yes", 1, 0.5])
def test_allow_self_validation_refuses_non_bools(bad):
    with pytest.raises(ValueError, match="allow_self"):
        _ee_conn(allow_self=bad)


def test_allow_self_helper_sets_one_rule_key():
    cfg = _bridge(_single_area_tensor([_nt("E")], [_ee_conn()], n_neurons=4))
    (before,) = _rules(cfg)
    assert "allow_self_connections" not in before
    patched = with_rule_allow_self_connections(cfg, before["name"], True)
    (after,) = _rules(patched)
    assert after["allow_self_connections"] is True
    assert {k: v for k, v in after.items() if k != "allow_self_connections"} == before
    assert "allow_self_connections" not in _rules(cfg)[0], "helper copies, never mutates"


def test_allow_self_helper_refusals():
    cfg = _bridge(_single_area_tensor([_nt("E")], [_ee_conn()], n_neurons=4))
    (rule,) = _rules(cfg)
    with pytest.raises(ValueError, match="must be a bool"):
        with_rule_allow_self_connections(cfg, rule["name"], "yes")
    with pytest.raises(ValueError, match="exactly once"):
        with_rule_allow_self_connections(cfg, "no_such_rule", True)


def test_allow_self_helper_stays_out_of_all():
    assert "with_rule_allow_self_connections" not in connectivity.__all__
    assert callable(with_rule_allow_self_connections)


def test_allow_self_true_keeps_self_edges():
    dropped = _single_area_tensor([_nt("E")], [_ee_conn()], n_neurons=3,
                                  name="lane6b-drop")
    kept = _single_area_tensor([_nt("E")], [_ee_conn(allow_self=True)],
                               n_neurons=3, name="lane6b-keep")
    base = _construct(dropped).params["edge_list"]
    got = _construct(kept).params["edge_list"]
    assert len(np.asarray(base.pre)) == 9 - 3
    assert len(np.asarray(got.pre)) == 9
    pairs = set(zip(np.asarray(got.pre).tolist(), np.asarray(got.post).tolist()))
    assert all((i, i) in pairs for i in range(3))


def test_allow_self_false_states_the_default():
    tensor = _single_area_tensor([_nt("E")], [_ee_conn(allow_self=False)],
                                 n_neurons=3, name="lane6b-explicit-drop")
    edges = _construct(tensor).params["edge_list"]
    assert len(np.asarray(edges.pre)) == 9 - 3
    assert bool((np.asarray(edges.pre) != np.asarray(edges.post)).all())


def test_spatial_sigma_refused_on_the_adex_route():
    types = [_nt("E", emitter="adex")]
    tensor = _single_area_tensor(types, [_ee_conn(spatial_sigma=0.2)],
                                 n_neurons=2, name="lane6b-adex-sigma")
    with pytest.raises(ValueError, match="max_in_degree"):
        _construct(tensor, emitter="adex")


def test_spatial_sigma_point_one_probability_rule_stays_refused_on_adex():
    conn = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(), probability=0.5,
        delay_ms=1.0, spatial_sigma=0.1)
    tensor = nt.NeuronalTensor(
        name="lane6b-adex-sigma01",
        areas=[nt.Area(name="V1", layers=[nt.Layer(
            name="L4", n_neurons=2,
            neuron_types=[_nt("E", emitter="adex")])]),
               nt.Area(name="V4", layers=[nt.Layer(
                   name="L4", n_neurons=2,
                   neuron_types=[_nt("E", emitter="adex")])])],
        area_connections=[conn])
    with pytest.raises(ValueError, match="max_in_degree"):
        _construct(tensor, emitter="adex")


def test_allow_self_forwarded_on_the_adex_route():
    def tensor(allow_self):
        return _single_area_tensor(
            [_nt("E", emitter="adex")],
            [_ee_conn(allow_self=allow_self)] if allow_self is not None else [_ee_conn()],
            n_neurons=2, name="lane6b-adex-self")
    assert len(_construct(tensor(None), emitter="adex").static["adex"]["pre"]) == 2
    assert len(_construct(tensor(True), emitter="adex").static["adex"]["pre"]) == 4


def test_adex_allow_self_with_velocity_refused():
    """A zero-distance self edge realizes to 0 delay steps: refused on AdEx."""
    tensor = _single_area_tensor(
        [_nt("E", emitter="adex")],
        [nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(),
            conduction_velocity_m_per_s=1.0, allow_self=True)],
        n_neurons=4, name="lane6b-adex-velocity-self")
    with pytest.raises(ValueError, match="0 delay steps"):
        _construct(tensor, emitter="adex")


def test_izhi_allow_self_with_velocity_keeps_zero_step_self_edges():
    """Documented Izhikevich behaviour: self pairs keep a 0-step delay."""
    tensor = _single_area_tensor(
        [_nt("E")],
        [nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(),
            conduction_velocity_m_per_s=1.0, allow_self=True)],
        n_neurons=3, name="lane6b-izhi-velocity-self")
    steps = np.asarray(_construct(tensor).params["edge_list"].delay_steps).tolist()
    assert len(steps) == 9
    assert sorted(steps) == [0, 0, 0, 1, 1, 1, 1, 1, 1]


def _two_layer_tensor(conn, name="lane6e-self"):
    return nt.NeuronalTensor(
        name=name,
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=2, neuron_types=[_nt("E")]),
                    nt.Layer(name="L23", n_neurons=2, neuron_types=[_nt("E")])],
            inter_connections=[conn],
        )],
    )


def test_impossible_allow_self_refused_across_layers():
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L23", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(), delay_ms=1.0, allow_self=True)
    with pytest.raises(ValueError, match="no self pair"):
        _construct(_two_layer_tensor(conn))


def test_impossible_allow_self_refused_across_types():
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA", static=_ampa_static(), delay_ms=1.0, allow_self=True)
    tensor = nt.NeuronalTensor(
        name="lane6e-self-type",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=4,
                             neuron_types=[_nt("E"), _nt("PV")])],
            inter_connections=[conn],
        )],
    )
    with pytest.raises(ValueError, match="no self pair"):
        _construct(tensor)


def test_impossible_allow_self_refused_across_areas():
    conn = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(), delay_ms=1.0, allow_self=True)
    tensor = nt.NeuronalTensor(
        name="lane6e-self-area",
        areas=[nt.Area(name="V1", layers=[nt.Layer(
            name="L4", n_neurons=2, neuron_types=[_nt("E")])]),
               nt.Area(name="V4", layers=[nt.Layer(
                   name="L4", n_neurons=2, neuron_types=[_nt("E")])])],
        area_connections=[conn])
    with pytest.raises(ValueError, match="no self pair"):
        _construct(tensor, emitter="izhikevich")


@pytest.mark.parametrize("unit", ["um", "m", "ft"])
def test_non_mm_length_unit_refused(unit):
    with pytest.raises(ValueError, match="length_unit"):
        _ee_conn(length_unit=unit)


# G2/G4/G6 pins: reversal metadata-only, pose-driven delays, velocity rule median.
def test_reversal_is_metadata_only_on_the_current_route():
    def tensor(reversal):
        conn = nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(reversal=reversal),
            delay_ms=1.0)
        return _single_area_tensor([_nt("E"), _nt("PV")], [conn], n_neurons=6)
    lo, hi = _bridge(tensor(0.0)), _bridge(tensor(25.0))
    assert lo.metadata["circuit"]["mechanisms"][0]["params"]["reversal_mV"] == 0.0
    assert hi.metadata["circuit"]["mechanisms"][0]["params"]["reversal_mV"] == 25.0
    sig_lo = jtfne.simulate(_construct(tensor(0.0)))
    sig_hi = jtfne.simulate(_construct(tensor(25.0)))
    assert np.array_equal(np.asarray(sig_lo.spikes), np.asarray(sig_hi.spikes))


def test_pose_translation_grows_velocity_delays():
    def tensor(offset):
        area = lambda name, dx: nt.Area(
            name=name, layers=[nt.Layer(
                name="L4", n_neurons=4, neuron_types=[_nt("E")])],
            pose=nt.Pose3D(translation=(dx, 0.0, 0.0)))
        conn = nt.AreaConnection(
            source_area="V1", source_layer="L4", source_neuron_type="E",
            target_area="V4", target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(),
            conduction_velocity_m_per_s=1.0)
        return nt.NeuronalTensor(
            name="lane6b-pose", areas=[area("V1", 0.0), area("V4", offset)],
            area_connections=[conn])
    near = np.asarray(_construct(tensor(10.0)).params["edge_list"].delay_steps)
    far = np.asarray(_construct(tensor(30.0)).params["edge_list"].delay_steps)
    assert int(np.max(far)) > int(np.max(near))


def test_velocity_delays_match_pairwise_distance():
    mk = lambda name, dx: nt.Area(
        name=name, layers=[nt.Layer(
            name="L4", n_neurons=4, neuron_types=[_nt("E")])],
        pose=nt.Pose3D(translation=(dx, 0.0, 0.0)))
    conn = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(),
        conduction_velocity_m_per_s=2.0)
    tensor = nt.NeuronalTensor(
        name="lane6b-velocity",
        areas=[mk("V1", 0.0), mk("V4", 10.0)],
        area_connections=[conn])
    model = _construct(tensor)
    pos = np.asarray(model.params["positions"], dtype=float)
    pre = np.asarray(model.params["edge_list"].pre)
    post = np.asarray(model.params["edge_list"].post)
    dist = np.linalg.norm(pos[pre] - pos[post], axis=1)
    expected = [delay_steps_from_ms(d / 2.0, DT_MS) for d in dist.tolist()]
    assert np.asarray(model.params["edge_list"].delay_steps).tolist() == expected


def test_zero_gain_edge_refused_on_the_adex_route():
    """A declared edge realizing to 0 nS carries no conductance: refused, not compiled."""
    conn = _ee_conn()
    conn.plastic.w_mech = 0.0
    tensor = _single_area_tensor(
        [_nt("E", emitter="adex")], [conn],
        n_neurons=2, name="lane6b-adex-zero-gain")
    with pytest.raises(ValueError, match="0 nS"):
        _construct(tensor, emitter="adex")


def test_zero_conductance_edge_refused_on_the_adex_route():
    """g_mech = 0 for the connection's mechanism likewise realizes to 0 nS: refused."""
    conn = _ee_conn()
    conn.static.g_mech = {"AMPA": 0.0}
    tensor = _single_area_tensor(
        [_nt("E", emitter="adex")], [conn],
        n_neurons=2, name="lane6b-adex-zero-conductance")
    with pytest.raises(ValueError, match="0 nS"):
        _construct(tensor, emitter="adex")


# Lane 6e: existing callers stay bit-identical; new fields round-trip.
def test_canonical_default_column_build_is_bit_identical():
    import hashlib
    tensor = jtfne.load_canonical_neuronal_tensor("default-column")
    model = _construct(tensor)
    edge = model.params["edge_list"]
    digest = hashlib.sha256()
    for arr in (np.asarray(edge.pre), np.asarray(edge.post),
                np.asarray(edge.delay_steps),
                np.asarray(model.params["positions"])):
        digest.update(np.ascontiguousarray(arr).tobytes())
    assert len(np.asarray(edge.pre)) == 3200
    assert digest.hexdigest() == \
        "716b415fa8d939b2534c5e9b075ffd048029db40e858cc385abd6abd00a3d607"


def test_new_connection_fields_survive_save_load(tmp_path):
    inter = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(), delay_ms=1.0,
        spatial_sigma=0.25, max_in_degree=2, allow_self=True)
    area_conn = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(), delay_ms=1.0,
        spatial_sigma=0.3, allow_self=False)
    tensor = nt.NeuronalTensor(
        name="lane6e-roundtrip",
        areas=[nt.Area(name="V1", layers=[nt.Layer(
            name="L4", n_neurons=2, neuron_types=[_nt("E")])],
            inter_connections=[inter]),
               nt.Area(name="V4", layers=[nt.Layer(
                   name="L4", n_neurons=2, neuron_types=[_nt("E")])])],
        area_connections=[area_conn])
    path = tmp_path / "tensor.json"
    nt.save_neuronal_tensor(tensor, path)
    back = nt.load_neuronal_tensor(path)
    (got_inter,) = back.areas[0].inter_connections
    assert got_inter.spatial_sigma == pytest.approx(0.25)
    assert got_inter.max_in_degree == 2
    assert got_inter.allow_self is True
    (got_area,) = back.area_connections
    assert got_area.spatial_sigma == pytest.approx(0.3)
    assert got_area.max_in_degree is None
    assert got_area.allow_self is False
