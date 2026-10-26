"""v0.3.30 connectivity compiler: rule compilation, selectors, sampling, interop."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


NEURONS = [
    {"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E"},
    {"neuron_id": 1, "area": "V1", "layer": "L4", "cell_type": "E"},
    {"neuron_id": 2, "area": "V1", "layer": "L4", "cell_type": "PV"},
    {"neuron_id": 3, "area": "V4", "layer": "L2", "cell_type": "E"},
]
MECHS = [{"name": "ampa", "kind": "synapse", "params": {"tau_ms": 5.0, "sign": 1.0}}]


def _compile(conns, **kw):
    return jtfne.compile_connection_rules(NEURONS, conns, MECHS, **kw)


def test_in_area_selector_resolves():
    r = _compile([{"name": "E_PV", "source": {"cell_type": "E", "area": "V1"},
                   "target": {"cell_type": "PV"}, "weight": 0.1, "mechanism": "ampa"}])
    # source E in V1 = {0,1}; target PV = {2}; all-to-all => 2 edges
    assert r.n_edges == 2
    assert set(int(x) for x in np.asarray(r.edge_pre)) == {0, 1}
    assert set(int(x) for x in np.asarray(r.edge_post)) == {2}


def test_cross_area_selector_resolves():
    r = _compile([{"name": "fb", "source": {"area": "V4"}, "target": {"area": "V1", "cell_type": "PV"},
                   "weight": 0.4, "mechanism": "ampa"}])
    assert set(int(x) for x in np.asarray(r.edge_pre)) == {3}     # V4
    assert set(int(x) for x in np.asarray(r.edge_post)) == {2}    # V1 PV


def test_sparse_sampling_is_deterministic():
    conn = [{"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
             "probability": 0.5, "weight": 0.1, "mechanism": "ampa", "allow_self_connections": True}]
    a = _compile(conn, seed=3)
    b = _compile(conn, seed=3)
    assert np.array_equal(np.asarray(a.edge_pre), np.asarray(b.edge_pre))
    assert np.array_equal(np.asarray(a.edge_post), np.asarray(b.edge_post))


def test_different_seed_changes_sampling():
    conn = [{"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
             "probability": 0.5, "weight": 0.1, "mechanism": "ampa", "allow_self_connections": True}]
    a = _compile(conn, seed=1)
    b = _compile(conn, seed=99)
    # Very likely different edge sets; assert not identical.
    assert not (np.array_equal(np.asarray(a.edge_pre), np.asarray(b.edge_pre))
                and np.array_equal(np.asarray(a.edge_post), np.asarray(b.edge_post)))


def test_self_connections_excluded_by_default():
    conn = [{"name": "rec", "source": {"area": "V1", "cell_type": "E"},
             "target": {"area": "V1", "cell_type": "E"}, "weight": 0.1, "mechanism": "ampa"}]
    r = _compile(conn)  # E in V1 = {0,1}; all-to-all minus self => (0,1),(1,0)
    pairs = set(zip((int(x) for x in np.asarray(r.edge_pre)), (int(x) for x in np.asarray(r.edge_post))))
    assert (0, 0) not in pairs and (1, 1) not in pairs
    assert pairs == {(0, 1), (1, 0)}


def test_self_connections_allowed_when_opted_in():
    conn = [{"name": "rec", "source": {"cell_type": "E", "area": "V1"},
             "target": {"cell_type": "E", "area": "V1"}, "weight": 0.1, "mechanism": "ampa",
             "allow_self_connections": True}]
    r = _compile(conn)
    pairs = set(zip((int(x) for x in np.asarray(r.edge_pre)), (int(x) for x in np.asarray(r.edge_post))))
    assert (0, 0) in pairs and (1, 1) in pairs


def test_empty_selector_hard_error():
    with pytest.raises(ValueError, match="zero neurons"):
        _compile([{"name": "x", "source": {"cell_type": "SST"}, "target": {"cell_type": "E"},
                   "weight": 0.1, "mechanism": "ampa"}])


def test_empty_selector_skipped_when_allowed():
    r = _compile([{"name": "x", "source": {"cell_type": "SST"}, "target": {"cell_type": "E"},
                   "weight": 0.1, "mechanism": "ampa"}], allow_empty=True)
    assert r.n_edges == 0
    assert r.diagnostics["skipped_rules"] == ["x"]


def test_structural_selector_error_not_swallowed_under_allow_empty():
    """A non-mapping selector must raise loudly even with allow_empty=True."""
    with pytest.raises(ValueError, match="must be a mapping"):
        _compile([{"name": "bad", "source": ["cell_type", "E"], "target": {"cell_type": "PV"},
                   "weight": 0.1, "mechanism": "ampa"}], allow_empty=True)


def test_probability_one_yields_full_connectivity():
    """probability=1.0 must produce all-to-all (minus self), not ~63%."""
    conn = [{"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
             "probability": 1.0, "weight": 0.1, "mechanism": "ampa", "allow_self_connections": True}]
    r = _compile(conn)
    n_v1 = sum(1 for row in NEURONS if row["area"] == "V1")  # 3
    assert r.n_edges == n_v1 * n_v1  # full bipartite incl self = 9


def test_probability_hits_target_density_count():
    """Realized unique edge count == round(p * n_pre * n_post) in sparse regime."""
    # Build a larger same-area population for a meaningful density target.
    neurons = [{"neuron_id": i, "area": "X", "layer": "L", "cell_type": "E"} for i in range(10)]
    conn = [{"name": "rec", "source": {"area": "X"}, "target": {"area": "X"},
             "probability": 0.2, "weight": 0.1, "mechanism": "ampa", "allow_self_connections": True}]
    r = jtfne.compile_connection_rules(neurons, conn, MECHS, seed=0)
    assert r.n_edges == round(0.2 * 10 * 10)  # exactly 20, not birthday-reduced


def test_edges_finite_and_diagnostics_json_safe():
    r = _compile([{"name": "E_PV", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"},
                   "weight": 0.1, "mechanism": "ampa"}])
    assert jnp.all(jnp.isfinite(r.edge_weight))
    json.dumps(r.diagnostics, allow_nan=False)
    json.dumps(r.connection_table, allow_nan=False)
    json.dumps(r.mechanism_table, allow_nan=False)


def test_edge_list_interop():
    r = _compile([{"name": "E_PV", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"},
                   "weight": 0.1, "mechanism": "ampa"}])
    el = r.to_edge_list()
    assert el.pre.shape == el.post.shape == el.weight.shape == el.receptor_index.shape
    assert float(el.tau_ms[0]) == 5.0
    assert el.source_calibration_status == "uncalibrated_proxy_compiled"


def test_model_wrapper_equivalence():
    """Model.compile_connections() matches the free function on the model's config."""
    cfg = (
        jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
        .mechanisms(name="ampa", kind="synapse", params={"tau_ms": 5.0, "sign": 1.0})
        .connections(name="E_to_E", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.3, weight=0.01, mechanism="ampa")
    )
    model = jtfne.construct(cfg)
    via_model = model.compile_connections(seed=0)
    circuit = cfg.metadata["circuit"]
    via_free = jtfne.compile_connection_rules(
        model.neuron_table(), circuit["connections"], circuit["mechanisms"], seed=0
    )
    assert np.array_equal(np.asarray(via_model.edge_pre), np.asarray(via_free.edge_pre))
    assert np.array_equal(np.asarray(via_model.edge_weight), np.asarray(via_free.edge_weight))


def test_truth_gates_in_diagnostics():
    r = _compile([{"name": "E_PV", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"},
                   "weight": 0.1, "mechanism": "ampa"}])
    assert r.diagnostics["field_solver_status"] == "linear_solver"
    assert r.diagnostics["physical_amplitude_calibrated"] is False
    assert r.mechanism_table[0]["status"] == "declared_not_simulated"


def test_dense_regime_shortfall_warns_on_self_exclusion():
    """Dense regime with allow_self=False and overlapping populations warns when target can't be met."""
    # V1 has 3 neurons (0, 1, 2): probability=0.9 → target = round(0.9*9) = 8
    # But with allow_self=False, max pairs = 9 - 3 = 6 (excluding diagonal)
    # Dense regime should warn and realize only 6 edges.
    conn = [{
        "name": "rec_high_prob",
        "source": {"area": "V1"},
        "target": {"area": "V1"},
        "probability": 0.9,
        "weight": 0.05,
        "mechanism": "ampa",
        "allow_self_connections": False,  # Exclude self-connections
    }]
    with pytest.warns(UserWarning, match="Dense-regime edge target.*exceeds non-self"):
        r = _compile(conn, seed=0)
    # The compiler should realize exactly 6 edges (3*3 - 3 diagonal)
    assert r.n_edges == 6
    # Verify no self-connections were added
    pre = np.asarray(r.edge_pre, dtype=int)
    post = np.asarray(r.edge_post, dtype=int)
    assert not np.any(pre == post)


def _spread_neurons(n=20):
    return (
        [{"neuron_id": i, "area": "V1", "layer": "L4", "cell_type": "E"}
         for i in range(n)]
        + [{"neuron_id": n + j, "area": "V1", "layer": "L4", "cell_type": "PV"}
           for j in range(n)]
    )


def _compile_delay_spread(delay_ms, delay_std_ms, seed=7, n=20):
    conn = [{"name": "E_PV", "source": {"cell_type": "E"},
             "target": {"cell_type": "PV"}, "weight": 0.1,
             "mechanism": "ampa", "delay_ms": delay_ms,
             "delay_std_ms": delay_std_ms}]
    return jtfne.compile_connection_rules(_spread_neurons(n), conn, MECHS,
                                          seed=seed)


def test_delay_std_spread_snapped_to_grid():
    """G-DDIST (compiler): per-edge Gaussian delays snap half-up to dt steps."""
    import math as _math

    a = _compile_delay_spread(1.5, 0.75, seed=7)
    b = _compile_delay_spread(1.5, 0.75, seed=7)
    d = np.asarray(a.edge_delay_ms, dtype=np.float64)
    assert a.n_edges == 20 * 20, "spread changes no pairing"
    assert np.array_equal(d, np.asarray(b.edge_delay_ms, dtype=np.float64))
    assert bool((d >= 0.0).all()), "folded draws stay non-negative"
    assert bool((d != d[0]).any()), "spread realizes per-edge delays"
    el = a.to_edge_list(dt_ms=0.5)
    steps = np.asarray(el.delay_steps, dtype=np.int64)
    expect = np.array([0 if v == 0.0 else max(1, int(_math.floor(v / 0.5 + 0.5)))
                       for v in d], dtype=np.int64)
    assert np.array_equal(steps, expect), "steps snap half-up, minimum 1"
    assert bool((steps >= 1).all()), "positive draws keep a positive delay"
    assert a.connection_table[0].get("delay_std_ms") == 0.75


def test_delay_std_folds_negative_draws():
    r = _compile_delay_spread(0.5, 1.0, seed=7)
    d = np.asarray(r.edge_delay_ms, dtype=np.float64)
    assert bool((d >= 0.0).all()), "folded draws stay non-negative"
    el = r.to_edge_list(dt_ms=0.5)
    assert bool((np.asarray(el.delay_steps) >= 0).all())


def test_delay_std_requires_delay_ms():
    conn = [{"name": "E_PV", "source": {"cell_type": "E"},
             "target": {"cell_type": "PV"}, "weight": 0.1,
             "mechanism": "ampa", "delay_std_ms": 0.75}]
    with pytest.raises(ValueError, match="delay_std_ms"):
        _compile(conn)


def test_delay_std_negative_refused():
    with pytest.raises(ValueError, match="delay_std_ms"):
        _compile_delay_spread(1.5, -0.1)


def test_spread_keys_refused_on_sign_only_fallback():
    """The sign-only fallback compiler has no spread consumer (fail-closed)."""
    from jaxfne._construct_connectivity import _compile_connection_rules as _fallback

    labels = (["V1"] * 4, ["L4"] * 4, ["E", "E", "PV", "PV"])
    sign = [1.0, 1.0, -1.0, -1.0]
    with pytest.raises(ValueError, match="mechanism-aware"):
        _fallback([{"name": "r", "source": {}, "target": {}, "weight": 0.1,
                    "weight_std": 0.01}], *labels, sign, 4, None, 0, dt_ms=0.5)
    with pytest.raises(ValueError, match="mechanism-aware"):
        _fallback([{"name": "r", "source": {}, "target": {}, "weight": 0.1,
                    "delay_ms": 1.5, "delay_std_ms": 0.1}], *labels, sign, 4,
                   None, 0, dt_ms=0.5)


def _lif_spread_tensor(**kw):
    from jaxfne import neuronal_tensor as nt

    delay_kw = {"delay_ms": 1.5} if kw.pop("with_delay", True) else {}
    return nt.NeuronalTensor(
        name="spread",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=40,
            neuron_types=[
                nt.NeuronType.make("E", fraction=0.5, emitter="lif"),
                nt.NeuronType.make("PV", fraction=0.5, emitter="lif"),
            ])])],
        area_connections=[nt.AreaConnection(
            source_area="mc", source_layer="L23", source_neuron_type="E",
            target_area="mc", target_layer="L23", target_neuron_type="PV",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=1.0,
                                     w_std=kw.pop("w_std", 0.1)),
            sampling="density", probability=0.5,
            delay_std_ms=kw.pop("delay_std_ms", 0.75), **delay_kw,
            **kw)],
    )


def _lif_spread_model(tensor, seed=3):
    import jaxfne as jtfne
    from jaxfne import neuronal_tensor as nt

    return jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=seed, duration_ms=10.0,
                                        dt_ms=0.5, emitter="lif"))


def test_tensor_spread_lif_route_per_edge():
    """G-WDIST/G-DDIST (tensor): lif route realizes per-edge spreads."""
    model = _lif_spread_model(_lif_spread_tensor())
    el = model.params["edge_list"]
    w = np.asarray(el.weight, dtype=np.float64)
    steps = np.asarray(el.delay_steps, dtype=np.int64)
    assert w.size > 0, "fixture must wire edges"
    assert bool((w != w[0]).any()), "weights vary per edge"
    assert bool((w > 0).all()), "E-source weights keep their sign"
    assert bool((steps != steps[0]).any()), "delays vary per edge"
    assert bool((steps >= 1).all()), "spread delays stay positive"
    again = _lif_spread_model(_lif_spread_tensor())
    assert np.array_equal(w, np.asarray(again.params["edge_list"].weight,
                                        dtype=np.float64)), "fixed seed replays"


def test_tensor_spread_refusals():
    from jaxfne import neuronal_tensor as nt

    with pytest.raises(ValueError, match="w_std"):
        _lif_spread_model(_lif_spread_tensor(w_std=-0.1))
    with pytest.raises(ValueError, match="delay_std_ms"):
        _lif_spread_model(_lif_spread_tensor(delay_std_ms=-0.1))
    with pytest.raises(ValueError, match="delay_std_ms"):
        _lif_spread_model(_lif_spread_tensor(with_delay=False))
    with pytest.raises(ValueError, match="delay_std_ms"):
        nt.AreaConnection(
            source_area="mc", source_layer="L23", source_neuron_type="E",
            target_area="mc", target_layer="L23", target_neuron_type="E",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=1.0),
            delay_ms=1.5, delay_std_ms=0.75,
            conduction_velocity_m_per_s=1.0,
            sampling="density", probability=0.5)
