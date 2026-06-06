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
    assert r.diagnostics["field_solver_status"] == "laminar_proxy_no_pde"
    assert r.diagnostics["physical_amplitude_claim_allowed"] is False
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
