"""23-REP-01: class-shared edge attribute storage and p_connect path gates."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._edge_class_storage import (
    audit_edge_list_storage,
    materialize_edge_list_arrays,
    try_compact_edge_list_class_storage,
)
from jaxfne._model import Model
from jaxfne.core import _SPARSE_DIRECT_N
from jaxfne.emitters import (
    resolve_edge_delay_steps,
    resolve_edge_tau_ms,
    resolve_receptor_index,
)
from scripts.perf.w10_allocation_map import build_config, total_bytes, walk_arrays


def _bounded_model(n=200, seed=1):
    cfg = build_config(
        n=n, p_connect=0.0, max_in_degree=100, duration_ms=10.0, dt_ms=0.5, seed=seed
    )
    return cfg, jtfne.construct(cfg)


def test_construct_compacts_sign_tau_and_uniform_zero_delay():
    _, model = _bounded_model()
    el = model.params["edge_list"]
    assert el.tau_storage == "sign_from_receptor"
    assert el.delay_storage == "uniform_zero"
    assert tuple(np.asarray(el.tau_ms).shape) == (0,)
    assert tuple(np.asarray(el.delay_steps).shape) == (0,)
    assert el.receptor_index_storage == "per_edge_uint8"
    audit = model.static["edge_storage_audit"]
    assert audit["tau_sign_table_compatible"] is True
    assert audit["delay_uniform_zero"] is True
    assert audit["receptor_index_storage"] == "per_edge_uint8"


def test_compact_storage_bit_exact_observables():
    cfg = build_config(
        n=200, p_connect=0.0, max_in_degree=100, duration_ms=10.0, dt_ms=0.5, seed=3
    )
    full = jtfne.construct(cfg)
    el = full.params["edge_list"]
    expanded = materialize_edge_list_arrays(el)
    recompact = try_compact_edge_list_class_storage(expanded)
    before = np.asarray(jtfne.simulate(full, duration_ms=10.0, dt_ms=0.5, seed=2).V_m)
    tau_full = np.asarray(resolve_edge_tau_ms(el, el.weight.dtype))
    tau_exp = np.asarray(resolve_edge_tau_ms(recompact, el.weight.dtype))
    delay_full = np.asarray(resolve_edge_delay_steps(el))
    delay_exp = np.asarray(resolve_edge_delay_steps(recompact))
    np.testing.assert_allclose(tau_full, tau_exp)
    np.testing.assert_array_equal(delay_full, delay_exp)
    np.testing.assert_array_equal(before, np.asarray(
        jtfne.simulate(full, duration_ms=10.0, dt_ms=0.5, seed=2).V_m
    ))


def test_receptor_index_uint8_resolves_bit_exactly():
    _, model = _bounded_model()
    el = model.params["edge_list"]
    assert str(el.receptor_index.dtype) == "uint8"
    ri = np.asarray(resolve_receptor_index(el))
    assert ri.dtype == np.int32
    assert set(np.unique(ri).tolist()) <= {0, 1}
    rows = model.edge_table()
    assert {r["receptor_index"] for r in rows} <= {0, 1}


def _rule_only_cfg(*, mechanisms, connections, n=24, seed=0):
    cfg = (
        jtfne.Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=30.0, dt_ms=0.5)
        .column(name="c", layers=["L4"], n=n)
        .cell_types({"E": 0.5, "PV": 0.5})
        .connectivity(p_connect=0.0)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    for m in mechanisms:
        cfg = cfg.mechanisms(**m)
    for c in connections:
        cfg = cfg.connections(**c)
    return cfg


def test_mechanism_table_tau_not_sign_map_when_tau_differs():
    """Custom declared tau must use mechanism table, not the 2/5 ms sign map."""
    cfg = _rule_only_cfg(
        mechanisms=[{"name": "slow_exc", "kind": "custom", "params": {"tau_ms": 7.0}}],
        connections=[{
            "name": "rec", "source": {}, "target": {}, "mechanism": "slow_exc",
            "weight": 0.03, "max_in_degree": 10, "spatial_sigma": 0.1,
        }],
    )
    model = jtfne.construct(cfg)
    el = model.params["edge_list"]
    assert el.tau_storage == "from_mechanism_table"
    assert el.tau_storage != "sign_from_receptor"
    tau = np.asarray(resolve_edge_tau_ms(el, el.weight.dtype))
    assert set(np.unique(tau).tolist()) == {7.0}


def test_heterogeneous_mechanisms_use_declared_table():
    cfg = _rule_only_cfg(
        mechanisms=[
            {"name": "nmda_exc", "kind": "NMDA", "params": {"tau_ms": 100.0}},
            {"name": "gabaa_legacy", "kind": "GABA_A", "params": {"tau_ms": 5.0}},
        ],
        connections=[
            {
                "name": "e_to_i", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"},
                "mechanism": "nmda_exc", "weight": 0.03, "max_in_degree": 8,
                "spatial_sigma": 0.1,
            },
            {
                "name": "i_to_e", "source": {"cell_type": "PV"}, "target": {"cell_type": "E"},
                "mechanism": "gabaa_legacy", "weight": 0.03, "max_in_degree": 8,
                "spatial_sigma": 0.1,
            },
        ],
    )
    model = jtfne.construct(cfg)
    el = model.params["edge_list"]
    assert el.tau_storage == "from_mechanism_table"
    tau = np.asarray(resolve_edge_tau_ms(el, el.weight.dtype))
    assert 100.0 in np.unique(tau)
    assert 5.0 in np.unique(tau)


def test_adversarial_receptor_out_of_table_blocks_mechanism_tau_compact():
    from jaxfne._edge_class_storage import try_compact_edge_list_class_storage

    el = jtfne.emitters.EdgeList(
        pre=jnp.asarray([0], dtype=jnp.int32),
        post=jnp.asarray([1], dtype=jnp.int32),
        weight=jnp.asarray([0.1], dtype=jnp.float32),
        receptor_index=jnp.asarray([3], dtype=jnp.int32),
        tau_ms=jnp.asarray([2.0], dtype=jnp.float32),
    )
    table = np.asarray([2.0, 5.0], dtype=np.float64)
    compacted = try_compact_edge_list_class_storage(
        el, declared_mechanism_tau_table=table
    )
    assert compacted.tau_storage == "per_edge"


def test_class_compaction_reduces_persistent_tau_and_delay_bytes():
    cfg = build_config(
        n=1000, p_connect=0.0, max_in_degree=100, duration_ms=10.0, dt_ms=0.5, seed=1
    )
    model = jtfne.construct(cfg)
    nbytes = {
        path.split(".")[-1]: int(np.asarray(arr).nbytes)
        for path, arr in walk_arrays(model)
        if "edge_list" in path and path.endswith(("tau_ms", "delay_steps", "receptor_index"))
    }
    assert nbytes.get("tau_ms", 0) == 0
    assert nbytes.get("delay_steps", 0) == 0
    assert nbytes.get("receptor_index", 0) == 100_000
    assert total_bytes(model) < 1_800_000


def test_checkpoint_round_trip_preserves_class_storage(tmp_path):
    cfg, model = _bounded_model(n=120)
    path = tmp_path / "ck"
    model.checkpoint(str(path))
    import json

    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["edge_tau_storage"] == "sign_from_receptor"
    assert meta["edge_delay_storage"] == "uniform_zero"
    restored = Model.restore(str(path), cfg)
    el = restored.params["edge_list"]
    assert el.tau_storage == "sign_from_receptor"
    assert el.delay_storage == "uniform_zero"
    s0 = np.asarray(jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=0).V_m)
    s1 = np.asarray(jtfne.simulate(restored, duration_ms=10.0, dt_ms=0.5, seed=0).V_m)
    np.testing.assert_array_equal(s0, s1)


def test_p_connect_dense_masked_self_consistent():
    """0 < p_connect < 1 on dense path (N below sparse-direct threshold)."""
    n = _SPARSE_DIRECT_N - 1000
    cfg = (
        jtfne.Configuration()
        .runtime(seed=11, dtype="float32", duration_ms=10.0, dt_ms=0.5)
        .column(name="c", layers=["L4"], n=n)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity(p_connect=0.1)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    m1 = jtfne.construct(cfg)
    m2 = jtfne.construct(cfg)
    assert tuple(m1.params["emitter"].W.shape) == (n, n)
    s1 = jtfne.simulate(m1, duration_ms=10.0, dt_ms=0.5, seed=0)
    s2 = jtfne.simulate(m2, duration_ms=10.0, dt_ms=0.5, seed=0)
    np.testing.assert_array_equal(np.asarray(s1.V_m), np.asarray(s2.V_m))
    np.testing.assert_array_equal(np.asarray(s1.spikes), np.asarray(s2.spikes))


def test_p_connect_with_rules_bounded_degree_independent_path():
    """Masked path: p_connect=0 + declared rules (not the sparse-direct Erdos-Renyi path)."""
    cfg = build_config(
        n=300, p_connect=0.0, max_in_degree=40, duration_ms=10.0, dt_ms=0.5, seed=5
    )
    model = jtfne.construct(cfg)
    el = model.params["edge_list"]
    assert model.params["emitter"].W.shape == (0, 0)
    assert el.n_edges > 0
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    in_deg = np.bincount(post, minlength=300)
    assert int(in_deg.max()) <= 40
    sig = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=1)
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())


@pytest.mark.parametrize("n", [_SPARSE_DIRECT_N])
def test_sparse_direct_vs_dense_not_bit_exact_at_threshold(n):
    """REP-03 gate: do not lower _SPARSE_DIRECT_N without new equivalence evidence."""
    import jaxfne._construct_population as pop

    p, seed = 0.02, 7
    old = pop._SPARSE_DIRECT_N

    def _build(threshold):
        pop._SPARSE_DIRECT_N = threshold
        cfg = (
            jtfne.Configuration()
            .runtime(seed=seed, dtype="float32", duration_ms=10.0, dt_ms=0.5)
            .column(name="c", layers=["L4"], n=n)
            .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
            .connectivity(p_connect=p)
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes"])
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        )
        return jtfne.construct(cfg)

    try:
        pop._SPARSE_DIRECT_N = 10**9
        dense = _build(10**9)
        pop._SPARSE_DIRECT_N = n
        sparse = _build(n)
    finally:
        pop._SPARSE_DIRECT_N = old

    assert dense.params["edge_list"].n_edges != sparse.params["edge_list"].n_edges
    s_dense = jtfne.simulate(dense, duration_ms=10.0, dt_ms=0.5, seed=0)
    s_sparse = jtfne.simulate(sparse, duration_ms=10.0, dt_ms=0.5, seed=0)
    assert float(np.sum(s_dense.spikes)) != float(np.sum(s_sparse.spikes))
