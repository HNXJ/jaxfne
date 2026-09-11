"""Connectivity construction scaling + transparency (factor 13).

- The TCM population mask is built vectorized (no O(N^2) Python double loop);
  covered functionally by test_tcm_v1_6pop.
- Dense all-to-all within-area connectivity is O(N^2) by nature (the requested
  topology, not a defect). At scale it must SELF-REPORT the cost rather than
  silently materialize a huge matrix; sparse (p_connect<1) suppresses the warning.
"""
import warnings

import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne.core import _apply_connectivity, _DENSE_CONNECTIVITY_WARN_N
from jaxfne.emitters import IzhikevichParams


def _params(n):
    z = jnp.zeros(n)
    return IzhikevichParams(
        a=z, b=z, c=z, d=z, drive=z, sign=jnp.ones(n), W=jnp.zeros((1, 1)),
        v0=z, u0=z, source_scale=jnp.asarray(1.0),
        labels=tuple("E" for _ in range(n)),
        layer_labels=tuple("L4" for _ in range(n)),
        source_calibration_status="uncalibrated_proxy",
    )


def _apply(n, conn):
    return _apply_connectivity(
        _params(n), ["V1"] * n, ["L4"] * n, ["E"] * n,
        {"connectivity": conn}, seed=0, dtype="float32")


def test_dense_alltoall_warns_at_scale():
    n = _DENSE_CONNECTIVITY_WARN_N
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _apply(n, {"within_gain": 0.45})
    assert any("O(N^2)" in str(x.message) for x in w)


def test_small_dense_does_not_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _apply(64, {"within_gain": 0.45})
    assert not any("O(N^2)" in str(x.message) for x in w)


def test_sparse_p_connect_suppresses_warning():
    n = _DENSE_CONNECTIVITY_WARN_N
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _apply(n, {"within_gain": 0.45, "p_connect": 0.1})
    assert not any("O(N^2)" in str(x.message) for x in w)


# ── Sparse-direct edge construction (factor 13 deep refactor) ────────────────
from jaxfne.core import _SPARSE_DIRECT_N


def _sparse_model(n, p, seed=1):
    cfg = (jtfne.Configuration().runtime(seed=seed, dtype="float32", duration_ms=40.0, dt_ms=0.5)
           .column(name="c", layers=["L4"], n=n)
           .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
           .connectivity(p_connect=p).set_emitter("izhikevich", "cortical_eig")
           .probes(["spikes"])
           .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
    return cfg, jtfne.construct(cfg)


def test_sparse_direct_skips_dense_W_at_scale():
    import numpy as np
    _, m = _sparse_model(_SPARSE_DIRECT_N, 0.02)
    W = np.asarray(m.params["emitter"].W)
    assert W.shape == (0, 0)                       # placeholder, no (n,n) materialization
    n_edges = int(m.params["edge_list"].n_edges)
    expected = 0.02 * _SPARSE_DIRECT_N * (_SPARSE_DIRECT_N - 1)
    assert 0.7 * expected <= n_edges <= 1.3 * expected   # ~Erdos-Renyi density


def test_sparse_direct_runs_on_edge_list_backend():
    import numpy as np
    _, m = _sparse_model(_SPARSE_DIRECT_N, 0.02)
    sig = jtfne.simulate(m, duration_ms=40.0, dt_ms=0.5, seed=0)
    assert sig.metadata["recurrent_backend"] == "edge_list"
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())


def test_sparse_direct_deterministic():
    import numpy as np
    _, m1 = _sparse_model(_SPARSE_DIRECT_N, 0.02, seed=7)
    _, m2 = _sparse_model(_SPARSE_DIRECT_N, 0.02, seed=7)
    assert np.array_equal(np.asarray(m1.params["edge_list"].weight),
                          np.asarray(m2.params["edge_list"].weight))


def test_below_threshold_keeps_dense_W():
    import numpy as np
    _, m = _sparse_model(_SPARSE_DIRECT_N - 1000, 0.1)
    W = np.asarray(m.params["emitter"].W)
    assert W.shape[0] == _SPARSE_DIRECT_N - 1000   # dense path preserved below threshold


def test_sparse_direct_not_bit_exact_to_dense_at_threshold():
    """REP-03 gate: lowering _SPARSE_DIRECT_N needs separate equivalence evidence."""
    import numpy as np
    import jaxfne._construct_population as pop

    n, p, seed = _SPARSE_DIRECT_N, 0.02, 7
    old = pop._SPARSE_DIRECT_N

    def _construct(threshold):
        pop._SPARSE_DIRECT_N = threshold
        cfg = (jtfne.Configuration().runtime(seed=seed, dtype="float32", duration_ms=10.0, dt_ms=0.5)
               .column(name="c", layers=["L4"], n=n)
               .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
               .connectivity(p_connect=p)
               .set_emitter("izhikevich", "cortical_eig")
               .probes(["spikes"])
               .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
        return jtfne.construct(cfg)

    try:
        pop._SPARSE_DIRECT_N = 10**9
        dense = _construct(10**9)
        pop._SPARSE_DIRECT_N = n
        sparse = _construct(n)
    finally:
        pop._SPARSE_DIRECT_N = old

    assert tuple(dense.params["emitter"].W.shape) == (n, n)
    assert tuple(sparse.params["emitter"].W.shape) == (0, 0)
    assert dense.params["edge_list"].n_edges != sparse.params["edge_list"].n_edges
    s_dense = jtfne.simulate(dense, duration_ms=10.0, dt_ms=0.5, seed=0)
    s_sparse = jtfne.simulate(sparse, duration_ms=10.0, dt_ms=0.5, seed=0)
    assert float(np.sum(s_dense.spikes)) != float(np.sum(s_sparse.spikes))


def _bounded_degree_zero_p_model(n=1000, seed=1):
    from scripts.perf.w10_allocation_map import build_config

    cfg = build_config(
        n=n, p_connect=0.0, max_in_degree=100, duration_ms=10.0, dt_ms=0.5, seed=seed
    )
    return cfg, jtfne.construct(cfg)


def test_zero_p_connect_bounded_degree_omits_dense_W():
    """p_connect=0 with declared rules: edge_list authoritative, no (n,n) emitter.W."""
    import numpy as np

    _, model = _bounded_degree_zero_p_model()
    W = np.asarray(model.params["emitter"].W)
    assert W.shape == (0, 0)
    assert model.params["edge_list"].n_edges > 0
    sig = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=1)
    assert sig.metadata["recurrent_backend"] == "edge_list"
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())
    assert bool(np.isfinite(np.asarray(sig.spikes)).all())


def test_placeholder_W_lazy_materialization_is_transient():
    """dense_recurrent_weights() must not persist a dense emitter.W on the model."""
    import numpy as np
    from jaxfne.emitters import is_placeholder_dense_W

    _, model = _bounded_degree_zero_p_model(n=200)
    emitter = model.params["emitter"]
    assert is_placeholder_dense_W(emitter.W, emitter.n_neurons)
    dense = np.asarray(model.dense_recurrent_weights())
    assert dense.shape == (200, 200)
    assert np.asarray(model.params["emitter"].W).shape == (0, 0)
    assert int(np.count_nonzero(dense)) > 0


def test_construct_records_representation_authority():
    """REP-02: realized topology is edge_list-authoritative when W is a placeholder."""
    _, model = _bounded_degree_zero_p_model(n=120)
    rep = model.static["representation"]
    assert rep["topology_authoritative"] == "edge_list"
    assert rep["emitter_W_storage"] == "placeholder"
    assert rep["dense_W_role"] == "execution_layout_on_demand"
    assert rep["edge_list_role"] == "authoritative"


def test_dense_path_records_emitter_W_authority():
    """REP-02: below sparse-direct threshold, materialized W is topology authority."""
    import numpy as np

    n = _SPARSE_DIRECT_N - 1000
    cfg = (
        jtfne.Configuration()
        .runtime(seed=11, dtype="float32", duration_ms=10.0, dt_ms=0.5)
        .column(name="c", layers=["L4"], n=n)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity(p_connect=0.1)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
    )
    model = jtfne.construct(cfg)
    rep = model.static["representation"]
    assert rep["topology_authoritative"] == "emitter_W"
    assert rep["emitter_W_storage"] == "materialized"
    assert rep["dense_W_role"] == "execution_layout_materialized"
    assert rep["edge_list_role"] == "execution_layout_derived"
    assert tuple(model.params["emitter"].W.shape) == (n, n)
    assert model.params["edge_list"].n_edges > 0
    sig = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=0)
    assert sig.metadata["recurrent_backend"] == "dense"
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())


def test_simulate_refuses_explicit_dense_backend_on_placeholder_W():
    """REP-02: contradicted recurrent_backend must raise, not silently drop edges."""
    import pytest
    from dataclasses import replace

    from jaxfne._model import Model

    _, model = _bounded_degree_zero_p_model(n=80)
    contradicted = replace(
        model.cfg,
        metadata={**model.cfg.metadata, "recurrent_backend": "dense"},
    )
    model = Model(cfg=contradicted, params=model.params, static=model.static)
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=0)


def test_placeholder_W_checkpoint_records_topology_authority(tmp_path):
    """Checkpoint keeps placeholder W and records edge_list as authoritative."""
    import json
    import numpy as np
    from jaxfne._model import Model

    cfg, model = _bounded_degree_zero_p_model(n=120)
    path = tmp_path / "ck"
    model.checkpoint(str(path))
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["topology_authoritative"] == "edge_list"
    assert meta["emitter_W_storage"] == "placeholder"
    with np.load(path.with_suffix(".npz")) as archive:
        assert archive["emitter_W"].shape == (0, 0)
    restored = Model.restore(str(path), cfg)
    assert np.asarray(restored.params["emitter"].W).shape == (0, 0)
    assert restored.params["edge_list"].n_edges == model.params["edge_list"].n_edges


def test_synaptic_gain_scales_edge_list_when_placeholder_W():
    """Tuning synaptic_gain must affect dynamics via edge_list, not a (0,0) W."""
    import numpy as np
    from jaxfne.core import _model_with_scalar_parameter

    _, model = _bounded_degree_zero_p_model(n=200)
    before = np.asarray(jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=3).V_m)
    scaled = _model_with_scalar_parameter(model, "synaptic_gain", 0.5)
    assert np.asarray(scaled.params["emitter"].W).shape == (0, 0)
    weights = np.asarray(scaled.params["edge_list"].weight)
    base_weights = np.asarray(model.params["edge_list"].weight)
    np.testing.assert_allclose(weights, base_weights * 0.5)
    after = np.asarray(jtfne.simulate(scaled, duration_ms=10.0, dt_ms=0.5, seed=3).V_m)
    assert not np.array_equal(before, after)
