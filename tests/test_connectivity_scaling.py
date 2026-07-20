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
