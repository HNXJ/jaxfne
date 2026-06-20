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
from jaxfne.core import _suite2_apply_connectivity, _DENSE_CONNECTIVITY_WARN_N
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
    return _suite2_apply_connectivity(
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
