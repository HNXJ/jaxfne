"""23-COMPAT-JOM-01: independent reproduction of Jomission-reported regressions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import (
    EdgeList,
    resolve_edge_delay_steps,
    resolve_edge_tau_ms,
)
from scripts.perf.w10_allocation_map import build_config


def _bounded_model(n: int = 200, seed: int = 1):
    cfg = build_config(
        n=n, p_connect=0.0, max_in_degree=100, duration_ms=10.0, dt_ms=0.5, seed=seed
    )
    return cfg, jtfne.construct(cfg)


def test_compact_tau_ms_shape_zero_resolves_not_validates_raw():
    """Compact tau_ms placeholders are valid when tau_storage is derivable."""
    _, model = _bounded_model()
    el = model.params["edge_list"]
    assert el.tau_storage == "sign_from_receptor"
    assert tuple(np.asarray(el.tau_ms).shape) == (0,)
    tau = np.asarray(resolve_edge_tau_ms(el, el.weight.dtype))
    assert tau.shape == (el.n_edges,)
    assert set(np.unique(tau).tolist()) <= {2.0, 5.0}


def test_compact_edge_list_round_trip_to_dict_from_dict():
    """Serialization must preserve compact storage metadata (not only raw arrays)."""
    _, model = _bounded_model()
    el = model.params["edge_list"]
    restored = EdgeList.from_dict(el.to_dict())
    assert restored.tau_storage == el.tau_storage
    assert restored.delay_storage == el.delay_storage
    assert restored.weight_storage == el.weight_storage
    tau = np.asarray(resolve_edge_tau_ms(restored, restored.weight.dtype))
    tau0 = np.asarray(resolve_edge_tau_ms(el, el.weight.dtype))
    np.testing.assert_allclose(tau, tau0)


def test_jit_simulate_compact_uniform_zero_delay_no_tracer_conversion():
    """simulate must not call np.asarray on traced resolve_edge_delay_steps output."""
    model = _bounded_model()[1]
    el = model.params["edge_list"]
    assert el.delay_storage == "uniform_zero"

    eager = np.asarray(
        jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=2).V_m
    )

    @jax.jit
    def jitted(v0: jax.Array):
        sig = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=2)
        return jnp.sum(sig.V_m) + v0

    traced = float(jitted(jnp.asarray(0.0)))
    assert np.isfinite(traced)
    np.testing.assert_allclose(traced, float(np.sum(eager)), rtol=1e-5, atol=1e-3)


def test_resolve_edge_delay_steps_safe_under_jit_np_asarray_fails():
    """Document tracer hazard: np.asarray(resolve_edge_delay_steps) is invalid under jit."""
    _, model = _bounded_model(n=20)
    el = model.params["edge_list"]

    @jax.jit
    def sum_delays():
        return jnp.sum(resolve_edge_delay_steps(el))

    assert int(sum_delays()) == 0

    @jax.jit
    def broken():
        return jnp.sum(jnp.asarray(np.asarray(resolve_edge_delay_steps(el))))

    with pytest.raises(jax.errors.TracerArrayConversionError):
        broken()
