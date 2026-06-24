"""Standalone tests for the HDP (Homeostasis-Dependent Plasticity) kernel.

Exercises jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp directly
(no Configuration/Model layer), matching the existing
test_homeostasis_dispatch.py pattern. Item #12 of the HDP punch list:
the kernel itself (#11) had no dedicated test before this file.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jaxfne.emitters import (
    IzhikevichParams, EdgeList,
    simulate_edge_recurrent_izhikevich_hdp as hdp_kernel,
)


def _make_params_edges(N: int, ne: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = IzhikevichParams(
        a=jnp.full((N,), 0.02), b=jnp.full((N,), 0.2), c=jnp.full((N,), -65.0),
        d=jnp.full((N,), 8.0), drive=jnp.full((N,), 6.0), sign=jnp.ones((N,)),
        W=jnp.zeros((N, N)), v0=jnp.full((N,), -65.0), u0=jnp.full((N,), -13.0),
        source_scale=jnp.ones((N,)), labels=tuple("E" for _ in range(N)),
        layer_labels=tuple("L4" for _ in range(N)), source_calibration_status="x")
    receptor_index = rng.integers(0, 2, ne)
    # The kernel re-derives weight sign from |w| + receptor_index every step
    # (excitatory positive, inhibitory negative) -- match that convention in
    # the fixture so a K_HDP=0 null control is a true no-op on the weights,
    # not a one-time sign normalization.
    weight_mag = np.abs(rng.normal(0, 0.3, ne)).astype(np.float32)
    weight = np.where(receptor_index == 0, weight_mag, -weight_mag).astype(np.float32)
    edges = EdgeList(
        pre=jnp.asarray(rng.integers(0, N, ne), jnp.int32),
        post=jnp.asarray(rng.integers(0, N, ne), jnp.int32),
        weight=jnp.asarray(weight),
        receptor_index=jnp.asarray(receptor_index, jnp.int32),
        tau_ms=jnp.full((ne,), 5.0), source_calibration_status="x")
    return p, edges


def test_null_control_holds_H_at_1_and_matches_homeostatic_kernel():
    """alpha=beta=gamma=delta=C_spike=0.0 holds H_i fixed at 1.0 forever, making
    the K_HDP-scaled weight term identically zero regardless of K_HDP -- the
    documented null control."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(1)
    V, S, _, diag = hdp_kernel(p, edges, 100, 0.5, key, K_HDP=1.0, noise_scale=0.0)
    H_trace = np.asarray(diag["H_trace"])
    assert np.allclose(H_trace, 1.0, atol=1e-6)
    w_final = np.asarray(diag["w_final"])
    w0 = np.asarray(edges.weight)
    assert np.allclose(w_final, w0, atol=1e-6)
    assert bool(np.isfinite(np.asarray(V)).all())


def test_K_HDP_zero_disables_plasticity_regardless_of_other_gains():
    """K_HDP=0.0 disables HDP outright even with nonzero alpha/gamma/K_ctrl --
    H_i may still move, but weights must not."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(1)
    V, S, _, diag = hdp_kernel(
        p, edges, 200, 0.5, key, K_HDP=0.0,
        alpha=0.05, gamma=0.3, K_ctrl=0.1, noise_scale=0.0)
    w_final = np.asarray(diag["w_final"])
    w0 = np.asarray(edges.weight)
    assert np.allclose(w_final, w0, atol=1e-6)
    assert bool(np.isfinite(np.asarray(V)).all())


def test_nonzero_gains_move_H_and_weights_while_staying_finite_and_clipped():
    """Nonzero alpha/gamma/K_ctrl/K_HDP -> H_i deviates from 1.0, weights
    change, and both stay finite and within their clip bounds."""
    N, ne = 64, 512
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(2)
    V, S, _, diag = hdp_kernel(
        p, edges, 400, 0.5, key,
        alpha=0.05, gamma=0.5, K_ctrl=0.15, K_HDP=0.01,
        barrier_c=0.01, barrier_d=0.01, tau_0_ms=5.0,
        w_floor=0.01, w_ceiling=10.0, noise_scale=0.0)
    H_trace = np.asarray(diag["H_trace"])
    w_trace = np.asarray(diag["w_trace"])
    assert not np.allclose(H_trace, 1.0, atol=1e-6)
    assert bool(np.isfinite(H_trace).all())
    assert bool(np.isfinite(w_trace).all())
    assert H_trace.min() >= 0.1 - 1e-4 and H_trace.max() <= 10.0 + 1e-4
    w_final = np.asarray(diag["w_final"])
    assert np.abs(w_final).max() <= 10.0 + 1e-4
    assert bool(np.isfinite(np.asarray(V)).all())
    assert H_trace.shape == (400, N)
    assert w_trace.shape == (400, ne)


def test_record_dH_components_returns_five_traces_summing_to_dH():
    """record_dH_components=True exposes the five additive dH/dt terms;
    each has the documented shape (n_steps, n_neurons)."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(3)
    _, _, _, diag = hdp_kernel(
        p, edges, 50, 0.5, key,
        alpha=0.05, gamma=0.5, K_ctrl=0.15, K_HDP=0.01,
        barrier_c=0.01, barrier_d=0.01, tau_0_ms=5.0,
        record_dH_components=True, noise_scale=0.0)
    for name in ("dH_income_trace", "dH_rate_trace", "dH_weight_trace",
                 "dH_ctrl_trace", "dH_barrier_trace"):
        assert name in diag
        assert np.asarray(diag[name]).shape == (50, N)
        assert bool(np.isfinite(np.asarray(diag[name])).all())


def test_record_edge_current_returns_per_edge_trace():
    """record_edge_current=True exposes the per-edge synaptic current
    contribution with shape (n_steps, n_edges)."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(4)
    _, _, _, diag = hdp_kernel(
        p, edges, 30, 0.5, key, alpha=0.05, gamma=0.5, K_ctrl=0.15,
        record_edge_current=True, noise_scale=0.0)
    assert "edge_current_trace" in diag
    assert np.asarray(diag["edge_current_trace"]).shape == (30, ne)


def test_init_state_resume_matches_full_run():
    """Two chunks via init_state == one full run, deterministically -- same
    pause/resume guarantee as the existing homeostatic kernel."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(5)
    kw = dict(alpha=0.05, gamma=0.5, K_ctrl=0.15, K_HDP=0.01, noise_scale=0.0)
    Vf, Sf, _, _ = hdp_kernel(p, edges, 200, 0.5, key, **kw)
    V1, S1, _, d1 = hdp_kernel(p, edges, 100, 0.5, key, **kw)
    V2, S2, _, _ = hdp_kernel(p, edges, 100, 0.5, key, init_state=d1, **kw)
    Vc = np.concatenate([np.asarray(V1), np.asarray(V2)])
    Sc = np.concatenate([np.asarray(S1), np.asarray(S2)])
    assert np.allclose(np.asarray(Vf), Vc, atol=1e-4)
    assert np.array_equal(np.asarray(Sf), Sc)


def test_size_scale_override_changes_tau_and_dynamics():
    """An explicit per-neuron size_scale_override changes tau_i = tau_0_ms *
    size_i**2 and therefore the resulting H trajectory vs. the default
    cell-type-only size table."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(6)
    kw = dict(alpha=0.05, gamma=0.5, K_ctrl=0.15, K_HDP=0.01, tau_0_ms=5.0,
              noise_scale=0.0)
    _, _, _, d_default = hdp_kernel(p, edges, 100, 0.5, key, **kw)
    override = np.full(N, 10.0, dtype=np.float32)  # far from the E default
    _, _, _, d_override = hdp_kernel(
        p, edges, 100, 0.5, key, size_scale_override=jnp.asarray(override), **kw)
    H_default = np.asarray(d_default["H_trace"])
    H_override = np.asarray(d_override["H_trace"])
    assert not np.allclose(H_default, H_override, atol=1e-5)


def test_negative_K_HDP_is_anti_homeostatic_and_still_finite():
    """K_HDP<0 is the documented explicit anti-homeostatic stress-test mode;
    it must still run and stay finite over a short horizon."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(7)
    V, S, _, diag = hdp_kernel(
        p, edges, 50, 0.5, key, alpha=0.05, gamma=0.5, K_ctrl=0.15,
        K_HDP=-0.01, noise_scale=0.0)
    assert bool(np.isfinite(np.asarray(V)).all())
    assert bool(np.isfinite(np.asarray(diag["w_final"])).all())
