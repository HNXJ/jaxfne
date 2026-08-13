"""Protocol W1b — RBD-generated shadow plasticity on A⟷B."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.w1a_omega_plasticity import W1aConfig, integrate_prescribed_delta_h
from jaxfne.w1b_shadow_plasticity import (
    W1bConfig,
    build_two_node_circuit,
    f1_delta_h_continuous,
    integrate_shadow_two_edge,
    omega_f1_continuous_convolution,
    omega_f1_no_forgetting_limit,
    run_rbd_shadow_w1b,
)

ATOL = 1e-12


def _cfg_f1_analytic(**kw) -> W1bConfig:
    defaults = dict(
        w1a=W1aConfig(tau_w=200.0, kappa_w=1.5, lambda_w=0.05, w0=1.0, dt=1.0),
        delta_h=0.25,
        tau_h_ms=40.0,
        beta_h=0.0,
        kappa_h=0.0,
        edge_weight=6.0,
    )
    defaults.update(kw)
    return W1bConfig(**defaults)


def test_composition_equivalence_shadow_vs_offline_w1a():
    cfg = _cfg_f1_analytic()
    params, edges = build_two_node_circuit(cfg)
    out = run_rbd_shadow_w1b(params, edges, n_steps=80, seed=7, cfg=cfg)
    np.testing.assert_allclose(
        out["shadow"]["omega_ab"],
        out["offline_omega_ab"],
        rtol=0,
        atol=ATOL,
    )


def test_two_edge_antisymmetry():
    cfg = _cfg_f1_analytic()
    params, edges = build_two_node_circuit(cfg)
    out = run_rbd_shadow_w1b(params, edges, n_steps=60, seed=3, cfg=cfg)
    sh = out["shadow"]
    np.testing.assert_allclose(sh["omega_ab"], -sh["omega_ba"], rtol=0, atol=ATOL)
    np.testing.assert_allclose(
        sh["weight_ab"] * sh["weight_ba"],
        cfg.w0_ab * cfg.w0_ba,
        rtol=1e-12,
        atol=ATOL,
    )


def test_null_delta_h_zero():
    cfg = _cfg_f1_analytic(delta_h=0.0)
    params, edges = build_two_node_circuit(cfg)
    out = run_rbd_shadow_w1b(params, edges, n_steps=50, seed=1, cfg=cfg)
    np.testing.assert_allclose(out["shadow"]["omega_ab"], 0.0, atol=ATOL)


def test_null_kappa_w_zero():
    cfg = _cfg_f1_analytic(w1a=W1aConfig(kappa_w=0.0, lambda_w=0.1, tau_w=100.0, dt=1.0))
    params, edges = build_two_node_circuit(cfg)
    out = run_rbd_shadow_w1b(params, edges, n_steps=50, seed=2, cfg=cfg)
    np.testing.assert_allclose(out["shadow"]["omega_ab"], 0.0, atol=ATOL)


def test_f1_analytic_convolution_matches_numeric():
    cfg = _cfg_f1_analytic()
    w1a = cfg.w1a
    tau_h = cfg.tau_h_ms
    delta = cfg.delta_h
    n = 120
    times = np.arange(1, n + 1) * w1a.dt
    delta_h = jnp.asarray([f1_delta_h_continuous(t, delta, tau_h) for t in times])
    shadow = integrate_shadow_two_edge(delta_h, cfg=cfg)
    for i, t in enumerate(times):
        expected = omega_f1_continuous_convolution(t, delta, cfg=w1a, tau_h=tau_h)
        assert float(shadow["omega_ab"][i]) == pytest.approx(expected, rel=0.02)


def test_lambda_zero_permanent_integral_f1():
    w1a = W1aConfig(tau_w=100.0, kappa_w=2.0, lambda_w=0.0, dt=1.0)
    cfg = _cfg_f1_analytic(w1a=w1a, tau_h_ms=50.0, delta_h=0.3)
    tau_h = cfg.tau_h_ms
    n = 400
    times = np.arange(1, n + 1) * w1a.dt
    delta_h = jnp.asarray([f1_delta_h_continuous(t, cfg.delta_h, tau_h) for t in times])
    shadow = integrate_shadow_two_edge(delta_h, cfg=cfg)
    expected = omega_f1_no_forgetting_limit(cfg.delta_h, cfg=w1a, tau_h=tau_h)
    assert float(shadow["omega_ab"][-1]) == pytest.approx(expected, rel=0.02)


def test_timescale_separated_retention():
    """H gradient relaxed while omega remains measurable (tau_mem >> tau_H)."""
    w1a = W1aConfig(tau_w=100.0, kappa_w=3.0, lambda_w=0.01, dt=1.0)
    cfg = _cfg_f1_analytic(w1a=w1a, tau_h_ms=20.0, delta_h=0.4)
    params, edges = build_two_node_circuit(cfg)
    n_steps = 200
    out = run_rbd_shadow_w1b(params, edges, n_steps=n_steps, seed=11, cfg=cfg)
    late = n_steps - 1
    assert abs(float(out["delta_h_ab"][late])) < 0.05 * cfg.delta_h
    assert abs(float(out["shadow"]["omega_ab"][late])) > 0.05
    assert cfg.w1a.memory_timescale > 5.0 * cfg.tau_h_ms


def test_lambda_positive_asymptotic_decay_to_zero():
    cfg = _cfg_f1_analytic(
        w1a=W1aConfig(tau_w=100.0, kappa_w=1.0, lambda_w=0.5, dt=1.0)
    )
    params, edges = build_two_node_circuit(cfg)
    out = run_rbd_shadow_w1b(params, edges, n_steps=2000, seed=4, cfg=cfg)
    assert abs(float(out["delta_h_ab"][-1])) < 1e-3
    assert abs(float(out["shadow"]["omega_ab"][-1])) < 1e-3


def test_offline_w1a_matches_integrate_shadow_ab_only():
    cfg = _cfg_f1_analytic()
    params, edges = build_two_node_circuit(cfg)
    out = run_rbd_shadow_w1b(params, edges, n_steps=40, seed=8, cfg=cfg)
    offline, _ = integrate_prescribed_delta_h(
        out["delta_h_ab"], omega_0=0.0, cfg=cfg.w1a_ab()
    )
    np.testing.assert_allclose(out["shadow"]["omega_ab"], offline, rtol=0, atol=ATOL)
