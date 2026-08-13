"""Protocol W1a — single-edge ω integrator analytic and Euler receipts."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxfne.w1a_omega_plasticity import (
    W1aConfig,
    integrate_prescribed_delta_h,
    omega_at_pulse_offset,
    omega_continuous_forgetting,
    omega_continuous_writing,
    omega_discrete_exact_constant_delta,
    omega_infinity,
    rectangular_delta_h_schedule,
    simulate_rectangular_pulse,
    weight_from_omega,
)

ATOL = 1e-12
RTOL = 1e-10
CONVERGENCE_RTOL = 5e-3


def _cfg(**kw) -> W1aConfig:
    defaults = dict(tau_w=100.0, kappa_w=2.0, lambda_w=0.2, w0=3.0, dt=1.0)
    defaults.update(kw)
    return W1aConfig(**defaults)


def test_zero_drive_pure_relaxation_kappa_zero():
    cfg = _cfg(kappa_w=0.0, lambda_w=0.5)
    omega0 = 0.4
    n = 20
    sched = jnp.zeros(n)
    omega_tr, _ = integrate_prescribed_delta_h(sched, omega_0=omega0, cfg=cfg)
    expected = omega0 * (cfg.euler_a ** np.arange(1, n + 1))
    np.testing.assert_allclose(omega_tr, expected, rtol=0, atol=ATOL)


def test_zero_drive_pure_relaxation_delta_zero():
    cfg = _cfg()
    omega0 = -0.3
    n = 15
    sched = jnp.full(n, 0.0)
    omega_tr, _ = integrate_prescribed_delta_h(sched, omega_0=omega0, cfg=cfg)
    expected = omega0 * (cfg.euler_a ** np.arange(1, n + 1))
    np.testing.assert_allclose(omega_tr, expected, rtol=0, atol=ATOL)


def test_no_forgetting_limit_lambda_zero():
    cfg = _cfg(lambda_w=0.0)
    delta = 0.15
    tp_steps = 10
    post = 5
    out = simulate_rectangular_pulse(
        pulse_duration_steps=tp_steps,
        post_steps=post,
        delta_h=delta,
        omega_0=0.1,
        cfg=cfg,
    )
    expected_end = 0.1 + cfg.kappa_w * delta * tp_steps * cfg.dt / cfg.tau_w
    assert out["omega_pulse_end"] == pytest.approx(expected_end, rel=0, abs=ATOL)
    np.testing.assert_allclose(
        out["omega_trace"][tp_steps:],
        expected_end,
        rtol=0,
        atol=ATOL,
    )


def test_sign_symmetry_omega_and_weight_product():
    cfg = _cfg()
    tp, post = 8, 12
    delta = 0.25
    pos = simulate_rectangular_pulse(
        pulse_duration_steps=tp, post_steps=post, delta_h=delta, omega_0=0.0, cfg=cfg
    )
    neg = simulate_rectangular_pulse(
        pulse_duration_steps=tp, post_steps=post, delta_h=-delta, omega_0=0.0, cfg=cfg
    )
    np.testing.assert_allclose(pos["omega_trace"], -neg["omega_trace"], atol=ATOL, rtol=0)
    product = pos["weight_trace"] * neg["weight_trace"]
    np.testing.assert_allclose(product, cfg.w0**2, rtol=RTOL, atol=ATOL)


def test_structural_positivity_large_omega():
    cfg = _cfg(w0=2.0)
    for omega in (-50.0, -1.0, 0.0, 1.0, 50.0):
        w = float(weight_from_omega(omega, cfg))
        assert w > 0.0
        assert math.isfinite(w)


def test_reference_state_omega_zero():
    cfg = _cfg(w0=4.5)
    assert float(weight_from_omega(0.0, cfg)) == pytest.approx(4.5)


def test_memory_timescale_one_over_e_decay():
    cfg = _cfg(lambda_w=0.25, tau_w=50.0, dt=1.0)
    tau_mem = cfg.memory_timescale
    assert tau_mem == pytest.approx(200.0)
    delta = 0.3
    tp = 30
    post = 500
    out = simulate_rectangular_pulse(
        pulse_duration_steps=tp, post_steps=post, delta_h=delta, cfg=cfg
    )
    omega_p = float(out["omega_pulse_end"])
    target = omega_p / math.e
    after = out["omega_trace"][tp:]
    idx = int(np.argmin(np.abs(np.asarray(after) - target)))
    measured_steps = idx + 1
    assert measured_steps == pytest.approx(tau_mem, rel=0.02)


def test_pulse_duration_short_linear_and_long_saturation():
    cfg = _cfg(tau_w=10.0, lambda_w=0.2)
    delta = 0.2
    omega_inf = omega_infinity(delta, cfg)
    short_tp = 2
    long_tp = 400
    omega_short = omega_at_pulse_offset(short_tp * cfg.dt, delta, cfg=cfg)
    omega_long = omega_at_pulse_offset(long_tp * cfg.dt, delta, cfg=cfg)
    linear_est = cfg.kappa_w * delta * short_tp * cfg.dt / cfg.tau_w
    assert omega_short == pytest.approx(linear_est, rel=0.05)
    assert omega_long == pytest.approx(omega_inf, rel=0.01)
    assert omega_long > omega_short


def test_discrete_euler_bit_exact_constant_delta():
    cfg = _cfg()
    delta = 0.18
    omega0 = 0.05
    for n in (1, 5, 17, 40):
        sched = jnp.full(n, delta)
        omega_tr, _ = integrate_prescribed_delta_h(sched, omega_0=omega0, cfg=cfg)
        expected = omega_discrete_exact_constant_delta(n, delta, omega_0=omega0, cfg=cfg)
        assert float(omega_tr[-1]) == pytest.approx(expected, rel=0, abs=ATOL)


def test_continuous_limit_convergence_refined_dt():
    delta = 0.22
    tp = 0.4
    t_eval = 0.9
    tau_w, kappa_w, lambda_w = 80.0, 1.5, 0.3
    cont = omega_continuous_writing(tp, delta, cfg=W1aConfig(tau_w=tau_w, kappa_w=kappa_w, lambda_w=lambda_w, dt=1.0))
    cont_forget = omega_continuous_forgetting(
        t_eval,
        omega_at_pulse_offset(tp, delta, cfg=W1aConfig(tau_w=tau_w, kappa_w=kappa_w, lambda_w=lambda_w, dt=1e-6)),
        tp,
        cfg=W1aConfig(tau_w=tau_w, kappa_w=kappa_w, lambda_w=lambda_w, dt=1e-6),
    )
    errors = []
    for dt in (0.1, 0.05, 0.025):
        cfg = W1aConfig(tau_w=tau_w, kappa_w=kappa_w, lambda_w=lambda_w, dt=dt)
        n_pulse = int(round(tp / dt))
        n_after = int(round((t_eval - tp) / dt))
        out = simulate_rectangular_pulse(
            pulse_duration_steps=n_pulse,
            post_steps=n_after,
            delta_h=delta,
            cfg=cfg,
        )
        omega_at_tp = float(out["omega_trace"][n_pulse - 1])
        omega_at_eval = float(out["omega_trace"][-1])
        errors.append(abs(omega_at_tp - cont))
        if n_after > 0:
            errors.append(abs(omega_at_eval - cont_forget))
    assert errors[-1] < errors[0]
    assert errors[-1] < CONVERGENCE_RTOL


def test_euler_stability_ratio_documented():
    cfg = _cfg(lambda_w=0.5, tau_w=100.0, dt=1.0)
    r = cfg.euler_stability_ratio
    assert 0.0 < r < 2.0
    assert r <= 1.0


def test_omega_infinity_requires_positive_lambda():
    with pytest.raises(ValueError, match="lambda_w"):
        omega_infinity(0.1, _cfg(lambda_w=0.0))


def test_w0_must_be_positive():
    with pytest.raises(ValueError, match="w0"):
        W1aConfig(w0=0.0)
