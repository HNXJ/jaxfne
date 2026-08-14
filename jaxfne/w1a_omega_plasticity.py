"""Protocol W1a — single-edge log-weight plasticity (prescribed ΔH, no emitters).

Scalar state equation (W0 frozen):

    τ_W ω̇ = κ_W ΔH(t) − λ_W ω(t),    W(t) = W_0 exp(ω(t)).

See ``docs/doctrine/protocol_w_hdp_parameter_memory.md`` and
``artifacts/protocol_w/w0_mathematical_contract.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

_JDTYPE = jnp.float64


@dataclass(frozen=True)
class W1aConfig:
    """Frozen W1a scalar/single-edge integrator parameters."""

    tau_w: float = 100.0
    kappa_w: float = 1.0
    lambda_w: float = 0.1
    w0: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.tau_w <= 0:
            raise ValueError("tau_w must be > 0")
        if self.w0 <= 0:
            raise ValueError("w0 must be > 0")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.lambda_w < 0:
            raise ValueError("lambda_w must be >= 0")

    @property
    def euler_a(self) -> float:
        """Discrete multiplier ``a = 1 - λ_W Δt / τ_W``."""
        return 1.0 - self.lambda_w * self.dt / self.tau_w

    @property
    def euler_stability_ratio(self) -> float:
        """``λ_W Δt / τ_W`` — explicit Euler stability parameter."""
        return self.lambda_w * self.dt / self.tau_w

    @property
    def memory_timescale(self) -> float:
        """``τ_mem,W = τ_W / λ_W``; ``inf`` when ``λ_W=0``."""
        if self.lambda_w == 0.0:
            return float("inf")
        return self.tau_w / self.lambda_w


def omega_infinity(delta_h: float, cfg: W1aConfig) -> float:
    """Steady ω under constant drive; only valid when ``λ_W > 0``."""
    if cfg.lambda_w == 0.0:
        raise ValueError("omega_infinity requires lambda_w > 0")
    return cfg.kappa_w * float(delta_h) / cfg.lambda_w


def weight_from_omega(omega: jax.Array | float, cfg: W1aConfig) -> jax.Array:
    """``W = W_0 exp(ω)`` — structural positivity for finite ω."""
    return jnp.asarray(cfg.w0, dtype=jnp.result_type(jnp.asarray(omega))) * jnp.exp(
        jnp.asarray(omega)
    )


def euler_step_omega(
    omega: jax.Array | float,
    delta_h: jax.Array | float,
    cfg: W1aConfig,
) -> jax.Array:
    """One explicit-Euler step of the W1a ω equation."""
    omega_arr = jnp.asarray(omega)
    dh = jnp.asarray(delta_h)
    dt = jnp.asarray(cfg.dt)
    tau = jnp.asarray(cfg.tau_w)
    if cfg.lambda_w == 0.0:
        return omega_arr + cfg.kappa_w * dh * dt / tau
    a = 1.0 - cfg.lambda_w * dt / tau
    drive = cfg.kappa_w * dh * dt / tau
    return a * omega_arr + drive


def omega_discrete_exact_constant_delta(
    n_steps: int,
    delta_h: float,
    *,
    omega_0: float = 0.0,
    cfg: W1aConfig,
) -> float:
    """Exact n-step Euler recurrence under constant ``ΔH=δ`` (bit-exact contract)."""
    n = int(n_steps)
    if n < 0:
        raise ValueError("n_steps must be >= 0")
    if cfg.lambda_w == 0.0:
        return float(omega_0) + n * cfg.kappa_w * float(delta_h) * cfg.dt / cfg.tau_w
    a = cfg.euler_a
    omega_inf = omega_infinity(delta_h, cfg)
    return float(a**n * omega_0 + omega_inf * (1.0 - a**n))


def omega_continuous_writing(
    t: float,
    delta_h: float,
    *,
    omega_0: float = 0.0,
    cfg: W1aConfig,
) -> float:
    """Continuous solution during constant drive ``ΔH=δ`` for ``t >= 0``."""
    if t < 0:
        raise ValueError("t must be >= 0")
    if cfg.lambda_w == 0.0:
        return float(omega_0) + cfg.kappa_w * float(delta_h) * t / cfg.tau_w
    omega_inf = omega_infinity(delta_h, cfg)
    decay = np.exp(-cfg.lambda_w * t / cfg.tau_w)
    return float(omega_inf + (omega_0 - omega_inf) * decay)


def omega_at_pulse_offset(
    pulse_duration: float,
    delta_h: float,
    *,
    omega_0: float = 0.0,
    cfg: W1aConfig,
) -> float:
    """ω at ``t = T_p`` after rectangular pulse."""
    return omega_continuous_writing(
        pulse_duration, delta_h, omega_0=omega_0, cfg=cfg
    )


def omega_continuous_forgetting(
    t: float,
    omega_p: float,
    pulse_duration: float,
    *,
    cfg: W1aConfig,
) -> float:
    """Continuous solution for ``t >= T_p`` with drive off."""
    if t < pulse_duration:
        raise ValueError("t must be >= pulse_duration in forgetting phase")
    if cfg.lambda_w == 0.0:
        return float(omega_p)
    dt_forget = t - pulse_duration
    return float(omega_p * np.exp(-cfg.lambda_w * dt_forget / cfg.tau_w))


def rectangular_delta_h_schedule(
    n_pulse_steps: int,
    n_after_steps: int,
    delta_h: float,
    *,
    cfg: W1aConfig,
) -> jax.Array:
    """Prescribed ``ΔH`` pulse then zero drive."""
    pulse = jnp.full((int(n_pulse_steps),), float(delta_h), dtype=_JDTYPE)
    after = jnp.zeros((int(n_after_steps),), dtype=_JDTYPE)
    return jnp.concatenate([pulse, after], axis=0)


def integrate_prescribed_delta_h(
    delta_h_schedule: jax.Array,
    *,
    omega_0: float = 0.0,
    cfg: W1aConfig,
) -> tuple[jax.Array, jax.Array]:
    """Integrate ω over a prescribed ``ΔH(t)`` schedule; return ``(omega_trace, W_trace)``."""
    schedule = jnp.asarray(delta_h_schedule, dtype=_JDTYPE)

    def step(omega: jax.Array, dh: jax.Array) -> tuple[jax.Array, jax.Array]:
        omega_next = euler_step_omega(omega, dh, cfg)
        return omega_next, omega_next

    omega_final, omega_trace = jax.lax.scan(
        step, jnp.asarray(float(omega_0), dtype=_JDTYPE), schedule
    )
    w_trace = weight_from_omega(omega_trace, cfg)
    return omega_trace, w_trace


def simulate_rectangular_pulse(
    *,
    pulse_duration_steps: int,
    post_steps: int,
    delta_h: float,
    omega_0: float = 0.0,
    cfg: W1aConfig,
) -> dict[str, jax.Array | float | W1aConfig]:
    """Full W1a receipt for a rectangular ``ΔH`` pulse."""
    schedule = rectangular_delta_h_schedule(
        pulse_duration_steps, post_steps, delta_h, cfg=cfg
    )
    omega_trace, w_trace = integrate_prescribed_delta_h(
        schedule, omega_0=omega_0, cfg=cfg
    )
    omega_p = float(omega_trace[pulse_duration_steps - 1]) if pulse_duration_steps > 0 else float(omega_0)
    return {
        "config": cfg,
        "delta_h_schedule": schedule,
        "omega_trace": omega_trace,
        "weight_trace": w_trace,
        "omega_pulse_end": omega_p,
        "pulse_duration_steps": int(pulse_duration_steps),
        "post_steps": int(post_steps),
        "delta_h": float(delta_h),
        "omega_0": float(omega_0),
    }
