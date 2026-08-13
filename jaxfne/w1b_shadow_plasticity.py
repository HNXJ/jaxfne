"""Protocol W1b — RBD-generated shadow plasticity on minimal A⟷B pair.

Shadow contract: recurrent simulation uses fixed ``W_0``; ``omega`` integrates
``Delta H = H_pre - H_post`` passively and does **not** feed back.

Update ordering (frozen): ``omega_{n+1} = F(omega_n, H_n)`` using pre-step ``H_n``.

See ``docs/doctrine/protocol_w_hdp_parameter_memory.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.emitters import EdgeList, IzhikevichParams, simulate_edge_recurrent_izhikevich_rbd
from jaxfne.h3_decodability import localized_h_state, trial_drive_schedule
from jaxfne.w1a_omega_plasticity import (
    W1aConfig,
    euler_step_omega,
    integrate_prescribed_delta_h,
    weight_from_omega,
)

_JDTYPE = jnp.float64
IDX_A = 0
IDX_B = 1


@dataclass(frozen=True)
class W1bConfig:
    """Frozen W1b shadow-plasticity contract (no ``omega -> W -> X``)."""

    w1a: W1aConfig = field(default_factory=W1aConfig)
    w0_ab: float = 1.0
    w0_ba: float = 1.0
    delta_h: float = 0.2
    perturbation_step: int = 0
    tau_h_ms: float = 80.0
    rbd_family: str = "f1"
    beta_h: float = 0.0
    kappa_h: float = 0.0
    edge_weight: float = 6.0
    syn_tau_ms: float = 3.0
    delay_steps: int = 0

    def w1a_ab(self) -> W1aConfig:
        return W1aConfig(
            tau_w=self.w1a.tau_w,
            kappa_w=self.w1a.kappa_w,
            lambda_w=self.w1a.lambda_w,
            w0=self.w0_ab,
            dt=self.w1a.dt,
        )

    def w1a_ba(self) -> W1aConfig:
        return W1aConfig(
            tau_w=self.w1a.tau_w,
            kappa_w=self.w1a.kappa_w,
            lambda_w=self.w1a.lambda_w,
            w0=self.w0_ba,
            dt=self.w1a.dt,
        )


def build_two_node_circuit(
    cfg: W1bConfig,
) -> tuple[IzhikevichParams, EdgeList]:
    """Directed pair ``A -> B``, ``B -> A`` with fixed baseline weights."""
    jdtype = jnp.float32
    params = IzhikevichParams(
        v0=jnp.full((2,), -65.0, dtype=jdtype),
        u0=jnp.zeros((2,), dtype=jdtype),
        a=jnp.full((2,), 0.02, dtype=jdtype),
        b=jnp.full((2,), 0.2, dtype=jdtype),
        c=jnp.full((2,), -65.0, dtype=jdtype),
        d=jnp.full((2,), 8.0, dtype=jdtype),
        drive=jnp.zeros((2,), dtype=jdtype),
        sign=jnp.ones((2,), dtype=jdtype),
        W=jnp.zeros((2, 2), dtype=jdtype),
        source_scale=jnp.ones((2,), dtype=jdtype),
        labels=("A", "B"),
        layer_labels=("L4", "L4"),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    ds = jnp.asarray([cfg.delay_steps, cfg.delay_steps], dtype=jnp.int32)
    edges = EdgeList(
        pre=jnp.asarray([IDX_A, IDX_B], dtype=jnp.int32),
        post=jnp.asarray([IDX_B, IDX_A], dtype=jnp.int32),
        weight=jnp.asarray([cfg.edge_weight, cfg.edge_weight], dtype=jdtype),
        receptor_index=jnp.asarray([0, 0], dtype=jnp.int32),
        tau_ms=jnp.asarray([cfg.syn_tau_ms, cfg.syn_tau_ms], dtype=jdtype),
        delay_steps=ds,
    )
    return params, edges


def h_pre_step_trace(
    h_post_step: jax.Array,
    h_init: jax.Array,
) -> jax.Array:
    """``H_n`` before step ``n``: ``H_0 = h_init``, ``H_n = h_post_step[n-1]``."""
    h_post = jnp.asarray(h_post_step, dtype=_JDTYPE)
    h0 = jnp.asarray(h_init, dtype=_JDTYPE)
    return jnp.concatenate([h0[None, :], h_post[:-1]], axis=0)


def delta_h_ab_from_h_pre(h_pre: jax.Array) -> jax.Array:
    return h_pre[:, IDX_A] - h_pre[:, IDX_B]


def integrate_shadow_two_edge(
    delta_h_ab: jax.Array,
    *,
    omega_ab_0: float = 0.0,
    omega_ba_0: float = 0.0,
    cfg: W1bConfig,
) -> dict[str, jax.Array]:
    """Integrate shadow ``omega_AB`` and ``omega_BA`` from prescribed ``Delta H_AB(t)``."""
    dh_ab = jnp.asarray(delta_h_ab, dtype=_JDTYPE)
    dh_ba = -dh_ab
    cfg_ab = cfg.w1a_ab()
    cfg_ba = cfg.w1a_ba()

    def step_ab(carry: jax.Array, dh: jax.Array) -> tuple[jax.Array, jax.Array]:
        omega_next = euler_step_omega(carry, dh, cfg_ab)
        return omega_next, omega_next

    def step_ba(carry: jax.Array, dh: jax.Array) -> tuple[jax.Array, jax.Array]:
        omega_next = euler_step_omega(carry, dh, cfg_ba)
        return omega_next, omega_next

    _, omega_ab = jax.lax.scan(
        step_ab, jnp.asarray(float(omega_ab_0), dtype=_JDTYPE), dh_ab
    )
    _, omega_ba = jax.lax.scan(
        step_ba, jnp.asarray(float(omega_ba_0), dtype=_JDTYPE), dh_ba
    )
    w_ab = weight_from_omega(omega_ab, cfg_ab)
    w_ba = weight_from_omega(omega_ba, cfg_ba)
    return {
        "delta_h_ab": dh_ab,
        "delta_h_ba": dh_ba,
        "omega_ab": omega_ab,
        "omega_ba": omega_ba,
        "weight_ab": w_ab,
        "weight_ba": w_ba,
    }


def run_rbd_shadow_w1b(
    params: IzhikevichParams,
    edges: EdgeList,
    *,
    n_steps: int,
    seed: int,
    cfg: W1bConfig,
    drive_schedule: jax.Array | None = None,
) -> dict[str, Any]:
    """Run RBD at fixed ``W_0`` and integrate passive shadow ``omega`` from recorded ``H``."""
    t0 = int(cfg.perturbation_step)
    if t0 < 0 or t0 >= n_steps:
        raise ValueError(f"perturbation_step must be in [0, {n_steps})")

    if drive_schedule is None:
        drive_schedule = trial_drive_schedule(n_steps, 2, seed)

    h_pert = localized_h_state(2, IDX_A, delta_h=cfg.delta_h, dtype=jnp.float32)
    h_rest = jnp.ones((2,), dtype=jnp.float32)

    key = jax.random.PRNGKey(int(seed))
    rbd_kw = dict(
        rbd_family=cfg.rbd_family,
        beta_h=cfg.beta_h,
        kappa_h=cfg.kappa_h,
        tau_h_ms=cfg.tau_h_ms,
        dtype="float32",
        noise_scale=0.0,
    )

    if t0 == 0:
        init_state = {"H": h_pert}
        v, s, q, st = simulate_edge_recurrent_izhikevich_rbd(
            params,
            edges,
            n_steps,
            cfg.w1a.dt,
            key,
            drive_schedule=drive_schedule,
            init_state=init_state,
            **rbd_kw,
        )
        h_init = h_pert
    else:
        v0, s0, q0, st0 = simulate_edge_recurrent_izhikevich_rbd(
            params,
            edges,
            t0,
            cfg.w1a.dt,
            key,
            drive_schedule=drive_schedule[:t0],
            init_state={"H": h_rest},
            **rbd_kw,
        )
        cont = dict(st0)
        cont["H"] = h_pert
        cont["H_final"] = h_pert
        key, key2 = jax.random.split(key)
        v1, s1, q1, st1 = simulate_edge_recurrent_izhikevich_rbd(
            params,
            edges,
            n_steps - t0,
            cfg.w1a.dt,
            key2,
            drive_schedule=drive_schedule[t0:],
            init_state=cont,
            **rbd_kw,
        )
        v = jnp.concatenate([v0, v1], axis=0)
        s = jnp.concatenate([s0, s1], axis=0)
        q = jnp.concatenate([q0, q1], axis=0)
        st = dict(st1)
        st["H_trace"] = jnp.concatenate([st0["H_trace"], st1["H_trace"]], axis=0)
        h_init = h_rest

    h_post = jnp.asarray(st["H_trace"], dtype=_JDTYPE)
    h_pre = h_pre_step_trace(h_post, jnp.asarray(h_init, dtype=_JDTYPE))
    delta_h_ab = delta_h_ab_from_h_pre(h_pre)

    shadow = integrate_shadow_two_edge(delta_h_ab, cfg=cfg)
    offline_ab, _ = integrate_prescribed_delta_h(
        delta_h_ab, omega_0=0.0, cfg=cfg.w1a_ab()
    )

    return {
        "config": cfg,
        "voltages": v,
        "spikes": s,
        "h_pre": h_pre,
        "h_post": h_post,
        "delta_h_ab": delta_h_ab,
        "shadow": shadow,
        "offline_omega_ab": offline_ab,
        "drive_schedule": drive_schedule,
        "seed": int(seed),
    }


def omega_f1_continuous_convolution(
    t: float,
    delta: float,
    *,
    cfg: W1aConfig,
    tau_h: float,
) -> float:
    r"""Continuous ``omega(t)`` for ``Delta H(t)=delta*exp(-t/tau_H)`` and ``lambda_W>0``."""
    if t < 0:
        raise ValueError("t must be >= 0")
    if cfg.lambda_w == 0.0:
        raise ValueError("use omega_f1_no_forgetting_limit for lambda_w=0")
    lam_w = cfg.lambda_w / cfg.tau_w
    inv_tau_h = 1.0 / float(tau_h)
    coeff = cfg.kappa_w * float(delta) / cfg.tau_w
    denom = lam_w - inv_tau_h
    if abs(denom) < 1e-14:
        return float(coeff * t * np.exp(-t / float(tau_h)))
    return float(
        coeff
        * (np.exp(-t / float(tau_h)) - np.exp(-cfg.lambda_w * t / cfg.tau_w))
        / denom
    )


def omega_f1_no_forgetting_limit(
    delta: float,
    *,
    cfg: W1aConfig,
    tau_h: float,
) -> float:
    r"""``omega(infty) = kappa_W * delta * tau_H / tau_W`` when ``lambda_W=0``."""
    if cfg.lambda_w != 0.0:
        raise ValueError("omega_f1_no_forgetting_limit requires lambda_w=0")
    return float(cfg.kappa_w * float(delta) * float(tau_h) / cfg.tau_w)


def f1_delta_h_continuous(t: float, delta: float, tau_h: float) -> float:
    """Independent F1 relaxation: ``H_A(0)=1+delta``, ``H_B(0)=1`` => ``Delta H = delta*exp(-t/tau_H)``."""
    return float(delta * np.exp(-t / float(tau_h)))
