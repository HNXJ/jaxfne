"""Protocol W2 — frozen-omega parameter expression (no ``H -> omega`` during run).

Causal graph: ``omega* -> W* -> X`` with ``dot omega = 0``.

Canonical stored plastic state is ``omega``; effective signed coupling at the edge
boundary is ``J = s * W0 * exp(omega)`` (magnitude ``W = W0*exp(omega) > 0``).

See ``docs/doctrine/protocol_w_hdp_parameter_memory.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.emitters import EdgeList, IzhikevichParams, simulate_edge_recurrent_izhikevich_rbd
from jaxfne.w1a_omega_plasticity import weight_from_omega, W1aConfig
from jaxfne.w1b_shadow_plasticity import (
    IDX_A,
    IDX_B,
    W1bConfig,
    build_two_node_circuit,
    run_rbd_shadow_w1b,
)

Sign = Literal[-1, 1]


@dataclass(frozen=True)
class W2FrozenOmegaState:
    """Immutable plastic readout state for W2 (not updated during expression)."""

    omega_ab: float = 0.0
    omega_ba: float = 0.0
    w0_ab: float = 6.0
    w0_ba: float = 6.0

    def magnitude_ab(self) -> float:
        return float(weight_from_omega(self.omega_ab, W1aConfig(w0=self.w0_ab)))

    def magnitude_ba(self) -> float:
        return float(weight_from_omega(self.omega_ba, W1aConfig(w0=self.w0_ba)))

    def coupling_ab(self, sign: Sign = 1) -> float:
        return float(sign) * self.magnitude_ab()

    def coupling_ba(self, sign: Sign = 1) -> float:
        return float(sign) * self.magnitude_ba()


@dataclass(frozen=True)
class W2ProtocolConfig:
    """Frozen W2 expression experiment contract."""

    dt_ms: float = 1.0
    n_steps: int = 80
    pulse_step: int = 8
    pulse_amp: float = 38.0
    response_start: int = 8
    response_end: int = 35
    omega_levels: tuple[float, ...] = (-0.25, 0.0, 0.25)
    rbd_family: str = "f1"
    beta_h: float = 0.0
    kappa_h: float = 0.0
    tau_h_ms: float = 80.0
    syn_tau_ms: float = 3.0
    delay_steps: int = 0
    w1b_for_memory: W1bConfig = field(default_factory=W1bConfig)
    memory_pick_step: int = 40


# Prospective frozen W2 protocol (write-once receipt: artifacts/protocol_w/w2_expression/).
FROZEN_W2_CONFIG = W2ProtocolConfig(
    pulse_amp=38.0,
    omega_levels=(-0.25, 0.0, 0.25),
)


def build_w2_circuit(
    omega: W2FrozenOmegaState,
    *,
    sign_ab: Sign = 1,
    sign_ba: Sign = 1,
    isolate_ab: bool = True,
    syn_tau_ms: float = 3.0,
    delay_steps: int = 0,
) -> tuple[IzhikevichParams, EdgeList]:
    """Build A⟷B with typed ``J = s * W0 * exp(omega)`` edge couplings."""
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
    w_ab = omega.coupling_ab(sign_ab)
    w_ba = 0.0 if isolate_ab else omega.coupling_ba(sign_ba)
    ds = jnp.asarray([delay_steps, delay_steps], dtype=jnp.int32)
    edges = EdgeList(
        pre=jnp.asarray([IDX_A, IDX_B], dtype=jnp.int32),
        post=jnp.asarray([IDX_B, IDX_A], dtype=jnp.int32),
        weight=jnp.asarray([w_ab, w_ba], dtype=jdtype),
        receptor_index=jnp.asarray([0, 0], dtype=jnp.int32),
        tau_ms=jnp.asarray([syn_tau_ms, syn_tau_ms], dtype=jdtype),
        delay_steps=ds,
    )
    return params, edges


def a_pulse_drive_schedule(
    cfg: W2ProtocolConfig,
    *,
    seed: int = 0,
) -> jax.Array:
    """Subthreshold-oriented pulse on A only; shared across W2 conditions."""
    drive = jnp.zeros((cfg.n_steps, 2), dtype=jnp.float32)
    drive = drive.at[cfg.pulse_step, IDX_A].set(cfg.pulse_amp)
    return drive


def postsynaptic_response_b(
    voltages: jax.Array,
    *,
    cfg: W2ProtocolConfig,
    sign_ab: Sign = 1,
    v_rest: float = -65.0,
) -> float:
    """Primary W2 metric: subthreshold postsynaptic deflection on B."""
    v_b = np.asarray(voltages[cfg.response_start : cfg.response_end, IDX_B], dtype=np.float64)
    if sign_ab < 0:
        return float(v_rest - np.min(v_b))
    return float(np.max(v_b) - v_rest)


def spike_count_b(spikes: jax.Array, *, cfg: W2ProtocolConfig) -> float:
    s_b = np.asarray(spikes[cfg.response_start : cfg.response_end, IDX_B], dtype=np.float64)
    return float(np.sum(s_b))


def run_w2_expression(
    omega: W2FrozenOmegaState,
    *,
    cfg: W2ProtocolConfig,
    seed: int = 0,
    sign_ab: Sign = 1,
    sign_ba: Sign = 1,
    isolate_ab: bool = True,
    drive_schedule: jax.Array | None = None,
) -> dict[str, Any]:
    """Simulate ``omega* -> W* -> X`` with ``dot omega = 0`` (weights frozen at build)."""
    params, edges = build_w2_circuit(
        omega,
        sign_ab=sign_ab,
        sign_ba=sign_ba,
        isolate_ab=isolate_ab,
        syn_tau_ms=cfg.syn_tau_ms,
        delay_steps=cfg.delay_steps,
    )
    if drive_schedule is None:
        drive_schedule = a_pulse_drive_schedule(cfg, seed=seed)

    key = jax.random.PRNGKey(int(seed))
    v, s, q, st = simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        cfg.n_steps,
        cfg.dt_ms,
        key,
        drive_schedule=drive_schedule,
        init_state={"H": jnp.ones((2,), dtype=jnp.float32)},
        rbd_family=cfg.rbd_family,
        beta_h=cfg.beta_h,
        kappa_h=cfg.kappa_h,
        tau_h_ms=cfg.tau_h_ms,
        dtype="float32",
        noise_scale=0.0,
    )
    return {
        "config": cfg,
        "omega_state": omega,
        "effective_weight_ab": float(edges.weight[0]),
        "effective_weight_ba": float(edges.weight[1]),
        "voltages": v,
        "spikes": s,
        "sources": q,
        "state": st,
        "drive_schedule": drive_schedule,
        "response_b": postsynaptic_response_b(v, cfg=cfg, sign_ab=sign_ab),
        "spike_count_b": spike_count_b(s, cfg=cfg),
        "seed": int(seed),
    }


def run_w2_monotonic_sweep(
    cfg: W2ProtocolConfig | None = None,
    *,
    seed: int = 0,
    sign_ab: Sign = 1,
) -> dict[str, Any]:
    """Prescribed ``omega in {-a,0,+a}`` monotonicity receipt on A->B."""
    cfg = cfg or W2ProtocolConfig()
    drive = a_pulse_drive_schedule(cfg, seed=seed)
    w0 = cfg.w1b_for_memory.edge_weight
    results: dict[float, dict[str, Any]] = {}
    for level in cfg.omega_levels:
        omega = W2FrozenOmegaState(omega_ab=float(level), omega_ba=0.0, w0_ab=w0, w0_ba=w0)
        results[float(level)] = run_w2_expression(
            omega, cfg=cfg, seed=seed, sign_ab=sign_ab, drive_schedule=drive
        )
    return {"config": cfg, "results": results, "drive_schedule": drive}


def run_w2_w1b_memory_contrast(
    cfg: W2ProtocolConfig | None = None,
    *,
    seed: int = 7,
) -> dict[str, Any]:
    """Compare ``X(t; W0*exp(omega*))`` vs ``X(t; W0)`` with identical drive."""
    cfg = cfg or W2ProtocolConfig()
    w1b_cfg = cfg.w1b_for_memory
    params, edges = build_two_node_circuit(w1b_cfg)
    w1b = run_rbd_shadow_w1b(params, edges, n_steps=cfg.n_steps, seed=seed, cfg=w1b_cfg)
    pick = int(cfg.memory_pick_step)
    omega_star = float(w1b["shadow"]["omega_ab"][pick])
    w0 = w1b_cfg.edge_weight
    drive = a_pulse_drive_schedule(cfg, seed=seed + 1000)

    memory_omega = W2FrozenOmegaState(
        omega_ab=omega_star, omega_ba=0.0, w0_ab=w0, w0_ba=w0
    )
    null_omega = W2FrozenOmegaState(omega_ab=0.0, omega_ba=0.0, w0_ab=w0, w0_ba=w0)
    mem = run_w2_expression(memory_omega, cfg=cfg, seed=seed, drive_schedule=drive)
    null = run_w2_expression(null_omega, cfg=cfg, seed=seed, drive_schedule=drive)
    return {
        "config": cfg,
        "omega_star": omega_star,
        "omega_pick_step": pick,
        "w1b_shadow_omega_trace": w1b["shadow"]["omega_ab"],
        "memory": mem,
        "null": null,
    }
