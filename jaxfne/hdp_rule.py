"""Registrable finite-state HDP rule surface (23-HDP-01).

Rules map declared state coordinates and plastic targets through a single step
operator:

    (H, X, B, Theta) --P--> (dH, dTheta)

Each rule declares only the coordinates it needs via :class:`HDPRuleDescriptor`.
Built-in legacy rules remain in :func:`simulate_edge_recurrent_izhikevich_hdp`;
registered names route through :func:`simulate_edge_recurrent_izhikevich_hdp_registered`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import jax.numpy as jnp

from ._hdp_adaptive import normalize_hdp_params_boundary, resolve_h_state_locality

RuleStepFn = Callable[["HDPRuleContext"], "HDPRuleUpdate"]


@dataclass(frozen=True)
class HDPRuleDescriptor:
    """Declarative metadata for a registrable plasticity rule."""

    name: str
    h_coords: tuple[str, ...] = ("H",)
    theta_targets: tuple[str, ...] = ("edge_weight",)
    aux_coords: tuple[str, ...] = ()
    scope: str = "node"
    h_bounds: tuple[float, float] = (0.1, 10.0)
    w_bounds: tuple[float, float] = (1e-3, 50.0)
    default_params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HDPRuleContext:
    """Per-step inputs presented to a registered rule."""

    H: jnp.ndarray
    aux: jnp.ndarray
    v: jnp.ndarray
    u: jnp.ndarray
    spikes: jnp.ndarray
    prev_spikes: jnp.ndarray
    syn_state: jnp.ndarray
    w: jnp.ndarray
    pre: jnp.ndarray
    post: jnp.ndarray
    exc_mask: jnp.ndarray
    dt: jnp.ndarray
    n_neurons: int
    rule_params: Mapping[str, Any]
    # Per-edge delayed presynaptic drive at step t (23-DELAY-01):
    # ``spikes_{t-d[e]}[pre[e]]`` under the Protocol D ring convention
    # (d=0 recovers ``spikes[pre]``). ``None`` on the legacy zero-delay
    # path; rules must fall back to ``spikes[pre]`` so zero-delay
    # qualification stays bit-exact.
    pre_sp: jnp.ndarray | None = None


@dataclass(frozen=True)
class HDPRuleUpdate:
    """Infinitesimal updates returned by a registered rule."""

    dH: jnp.ndarray
    d_aux: jnp.ndarray | None = None
    d_theta: Mapping[str, jnp.ndarray] | None = None


_REGISTRY: dict[str, tuple[HDPRuleDescriptor, RuleStepFn]] = {}


def register_hdp_rule(descriptor: HDPRuleDescriptor, step_fn: RuleStepFn) -> None:
    """Register a JIT-safe HDP rule by name."""
    if descriptor.name in _REGISTRY:
        raise ValueError(f"HDP rule {descriptor.name!r} is already registered")
    _REGISTRY[descriptor.name] = (descriptor, step_fn)


def get_hdp_rule(name: str) -> tuple[HDPRuleDescriptor, RuleStepFn]:
    """Return descriptor and step function for a registered rule name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown registered hdp_rule {name!r}; "
            f"known: {sorted(_REGISTRY)}"
        ) from None


def is_registered_hdp_rule(name: str | None) -> bool:
    return bool(name) and name in _REGISTRY


def list_registered_hdp_rules() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def hdp_params_are_identity(hp: Mapping[str, Any] | None) -> bool:
    """True when builtin HDP parameters reduce to static edge-list dynamics."""
    hp = normalize_hdp_params_boundary(dict(hp or {}))
    if is_registered_hdp_rule(hp.get("hdp_rule")):
        return False
    if resolve_h_state_locality(hp) == "population":
        return False
    if bool(hp.get("enable_boundary_stabilization", False)):
        return False
    h_driver_keys = (
        "alpha",
        "beta",
        "gamma",
        "delta",
        "C_spike",
        "K_ctrl",
        "rho_passive",
        "barrier_c",
        "barrier_d",
        "H_boost_gain",
    )
    if any(float(hp.get(k, 0.0) or 0.0) != 0.0 for k in h_driver_keys):
        return False
    if float(hp.get("K_w_ctrl", 0.0) or 0.0) != 0.0:
        return False
    # Legacy documented null may retain K_HDP>0 while H stays at 1 and dw=0.
    # Registered rules and any H-driver activity are never identity.
    return True


def _segment_sum(values: jnp.ndarray, indices: jnp.ndarray, n: int) -> jnp.ndarray:
    return jnp.zeros((n,), dtype=values.dtype).at[indices].add(values)


def _synthetic_presyn_gain_step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    """Qualification rule: event-coupled H and multiplicative edge-weight drive.

    With ``gamma=0`` and controlled presynaptic spikes, ``H`` and ``|w|`` evolve
    predictably from the discrete update equations exercised in tests.
    """
    k_h = jnp.asarray(ctx.rule_params.get("k_h", 0.0), dtype=ctx.H.dtype)
    k_w = jnp.asarray(ctx.rule_params.get("k_w", 0.0), dtype=ctx.H.dtype)
    gamma = jnp.asarray(ctx.rule_params.get("gamma", 0.0), dtype=ctx.H.dtype)
    # 23-DELAY-01: delayed arrivals drive H at time t (event_{t-d} -> H_t).
    # Zero-delay path passes pre_sp=None -> falls back to spikes[pre],
    # preserving the qualified HDP-01 discrete equations bit-exactly.
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    dH = -gamma * (ctx.H - 1.0)
    dH = dH + _segment_sum(k_h * pre_sp, ctx.post, ctx.n_neurons)
    H_post = ctx.H[ctx.post]
    dw = k_w * H_post * pre_sp * jnp.abs(ctx.w)
    return HDPRuleUpdate(dH=dH, d_theta={"edge_weight": dw})


register_hdp_rule(
    HDPRuleDescriptor(
        name="synthetic_presyn_gain",
        h_coords=("H",),
        theta_targets=("edge_weight",),
        scope="node",
        default_params={"k_h": 0.0, "k_w": 0.0, "gamma": 0.0},
    ),
    _synthetic_presyn_gain_step,
)
