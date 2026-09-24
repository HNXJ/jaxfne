"""Registrable finite-state HDP rule surface (23-HDP-01).

Rules map declared state coordinates and plastic targets through a single step
operator:

    (H, X, B, Theta) --P--> (dH, dTheta)

Each rule declares only the coordinates it needs via :class:`HDPRuleDescriptor`.
Built-in legacy rules remain in :func:`simulate_edge_recurrent_izhikevich_hdp`;
registered names route through :func:`simulate_edge_recurrent_izhikevich_hdp_registered`.
"""

from __future__ import annotations

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import jax.numpy as jnp

from ._hdp_adaptive import normalize_hdp_params_boundary, resolve_h_state_locality
from .emitters import _segment_sum

RuleStepFn = Callable[["HDPRuleContext"], "HDPRuleUpdate"]


@dataclass(frozen=True)
class HDPRuleDescriptor:
    """Declarative metadata for a registrable plasticity rule."""

    name: str
    h_coords: tuple[str, ...] = ("H",)
    # Trailing H-carry shape: H has shape ``(n_neurons, *h_shape)`` so rules
    # may declare vector coordinates (d_H > 1) without a simulator branch.
    h_shape: tuple[int, ...] = ()
    # Declared plastic targets, subset of _KNOWN_THETA_TARGETS. Rules return
    # infinitesimal drives in ``d_theta``; the kernel integrates the declared
    # ones and rejects undeclared keys loudly (never silently dropped).
    theta_targets: tuple[str, ...] = ("edge_weight",)
    aux_coords: tuple[str, ...] = ()
    # Flat auxiliary-carry layout for ``aux``/``d_aux``: "none" -> ``(0,)``,
    # "scalar" -> ``()`` (single global/population-shared coordinate),
    # "per_neuron" -> ``(n_neurons,)``, "per_edge" -> ``(n_edges,)``.
    # Declaring more than one aux coordinate widens the carry with a trailing
    # dimension (``(n, k)``), so multi-timescale cascades need no new branch.
    aux_layout: str = "none"
    scope: str = "node"
    h_bounds: tuple[float, float] = (0.1, 10.0)
    w_bounds: tuple[float, float] = (1e-3, 50.0)
    # Bounds for the per-neuron drive-bias coordinate ("drive_bias" target):
    # additive native current, clipped each step like the other coordinates.
    b_bounds: tuple[float, float] = (-50.0, 50.0)
    default_params: Mapping[str, float] = field(default_factory=dict)


# Plastic targets the registrable kernel integrates. "edge_weight" is the
# sign-preserving per-edge efficacy magnitude; "drive_bias" is a per-neuron
# additive drive (intrinsic excitability) coordinate owned by the kernel
# carry, continued and traced like H and w.
_KNOWN_THETA_TARGETS = ("edge_weight", "drive_bias")


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
    # Per-step deterministic key for stochastic rules (fold_in of the run
    # key and global step index): same seed -> same rule-noise stream.
    # Rules that never touch it stay exactly deterministic. Threading
    # convention matches membrane noise: Model-level segments share the
    # key chain and global step index, so chunked runs reproduce full runs;
    # manual kernel-level chunk callers thread (key, step offset) likewise.
    key: jnp.ndarray
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


_AUX_LAYOUTS = ("none", "scalar", "per_neuron", "per_edge")


def expected_aux_shape(
    descriptor: HDPRuleDescriptor, *, n_neurons: int, n_edges: int
) -> tuple[int, ...]:
    """Static ``aux`` carry shape for a descriptor (23-LAW-01)."""
    k = len(descriptor.aux_coords)
    tail = (int(k),) if k > 1 else ()
    if descriptor.aux_layout == "none":
        return (0,)
    if descriptor.aux_layout == "scalar":
        return tail
    if descriptor.aux_layout == "per_neuron":
        return (int(n_neurons),) + tail
    if descriptor.aux_layout == "per_edge":
        return (int(n_edges),) + tail
    raise ValueError(
        f"HDP rule {descriptor.name!r} declares unknown aux_layout "
        f"{descriptor.aux_layout!r}; expected one of {_AUX_LAYOUTS}"
    )


def register_hdp_rule(descriptor: HDPRuleDescriptor, step_fn: RuleStepFn) -> None:
    """Register a JIT-safe HDP rule by name."""
    if descriptor.name in _REGISTRY:
        raise ValueError(f"HDP rule {descriptor.name!r} is already registered")
    if descriptor.aux_layout not in _AUX_LAYOUTS:
        raise ValueError(
            f"HDP rule {descriptor.name!r} declares unknown aux_layout "
            f"{descriptor.aux_layout!r}; expected one of {_AUX_LAYOUTS}"
        )
    unknown_theta = [t for t in descriptor.theta_targets
                     if t not in _KNOWN_THETA_TARGETS]
    if unknown_theta:
        raise ValueError(
            f"HDP rule {descriptor.name!r} declares unknown theta_targets "
            f"{unknown_theta!r}; expected subset of {_KNOWN_THETA_TARGETS}"
        )
    if descriptor.aux_layout == "none" and descriptor.aux_coords:
        raise ValueError(
            f"HDP rule {descriptor.name!r} declares aux_coords "
            f"{descriptor.aux_coords!r} with aux_layout 'none'"
        )
    for bound_name in ("h_bounds", "w_bounds", "b_bounds"):
        lo, hi = getattr(descriptor, bound_name)
        if not (float(lo) <= float(hi)):
            raise ValueError(
                f"HDP rule {descriptor.name!r} declares inverted "
                f"{bound_name} {(lo, hi)!r}; expected (min, max)"
            )
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


def check_hdp_rule_params(
    rule_name: str, rule_params: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Fail-closed validation of per-rule parameters (0.5.3 item 7b, H7).

    Returns ``{**descriptor.default_params, **rule_params}``. Unknown keys
    (not in the rule's ``default_params``) raise ``ValueError`` instead of
    merging silently into ``ctx.rule_params`` where ``.get()`` reads would
    ignore them: a typo'd gain must never simulate as the default.
    Non-mapping ``rule_params`` also fails closed (``None`` means
    "defaults"). Unknown ``rule_name`` raises via :func:`get_hdp_rule`.
    """
    descriptor, _ = get_hdp_rule(rule_name)
    if rule_params is None:
        return dict(descriptor.default_params)
    if not isinstance(rule_params, _Mapping):
        raise ValueError(
            f"hdp_rule_params for rule {rule_name!r} must be a mapping, "
            f"got {type(rule_params).__name__}"
        )
    unknown = [k for k in rule_params if k not in descriptor.default_params]
    if unknown:
        raise ValueError(
            f"hdp_rule_params for rule {rule_name!r} contains unrecognized keys "
            f"{sorted(unknown, key=repr)}; declared params: "
            f"{sorted(descriptor.default_params)}"
        )
    return {**dict(descriptor.default_params), **dict(rule_params)}


def reject_unknown_hdp_kwargs(
    hdp_kwargs: Mapping[str, Any], *, kernel: str
) -> None:
    """Fail-closed unknown-key policy for ``compile_step_fn **hdp_kwargs``.

    0.5.3 item 7b (H7): every key must be consumed either by the selected
    kernel's signature or by ``compile_step_fn`` itself. Call-site-owned
    internals (``drive_schedule``, ``dtype``, ``init_state``,
    ``silence_mask``, ``noise_schedule``, ``step_indices``) are rejected:
    passing them would be silently overridden per step, so they fail closed
    with a direction to the owning API instead. ``hdp_rule_params`` is
    accepted on the ``"hdp"`` path (consumed by the registered-rule branch)
    but rejected for ``kernel="baseline"``, where plasticity is inert by
    design (item 5 fixed-W identity). Raises ``ValueError`` listing the
    unknown keys; returns ``None`` when clean.
    """
    import inspect

    from . import emitters as _emitters

    if kernel == "hdp":
        kernel_fn = _emitters.simulate_edge_recurrent_izhikevich_hdp
        local_keys = {"hdp_rule_params", "h_state_dim", "h_state_locality"}
    elif kernel == "baseline":
        kernel_fn = _emitters.simulate_edge_recurrent_izhikevich
        local_keys = {"h_state_dim", "h_state_locality"}
    else:
        raise ValueError(f"kernel must be 'hdp' or 'baseline', got {kernel!r}")
    accepted = {
        p.name
        for p in inspect.signature(kernel_fn).parameters.values()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    call_site_owned = {
        "drive_schedule",
        "dtype",
        "init_state",
        "silence_mask",
        "noise_schedule",
        "step_indices",
    }
    allowed = (accepted - call_site_owned) | local_keys
    unknown = [k for k in hdp_kwargs if k not in allowed]
    if unknown:
        raise ValueError(
            f"compile_step_fn(kernel={kernel!r}) got unrecognized hdp_kwargs keys "
            f"{sorted(unknown, key=repr)}; allowed: {sorted(allowed)}"
        )


def hdp_is_engaged(
    hp: Mapping[str, Any] | None,
    params: Mapping[str, Any] | None,
    *,
    enable_hdp: bool = False,
) -> bool:
    """Single routing predicate for HDP execution (bulk + continuation).

    Identity parameters alone route to the baseline kernel — except when the
    model carries explicitly seeded HDP state (``with_hdp_initial_state``):
    seeded non-uniform ``H`` drives real weight dynamics (``dw ~ ΔH``) even
    with null H-gains, so dropping it would silently discard user state the
    kernel machinery (`init_state`) exists to carry. Seeded-but-inert
    (``enable_hdp=False``) stays inert per the documented contract.
    """
    if not bool(enable_hdp):
        return False
    if not hdp_params_are_identity(hp):
        return True
    if params is None:
        return False
    return params.get("hdp_initial_H") is not None or params.get("hdp_initial_w") is not None


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


def _eligibility_trace_gain_step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    """23-LAW-01 structured-law exercise: event-driven eligibility + H gate.

    Per-edge eligibility ``E`` integrates pre/post coincidence with decay
    ``tau_e``; the weight drive is eligibility gated by postsynaptic ``H``.
    ``H`` keeps the event-coupled drive so the delayed chain
    ``event_{t-d} -> H_t -> P -> Theta_t -> I_t`` holds as for the
    qualification rule. Not an STDP mechanism claim — an expressivity probe
    for the aux-carrying primitive (non-empty ``aux`` continuation).
    """
    k_h = jnp.asarray(ctx.rule_params.get("k_h", 0.0), dtype=ctx.H.dtype)
    k_w = jnp.asarray(ctx.rule_params.get("k_w", 0.0), dtype=ctx.H.dtype)
    gamma = jnp.asarray(ctx.rule_params.get("gamma", 0.0), dtype=ctx.H.dtype)
    tau_e = jnp.asarray(ctx.rule_params.get("tau_e", 20.0), dtype=ctx.H.dtype)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    post_sp = ctx.spikes[ctx.post]
    dH = -gamma * (ctx.H - 1.0)
    dH = dH + _segment_sum(k_h * pre_sp, ctx.post, ctx.n_neurons)
    dE = -ctx.aux / jnp.maximum(tau_e, jnp.asarray(1e-6, dtype=ctx.H.dtype))
    dE = dE + pre_sp * post_sp
    H_post = ctx.H[ctx.post]
    dw = k_w * H_post * ctx.aux * jnp.abs(ctx.w)
    return HDPRuleUpdate(dH=dH, d_aux=dE, d_theta={"edge_weight": dw})


register_hdp_rule(
    HDPRuleDescriptor(
        name="eligibility_trace_gain",
        h_coords=("H",),
        theta_targets=("edge_weight",),
        aux_coords=("eligibility",),
        aux_layout="per_edge",
        scope="node",
        default_params={"k_h": 0.0, "k_w": 0.0, "gamma": 0.0, "tau_e": 20.0},
    ),
    _eligibility_trace_gain_step,
)
