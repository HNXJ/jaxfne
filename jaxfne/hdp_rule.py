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
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass, field
import enum
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp

from ._hdp_adaptive import (
    bind_theta_to_plant as bind_theta_to_plant,
    normalize_hdp_params_boundary,
    resolve_h_state_locality,
)
from .emitters import _segment_sum

RuleStepFn = Callable[["HDPRuleContext"], "HDPRuleUpdate"]


class QuantityKind(str, enum.Enum):
    """Quantity kinds of the relative-quantity grammar (D28).

    Vocabulary only: the doctrine names what each numerical quantity
    is (base, relative, effective, time). No mapping or use lives here.
    """

    B = "B"  # base/model quantity: physical baseline the relative state modulates around
    R = "R"  # relative quantity: dimensionless internal dynamic state or modulation
    E = "E"  # effective quantity: mapped output C_p(p0, r_p) actually used in dynamics
    T = "T"  # physical time quantity: dimensional clock, delays and time constants


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
    # Flat rules map param name -> default; superposed composites map
    # component name -> that component's param dict (namespaced overrides).
    default_params: Mapping[str, Any] = field(default_factory=dict)


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
    """Updates returned by a registered rule.

    ``dH``, ``d_aux`` and ``d_theta`` are rates: the kernel adds ``dt * d``.
    ``jH``, ``j_aux`` and ``j_theta`` are per-event jumps: the kernel adds the
    value once, with no ``dt`` factor, after the rate term and before the clip
    where the coordinate has one. ``None`` means no jump.
    """

    dH: jnp.ndarray
    d_aux: jnp.ndarray | None = None
    d_theta: Mapping[str, jnp.ndarray] | None = None
    jH: jnp.ndarray | None = None
    j_aux: jnp.ndarray | None = None
    j_theta: Mapping[str, jnp.ndarray] | None = None


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


def apply_registered_rule_update(
    *,
    descriptor: HDPRuleDescriptor,
    rule_step: Callable[[HDPRuleContext], Any],
    H: jax.Array,
    aux: jax.Array,
    bias: jax.Array,
    v: jax.Array,
    u: jax.Array,
    spikes: jax.Array,
    prev_spikes: jax.Array,
    syn_state: jax.Array,
    w: jax.Array,
    pre: jax.Array,
    post: jax.Array,
    exc_mask: jax.Array | None,
    dt: jax.Array,
    n_neurons: int,
    rule_params: Mapping[str, Any],
    key: jax.Array,
    pre_sp: jax.Array | None,
    rule_name: str,
    plasticity_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Integrate one registered-rule update; shared by every family kernel.

    ``w`` is the family's plastic coordinate: per-edge signed efficacy when
    ``exc_mask`` is given (Izhikevich sign-preserving magnitude update), or
    per-edge magnitude (``|w|``, AdEx conductance) when ``exc_mask`` is
    ``None``. ``plasticity_mask`` (per-edge bool, or ``None`` for all-plastic)
    gates the weight update. Undeclared ``d_theta`` and ``j_theta`` keys fail
    closed naming the rule. Rate terms integrate as ``x + dt * dx``; jump
    terms add once, ``x + dt * dx + jx``. Returns
    ``(H_next, w_next, aux_next, bias_next)``.
    """
    ctx = HDPRuleContext(
        H=H,
        aux=aux,
        v=v,
        u=u,
        spikes=spikes,
        prev_spikes=prev_spikes,
        syn_state=syn_state,
        w=w,
        pre=pre,
        post=post,
        exc_mask=exc_mask if exc_mask is not None else jnp.ones_like(w, dtype=bool),
        dt=dt,
        n_neurons=n_neurons,
        rule_params=rule_params,
        key=key,
        pre_sp=pre_sp,
    )
    upd = rule_step(ctx)
    undeclared = [
        k for k in (*(upd.d_theta or {}), *(upd.j_theta or {}))
        if k not in descriptor.theta_targets
    ]
    if undeclared:
        raise ValueError(
            f"hdp_rule {rule_name!r} returned undeclared theta_targets "
            f"{undeclared!r}; declared: {list(descriptor.theta_targets)}"
        )
    h_min_arr = jnp.asarray(descriptor.h_bounds[0], dtype=H.dtype)
    h_max_arr = jnp.asarray(descriptor.h_bounds[1], dtype=H.dtype)
    w_floor_arr = jnp.asarray(descriptor.w_bounds[0], dtype=w.dtype)
    w_ceiling_arr = jnp.asarray(descriptor.w_bounds[1], dtype=w.dtype)
    H_drift = H + dt * upd.dH
    if upd.jH is not None:
        H_drift = H_drift + upd.jH
    H_next = jnp.clip(H_drift, h_min_arr, h_max_arr)
    aux_next = aux
    if upd.d_aux is not None:
        aux_next = aux_next + dt * upd.d_aux
    if upd.j_aux is not None:
        aux_next = aux_next + upd.j_aux
    dw = (upd.d_theta or {}).get("edge_weight")
    jw = (upd.j_theta or {}).get("edge_weight")
    if dw is not None or jw is not None:
        # Magnitude: rate term scaled by dt, then the event jump, then the clip.
        mag = w if exc_mask is None else jnp.abs(w)
        if dw is not None:
            mag = mag + dt * dw
        if jw is not None:
            mag = mag + jw
        mag = jnp.clip(mag, w_floor_arr, w_ceiling_arr)
        w_next = mag if exc_mask is None else jnp.where(exc_mask, mag, -mag)
        if plasticity_mask is not None:
            w_next = jnp.where(plasticity_mask, w_next, w)
    else:
        w_next = w
    if "drive_bias" in descriptor.theta_targets:
        db = (upd.d_theta or {}).get("drive_bias")
        jb = (upd.j_theta or {}).get("drive_bias")
        if db is None and jb is None:
            b_next = bias
        else:
            b_drift = bias if db is None else bias + dt * db
            if jb is not None:
                b_drift = b_drift + jb
            b_next = jnp.clip(
                b_drift,
                jnp.asarray(descriptor.b_bounds[0], dtype=bias.dtype),
                jnp.asarray(descriptor.b_bounds[1], dtype=bias.dtype),
            )
    else:
        b_next = bias
    return H_next, w_next, aux_next, b_next

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

    Identity parameters alone route to the baseline kernel; except when the
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
    predictably from the discrete update equations exercised in tests. The
    spike terms are per-event jumps (``jH``, ``j_theta``); only the decay of
    ``H`` is a rate.
    """
    k_h = jnp.asarray(ctx.rule_params.get("k_h", 0.0), dtype=ctx.H.dtype)
    k_w = jnp.asarray(ctx.rule_params.get("k_w", 0.0), dtype=ctx.H.dtype)
    gamma = jnp.asarray(ctx.rule_params.get("gamma", 0.0), dtype=ctx.H.dtype)
    # 23-DELAY-01: delayed arrivals drive H at time t (event_{t-d} -> H_t).
    # Zero-delay path passes pre_sp=None -> falls back to spikes[pre],
    # preserving the qualified HDP-01 discrete equations bit-exactly.
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    dH = -gamma * (ctx.H - 1.0)
    jH = _segment_sum(k_h * pre_sp, ctx.post, ctx.n_neurons)
    H_post = ctx.H[ctx.post]
    jw = k_w * H_post * pre_sp * jnp.abs(ctx.w)
    return HDPRuleUpdate(dH=dH, jH=jH, j_theta={"edge_weight": jw})


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
    qualification rule. Not an STDP mechanism claim; an expressivity probe
    for the aux-carrying primitive (non-empty ``aux`` continuation).
    """
    k_h = jnp.asarray(ctx.rule_params.get("k_h", 0.0), dtype=ctx.H.dtype)
    k_w = jnp.asarray(ctx.rule_params.get("k_w", 0.0), dtype=ctx.H.dtype)
    gamma = jnp.asarray(ctx.rule_params.get("gamma", 0.0), dtype=ctx.H.dtype)
    tau_e = jnp.asarray(ctx.rule_params.get("tau_e", 20.0), dtype=ctx.H.dtype)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    post_sp = ctx.spikes[ctx.post]
    dH = -gamma * (ctx.H - 1.0)
    jH = _segment_sum(k_h * pre_sp, ctx.post, ctx.n_neurons)
    dE = -ctx.aux / jnp.maximum(tau_e, jnp.asarray(1e-6, dtype=ctx.H.dtype))
    jE = pre_sp * post_sp
    H_post = ctx.H[ctx.post]
    dw = k_w * H_post * ctx.aux * jnp.abs(ctx.w)
    return HDPRuleUpdate(dH=dH, jH=jH, d_aux=dE, j_aux=jE, d_theta={"edge_weight": dw})


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


# --- Trace rules (reader STDP scaffold) --------------------------------------
# Per-neuron spike traces with jump-then-decay exact-exponential updates:
# each step integrates ``H <- decay * (H + spike)`` with
# ``decay = exp(-dt/tau)`` via the infinitesimal drive
# ``dH = (decay * (H + spike) - H) / dt``, so a pre spike at step 0 reads
# ``decay**m`` at a post spike m steps later (and 0 for a same-step pair).
# ``dw`` is a rate driven by the spike train (``spike / dt`` per step), so
# one pair changes the weight by exactly ``A * exp(-Delta t / tau)`` at any
# dt. ``trace_ltp`` carries the presynaptic trace (potentiation on post
# spikes), ``trace_ltd`` the postsynaptic trace (depression on pre
# arrivals), ``trace_stdp`` both.
_TRACE_TAU_PLUS = 17.0
_TRACE_TAU_MINUS = 34.0
_TRACE_A_PLUS = 0.005
_TRACE_A_MINUS = 0.00525
_TRACE_H_BOUNDS = (0.0, 1000000.0)


def _trace_decay_dH(H: jnp.ndarray, spikes: jnp.ndarray, dt: jnp.ndarray, tau: float) -> jnp.ndarray:
    decay = jnp.exp(-dt / jnp.asarray(tau, dtype=H.dtype))
    return (decay * (H + spikes) - H) / jnp.maximum(dt, jnp.asarray(1e-9, dtype=H.dtype))


def _trace_ltp_step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    tau = float(ctx.rule_params.get("tau_plus", _TRACE_TAU_PLUS))
    a = jnp.asarray(ctx.rule_params.get("A_plus", _TRACE_A_PLUS), dtype=ctx.H.dtype)
    dt_safe = jnp.maximum(ctx.dt, jnp.asarray(1e-9, dtype=ctx.H.dtype))
    dH = _trace_decay_dH(ctx.H, ctx.spikes, ctx.dt, tau)
    dw = a * ctx.H[ctx.pre] * ctx.spikes[ctx.post] / dt_safe
    return HDPRuleUpdate(dH=dH, d_theta={"edge_weight": dw})


register_hdp_rule(
    HDPRuleDescriptor(
        name="trace_ltp",
        h_coords=("pre_trace",),
        theta_targets=("edge_weight",),
        scope="node",
        h_bounds=_TRACE_H_BOUNDS,
        default_params={"tau_plus": _TRACE_TAU_PLUS, "A_plus": _TRACE_A_PLUS},
    ),
    _trace_ltp_step,
)


def _trace_ltd_step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    tau = float(ctx.rule_params.get("tau_minus", _TRACE_TAU_MINUS))
    a = jnp.asarray(ctx.rule_params.get("A_minus", _TRACE_A_MINUS), dtype=ctx.H.dtype)
    dt_safe = jnp.maximum(ctx.dt, jnp.asarray(1e-9, dtype=ctx.H.dtype))
    dH = _trace_decay_dH(ctx.H, ctx.spikes, ctx.dt, tau)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    dw = -a * ctx.H[ctx.post] * pre_sp / dt_safe
    return HDPRuleUpdate(dH=dH, d_theta={"edge_weight": dw})


register_hdp_rule(
    HDPRuleDescriptor(
        name="trace_ltd",
        h_coords=("post_trace",),
        theta_targets=("edge_weight",),
        scope="node",
        h_bounds=_TRACE_H_BOUNDS,
        default_params={"tau_minus": _TRACE_TAU_MINUS, "A_minus": _TRACE_A_MINUS},
    ),
    _trace_ltd_step,
)


def _trace_stdp_step(ctx: HDPRuleContext) -> HDPRuleUpdate:
    tau_p = float(ctx.rule_params.get("tau_plus", _TRACE_TAU_PLUS))
    tau_m = float(ctx.rule_params.get("tau_minus", _TRACE_TAU_MINUS))
    a_p = jnp.asarray(ctx.rule_params.get("A_plus", _TRACE_A_PLUS), dtype=ctx.H.dtype)
    a_m = jnp.asarray(ctx.rule_params.get("A_minus", _TRACE_A_MINUS), dtype=ctx.H.dtype)
    pre_trace = ctx.H[..., 0]
    post_trace = ctx.H[..., 1]
    dH = jnp.stack(
        [
            _trace_decay_dH(pre_trace, ctx.spikes, ctx.dt, tau_p),
            _trace_decay_dH(post_trace, ctx.spikes, ctx.dt, tau_m),
        ],
        axis=-1,
    )
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    dt_safe = jnp.maximum(ctx.dt, jnp.asarray(1e-9, dtype=ctx.H.dtype))
    dw = a_p * pre_trace[ctx.pre] * ctx.spikes[ctx.post] / dt_safe - a_m * post_trace[ctx.post] * pre_sp / dt_safe
    return HDPRuleUpdate(dH=dH, d_theta={"edge_weight": dw})


register_hdp_rule(
    HDPRuleDescriptor(
        name="trace_stdp",
        h_coords=("pre_trace", "post_trace"),
        h_shape=(2,),
        theta_targets=("edge_weight",),
        scope="node",
        h_bounds=_TRACE_H_BOUNDS,
        default_params={
            "tau_plus": _TRACE_TAU_PLUS,
            "tau_minus": _TRACE_TAU_MINUS,
            "A_plus": _TRACE_A_PLUS,
            "A_minus": _TRACE_A_MINUS,
        },
    ),
    _trace_stdp_step,
)


def superpose_hdp_rules(
    name: str,
    components: _Sequence[str],
    *,
    w_bounds: tuple[float, float] | None = None,
) -> HDPRuleDescriptor:
    """Register ``name`` as the additive superposition of registered rules.

    The composite carries the concatenated ``H`` coordinates of its
    components (width = sum of ``prod(h_shape)``, 1 for scalar rules) and
    sums their ``d_theta`` drives per target. Parameters are namespaced:
    each component's step sees only its own defaults updated by
    ``params[component]`` (``{"trace_ltp": {"A_plus": 0.01}}``), so
    equal-named params never cross-talk. A flat key in the superposed
    params (not a component name) is refused by param validation, naming
    it. ``h_bounds``, ``b_bounds`` and ``scope`` must agree across
    components (one clip box drives the composite carry, so a split is
    refused rather than averaged). ``w_bounds`` overrides the unanimous
    component bound when given. Aux-carrying components are refused: the
    composite carry is H-only.
    """
    from dataclasses import replace as _replace

    names = [str(c) for c in components]
    if len(names) == 0:
        raise ValueError(f"superpose_hdp_rules({name!r}, ...): no components given")
    if len(set(names)) != len(names):
        raise ValueError(
            f"superpose_hdp_rules({name!r}, ...): duplicate components "
            f"{sorted(n for n in set(names) if names.count(n) > 1)}; "
            "superposing a rule with itself is refused"
        )
    resolved = [get_hdp_rule(n) for n in names]
    for comp_name, (desc, _) in zip(names, resolved):
        if desc.aux_layout != "none" or desc.aux_coords:
            raise ValueError(
                f"superpose_hdp_rules({name!r}, ...): component {comp_name!r} "
                f"carries aux {desc.aux_coords!r} (layout {desc.aux_layout!r}); "
                "only H-only rules superpose"
            )
    widths: list[int] = []
    for _, (desc, _) in zip(names, resolved):
        widths.append(int(_np_prod(desc.h_shape)) if desc.h_shape else 1)
    for comp_name, (desc, _), width in zip(names, resolved, widths):
        coords = len(desc.h_coords)
        if desc.h_shape:
            if coords != width:
                raise ValueError(
                    f"superpose_hdp_rules({name!r}, ...): component {comp_name!r} "
                    f"declares {coords} h_coords for h_shape {desc.h_shape} "
                    f"(width {width}); unmappable"
                )
        elif coords != 1:
            raise ValueError(
                f"superpose_hdp_rules({name!r}, ...): scalar component {comp_name!r} "
                f"declares h_coords {desc.h_coords!r}; expected one coordinate"
            )
    first = resolved[0][0]
    for attr in ("h_bounds", "b_bounds", "scope"):
        vals = [getattr(desc, attr) for desc, _ in resolved]
        if any(v != vals[0] for v in vals[1:]):
            raise ValueError(
                f"superpose_hdp_rules({name!r}, ...): components disagree on "
                f"{attr} {[ (n, v) for n, v in zip(names, vals) ]}; "
                "pass compatible rules"
            )
    comp_w = [desc.w_bounds for desc, _ in resolved]
    if w_bounds is None:
        if any(w != comp_w[0] for w in comp_w[1:]):
            raise ValueError(
                f"superpose_hdp_rules({name!r}, ...): components disagree on "
                f"w_bounds {[ (n, w) for n, w in zip(names, comp_w) ]}; "
                "pass w_bounds=... explicitly"
            )
        w_bounds = comp_w[0]
    # Namespaced defaults: {component_name: {own params}}. Equal-named
    # params across components never cross-talk; a flat key lands in no
    # component namespace and fails closed in check_hdp_rule_params.
    namespaced_defaults: dict[str, dict[str, float]] = {
        comp_name: dict(desc.default_params)
        for comp_name, (desc, _) in zip(names, resolved)
    }
    targets: list[str] = []
    for desc, _ in resolved:
        for t in desc.theta_targets:
            if t not in targets:
                targets.append(t)
    coords: list[str] = []
    for (desc, _), width in zip(resolved, widths):
        if not desc.h_shape:
            coords.extend(desc.h_coords)
        elif len(desc.h_coords) == width:
            coords.extend(desc.h_coords)
        else:  # pragma: no cover - guarded above
            coords.extend(f"{desc.h_coords[0]}[{i}]" for i in range(width))
    offsets: list[int] = []
    running = 0
    for width in widths:
        offsets.append(running)
        running += width

    def _superposed_step(ctx: HDPRuleContext) -> HDPRuleUpdate:
        n_ctx = int(ctx.H.shape[0])
        dH_parts: list[jnp.ndarray] = []
        dw_accum: dict[str, jnp.ndarray] = {}
        jH_parts: list[jnp.ndarray] = []
        jH_used = False
        jw_accum: dict[str, jnp.ndarray] = {}
        for (desc, step_fn), comp_name, width, off in zip(resolved, names, widths, offsets):
            part = ctx.H[..., off : off + width]
            if desc.h_shape:
                part = jnp.reshape(part, (n_ctx,) + tuple(desc.h_shape))
            else:
                part = part[..., 0]
            raw = ctx.rule_params.get(comp_name, {})
            if not isinstance(raw, _Mapping):
                raise ValueError(
                    f"hdp_rule_params for rule {name!r}: component {comp_name!r} "
                    f"params must be a mapping, got {type(raw).__name__}"
                )
            unknown = [k for k in raw if k not in desc.default_params]
            if unknown:
                raise ValueError(
                    f"hdp_rule_params for rule {name!r}: component {comp_name!r} "
                    f"contains unrecognized keys {sorted(unknown, key=repr)}; "
                    f"declared params: {sorted(desc.default_params)}"
                )
            sub = _replace(ctx, H=part,
                           rule_params={**dict(desc.default_params), **dict(raw)})
            upd = step_fn(sub)
            if int(jnp.size(upd.dH)) != n_ctx * width:
                raise ValueError(
                    f"hdp_rule {name!r}: component returned dH of size "
                    f"{int(jnp.size(upd.dH))}, expected {n_ctx * width}"
                )
            got = jnp.reshape(upd.dH, (n_ctx, width))
            dH_parts.append(got)
            if upd.jH is None:
                jH_parts.append(jnp.zeros_like(got))
            else:
                jH_used = True
                jH_parts.append(jnp.reshape(upd.jH, (n_ctx, width)))
            for target, drive in (upd.d_theta or {}).items():
                if target in dw_accum and dw_accum[target].shape != drive.shape:
                    raise ValueError(
                        f"hdp_rule {name!r}: components disagree on d_theta "
                        f"{target!r} shape {dw_accum[target].shape} vs {drive.shape}"
                    )
                dw_accum[target] = drive if target not in dw_accum else dw_accum[target] + drive
            for target, jump in (upd.j_theta or {}).items():
                if target in jw_accum and jw_accum[target].shape != jump.shape:
                    raise ValueError(
                        f"hdp_rule {name!r}: components disagree on j_theta "
                        f"{target!r} shape {jw_accum[target].shape} vs {jump.shape}"
                    )
                jw_accum[target] = jump if target not in jw_accum else jw_accum[target] + jump
        return HDPRuleUpdate(
            dH=jnp.concatenate(dH_parts, axis=-1),
            d_theta=dict(dw_accum) or None,
            jH=jnp.concatenate(jH_parts, axis=-1) if jH_used else None,
            j_theta=dict(jw_accum) or None,
        )

    descriptor = HDPRuleDescriptor(
        name=name,
        h_coords=tuple(coords),
        h_shape=(running,),
        theta_targets=tuple(targets),
        scope=first.scope,
        h_bounds=first.h_bounds,
        w_bounds=w_bounds,
        b_bounds=first.b_bounds,
        default_params=namespaced_defaults,
    )
    register_hdp_rule(descriptor, _superposed_step)
    return descriptor


def _np_prod(shape: tuple[int, ...]) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out
