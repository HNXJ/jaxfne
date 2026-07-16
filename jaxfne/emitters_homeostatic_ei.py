"""The second canonical HDP sanity emitter: a minimal 2-neuron E/I circuit.

Docs: ``docs/api/emitters.md`` — update that page when this module's public
API changes.

Companion to the Izhikevich/edge-list HDP scaffold in :mod:`jaxfne.emitters`
(the first canonical HDP sanity circuit is ``scripts/major_sanity_test.py``,
an 8-neuron Izhikevich/NeuronalTensor script). This module is the smallest
dynamical system that exercises HDP: state ``x = [E, I]``, an explicit
differentiable conductance matrix ``G`` (2x2 here, generalizes to N x N),
and HDP state ``H = [H_e, H_i]``, updated as three explicit, separately
staged dynamical systems rather than one fused rule:

    dx/dt = f(x, G, u)        fast neuronal dynamics
    dG/dt = f_G(x, H)         intermediate conductance adaptation
    dH/dt = f_H(x, H)         slow HDP homeostasis

``E``/``I`` are continuous, bounded, differentiable rate-like state (not a
hard Izhikevich threshold+reset) -- ``spikes`` in the returned tuple is a
threshold-crossing indicator derived from ``x`` for output parity with every
other emitter, not a non-differentiable reset; gradients flow through ``x``
itself with no surrogate-gradient machinery required on this circuit.

This is a computational scaffold: native units are not physical currents
unless a future calibration bridge declares otherwise (same caveat as every
other emitter in this package -- see ``jaxfne.emitters``'s module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Rule signatures (documented contract, not enforced at runtime beyond arity):
#   activation_rule(x, G, u)  -> dx   (shape (N,))
#   conductance_rule(x, H, G) -> dG   (shape (N,N))
#   homeostasis_rule(x, H)    -> dH   (shape (N,))
# Each rule is a plain JAX-traceable function -- pure, no side effects, no
# reliance on Python-level state beyond its own closed-over defaults.
# ---------------------------------------------------------------------------


def _activation_linear(x: jax.Array, G: jax.Array, u: jax.Array) -> jax.Array:
    """dx = -x + G @ x + u  (linear recurrent drive, leak to 0)."""
    return -x + G @ x + u


def _activation_cubic(x: jax.Array, G: jax.Array, u: jax.Array) -> jax.Array:
    """dx = -x^3 + G @ x + u  (cubic self-damping, bounds runaway growth)."""
    return -(x ** 3) + G @ x + u


def _activation_logistic(x: jax.Array, G: jax.Array, u: jax.Array) -> jax.Array:
    """dx = -x + G @ sigmoid(x) + u  (bounded logistic nonlinearity on drive)."""
    return -x + G @ jax.nn.sigmoid(x) + u


ACTIVATION_RULES: dict[str, Callable[[jax.Array, jax.Array, jax.Array], jax.Array]] = {
    "linear": _activation_linear,
    "cubic": _activation_cubic,
    "logistic": _activation_logistic,
}


def _conductance_hebbian(x: jax.Array, H: jax.Array, G: jax.Array) -> jax.Array:
    """dG_ij = H_i * x_i * x_j - G_ij  (Hebbian co-activation, H-gated, self-decaying)."""
    outer = jnp.outer(x, x)
    return H[:, None] * outer - G


def _conductance_bcm(x: jax.Array, H: jax.Array, G: jax.Array) -> jax.Array:
    """dG_ij = x_i * x_j * (x_j - H_i) - G_ij  (BCM sliding threshold set by H_i)."""
    outer_pre = x[None, :]
    outer_post = x[:, None]
    threshold = H[:, None]
    return outer_post * outer_pre * (outer_pre - threshold) - G


def _conductance_linear(x: jax.Array, H: jax.Array, G: jax.Array) -> jax.Array:
    """dG_ij = H_i * x_j - G_ij  (linear presynaptic-activity-gated drive)."""
    return H[:, None] * x[None, :] - G


CONDUCTANCE_RULES: dict[str, Callable[[jax.Array, jax.Array, jax.Array], jax.Array]] = {
    "hebbian": _conductance_hebbian,
    "bcm": _conductance_bcm,
    "linear": _conductance_linear,
}


def _homeostasis_linear(x: jax.Array, H: jax.Array) -> jax.Array:
    """dH = -(x - 1.0) * H  (linear rate-drain toward a unit activity setpoint)."""
    return -(x - 1.0) * H


def _homeostasis_logistic(x: jax.Array, H: jax.Array) -> jax.Array:
    """dH = -(sigmoid(x) - 0.5) * H  (bounded logistic rate-drain)."""
    return -(jax.nn.sigmoid(x) - 0.5) * H


HOMEOSTASIS_RULES: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "linear": _homeostasis_linear,
    "logistic": _homeostasis_logistic,
}


def _resolve_rule(rule, registry: dict[str, Callable], kind: str) -> Callable:
    if callable(rule):
        return rule
    if rule in registry:
        return registry[rule]
    raise ValueError(f"unknown {kind} rule {rule!r}; expected one of {sorted(registry)} or a callable")


@dataclass(frozen=True)
class HomeostaticEIParams:
    """Parameter/state container for the minimal homeostatic E/I circuit.

    Fields:
    - x0: initial state, shape (N,) -- N=2 for the canonical E/I sanity circuit.
    - G0: initial conductance matrix, shape (N,N).
    - H0: initial HDP H-factor, shape (N,).
    - drive: external bias current, shape (N,).
    - tau_x_ms/tau_G_ms/tau_H_ms: the three explicit timescales (fast/intermediate/slow).
    - G_min/G_max/H_min/H_max: clipping bounds for numerical stability.
    - source_scale: output-scaling field, shape (N,) -- parity with every other emitter.
    - labels: cell-type labels, e.g. ("E", "I").
    """

    x0: jax.Array
    G0: jax.Array
    H0: jax.Array
    drive: jax.Array
    tau_x_ms: jax.Array
    tau_G_ms: jax.Array
    tau_H_ms: jax.Array
    G_min: jax.Array
    G_max: jax.Array
    H_min: jax.Array
    H_max: jax.Array
    source_scale: jax.Array
    labels: tuple[str, ...] = ("E", "I")
    activation_rule_name: str = "linear"
    conductance_rule_name: str = "hebbian"
    homeostasis_rule_name: str = "linear"
    source_calibration_status: str = "uncalibrated_homeostatic_ei_native_current"

    @property
    def n_neurons(self) -> int:
        """Documented public function `n_neurons`."""
        return int(self.x0.shape[0])


def _homeostatic_ei_params_flatten(params: HomeostaticEIParams):
    children = (
        params.x0,
        params.G0,
        params.H0,
        params.drive,
        params.tau_x_ms,
        params.tau_G_ms,
        params.tau_H_ms,
        params.G_min,
        params.G_max,
        params.H_min,
        params.H_max,
        params.source_scale,
    )
    aux_data = {
        "labels": params.labels,
        "activation_rule_name": params.activation_rule_name,
        "conductance_rule_name": params.conductance_rule_name,
        "homeostasis_rule_name": params.homeostasis_rule_name,
        "source_calibration_status": params.source_calibration_status,
    }
    return children, aux_data


def _homeostatic_ei_params_unflatten(aux_data, children):
    return HomeostaticEIParams(
        x0=children[0],
        G0=children[1],
        H0=children[2],
        drive=children[3],
        tau_x_ms=children[4],
        tau_G_ms=children[5],
        tau_H_ms=children[6],
        G_min=children[7],
        G_max=children[8],
        H_min=children[9],
        H_max=children[10],
        source_scale=children[11],
        **aux_data,
    )


jax.tree_util.register_pytree_node(
    HomeostaticEIParams,
    _homeostatic_ei_params_flatten,
    _homeostatic_ei_params_unflatten,
)


def simulate_homeostatic_ei(
    params: HomeostaticEIParams,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    activation_rule: str | Callable = "linear",
    conductance_rule: str | Callable = "hebbian",
    homeostasis_rule: str | Callable = "linear",
    drive_schedule: jax.Array | None = None,
    noise_scale: float = 0.5,
    freeze_G: bool = False,
    freeze_H: bool = False,
    dtype: str = "float32",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """Run the minimal homeostatic E/I circuit for ``n_steps``.

    Three explicit, separately staged updates per step (never fused into one
    rule): fast ``x`` (neuronal), intermediate ``G`` (conductance adaptation,
    skipped when ``freeze_G``), slow ``H`` (HDP homeostasis, skipped when
    ``freeze_H``). ``freeze_G``/``freeze_H`` are plain Python bools closed
    over at trace time (not traced arrays) -- each distinct pair of values is
    its own compile, which is expected (e.g. Milestone 1 freezes both,
    Milestone 2 freezes only H, Milestone 3 freezes neither).

    Returns ``(voltages, spikes, sources, G_history, H_history, diagnostics)``:
    - voltages: shape (n_steps, N) -- the raw ``x`` trajectory (membrane-like
      state; not source-scaled).
    - spikes: shape (n_steps, N) -- threshold-crossing indicator on ``x``
      (x > 0), NOT a hard reset; gradients flow through ``x`` unaffected.
    - sources: shape (n_steps, N) -- ``source_scale``-projected proxy of
      ``x``, the same role every other emitter's "source tensor" output
      plays for downstream field projection (see ``jaxfne.emitters``'s
      ``simulate_dynamic_ei_coupling`` for the analogous convention).
    - G_history: shape (n_steps, N, N).
    - H_history: shape (n_steps, N).
    - diagnostics: dict with "error" (bool, True if any non-finite value was
      produced) and "final_state" (dict of the last x/G/H).
    """
    act_fn = _resolve_rule(activation_rule, ACTIVATION_RULES, "activation")
    cond_fn = _resolve_rule(conductance_rule, CONDUCTANCE_RULES, "conductance")
    homeo_fn = _resolve_rule(homeostasis_rule, HOMEOSTASIS_RULES, "homeostasis")

    jnp_dtype = jnp.dtype(dtype)
    x0 = jnp.asarray(params.x0, dtype=jnp_dtype)
    G0 = jnp.asarray(params.G0, dtype=jnp_dtype)
    H0 = jnp.asarray(params.H0, dtype=jnp_dtype)
    drive = jnp.asarray(params.drive, dtype=jnp_dtype)
    dt_x = jnp.asarray(dt_ms, dtype=jnp_dtype) / jnp.asarray(params.tau_x_ms, dtype=jnp_dtype)
    dt_G = jnp.asarray(dt_ms, dtype=jnp_dtype) / jnp.asarray(params.tau_G_ms, dtype=jnp_dtype)
    dt_H = jnp.asarray(dt_ms, dtype=jnp_dtype) / jnp.asarray(params.tau_H_ms, dtype=jnp_dtype)
    G_min = jnp.asarray(params.G_min, dtype=jnp_dtype)
    G_max = jnp.asarray(params.G_max, dtype=jnp_dtype)
    H_min = jnp.asarray(params.H_min, dtype=jnp_dtype)
    H_max = jnp.asarray(params.H_max, dtype=jnp_dtype)

    if drive_schedule is None:
        xs = jnp.zeros((n_steps,) + x0.shape, dtype=jnp_dtype)
    else:
        xs = jnp.asarray(drive_schedule, dtype=jnp_dtype)

    def step(carry, xs_t):
        x, G, H, rng = carry
        rng, nkey = jax.random.split(rng)
        noise = jnp.asarray(noise_scale, dtype=jnp_dtype) * jax.random.normal(nkey, shape=x.shape, dtype=jnp_dtype)
        u = drive + xs_t + noise

        dx = act_fn(x, G, u)
        x_next = x + dt_x * dx

        dG_raw = cond_fn(x, H, G)
        dG = jnp.where(freeze_G, jnp.zeros_like(dG_raw), dG_raw)
        G_next = jnp.clip(G + dt_G * dG, G_min, G_max)

        dH_raw = homeo_fn(x, H)
        dH = jnp.where(freeze_H, jnp.zeros_like(dH_raw), dH_raw)
        H_next = jnp.clip(H + dt_H * dH, H_min, H_max)

        new_carry = (x_next, G_next, H_next, rng)
        outputs = (x_next, G_next, H_next)
        return new_carry, outputs

    init_carry = (x0, G0, H0, key)
    final_carry, (x_hist, G_hist, H_hist) = jax.lax.scan(step, init_carry, xs, length=n_steps)

    source_scale = jnp.asarray(params.source_scale, dtype=jnp_dtype)
    voltages = x_hist
    spikes = (x_hist > 0.0).astype(jnp_dtype)
    sources = x_hist * source_scale[None, :]

    x_final, G_final, H_final, _ = final_carry
    all_finite = jnp.all(jnp.isfinite(x_hist)) & jnp.all(jnp.isfinite(G_hist)) & jnp.all(jnp.isfinite(H_hist))
    diagnostics = {
        "error": jnp.logical_not(all_finite),
        "final_state": {"x": x_final, "G": G_final, "H": H_final},
    }
    return voltages, spikes, sources, G_hist, H_hist, diagnostics
