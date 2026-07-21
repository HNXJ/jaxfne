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

import functools
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Rule signatures (documented contract, not enforced at runtime beyond arity):
#   activation_rule(x, G, u)        -> dx   (shape (N,))
#   conductance_rule(x, H, G, is_e) -> dG   (shape (N,N))
#   homeostasis_rule(x, H, is_e)    -> dH   (shape (N,))
# `is_e` is a static-derived boolean population mask (True=E, False=I/other),
# always passed by simulate_homeostatic_ei; rules that don't need cross-
# population information (linear/logistic/cubic_penalty, plain hebbian/bcm/
# linear conductance rules) simply ignore it via a trailing default.
# Each rule is a plain JAX-traceable function -- pure, no side effects, no
# reliance on Python-level state beyond its own closed-over defaults.
# ---------------------------------------------------------------------------


def _soft_bound(v: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """Smooth, gradient-friendly bound: maps any finite input into (lo, hi).

    `mid + half*tanh((v-mid)/half)` -- near-linear for `v` close to `mid`
    (`tanh(z)~=z` for small z), saturates smoothly toward `lo`/`hi` as `v`
    moves away, and can never leave `(lo, hi)` regardless of how large `v`
    gets (tanh's codomain is exactly `(-1, 1)`). Unlike `jnp.clip`, this is
    differentiable everywhere (no zero-gradient plateau at the boundary) and
    -- the property that actually matters for numerical safety -- it is a
    bounded *codomain*, not a repulsive *force*: a force-based barrier (e.g.
    this module's `cubic_penalty`, or the main HDP kernel's `barrier_c`/
    `barrier_d`) can still be outrun by a large enough Euler step (verified
    this session: `x`'s N=16 divergence was exactly this -- no bound at all,
    so nothing stopped an overshoot from compounding to `inf`). A soft_bound
    applied to the post-step value cannot produce a non-finite result from
    any finite input, independent of step size, N, or gain.
    """
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return mid + half * jnp.tanh((v - mid) / half)


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


def _conductance_hebbian(x: jax.Array, H: jax.Array, G: jax.Array, is_e: jax.Array | None = None) -> jax.Array:
    """dG_ij = H_i * x_i * x_j - G_ij  (Hebbian co-activation, H-gated, self-decaying)."""
    outer = jnp.outer(x, x)
    return H[:, None] * outer - G


def _conductance_bcm(x: jax.Array, H: jax.Array, G: jax.Array, is_e: jax.Array | None = None) -> jax.Array:
    """dG_ij = x_i * x_j * (x_j - H_i) - G_ij  (BCM sliding threshold set by H_i)."""
    outer_pre = x[None, :]
    outer_post = x[:, None]
    threshold = H[:, None]
    return outer_post * outer_pre * (outer_pre - threshold) - G


def _conductance_linear(x: jax.Array, H: jax.Array, G: jax.Array, is_e: jax.Array | None = None) -> jax.Array:
    """dG_ij = H_i * x_j - G_ij  (linear presynaptic-activity-gated drive)."""
    return H[:, None] * x[None, :] - G


def make_hebbian_pairwise_rule(
    k_ee: float = 1.0, k_ei: float = 1.0, k_ie: float = 1.0, k_ii: float = 1.0,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    """Build a Hebbian conductance_rule with independent gains per population
    pair -- E-E/E-I/I-E/I-I edges each adjust at their own rate, instead of
    one flat Hebbian rate for every edge. ``G[i,j]`` is the edge from
    presynaptic neuron ``j`` to postsynaptic neuron ``i`` (matching this
    module's existing convention: ``G @ x`` in every activation_rule sums
    over ``j``, and `_conductance_hebbian`'s `H[:, None]` already gates by
    the postsynaptic neuron's own H). Pair naming is (pre)-(post): ``k_ei``
    is the gain for edges FROM an E neuron TO an I neuron, ``k_ie`` is
    I-to-E, etc.

    With all 4 gains at their default of 1.0 this is mathematically
    identical to plain ``hebbian`` (verified by
    tests/test_homeostatic_ei_pairwise_conductance.py's equivalence check) --
    the gains are the only thing this adds. Requires ``is_e`` (unlike
    ``hebbian``/``bcm``/``linear``, which accept and ignore it) -- calling it
    without ``is_e`` raises, since a pairwise rule with no population
    information cannot be evaluated.

    User-confirmed scope: these 4 gains are fixed, independently-settable
    constants (via this factory), not a self-adjusting controller -- pass a
    custom-gain closure directly as ``conductance_rule=`` (the existing
    "real callable" path every rule kind already supports); there is no
    Configuration-level plumbing for custom gains, matching the same
    documented limitation as custom activation/homeostasis rules.
    """
    def rule(x: jax.Array, H: jax.Array, G: jax.Array, is_e: jax.Array) -> jax.Array:
        if is_e is None:
            raise ValueError("hebbian_pairwise requires is_e (a population mask); "
                              "simulate_homeostatic_ei always passes it, so this "
                              "should only happen when calling the rule directly.")
        is_e_post = is_e[:, None]
        is_e_pre = is_e[None, :]
        gain = jnp.where(
            is_e_post & is_e_pre, k_ee,
            jnp.where(
                is_e_post & ~is_e_pre, k_ie,   # pre=I -> post=E
                jnp.where(~is_e_post & is_e_pre, k_ei,  # pre=E -> post=I
                          k_ii),               # pre=I -> post=I
            ),
        )
        outer = jnp.outer(x, x)
        return gain * (H[:, None] * outer) - G

    return rule


CONDUCTANCE_RULES: dict[str, Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]] = {
    "hebbian": _conductance_hebbian,
    "bcm": _conductance_bcm,
    "linear": _conductance_linear,
    "hebbian_pairwise": make_hebbian_pairwise_rule(),
}


def _homeostasis_linear(x: jax.Array, H: jax.Array, is_e: jax.Array | None = None) -> jax.Array:
    """dH = -(x - 1.0) * H  (linear rate-drain toward a unit activity setpoint)."""
    return -(x - 1.0) * H


def _homeostasis_logistic(x: jax.Array, H: jax.Array, is_e: jax.Array | None = None) -> jax.Array:
    """dH = -(sigmoid(x) - 0.5) * H  (bounded logistic rate-drain)."""
    return -(jax.nn.sigmoid(x) - 0.5) * H


_CUBIC_PENALTY_BASELINE = 1.0
_CUBIC_PENALTY_DK = 0.5
_CUBIC_PENALTY_CROSS_GAIN = 0.15


def _homeostasis_cubic_penalty(x: jax.Array, H: jax.Array, is_e: jax.Array | None = None) -> jax.Array:
    """dH = -(x - 1.0) * H  [activity-driven rate drain, same as
    `_homeostasis_linear`] - dK * (H - baseline)^3 / (H + baseline)
    [explicit restoring force pulling H back toward `baseline`].

    `_homeostasis_linear`/`_homeostasis_logistic` are pure x-driven rate
    drain with no term pulling H back up once it starts falling -- verified
    this session to collapse H to its H_min clip floor at N=8 rather than
    settle at an interior point (the same failure mode documented in
    skills/FRICTIONS_STACK.md's F-017 for the Izhikevich/edge-list HDP
    kernel's rho_passive/H^2 formula). A first version of this rule dropped
    the x-coupling entirely (pure H-autonomous relaxation) -- verified this
    session that this makes H completely unresponsive to activity
    perturbations (a drive pulse could not move H at all, since the rule
    never reads x), which defeats the point of H tracking activity. This
    version keeps the same activity-coupling as `_homeostasis_linear` and
    ADDS the cubic term as a second, independent restoring force: the
    restoring term is zero exactly at H=baseline and grows cubically
    (stronger correction for larger deviations, gentle for small ones) as H
    moves away from it -- a genuine two-sided attractor, not a one-sided
    drain. The `/(H+baseline)` denominator keeps the correction's magnitude
    comparable across different absolute H scales; both H and baseline are
    always > 0 here (H_min > 0), so it never divides by zero.
    """
    baseline = jnp.asarray(_CUBIC_PENALTY_BASELINE, dtype=H.dtype)
    dK = jnp.asarray(_CUBIC_PENALTY_DK, dtype=H.dtype)
    activity_drain = -(x - 1.0) * H
    restoring = -dK * (H - baseline) ** 3 / (H + baseline)
    return activity_drain + restoring


def _homeostasis_cubic_penalty_coupled(x: jax.Array, H: jax.Array, is_e: jax.Array) -> jax.Array:
    """`_homeostasis_cubic_penalty` plus an explicit E<->I cross-population
    coupling term, requires `is_e` (boolean population mask, derived from
    `HomeostaticEIParams.labels` by `simulate_homeostatic_ei`).

    Every other homeostasis_rule here (`linear`/`logistic`/`cubic_penalty`)
    gives each neuron's dH a term that depends ONLY on that neuron's own x --
    reasoned through with the user (2026-07-16, "the HDP problem of two
    groups"): under the fast-x/intermediate-G/slow-H timescale separation,
    the long-term system reduces (after adiabatically eliminating x and G)
    to a 2D field in (H_e, H_i). A rule with no cross-neuron term gives that
    reduced field a DIAGONAL Jacobian -- stable only because each population
    is independently damped, not because of any real E/I interaction (matches
    the observed data: E and PV/I settle to independently stable, uncoupled
    rates in both canonical sanity tests). This rule adds the missing
    off-diagonal coupling: I's H rises when E's population-mean activity
    exceeds its target (so I's Hebbian conductance gain grows, strengthening
    the inhibition that should bring E back down); E's H falls when I's
    population-mean activity exceeds its target (dampening E's own outgoing
    weight growth). `_CUBIC_PENALTY_CROSS_GAIN=0.15` was chosen empirically:
    small enough not to prevent the existing interior-equilibrium/recovery
    properties from `cubic_penalty` (same tests pass), large enough to give a
    perturbation to ONE population a measurable, verified effect on the
    OTHER's H (see tests/test_homeostatic_ei_cross_population_coupling.py).
    """
    baseline = jnp.asarray(_CUBIC_PENALTY_BASELINE, dtype=H.dtype)
    dK = jnp.asarray(_CUBIC_PENALTY_DK, dtype=H.dtype)
    activity_drain = -(x - 1.0) * H
    restoring = -dK * (H - baseline) ** 3 / (H + baseline)

    is_e_f = is_e.astype(H.dtype)
    n_e = jnp.maximum(jnp.sum(is_e_f), 1.0)
    n_i = jnp.maximum(jnp.sum(1.0 - is_e_f), 1.0)
    mean_x_e = jnp.sum(jnp.where(is_e, x, 0.0)) / n_e
    mean_x_i = jnp.sum(jnp.where(is_e, 0.0, x)) / n_i
    cross_gain = jnp.asarray(_CUBIC_PENALTY_CROSS_GAIN, dtype=H.dtype)
    cross = jnp.where(is_e, -cross_gain * (mean_x_i - 1.0), cross_gain * (mean_x_e - 1.0))

    return activity_drain + restoring + cross


HOMEOSTASIS_RULES: dict[str, Callable[[jax.Array, jax.Array, jax.Array], jax.Array]] = {
    "linear": _homeostasis_linear,
    "logistic": _homeostasis_logistic,
    "cubic_penalty": _homeostasis_cubic_penalty,
    "cubic_penalty_coupled": _homeostasis_cubic_penalty_coupled,
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
    - G_min/G_max/H_min/H_max: clipping/soft-bound range for numerical stability.
    - source_scale: output-scaling field, shape (N,) -- parity with every other emitter.
    - x_min/x_max: soft-bound range for `x` (only used when `bound_mode="stable"` --
      `x` is otherwise unbounded, matching every prior session's behavior). Default
      to a very wide range so callers that don't pass them see zero behavior change.
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
    x_min: jax.Array = field(default_factory=lambda: jnp.asarray(-1.0e6))
    x_max: jax.Array = field(default_factory=lambda: jnp.asarray(1.0e6))
    labels: tuple[str, ...] = ("E", "I")
    activation_rule_name: str = "linear"
    conductance_rule_name: str = "hebbian"
    homeostasis_rule_name: str = "linear"
    bound_mode: str = "minimal"
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
        params.x_min,
        params.x_max,
    )
    aux_data = {
        "labels": params.labels,
        "activation_rule_name": params.activation_rule_name,
        "conductance_rule_name": params.conductance_rule_name,
        "homeostasis_rule_name": params.homeostasis_rule_name,
        "bound_mode": params.bound_mode,
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
        x_min=children[12],
        x_max=children[13],
        **aux_data,
    )


jax.tree_util.register_pytree_node(
    HomeostaticEIParams,
    _homeostatic_ei_params_flatten,
    _homeostatic_ei_params_unflatten,
)


def make_minimal_ei_params(
    n: int = 8, e_fraction: float = 0.75, *,
    x0_value: float = 0.1, H0_value: float = 0.3,
    drive_e: float = 0.5, drive_i: float = 0.3,
    tau_x_ms: float = 5.0, tau_G_ms: float = 200.0, tau_H_ms: float = 1000.0,
    G_max: float | None = None, H_min: float = 0.1, H_max: float = 10.0,
    x_min: float = -10.0, x_max: float = 10.0,
    source_scale_value: float = 1.0,
) -> HomeostaticEIParams:
    """Build a minimal all-pairwise E/I HomeostaticEIParams for any N -- the
    same shape as scripts/snt_pipeline.py's canonical N=8 circuit and
    jaxfne/_construct.py's construct()-path default, generalized to N=2/4/8
    (or any N>=2) for ad hoc scripts/experiments.

    Consolidates the near-identical `_make_params`/`_make` helper duplicated
    across tests/test_homeostatic_ei_cubic_penalty_rule.py,
    tests/test_homeostatic_ei_cross_population_coupling.py, and
    tests/test_homeostatic_ei_n_generalization.py into one real, reusable
    function -- and fixes a real latent bug those copies all shared: none of
    them set `labels` explicitly, so at N>2 they silently kept
    HomeostaticEIParams's default `labels=("E","I")` (length 2) while
    x0/H0/drive were length N. Harmless only because no existing N>2 test
    call combined that with a homeostasis_rule/conductance_rule that
    actually reads `is_e` (cubic_penalty_coupled/hebbian_pairwise would have
    hit a real shape mismatch). This function sets `labels` correctly for
    every N.

    `G_max` defaults to `10.0/n` (not a flat constant) -- verified this
    session: G0 is column-normalized so its own aggregate is N-invariant, but
    Hebbian adaptation clips every one of N row-entries toward the same
    ceiling, so the aggregate per-row feedback scales with N under a flat
    G_max. Holding `n * G_max` constant (=10.0, matching the N=2 canonical
    default's `2*5.0`) keeps that aggregate ceiling comparable across N.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2 (need both an E and an I neuron), got {n}")
    n_e = 1 if n <= 2 else max(1, min(n - 1, round(n * e_fraction)))
    n_i = n - n_e
    is_e = jnp.array([True] * n_e + [False] * n_i)
    labels = ("E",) * n_e + ("I",) * n_i
    drive = jnp.where(is_e, drive_e, drive_i)
    col = jnp.where(is_e, 0.5 / n_e, -0.5 / max(n_i, 1))
    G0 = jnp.broadcast_to(col[None, :], (n, n))
    if G_max is None:
        G_max = 10.0 / n
    return HomeostaticEIParams(
        x0=jnp.full((n,), x0_value), G0=G0, H0=jnp.full((n,), H0_value), drive=drive,
        tau_x_ms=jnp.array(tau_x_ms), tau_G_ms=jnp.array(tau_G_ms), tau_H_ms=jnp.array(tau_H_ms),
        G_min=jnp.array(-G_max), G_max=jnp.array(G_max), H_min=jnp.array(H_min), H_max=jnp.array(H_max),
        source_scale=jnp.full((n,), source_scale_value),
        x_min=jnp.array(x_min), x_max=jnp.array(x_max),
        labels=labels,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "n_steps", "activation_rule", "conductance_rule", "homeostasis_rule",
        "freeze_G", "freeze_H", "dtype", "bound_mode",
    ),
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
    bound_mode: str = "minimal",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """Run the minimal homeostatic E/I circuit for ``n_steps``.

    Three explicit, separately staged updates per step (never fused into one
    rule): fast ``x`` (neuronal), intermediate ``G`` (conductance adaptation,
    skipped when ``freeze_G``), slow ``H`` (HDP homeostasis, skipped when
    ``freeze_H``). ``freeze_G``/``freeze_H`` are plain Python bools closed
    over at trace time (not traced arrays) -- each distinct pair of values is
    its own compile, which is expected (e.g. Milestone 1 freezes both,
    Milestone 2 freezes only H, Milestone 3 freezes neither).

    ``bound_mode`` (static, one of "minimal"/"stable"):
    - "minimal" (default): today's behavior, unchanged -- ``jnp.clip`` on
      ``G``/``H``, ``x`` left entirely unbounded. Zero behavior change for
      any existing caller that doesn't pass ``bound_mode``.
    - "stable": ``_soft_bound`` (smooth tanh remap, see its docstring) applied
      to ``x``, ``G``, and ``H`` every step instead of ``jnp.clip`` -- a
      bounded codomain that cannot be numerically outrun by a large Euler
      step, unlike a hard clip or a force-based barrier. Verified this
      session to close the N=16 divergence found earlier (``x`` had no bound
      of any kind under "minimal", and explicit Euler on the cubic
      activation term overshot past its numerical stability radius).

    The whole function is ``jax.jit``-compiled (``n_steps`` and every rule/
    mode/dtype argument are static) -- repeated calls with the same static
    configuration reuse one compiled program instead of retracing. The scan
    carry is a single flattened ``(x, G, H)`` array (plus the PRNG key kept
    separate) rather than 3 separate named arrays, reducing per-step carry
    overhead across the (potentially very long, e.g. 100s/200k-step) scan.

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
    if bound_mode not in ("minimal", "stable"):
        raise ValueError(f"unknown bound_mode {bound_mode!r}; expected 'minimal' or 'stable'")

    # Transparency: G is an explicit dense (N,N) conductance matrix by design (this
    # emitter has no sparse variant, unlike the Izhikevich/edge-list path in
    # jaxfne._construct_population), and every step's full G is retained in the
    # returned G_history -- (n_steps, N, N). That is O(n_steps * N^2) memory, not just
    # O(N^2), and it is not a broadcast_to laziness artifact: G is a real carried scan
    # array from step 0, verified via direct measurement (N=1000, 50 steps ->
    # G_history.nbytes = 2.0e8, matching n_steps*N*N*4 bytes exactly). Warn once at
    # trace time when this would be large (Phase 3 0.4.8-0.4.48 roadmap; closes
    # plans::homeostatic-ei-g0-broadcast-per-step-realization-risk).
    _n0 = params.x0.shape[0]
    _hist_mb = (n_steps * _n0 * _n0 * (8 if jnp.dtype(dtype) == jnp.float64 else 4)) / 1e6
    if _hist_mb >= 500:
        import warnings as _warnings
        _warnings.warn(
            f"homeostatic_ei G_history at N={_n0}, n_steps={n_steps} materializes a "
            f"dense (n_steps, N, N) array (~{_hist_mb:.0f} MB) -- this emitter's "
            f"conductance matrix has no sparse variant. Reduce n_steps or N, or discard "
            f"G_history downstream if only the final state is needed.",
            RuntimeWarning,
            stacklevel=2,
        )

    act_fn = _resolve_rule(activation_rule, ACTIVATION_RULES, "activation")
    cond_fn = _resolve_rule(conductance_rule, CONDUCTANCE_RULES, "conductance")
    homeo_fn = _resolve_rule(homeostasis_rule, HOMEOSTASIS_RULES, "homeostasis")

    jnp_dtype = jnp.dtype(dtype)
    n = params.x0.shape[0]
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
    x_min = jnp.asarray(params.x_min, dtype=jnp_dtype)
    x_max = jnp.asarray(params.x_max, dtype=jnp_dtype)
    # labels is static aux data (a Python tuple), so this is a trace-time
    # constant, not a traced computation.
    is_e = jnp.array([lbl == "E" for lbl in params.labels])

    if drive_schedule is None:
        xs = jnp.zeros((n_steps,) + x0.shape, dtype=jnp_dtype)
    else:
        xs = jnp.asarray(drive_schedule, dtype=jnp_dtype)

    def _unravel(state_flat):
        x = state_flat[:n]
        G = state_flat[n:n + n * n].reshape(n, n)
        H = state_flat[n + n * n:n + n * n + n]
        return x, G, H

    def _ravel(x, G, H):
        return jnp.concatenate([x.ravel(), G.ravel(), H.ravel()])

    def step(carry, xs_t):
        state, rng = carry
        x, G, H = _unravel(state)
        rng, nkey = jax.random.split(rng)
        # Euler-Maruyama scaling: noise enters additively into `u`, which is
        # itself multiplied by dt_x inside dx (see every ACTIVATION_RULES
        # entry -- u is always a plain additive term). For the net
        # contribution to x_next to equal the correct diffusion term
        # `noise_scale * sqrt(dt_x) * randn()` (not `noise_scale * dt_x *
        # randn()`, which is what a plain `noise_scale * randn()` here would
        # give after the dt_x multiply), the pre-multiply draw must be scaled
        # by `1/sqrt(dt_x)`. Without this, the simulated process's stationary
        # statistics depend on the arbitrary choice of dt_ms rather than
        # converging to a fixed continuous-time SDE.
        noise_std = jnp.asarray(noise_scale, dtype=jnp_dtype) / jnp.sqrt(dt_x)
        noise = noise_std * jax.random.normal(nkey, shape=x.shape, dtype=jnp_dtype)
        u = drive + xs_t + noise

        dx = act_fn(x, G, u)
        x_raw = x + dt_x * dx

        dG_raw_val = cond_fn(x, H, G, is_e)
        # freeze_G/freeze_H are static Python bools (closed over at trace
        # time, not traced arrays) -- a plain Python branch removes the
        # unused side entirely from the compiled graph, instead of tracing
        # both sides of a jnp.where every step for a condition already known
        # before tracing starts.
        dG = jnp.zeros_like(dG_raw_val) if freeze_G else dG_raw_val
        G_raw = G + dt_G * dG

        dH_raw_val = homeo_fn(x, H, is_e)
        dH = jnp.zeros_like(dH_raw_val) if freeze_H else dH_raw_val
        H_raw = H + dt_H * dH

        if bound_mode == "stable":
            x_next = _soft_bound(x_raw, x_min, x_max)
            G_next = _soft_bound(G_raw, G_min, G_max)
            H_next = _soft_bound(H_raw, H_min, H_max)
        else:
            x_next = x_raw
            G_next = jnp.clip(G_raw, G_min, G_max)
            H_next = jnp.clip(H_raw, H_min, H_max)

        new_carry = (_ravel(x_next, G_next, H_next), rng)
        outputs = (x_next, G_next, H_next)
        return new_carry, outputs

    init_carry = (_ravel(x0, G0, H0), key)
    final_carry, (x_hist, G_hist, H_hist) = jax.lax.scan(step, init_carry, xs, length=n_steps)

    source_scale = jnp.asarray(params.source_scale, dtype=jnp_dtype)
    voltages = x_hist
    # Edge-triggered: a spike fires only on the rising zero-crossing
    # (previous state <= 0, current state > 0), not for every step x stays
    # positive -- makes `spikes` a genuine countable pulse train (a per-second
    # rate is meaningful), matching the convention every other emitter's
    # spikes output follows.
    x_prev = jnp.concatenate([x0[None, :], x_hist[:-1]], axis=0)
    spikes = ((x_hist > 0.0) & (x_prev <= 0.0)).astype(jnp_dtype)
    sources = x_hist * source_scale[None, :]

    x_final, G_final, H_final = _unravel(final_carry[0])
    all_finite = jnp.all(jnp.isfinite(x_hist)) & jnp.all(jnp.isfinite(G_hist)) & jnp.all(jnp.isfinite(H_hist))
    diagnostics = {
        "error": jnp.logical_not(all_finite),
        "final_state": {"x": x_final, "G": G_final, "H": H_final},
    }
    return voltages, spikes, sources, G_hist, H_hist, diagnostics
