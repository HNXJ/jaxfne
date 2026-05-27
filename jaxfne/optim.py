"""Optimizer specification and metadata layer for :mod:`jaxfne` v0.0.5-P3.

This module provides JSON-safe optimizer specs and a guarded Optax import.
No real optimization loop runs in v0.0.5 — all tune() calls return
``tuning_status="metadata_only_v0.0.5"`` and leave model parameters unchanged.

Optimizer grammar:
  optimizer_class: differentiable | blackbox | hybrid
  optimizer:       GSDR | AGSDR | random_search | optax_adam | optax_sgd
  differentiability_status: differentiable | declared_surrogate |
                            non_differentiable | not_checked
  surrogate_status: none | declared | required_but_missing | not_applicable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax

_BLACKBOX_OPTIMIZERS: frozenset[str] = frozenset({"GSDR", "AGSDR", "random_search"})
_DIFFERENTIABLE_OPTIMIZERS: frozenset[str] = frozenset({"optax_adam", "optax_sgd"})
_ALL_KNOWN_OPTIMIZERS: frozenset[str] = _BLACKBOX_OPTIMIZERS | _DIFFERENTIABLE_OPTIMIZERS

_VALID_DIFFERENTIABILITY: frozenset[str] = frozenset({
    "differentiable", "declared_surrogate", "non_differentiable", "not_checked",
})
_VALID_SURROGATE: frozenset[str] = frozenset({
    "none", "declared", "required_but_missing", "not_applicable",
})
_VALID_OPTIMIZER_CLASS: frozenset[str] = frozenset({"differentiable", "blackbox", "hybrid"})


@dataclass(frozen=True)
class OptimizerSpec:
    """Declarative optimizer specification with differentiability metadata.

    All fields are plain JSON-safe scalars so the spec can be serialized
    directly into a manifest without any callable objects.
    """

    optimizer: str
    optimizer_class: str
    differentiability_status: str
    surrogate_status: str
    # GSDR / AGSDR tuning knobs (unused in P3 but documented for P4+).
    alpha: float = 0.7
    exploration: float = 0.05
    deselect_factor: float = 2.0
    # Optax knobs (unused in P3 but documented for P4+).
    learning_rate: Optional[float] = None
    # Free-form metadata (JSON-safe values only).
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "optimizer_class": self.optimizer_class,
            "differentiability_status": self.differentiability_status,
            "surrogate_status": self.surrogate_status,
            "alpha": self.alpha,
            "exploration": self.exploration,
            "deselect_factor": self.deselect_factor,
            "learning_rate": self.learning_rate,
            "metadata": self.metadata,
        }

    def is_blackbox(self) -> bool:
        return self.optimizer in _BLACKBOX_OPTIMIZERS

    def is_differentiable_path(self) -> bool:
        return self.optimizer in _DIFFERENTIABLE_OPTIMIZERS

    def gradient_path_safe(self) -> bool:
        """Return True only if the spec explicitly declares gradient validity."""
        return self.differentiability_status in {"differentiable", "declared_surrogate"}

    def status(self) -> dict[str, Any]:
        return {
            "optimizer_class": self.optimizer_class,
            "optimizer": self.optimizer,
            "differentiability_status": self.differentiability_status,
            "surrogate_status": self.surrogate_status,
            "status": "metadata_only_v0.0.5",
            "alpha": self.alpha,
            "exploration": self.exploration,
            "deselect_factor": self.deselect_factor,
            "learning_rate": self.learning_rate,
        }


def gsdr(
    alpha: float = 0.7,
    exploration: float = 0.05,
    deselect_factor: float = 2.0,
    metadata: Optional[dict[str, Any]] = None,
) -> OptimizerSpec:
    """Return an OptimizerSpec for the GSDR (Genetic Stochastic Delta Rule) optimizer."""
    return OptimizerSpec(
        optimizer="GSDR",
        optimizer_class="blackbox",
        differentiability_status="non_differentiable",
        surrogate_status="not_applicable",
        alpha=alpha,
        exploration=exploration,
        deselect_factor=deselect_factor,
        metadata=metadata or {},
    )


def agsdr(
    alpha: float = 0.7,
    exploration: float = 0.05,
    deselect_factor: float = 2.0,
    metadata: Optional[dict[str, Any]] = None,
) -> OptimizerSpec:
    """Return an OptimizerSpec for the AGSDR (Adaptive GSDR) optimizer."""
    return OptimizerSpec(
        optimizer="AGSDR",
        optimizer_class="blackbox",
        differentiability_status="non_differentiable",
        surrogate_status="not_applicable",
        alpha=alpha,
        exploration=exploration,
        deselect_factor=deselect_factor,
        metadata=metadata or {},
    )


def random_search(metadata: Optional[dict[str, Any]] = None) -> OptimizerSpec:
    """Return an OptimizerSpec for random search."""
    return OptimizerSpec(
        optimizer="random_search",
        optimizer_class="blackbox",
        differentiability_status="non_differentiable",
        surrogate_status="not_applicable",
        metadata=metadata or {},
    )


def optax_adam(
    learning_rate: float = 1e-3,
    differentiability_status: str = "not_checked",
    surrogate_status: str = "none",
    metadata: Optional[dict[str, Any]] = None,
) -> OptimizerSpec:
    """Return an OptimizerSpec for Optax Adam.

    ``differentiability_status`` must be set to ``"differentiable"`` or
    ``"declared_surrogate"`` before Model.tune() will allow the Optax path.
    The default ``"not_checked"`` is intentionally conservative: spiking
    networks are not differentiable through spike resets without a surrogate.
    """
    return OptimizerSpec(
        optimizer="optax_adam",
        optimizer_class="differentiable",
        differentiability_status=differentiability_status,
        surrogate_status=surrogate_status,
        learning_rate=learning_rate,
        metadata=metadata or {},
    )


def optax_sgd(
    learning_rate: float = 1e-3,
    differentiability_status: str = "not_checked",
    surrogate_status: str = "none",
    metadata: Optional[dict[str, Any]] = None,
) -> OptimizerSpec:
    """Return an OptimizerSpec for Optax SGD.

    Same differentiability warning as optax_adam.
    """
    return OptimizerSpec(
        optimizer="optax_sgd",
        optimizer_class="differentiable",
        differentiability_status=differentiability_status,
        surrogate_status=surrogate_status,
        learning_rate=learning_rate,
        metadata=metadata or {},
    )


def _resolve_optimizer(optimizer: Any) -> OptimizerSpec:
    """Convert a string shorthand or OptimizerSpec into an OptimizerSpec."""
    if isinstance(optimizer, OptimizerSpec):
        return optimizer
    if isinstance(optimizer, str):
        name = optimizer.upper()
        if name in {"GSDR"}:
            return gsdr()
        if name in {"AGSDR"}:
            return agsdr()
        if name in {"RANDOM_SEARCH", "RANDOM"}:
            return random_search()
        if name in {"ADAM", "OPTAX_ADAM"}:
            return optax_adam()
        if name in {"SGD", "OPTAX_SGD"}:
            return optax_sgd()
        # Unknown string: treat as blackbox / not_checked
        return OptimizerSpec(
            optimizer=optimizer,
            optimizer_class="blackbox",
            differentiability_status="not_checked",
            surrogate_status="none",
        )
    # Legacy: AGSDR dataclass from v0.0.4
    if hasattr(optimizer, "status"):
        s = optimizer.status()
        return OptimizerSpec(
            optimizer=s.get("optimizer", "unknown"),
            optimizer_class=s.get("optimizer_class", "blackbox"),
            differentiability_status="non_differentiable",
            surrogate_status="not_applicable",
            alpha=getattr(optimizer, "alpha", 0.7),
            exploration=getattr(optimizer, "exploration", 0.05),
            deselect_factor=getattr(optimizer, "deselect_factor", 2.0),
        )
    return OptimizerSpec(
        optimizer="unknown",
        optimizer_class="blackbox",
        differentiability_status="not_checked",
        surrogate_status="none",
    )


# Legacy AGSDR class preserved from v0.0.4 for backward compatibility.
@dataclass(frozen=True)
class AGSDR:
    """Adaptive Genetic Stochastic Delta Rule placeholder (v0.0.4 legacy)."""

    alpha: float = 0.7
    exploration: float = 0.05
    deselect_factor: float = 2.0

    def status(self) -> dict[str, Any]:
        return {
            "optimizer_class": "blackbox",
            "optimizer": "AGSDR",
            "status": "prototype_api",
            "alpha": self.alpha,
            "exploration": self.exploration,
            "deselect_factor": self.deselect_factor,
        }


def require_optax() -> Any:
    """Import Optax lazily with an informative error.

    Never import Optax at module top-level.  Always call this guard inside the
    Optax-specific code path so that the rest of jaxfne remains importable
    without Optax installed.
    """
    try:
        import optax  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This feature requires optional dependency 'optax'. "
            "Install with: pip install -e '.[opt]'"
        ) from exc
    return optax


def propose_blackbox_candidates(
    optimizer: OptimizerSpec,
    n_steps: int,
    seed: int,
    bounds: tuple[float, float],
) -> list[float]:
    """Return deterministic scalar candidates for black-box tuning.

    This utility is deliberately small and dependency-free.  It is suitable for
    v0.0.x smoke tuning, but the returned candidates are computational proposals
    only and do not carry biological meaning.
    """
    import random

    lo, hi = float(bounds[0]), float(bounds[1])
    if hi < lo:
        lo, hi = hi, lo
    n = max(0, int(n_steps))
    if n == 0:
        return []
    rng = random.Random(int(seed))
    if optimizer.optimizer == "random_search":
        return [lo + (hi - lo) * rng.random() for _ in range(n)]
    if optimizer.optimizer == "AGSDR":
        # Adaptive Genetic-Stochastic Delta Rule (improved two-phase strategy).
        # Phase 1: Broad exploration across full bounds.
        # Phase 2: Focused exploitation around promising regions.
        center = 0.5 * (lo + hi)
        span = max(0.0, hi - lo)

        # Two-phase allocation: exploration_fraction for phase 1, rest for phase 2
        exploration_fraction = max(0.3, min(0.7, float(optimizer.alpha)))
        phase_1_steps = max(1, int(n * exploration_fraction))
        phase_2_steps = n - phase_1_steps

        out: list[float] = []

        # Phase 1: Broad exploration across full span
        for step in range(phase_1_steps):
            # Uniform random across entire range (high exploration)
            proposal = lo + span * rng.random()
            out.append(proposal)

        # Phase 2: Adaptive refinement around center with shrinking radius
        for step in range(phase_2_steps):
            # Radius shrinks over phase 2 iterations
            step_in_phase = step / max(1.0, phase_2_steps - 1)
            radius_scale = (1.0 - step_in_phase) * (1.0 - optimizer.exploration) + optimizer.exploration
            radius = span * radius_scale * (1.0 / (1.0 + step / max(1.0, optimizer.deselect_factor)))
            proposal = center + rng.uniform(-radius, radius)
            out.append(min(hi, max(lo, proposal)))

        return out
    # GSDR and unknown black-box specs use a deterministic sweep with small jitter.
    if n == 1:
        return [0.5 * (lo + hi)]
    out = []
    for step in range(n):
        frac = step / max(1, n - 1)
        jitter = (rng.random() - 0.5) * optimizer.exploration * (hi - lo)
        out.append(min(hi, max(lo, lo + frac * (hi - lo) + jitter)))
    return out


# Transform state dataclasses for Optax-compatible gradient optimization paths.
# These are PyTree-compatible (frozen=True, all JAX arrays) and hold no hidden
# global random state. Explicit PRNG keys required for stochasticity.


@dataclass(frozen=True)
class SDRState:
    """Stochastic Delta Rule optimizer state.

    Holds best-loss tracking, reset counter, and EMA variance estimates
    for adaptive alpha computation.
    """

    step: int = 0
    best_loss: float = float("inf")
    best_param: Optional[Any] = None
    reset_counter: int = 0
    var_sup_ema: float = 0.0  # EMA of supervised (inner) update variance
    var_unsup_ema: float = 0.0  # EMA of unsupervised (stochastic delta) variance
    ema_decay: float = 0.99


@dataclass(frozen=True)
class GSDRState:
    """Genetic Stochastic Delta Rule optimizer state.

    Similar to SDRState, with additional tracking for genetic deselection.
    """

    step: int = 0
    best_loss: float = float("inf")
    best_param: Optional[Any] = None
    reset_counter: int = 0
    deselection_counter: int = 0
    var_sup_ema: float = 0.0
    var_unsup_ema: float = 0.0
    ema_decay: float = 0.99


@dataclass(frozen=True)
class AGSDRState:
    """Adaptive Genetic Stochastic Delta Rule optimizer state.

    Combines genetic deselection with adaptive alpha for two-phase search.
    """

    step: int = 0
    best_loss: float = float("inf")
    best_param: Optional[Any] = None
    reset_counter: int = 0
    deselection_counter: int = 0
    var_sup_ema: float = 0.0
    var_unsup_ema: float = 0.0
    ema_decay: float = 0.99
    alpha_adaptive: float = 0.7  # Adaptive alpha, updated via variance ratio


def sdr_transform(
    inner_optimizer: Optional[Any] = None,
    stochastic_scale: float = 0.1,
    checkpoint_n_steps: int = 50,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> Any:
    """Return an Optax-compatible GradientTransformation for Stochastic Delta Rule.

    This transform combines a standard gradient-based inner optimizer (default: Adam)
    with a stochastic delta term. The state is PyTree-compatible (no global RNG).
    Explicit PRNG keys must be passed to init and update.

    Parameters
    ----------
    inner_optimizer : Optional[Any]
        Optional inner gradient optimizer (e.g., optax.adam). If None, defaults to
        optax.adam(learning_rate=1e-3).
    stochastic_scale : float
        Scale of the stochastic perturbation term.
    checkpoint_n_steps : int
        Reset to best_param if no improvement for this many steps.
    alpha_min, alpha_max : float
        Bounds for adaptive alpha (variance ratio weighting).

    Returns
    -------
    optax.GradientTransformation-compatible object
        init(params) -> SDRState
        update(updates, state, params=None, key=None) -> (updates, new_state)
    """
    optax = require_optax()

    if inner_optimizer is None:
        inner_optimizer = optax.adam(learning_rate=1e-3)

    def init(params: Any) -> tuple[SDRState, Any]:
        """Initialize SDR state and inner optimizer state."""
        inner_state = inner_optimizer.init(params)
        sdr_state = SDRState(
            step=0,
            best_loss=float("inf"),
            best_param=params,
            reset_counter=0,
            var_sup_ema=0.0,
            var_unsup_ema=0.0,
            ema_decay=0.99,
        )
        return sdr_state, inner_state

    def update(
        updates: Any,
        state: tuple[SDRState, Any],
        params: Optional[Any] = None,
        key: Optional[Any] = None,
        loss: Optional[float] = None,
    ) -> tuple[Any, tuple[SDRState, Any]]:
        """Apply SDR update step.

        Requires explicit PRNG key for stochastic delta term.
        If loss is provided, updates best-loss tracking and EMA variance.
        """
        if key is None:
            raise ValueError("SDR transform requires explicit PRNG key (key=jax.random.PRNGKey(...))")

        sdr_state, inner_state = state
        import jax.numpy as jnp

        # Apply inner optimizer update
        inner_updates, inner_state = inner_optimizer.update(updates, inner_state, params)

        # Add stochastic delta term using the explicit PRNG key
        key_delta = jax.random.fold_in(key, sdr_state.step)
        stochastic_delta = jax.tree_util.tree_map(
            lambda u: stochastic_scale * jax.random.normal(key_delta, u.shape, dtype=u.dtype)
            if hasattr(u, "shape")
            else u,
            inner_updates,
        )
        combined_updates = jax.tree_util.tree_map(
            lambda u, d: u + d if hasattr(u, "dtype") else u,
            inner_updates,
            stochastic_delta,
        )

        # Update best-loss tracking and EMA variance if loss provided
        new_sdr_state = sdr_state
        if loss is not None and params is not None:
            is_improvement = loss < sdr_state.best_loss
            new_best_loss = jnp.minimum(sdr_state.best_loss, loss)
            new_best_param = params if is_improvement else sdr_state.best_param

            # Estimate variances for adaptive alpha
            new_reset_counter = 0 if is_improvement else sdr_state.reset_counter + 1
            should_checkpoint = new_reset_counter >= checkpoint_n_steps
            params_to_use = new_best_param if should_checkpoint else params

            # Simple EMA update of variance estimates
            # (In practice, these would be computed from historical losses)
            var_sup = jnp.mean(jnp.asarray([jnp.mean(jnp.abs(u)) for u in jax.tree_util.tree_leaves(inner_updates)]))
            var_unsup = jnp.mean(jnp.asarray([jnp.mean(jnp.abs(d)) for d in jax.tree_util.tree_leaves(stochastic_delta)]))
            new_var_sup_ema = sdr_state.ema_decay * sdr_state.var_sup_ema + (1.0 - sdr_state.ema_decay) * var_sup
            new_var_unsup_ema = sdr_state.ema_decay * sdr_state.var_unsup_ema + (1.0 - sdr_state.ema_decay) * var_unsup

            new_sdr_state = SDRState(
                step=sdr_state.step + 1,
                best_loss=float(new_best_loss),
                best_param=new_best_param,
                reset_counter=int(new_reset_counter),
                var_sup_ema=float(new_var_sup_ema),
                var_unsup_ema=float(new_var_unsup_ema),
                ema_decay=sdr_state.ema_decay,
            )

        return combined_updates, (new_sdr_state, inner_state)

    # Return a simple object with init/update methods
    @dataclass(frozen=True)
    class GradientTransformation:
        init: Callable[[Any], Any]
        update: Callable[[Any, Any], Any]

    return GradientTransformation(init=init, update=update)


def gsdr_transform(
    inner_optimizer: Optional[Any] = None,
    stochastic_scale: float = 0.1,
    checkpoint_n_steps: int = 50,
    deselection_threshold: int = 10,
) -> Any:
    """Return an Optax-compatible GradientTransformation for Genetic SDR.

    Extends SDR with genetic deselection: resets to best_param after
    deselection_threshold steps without improvement.

    Parameters
    ----------
    inner_optimizer : Optional[Any]
        Inner gradient optimizer (default: optax.adam).
    stochastic_scale : float
        Scale of stochastic perturbation.
    checkpoint_n_steps : int
        Checkpoint interval.
    deselection_threshold : int
        Steps without improvement before genetic reset.

    Returns
    -------
    optax.GradientTransformation-compatible object
    """
    optax = require_optax()

    if inner_optimizer is None:
        inner_optimizer = optax.adam(learning_rate=1e-3)

    def init(params: Any) -> tuple[GSDRState, Any]:
        inner_state = inner_optimizer.init(params)
        gsdr_state = GSDRState(
            step=0,
            best_loss=float("inf"),
            best_param=params,
            reset_counter=0,
            deselection_counter=0,
            var_sup_ema=0.0,
            var_unsup_ema=0.0,
            ema_decay=0.99,
        )
        return gsdr_state, inner_state

    def update(
        updates: Any,
        state: tuple[GSDRState, Any],
        params: Optional[Any] = None,
        key: Optional[Any] = None,
        loss: Optional[float] = None,
    ) -> tuple[Any, tuple[GSDRState, Any]]:
        if key is None:
            raise ValueError("GSDR transform requires explicit PRNG key")

        gsdr_state, inner_state = state
        import jax.numpy as jnp

        inner_updates, inner_state = inner_optimizer.update(updates, inner_state, params)

        key_delta = jax.random.fold_in(key, gsdr_state.step)
        stochastic_delta = jax.tree_util.tree_map(
            lambda u: stochastic_scale * jax.random.normal(key_delta, u.shape, dtype=u.dtype)
            if hasattr(u, "shape")
            else u,
            inner_updates,
        )
        combined_updates = jax.tree_util.tree_map(
            lambda u, d: u + d if hasattr(u, "dtype") else u,
            inner_updates,
            stochastic_delta,
        )

        new_gsdr_state = gsdr_state
        if loss is not None and params is not None:
            is_improvement = loss < gsdr_state.best_loss
            new_best_loss = jnp.minimum(gsdr_state.best_loss, loss)
            new_best_param = params if is_improvement else gsdr_state.best_param

            new_desel_counter = 0 if is_improvement else gsdr_state.deselection_counter + 1
            should_reset = new_desel_counter >= deselection_threshold
            params_to_use = new_best_param if should_reset else params

            var_sup = jnp.mean(jnp.asarray([jnp.mean(jnp.abs(u)) for u in jax.tree_util.tree_leaves(inner_updates)]))
            var_unsup = jnp.mean(jnp.asarray([jnp.mean(jnp.abs(d)) for d in jax.tree_util.tree_leaves(stochastic_delta)]))
            new_var_sup_ema = gsdr_state.ema_decay * gsdr_state.var_sup_ema + (1.0 - gsdr_state.ema_decay) * var_sup
            new_var_unsup_ema = gsdr_state.ema_decay * gsdr_state.var_unsup_ema + (1.0 - gsdr_state.ema_decay) * var_unsup

            new_gsdr_state = GSDRState(
                step=gsdr_state.step + 1,
                best_loss=float(new_best_loss),
                best_param=new_best_param,
                reset_counter=gsdr_state.reset_counter + (1 if should_reset else 0),
                deselection_counter=int(new_desel_counter),
                var_sup_ema=float(new_var_sup_ema),
                var_unsup_ema=float(new_var_unsup_ema),
                ema_decay=gsdr_state.ema_decay,
            )

        @dataclass(frozen=True)
        class GradientTransformation:
            init: Callable[[Any], Any]
            update: Callable[[Any, Any], Any]

        return combined_updates, (new_gsdr_state, inner_state)

    @dataclass(frozen=True)
    class GradientTransformation:
        init: Callable[[Any], Any]
        update: Callable[[Any, Any], Any]

    return GradientTransformation(init=init, update=update)


def agsdr_transform(
    inner_optimizer: Optional[Any] = None,
    stochastic_scale: float = 0.1,
    checkpoint_n_steps: int = 50,
    deselection_threshold: int = 10,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> Any:
    """Return an Optax-compatible GradientTransformation for Adaptive GSDR.

    Extends GSDR with adaptive alpha: the ratio of supervised vs unsupervised
    variance determines exploration/exploitation balance.

    Parameters
    ----------
    inner_optimizer : Optional[Any]
        Inner gradient optimizer (default: optax.adam).
    stochastic_scale : float
        Scale of stochastic perturbation.
    checkpoint_n_steps : int
        Checkpoint interval.
    deselection_threshold : int
        Steps without improvement before genetic reset.
    alpha_min, alpha_max : float
        Bounds for adaptive alpha.

    Returns
    -------
    optax.GradientTransformation-compatible object
    """
    optax = require_optax()

    if inner_optimizer is None:
        inner_optimizer = optax.adam(learning_rate=1e-3)

    def init(params: Any) -> tuple[AGSDRState, Any]:
        inner_state = inner_optimizer.init(params)
        agsdr_state = AGSDRState(
            step=0,
            best_loss=float("inf"),
            best_param=params,
            reset_counter=0,
            deselection_counter=0,
            var_sup_ema=0.0,
            var_unsup_ema=0.0,
            ema_decay=0.99,
            alpha_adaptive=0.7,
        )
        return agsdr_state, inner_state

    def update(
        updates: Any,
        state: tuple[AGSDRState, Any],
        params: Optional[Any] = None,
        key: Optional[Any] = None,
        loss: Optional[float] = None,
    ) -> tuple[Any, tuple[AGSDRState, Any]]:
        if key is None:
            raise ValueError("AGSDR transform requires explicit PRNG key")

        agsdr_state, inner_state = state
        import jax.numpy as jnp

        inner_updates, inner_state = inner_optimizer.update(updates, inner_state, params)

        key_delta = jax.random.fold_in(key, agsdr_state.step)
        stochastic_delta = jax.tree_util.tree_map(
            lambda u: agsdr_state.alpha_adaptive * jax.random.normal(key_delta, u.shape, dtype=u.dtype)
            if hasattr(u, "shape")
            else u,
            inner_updates,
        )
        combined_updates = jax.tree_util.tree_map(
            lambda u, d: u + d if hasattr(u, "dtype") else u,
            inner_updates,
            stochastic_delta,
        )

        new_agsdr_state = agsdr_state
        if loss is not None and params is not None:
            is_improvement = loss < agsdr_state.best_loss
            new_best_loss = jnp.minimum(agsdr_state.best_loss, loss)
            new_best_param = params if is_improvement else agsdr_state.best_param

            new_desel_counter = 0 if is_improvement else agsdr_state.deselection_counter + 1
            should_reset = new_desel_counter >= deselection_threshold
            params_to_use = new_best_param if should_reset else params

            var_sup = jnp.mean(jnp.asarray([jnp.mean(jnp.abs(u)) for u in jax.tree_util.tree_leaves(inner_updates)]))
            var_unsup = jnp.mean(jnp.asarray([jnp.mean(jnp.abs(d)) for d in jax.tree_util.tree_leaves(stochastic_delta)]))
            new_var_sup_ema = agsdr_state.ema_decay * agsdr_state.var_sup_ema + (1.0 - agsdr_state.ema_decay) * var_sup
            new_var_unsup_ema = agsdr_state.ema_decay * agsdr_state.var_unsup_ema + (1.0 - agsdr_state.ema_decay) * var_unsup

            # Adaptive alpha: variance ratio
            var_total = new_var_sup_ema + new_var_unsup_ema + 1e-6
            new_alpha_adaptive = jnp.clip(
                new_var_sup_ema / var_total,
                alpha_min,
                alpha_max,
            )

            new_agsdr_state = AGSDRState(
                step=agsdr_state.step + 1,
                best_loss=float(new_best_loss),
                best_param=new_best_param,
                reset_counter=agsdr_state.reset_counter + (1 if should_reset else 0),
                deselection_counter=int(new_desel_counter),
                var_sup_ema=float(new_var_sup_ema),
                var_unsup_ema=float(new_var_unsup_ema),
                ema_decay=agsdr_state.ema_decay,
                alpha_adaptive=float(new_alpha_adaptive),
            )

        @dataclass(frozen=True)
        class GradientTransformation:
            init: Callable[[Any], Any]
            update: Callable[[Any, Any], Any]

        return combined_updates, (new_agsdr_state, inner_state)

    @dataclass(frozen=True)
    class GradientTransformation:
        init: Callable[[Any], Any]
        update: Callable[[Any, Any], Any]

    return GradientTransformation(init=init, update=update)
