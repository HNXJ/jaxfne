"""Optimizer specifications and execution helpers for :mod:`jaxfne`.

The public tutorial grammar constructs optimizer specs, for example
``jtfne.agsdr(parameters=...)``, and passes them into ``Model.tune``. This
module owns the implementation details: candidate proposal, AGSDR bookkeeping,
optional Optax guards, and small JAX-native helper functions used by black-box
or differentiable paths.

Optimizer grammar:
  optimizer_class: differentiable | blackbox | hybrid | multiparameter_blackbox
  optimizer:       GSDR | AGSDR | random_search | optax_adam | optax_sgd
  differentiability_status: differentiable | declared_surrogate |
                            non_differentiable | not_checked
  surrogate_status: none | declared | required_but_missing | not_applicable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

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


@jax.jit
def _quadratic_target_loss(
    achieved: jax.Array,
    target: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """JAX-native weighted squared-relative-error loss.

    This helper is intentionally small so it can be inspected with
    ``jax.make_jaxpr`` and differentiated by tests or downstream optimizers.
    The AGSDR black-box path does not rely on the gradient, but exposing this
    pure function keeps objective arithmetic compatible with ``jax.grad``.
    """
    # Infer dtype from inputs; default to float32 if inputs are Python scalars
    dtype = jnp.result_type(achieved, target, weights)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float32
    achieved = jnp.asarray(achieved, dtype=dtype)
    target = jnp.asarray(target, dtype=dtype)
    weights = jnp.asarray(weights, dtype=dtype)
    denom = jnp.maximum(jnp.abs(target), jnp.asarray(1e-6, dtype=dtype))
    rel = (achieved - target) / denom
    return jnp.sum(weights * rel * rel)


quadratic_target_loss_grad = jax.jit(jax.grad(_quadratic_target_loss, argnums=0))


@jax.jit
def _agsdr_candidates_from_noise(
    center: jax.Array,
    lows: jax.Array,
    highs: jax.Array,
    exploration: float,
    noise: jax.Array,
) -> jax.Array:
    """Return a vectorized AGSDR candidate population from standard-normal noise.

    ``evaluate_fn`` remains a Python black-box callback, but candidate proposal is
    JAX-native: one ``jit``-compiled function and one ``vmap`` over population
    rows.  Shapes are fixed by ``noise`` and parameter arrays.
    """
    # Infer dtype from inputs; default to float32 if inputs are Python scalars
    dtype = jnp.result_type(center, lows, highs, noise)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float32
    center = jnp.asarray(center, dtype=dtype)
    lows = jnp.asarray(lows, dtype=dtype)
    highs = jnp.asarray(highs, dtype=dtype)
    exploration_val = jnp.asarray(exploration, dtype=dtype)
    span = jnp.maximum(highs - lows, jnp.asarray(0.0, dtype=dtype))
    proposals = center[None, :] + exploration_val * span[None, :] * noise.astype(dtype)

    candidates = jnp.clip(proposals, lows[None, :], highs[None, :])
    # Fuse center boundary alignment: Candidate 0 remains locked exactly to the clipped center array
    return candidates.at[0].set(jnp.clip(center, lows, highs))


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
            "status": "optimizer_spec",
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


@dataclass(frozen=True)
class AGSDROptimizerSpec:
    """Multi-parameter AGSDR optimizer specification with execution parameters.

    This spec defines both the AGSDR algorithm parameters (alpha, exploration)
    and the execution context (parameters, generations, population_size, seed).
    When passed to Model.tune(), it triggers multi-parameter optimization.
    """

    parameters: dict  # {"param_name": (lower, upper) or MatrixParameterSpec, ...}
    generations: int = 8
    population_size: int = 6
    alpha: float = 0.65
    exploration: float = 0.18
    deselect_factor: float = 2.0
    seed: int = 0
    # Two-level optimization extensions (Suite No. 4)
    inner_optimizer: Any = None
    inner_steps: int = 0
    inner_objective: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dictionary.

        Optax objects in inner_optimizer are replaced with metadata strings
        so the result is always JSON-safe.
        """
        # Serialize parameters: MatrixParameterSpec -> metadata dict, tuples -> lists
        params_serial: dict[str, Any] = {}
        for k, v in self.parameters.items():
            try:
                # Check if it is a MatrixParameterSpec-like object
                if hasattr(v, "mask") and hasattr(v, "bounds"):
                    params_serial[k] = {
                        "type": "MatrixParameterSpec",
                        "mask": str(v.mask),
                        "bounds": [float(v.bounds[0]), float(v.bounds[1])],
                        "init": str(v.init),
                        "trainable": bool(v.trainable),
                    }
                else:
                    params_serial[k] = [float(v[0]), float(v[1])]
            except Exception:
                params_serial[k] = str(v)

        # Serialize inner_optimizer: only store metadata, never the Optax object
        inner_opt_meta: Any = None
        if self.inner_optimizer is not None:
            inner_opt_meta = "optax_optimizer_object_not_serialized"

        return {
            "optimizer": "AGSDR",
            "optimizer_class": "multiparameter_blackbox",
            "parameters": params_serial,
            "generations": int(self.generations),
            "population_size": int(self.population_size),
            "alpha": float(self.alpha),
            "exploration": float(self.exploration),
            "deselect_factor": float(self.deselect_factor),
            "seed": int(self.seed),
            "inner_optimizer": inner_opt_meta,
            "inner_steps": int(self.inner_steps),
            "inner_objective": self.inner_objective,
        }


def agsdr(
    alpha: float = 0.7,
    exploration: float = 0.05,
    deselect_factor: float = 2.0,
    metadata: Optional[dict[str, Any]] = None,
    # Multi-parameter path
    parameters: Optional[dict] = None,
    generations: Optional[int] = None,
    population_size: Optional[int] = None,
    seed: int = 0,
    # Two-level optimization: inner Adam refinement (Suite No. 4)
    inner_optimizer: Any = None,
    inner_steps: int = 0,
    inner_objective: Optional[str] = None,
) -> Any:
    """Return an optimizer spec for AGSDR.

    Two paths:
    1. Single-parameter (legacy): returns OptimizerSpec for scalar parameter tuning
    2. Multi-parameter (new): returns AGSDROptimizerSpec for multi-param optimization

    Parameters
    ----------
    alpha : float
        Step size for delta-rule center update (default 0.7 for legacy, 0.65 for multi-param).
    exploration : float
        Standard deviation scale for proposal distribution (default 0.05 for legacy, 0.18 for multi-param).
    deselect_factor : float
        Deselection factor for genetic algorithm (default 2.0).
    metadata : dict, optional
        Custom metadata (legacy path only).
    parameters : dict, optional
        Multi-parameter bounds: {"param_name": (lower, upper) or MatrixParameterSpec, ...}.
        If provided, returns AGSDROptimizerSpec.
    generations : int, optional
        Number of generations for multi-parameter optimization (default 8).
    population_size : int, optional
        Population per generation for multi-parameter optimization (default 6).
    seed : int
        Random seed (default 0).
    inner_optimizer : Any, optional
        Optax optimizer instance for the inner Adam refinement loop
        (e.g. optax.adam(learning_rate=1e-2)).  Requires Optax installed.
        Only used when parameters contains :class: entries.
    inner_steps : int
        Number of gradient steps in the inner Adam loop per AGSDR candidate.
    inner_objective : str, optional
        Name of the inner surrogate objective; None uses soft-rate MSE.

    Returns
    -------
    OptimizerSpec or AGSDROptimizerSpec
        OptimizerSpec for single-parameter path, AGSDROptimizerSpec for multi-parameter path.
    """
    # Multi-parameter path
    if parameters is not None:
        return AGSDROptimizerSpec(
            parameters=parameters,
            generations=int(generations) if generations is not None else 8,
            population_size=int(population_size) if population_size is not None else 6,
            alpha=float(alpha) if alpha != 0.7 else 0.65,  # Use 0.65 default for multi-param
            exploration=float(exploration) if exploration != 0.05 else 0.18,  # Use 0.18 default for multi-param
            deselect_factor=float(deselect_factor),
            seed=int(seed),
            inner_optimizer=inner_optimizer,
            inner_steps=int(inner_steps),
            inner_objective=inner_objective,
        )

    # Single-parameter path (legacy)
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
    """Convert string shorthand or optimizer-spec objects into OptimizerSpec."""
    if isinstance(optimizer, OptimizerSpec):
        return optimizer
    if isinstance(optimizer, AGSDROptimizerSpec):
        return OptimizerSpec(
            optimizer="AGSDR",
            optimizer_class="multiparameter_blackbox",
            differentiability_status="non_differentiable",
            surrogate_status="not_applicable",
            alpha=float(optimizer.alpha),
            exploration=float(optimizer.exploration),
            deselect_factor=float(optimizer.deselect_factor),
            metadata={
                "parameters": {
                    k: ([float(v[0]), float(v[1])] if not hasattr(v, "mask") else {"type": "MatrixParameterSpec", "mask": v.mask, "bounds": list(v.bounds)})
                    for k, v in optimizer.parameters.items()
                },
                "generations": int(optimizer.generations),
                "population_size": int(optimizer.population_size),
                "seed": int(optimizer.seed),
            },
        )
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
    """Legacy AGSDR adapter retained for old notebooks and tests.

    Prefer ``jtfne.agsdr(...)``.  This class only exposes metadata and does not
    execute optimization directly.
    """

    alpha: float = 0.7
    exploration: float = 0.05
    deselect_factor: float = 2.0

    def status(self) -> dict[str, Any]:
        return {
            "optimizer_class": "blackbox",
            "optimizer": "AGSDR",
            "status": "legacy_adapter",
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


def _run_agsdr_optimization_loop(
    evaluate_fn: Callable[[dict[str, float]], float],
    parameter_bounds: dict[str, tuple[float, float]],
    n_generations: int,
    n_population: int,
    alpha: float = 0.65,
    exploration: float = 0.18,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a stateful multi-parameter AGSDR (Adaptive Genetic Stochastic Delta Rule) loop.

    This function encapsulates the full AGSDR optimization workflow for black-box
    tuning of multiple scalar parameters. It is designed for notebook-style usage
    where you have a callable that evaluates candidate parameter dicts and returns
    a scalar loss/score. The function manages PRNG, parameter center tracking, and
    best-result history internally.

    Parameters
    ----------
    evaluate_fn : Callable[[dict[str, float]], float]
        User-supplied scoring function. Takes a dict mapping parameter names to
        float values and returns a scalar loss (lower is better).
    parameter_bounds : dict[str, tuple[float, float]]
        Bounds for each parameter: {"param_name": (lower, upper), ...}.
        All parameters are scalar floats.
    n_generations : int
        Number of optimization generations to run.
    n_population : int
        Number of candidates to evaluate per generation.
    alpha : float
        Step size for delta-rule center update (default 0.65).
        After each generation, theta_center is updated as:
            theta_center += alpha * (best_theta - theta_center)
    exploration : float
        Standard deviation scale for normal-distribution proposals (default 0.18).
        Candidates are sampled from normal(theta_center, exploration * span) and clipped.
    seed : int
        Random seed for reproducibility (default 0).

    Returns
    -------
    dict[str, Any]
        A dict with keys:
        - "best_parameters": dict mapping parameter names to optimal float values
        - "best_score": best (lowest) score achieved
        - "generation_records": list of dicts, one per generation:
            {"generation": int, "best_score": float, "best_parameters": dict}
        - "all_scores": list of all scores evaluated, in order
        - "all_candidates": list of all candidate dicts evaluated, in order

    Notes
    -----
    The AGSDR strategy is two-phase:
      - Phase 1 (0 to n_population//5): Propose from bounds center with high variance
      - Phase 2 (remaining): Propose from evolving center with decaying variance

    The function uses a simple Numpy random.Random() generator and does not
    require JAX. It is suitable for integration into Jupyter notebooks and
    Model.tune() as the "multi-parameter" optimization path.

    Example
    -------
    >>> def score_model(params):
    ...     m = model.with_parameters(params)
    ...     signals = m.simulate(...)
    ...     return model.evaluate(signals, objective)
    >>> bounds = {"drive_scale_a": (0.35, 2.25), "drive_scale_b": (0.35, 2.25)}
    >>> result = _run_agsdr_optimization_loop(
    ...     evaluate_fn=score_model,
    ...     parameter_bounds=bounds,
    ...     n_generations=8,
    ...     n_population=6,
    ...     alpha=0.65,
    ...     exploration=0.18,
    ...     seed=42,
    ... )
    >>> print(f"Best score: {result['best_score']}")
    >>> print(f"Best params: {result['best_parameters']}")
    """
    import random

    # Validate inputs
    if not parameter_bounds:
        raise ValueError("parameter_bounds must be non-empty dict")
    if n_generations < 1 or n_population < 1:
        raise ValueError("n_generations and n_population must be >= 1")

    # Initialize deterministic JAX key for vectorized population proposal.
    base_key = jax.random.PRNGKey(int(seed))

    # Extract parameter names and bounds
    param_names = sorted(parameter_bounds.keys())
    bounds_list = [parameter_bounds[name] for name in param_names]

    # Validate and normalize bounds
    normalized_bounds: list[tuple[float, float]] = []
    for name, (lo_raw, hi_raw) in zip(param_names, bounds_list):
        lo = float(lo_raw)
        hi = float(hi_raw)
        if not (lo == lo and hi == hi):
            raise ValueError(f"non-finite bounds for parameter {name!r}: {(lo_raw, hi_raw)!r}")
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            raise ValueError(f"degenerate bounds for parameter {name!r}: {(lo, hi)!r}")
        normalized_bounds.append((lo, hi))
    bounds_list = normalized_bounds

    # v0.3.17: derive working dtype from bounds values so float64 inputs are
    # preserved end-to-end without silent downcasting to float32.
    _raw_bounds = [b[0] for b in bounds_list] + [b[1] for b in bounds_list]
    _inferred = jnp.result_type(*_raw_bounds)
    _wdtype = _inferred if jnp.issubdtype(_inferred, jnp.floating) else jnp.float32

    lows = jnp.asarray([b[0] for b in bounds_list], dtype=_wdtype)
    highs = jnp.asarray([b[1] for b in bounds_list], dtype=_wdtype)

    # Initialize center: start at midpoint of each parameter's bounds
    center_arr = 0.5 * (lows + highs)
    theta_center = {
        name: float(center_arr[i])
        for i, name in enumerate(param_names)
    }

    # Track best result
    best_score = float("inf")
    best_parameters: dict[str, float] = {}

    # Track history
    generation_records: list[dict[str, Any]] = []
    all_scores: list[float] = []
    all_candidates: list[dict[str, float]] = []

    # Main AGSDR loop.  Candidate proposal is JAX-native (jit + vmap) while
    # evaluate_fn remains a Python black-box callback over a Model simulation.
    for gen in range(int(n_generations)):
        gen_best_score = float("inf")
        gen_best_params: dict[str, float] = {}

        key = jax.random.fold_in(base_key, int(gen))
        noise = jax.random.normal(
            key,
            shape=(int(n_population), len(param_names)),
            dtype=_wdtype,
        )
        candidate_matrix = _agsdr_candidates_from_noise(
            center_arr,
            lows,
            highs,
            float(exploration),
            noise,
        )

        # Evaluate population for this generation
        for row in range(int(n_population)):
            values = candidate_matrix[row]
            candidate = {
                name: float(values[idx])
                for idx, name in enumerate(param_names)
            }

            score = float(evaluate_fn(candidate))
            all_scores.append(score)
            all_candidates.append(dict(candidate))

            if score < gen_best_score:
                gen_best_score = score
                gen_best_params = dict(candidate)

            if score < best_score:
                best_score = score
                best_parameters = dict(candidate)

        # Delta-rule update: move center toward best candidate of this generation
        if gen_best_params:
            gen_best_arr = jnp.asarray(
                [gen_best_params[name] for name in param_names],
                dtype=_wdtype,
            )
            center_arr = jnp.clip(center_arr + float(alpha) * (gen_best_arr - center_arr), lows, highs)
            theta_center = {
                name: float(center_arr[i])
                for i, name in enumerate(param_names)
            }

        # Record generation-local and best-so-far state.  The best_so_far field is
        # monotone non-increasing even when exploratory candidates worsen.
        generation_records.append({
            "generation": int(gen),
            "generation_best_score": gen_best_score,
            "generation_best_parameters": dict(gen_best_params),
            "best_score": best_score,
            "best_parameters": dict(best_parameters),
            "theta_center": dict(theta_center),
        })

    # Return results
    return {
        "best_parameters": best_parameters,
        "best_score": best_score,
        "generation_records": generation_records,
        "all_scores": all_scores,
        "all_candidates": all_candidates,
    }


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
    global_scale: float = 1.0,
    checkpoint_n_steps: int = 50,
    deselection_threshold: int = 10,
    epsilon: float = 1e-6,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> Any:
    """Return an Optax-compatible GradientTransformation for Adaptive GSDR.

    Mathematically implements:
    U_{t+1} = U_t + λ * [ α_t * (σ * R_t) + (1 - α_t) * D_t ]
    where α_t = (Var(R_t) + ε) / (Var(R_t) + Var(D_t) + 2ε)

    Parameters
    ----------
    inner_optimizer : Optional[Any]
        Inner gradient optimizer (default: optax.adam).
    stochastic_scale : float
        Scale σ of stochastic perturbation.
    global_scale : float
        Global scale λ applied to combined updates.
    checkpoint_n_steps : int
        Checkpoint interval.
    deselection_threshold : int
        Steps without improvement before genetic reset.
    epsilon : float
        Numerical stability term ε.
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
            alpha_adaptive=0.5,  # Start neutral
        )
        return agsdr_state, inner_state

    def _calc_tree_variance(tree: Any) -> jax.Array:
        """Compute true L2 variance of a PyTree."""
        leaves = [jnp.ravel(l) for l in jax.tree_util.tree_leaves(tree) if hasattr(l, "shape")]
        if not leaves:
            return jnp.asarray(0.0, dtype=jnp.float32)
        flat_concat = jnp.concatenate(leaves)
        return jnp.var(flat_concat)

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

        # 1. Compute D_t (Supervised Update Vector from Inner Optimizer)
        inner_updates, inner_state = inner_optimizer.update(updates, inner_state, params)

        # 2. Compute R_t (Stochastic Base Representation Vector)
        key_delta = jax.random.fold_in(key, agsdr_state.step)
        base_representation = jax.tree_util.tree_map(
            lambda u: jax.random.normal(key_delta, u.shape, dtype=u.dtype)
            if hasattr(u, "shape")
            else u,
            inner_updates,
        )

        # 3. Calculate True Mathematical Variances
        var_d = _calc_tree_variance(inner_updates)
        var_r = _calc_tree_variance(base_representation) * (stochastic_scale ** 2)

        # 4. Synthesize Combined Updates via Damped Convex Combination
        # U_{t+1} = U_t + λ * [ α_t * (σ * R_t) + (1 - α_t) * D_t ]
        alpha = agsdr_state.alpha_adaptive
        combined_updates = jax.tree_util.tree_map(
            lambda d, r: jnp.asarray(global_scale, dtype=d.dtype) * (
                alpha * (jnp.asarray(stochastic_scale, dtype=r.dtype) * r) +
                (jnp.asarray(1.0, dtype=d.dtype) - alpha) * d
            ) if hasattr(d, "dtype") else d,
            inner_updates,
            base_representation
        )

        new_agsdr_state = agsdr_state
        if loss is not None and params is not None:
            is_improvement = loss < agsdr_state.best_loss
            new_best_loss = jnp.minimum(agsdr_state.best_loss, loss)
            new_best_param = params if is_improvement else agsdr_state.best_param

            new_desel_counter = 0 if is_improvement else agsdr_state.deselection_counter + 1
            should_reset = new_desel_counter >= deselection_threshold

            # Apply Exponential Moving Average (EMA) to tracked variances
            new_var_sup_ema = agsdr_state.ema_decay * agsdr_state.var_sup_ema + (1.0 - agsdr_state.ema_decay) * var_d
            new_var_unsup_ema = agsdr_state.ema_decay * agsdr_state.var_unsup_ema + (1.0 - agsdr_state.ema_decay) * var_r

            # Mathematical Formula: α_t = (Var(R_t) + ε) / (Var(R_t) + Var(D_t) + 2ε)
            var_total_stable = new_var_unsup_ema + new_var_sup_ema + 2.0 * epsilon
            alpha_next = (new_var_unsup_ema + epsilon) / var_total_stable

            new_agsdr_state = AGSDRState(
                step=agsdr_state.step + 1,
                best_loss=float(new_best_loss),
                best_param=new_best_param,
                reset_counter=agsdr_state.reset_counter + (1 if should_reset else 0),
                deselection_counter=int(new_desel_counter),
                var_sup_ema=float(new_var_sup_ema),
                var_unsup_ema=float(new_var_unsup_ema),
                ema_decay=agsdr_state.ema_decay,
                alpha_adaptive=float(jnp.clip(alpha_next, alpha_min, alpha_max)),
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


def _tune_matrix_agsdr_optax(
    model: Any,
    objective: Any,
    parameters: Any,
    param_specs: Any,
    scalar_bounds: Any,
    inner_optimizer: Any,
    inner_steps: int,
    inner_objective: Any,
    spec: Any,
    generations: int,
    population_size: int,
    seed: int,
    strict: bool,
    simulation: Any,
    base_report: Any,
) -> Any:
    """Two-level matrix AGSDR + optax.adam optimization (proxy-scale calibration scaffold).

    Scope: Simulated/proxy readouts for a computational optimization scaffold.
    The inner-loop surrogate gradients use differentiable sigmoid approximations and
    do NOT represent biological learning mechanisms. Final scoring uses the real
    declared objective for selection.

    OUTER LOOP: AGSDR proposes candidate scale values (one per matrix parameter).
    INNER LOOP: Adam refines each candidate on a soft-rate surrogate loss using
                jax.value_and_grad (differentiable sigmoid spike approximation).
                NOTE: Hard spike resets in simulate() create zero-gradient artifacts;
                fallback stochastic steps are injected when gradients flatten.
    FINAL SCORING: Real objective (group_rate_targets) evaluates the refined candidate.
    BEST-STATE MEMORY: Tracks best parameters and loss across all generations.

    Parameters
    ----------
    model : Model
        Starting model.
    objective : Objective
        Real scoring objective (used for final selection).
    parameters : dict
        Full parameters dict (may contain MatrixParameterSpec values).
    param_specs : dict
        Only the MatrixParameterSpec entries.
    scalar_bounds : dict
        Bounds dict for AGSDR proposal (extracted from param_specs).
    inner_optimizer : Any
        Optax optimizer instance (e.g. optax.adam(1e-2)).
    inner_steps : int
        Gradient steps per candidate in the inner loop.
    inner_objective : str or None
        Inner surrogate objective name; None = soft_rate_mse.
    spec : OptimizerSpec
        Resolved AGSDR spec (for alpha/exploration).
    generations, population_size, seed, strict, simulation : ...
        Standard AGSDR parameters.
    base_report : dict
        Pre-built base report dict for the TuneResult summary.

    Returns
    -------
    TuneResult
    """
    import math as _math
    from dataclasses import replace as _replace

    # Guard: require Optax for inner loop
    optax = require_optax()

    try:
        from jaxfne.core import (
            TuneResult,
            MatrixParameterSpec,
            _evaluate_soft_rate_targets,
            _model_with_parameters,
            Simulation,
        )
        from jaxfne.io import json_safe
    except ImportError:
        # Fallback: lazy import from current package context
        import importlib
        _core = importlib.import_module("jaxfne.core")
        TuneResult = _core.TuneResult
        MatrixParameterSpec = _core.MatrixParameterSpec
        _evaluate_soft_rate_targets = _core._evaluate_soft_rate_targets
        _model_with_parameters = _core._model_with_parameters
        Simulation = _core.Simulation
        _io = importlib.import_module("jaxfne.io")
        json_safe = _io.json_safe

    import jax
    import jax.numpy as jnp
    import numpy as np

    # Extract group rate targets for inner soft-rate surrogate
    # Try to parse groups and target rates from the objective
    groups_for_inner: dict = {}
    targets_hz_for_inner: dict = {}
    if hasattr(objective, "gates"):
        for gate_spec in objective.gates:
            if isinstance(gate_spec, dict) and "metadata" in gate_spec:
                meta = gate_spec["metadata"]
                if "groups" in meta:
                    groups_for_inner = dict(meta.get("groups", {}))
                    targets_hz_for_inner = dict(meta.get("targets_hz", {}))
                    break

    def _inner_loss_fn(W_flat: jax.Array, candidate_model: Any) -> jax.Array:
        """Differentiable inner-loop loss using soft rate surrogate.

        Runs a forward pass by extracting V_m from the emitter and computing
        soft-spike approximation.  We approximate V_m by running the model
        and reading the V_m output.
        """
        # We cannot JIT over the full simulation easily, so use V_m from
        # a fast forward pass (no JIT, no spike reset in loss path).
        try:
            fast_sim = _replace(simulation, duration_ms=min(float(simulation.duration_ms), 50.0))
            sigs = candidate_model.simulate(fast_sim)
            if groups_for_inner:
                loss = _evaluate_soft_rate_targets(
                    V_m=sigs.V_m,
                    groups=groups_for_inner,
                    targets_hz=targets_hz_for_inner,
                    duration_ms=float(fast_sim.duration_ms),
                    dt_ms=float(fast_sim.dt_ms),
                )
            else:
                # Fallback: mean V_m should be near threshold
                loss = jnp.mean(jnp.abs(sigs.V_m - (-45.0)))
        except Exception:
            loss = jnp.asarray(float("inf"), dtype=_wdtype_outer)
        return loss

    # AGSDR outer loop setup
    param_names = sorted(scalar_bounds.keys())
    bounds_list = [(float(scalar_bounds[n][0]), float(scalar_bounds[n][1])) for n in param_names]

    # v0.3.17: derive working dtype from bounds to avoid silent float32 downcast.
    _raw_outer = [b[0] for b in bounds_list] + [b[1] for b in bounds_list]
    _inf_outer = jnp.result_type(*_raw_outer) if _raw_outer else jnp.float32
    _wdtype_outer = _inf_outer if jnp.issubdtype(_inf_outer, jnp.floating) else jnp.float32

    lows = jnp.asarray([b[0] for b in bounds_list], dtype=_wdtype_outer)
    highs = jnp.asarray([b[1] for b in bounds_list], dtype=_wdtype_outer)
    center_arr = 0.5 * (lows + highs)

    best_score = float("inf")
    best_parameters: dict = {}
    best_model = model
    generation_records = []
    all_scores = []

    base_key = jax.random.PRNGKey(int(seed))

    for gen in range(int(generations)):
        gen_best_score = float("inf")
        gen_best_params: dict = {}

        # Propose AGSDR candidates
        key = jax.random.fold_in(base_key, int(gen))
        noise = jax.random.normal(
            key,
            shape=(int(population_size), len(param_names)),
            dtype=_wdtype_outer,
        )
        candidate_matrix = _agsdr_candidates_from_noise(
            center_arr,
            lows,
            highs,
            float(spec.exploration),
            noise,
        )

        for row in range(int(population_size)):
            raw_values = candidate_matrix[row]
            candidate_scalars = {
                name: float(raw_values[idx])
                for idx, name in enumerate(param_names)
            }

            # Build candidate model (outer AGSDR scale)
            candidate_model = _model_with_parameters(model, candidate_scalars, param_specs)

            # INNER LOOP: Adam refinement on soft-rate surrogate
            if inner_steps > 0 and groups_for_inner:
                try:
                    emitter = candidate_model.params["emitter"]
                    W_init = jnp.asarray(emitter.W, dtype=emitter.W.dtype if hasattr(emitter.W, "dtype") else _wdtype_outer).reshape(-1)

                    opt_state = inner_optimizer.init(W_init)
                    current_W = W_init

                    for inner_step in range(int(inner_steps)):
                        def loss_and_grad_fn(W_flat: jax.Array) -> tuple:
                            # Rebuild emitter with updated W
                            new_W = W_flat.reshape(emitter.W.shape)
                            new_emitter = _replace(emitter, W=new_W)
                            new_params = dict(candidate_model.params)
                            new_params["emitter"] = new_emitter
                            from dataclasses import replace as _r
                            updated_model = _r(candidate_model, params=new_params)
                            return _inner_loss_fn(W_flat, updated_model), W_flat

                        try:
                            # Compute gradient via jax.grad on the soft-rate loss
                            def inner_loss_only(W_flat: jax.Array) -> jax.Array:
                                new_W = W_flat.reshape(emitter.W.shape)
                                new_emitter = _replace(emitter, W=new_W)
                                new_params = dict(candidate_model.params)
                                new_params["emitter"] = new_emitter
                                from dataclasses import replace as _r2
                                updated_model = _r2(candidate_model, params=new_params)
                                return _inner_loss_fn(W_flat, updated_model)

                            loss_val, grads = jax.value_and_grad(inner_loss_only)(current_W)

                            # GRADIENT FLATLINE HARDENING: Detect and mitigate zero-gradient artifacts
                            # from hard spike resets that block differentiable path
                            grads_flat = jnp.ravel(grads)
                            is_flatline = jnp.all(jnp.abs(grads_flat) < 1e-7)

                            # Adaptive fallback: inject stochastic step when gradient is flat
                            key_fallback = jax.random.fold_in(base_key, int(gen) * 1000 + int(row) * 100 + int(inner_step))
                            fallback_grads = jax.tree_util.tree_map(
                                lambda g: jnp.where(
                                    is_flatline,
                                    jax.random.uniform(key_fallback, g.shape, minval=-0.01, maxval=0.01, dtype=g.dtype),
                                    g
                                ) if hasattr(g, "shape") else g,
                                grads
                            )

                            updates, opt_state = inner_optimizer.update(fallback_grads, opt_state)
                            current_W = optax.apply_updates(current_W, updates)
                            # Clip to declared parameter bounds
                            # For gAMPA_w, use the bounds from MatrixParameterSpec
                            param_lower = float(param_specs.get("gAMPA_w").bounds[0]) if "gAMPA_w" in param_specs else -1000.0
                            param_upper = float(param_specs.get("gAMPA_w").bounds[1]) if "gAMPA_w" in param_specs else 1000.0
                            current_W = jnp.clip(current_W, param_lower, param_upper)
                        except Exception:
                            break  # Inner loop failed; use AGSDR candidate as-is

                    # Apply refined W to model
                    refined_W = current_W.reshape(emitter.W.shape)
                    new_emitter = _replace(emitter, W=refined_W)
                    new_params = dict(candidate_model.params)
                    new_params["emitter"] = new_emitter
                    from dataclasses import replace as _r3
                    candidate_model = _r3(candidate_model, params=new_params)
                except Exception:
                    pass  # Fall back to unrefined AGSDR candidate

            # FINAL SCORING: real objective
            try:
                candidate_signals = candidate_model.simulate(
                    _replace(simulation, seed=int(seed) + gen * population_size + row)
                )
                candidate_report = candidate_model.evaluate(candidate_signals, objective, strict=strict)
                score = candidate_report.get("total_loss")
                gates_pass = bool(candidate_report.get("all_gates_pass", False))
                if score is None:
                    score = 0.0 if gates_pass else float("inf")
                score = float(score)
            except Exception as e:
                score = float("inf")

            all_scores.append(score)

            if score < gen_best_score:
                gen_best_score = score
                gen_best_params = dict(candidate_scalars)

            if score < best_score:
                best_score = score
                # Preserve both scalar proposal values AND matrix parameters
                best_parameters = dict(candidate_scalars)
                # Extract tuned matrices from the refined model for ALL matrix parameters
                for matrix_name in param_specs.keys():
                    param_spec = param_specs[matrix_name]
                    if isinstance(param_spec, MatrixParameterSpec):
                        try:
                            tuned_W = np.asarray(candidate_model.params["emitter"].W, dtype=np.float32)
                            # Convert to JSON-safe nested list
                            best_parameters[matrix_name] = tuned_W.tolist()
                        except Exception:
                            # If extraction fails, keep scalar as placeholder
                            pass
                best_model = candidate_model

        # Delta-rule center update
        if gen_best_params:
            gen_best_arr = jnp.asarray(
                [gen_best_params[n] for n in param_names],
                dtype=_wdtype_outer,
            )
            center_arr = jnp.clip(
                center_arr + float(spec.alpha) * (gen_best_arr - center_arr),
                lows,
                highs,
            )

        generation_records.append({
            "generation": int(gen),
            "generation_best_score": gen_best_score,
            "generation_best_parameters": dict(gen_best_params),
            "best_score": best_score,
            "best_parameters": dict(best_parameters),
        })

    inner_meta = "optax_optimizer_object_not_serialized" if inner_optimizer is not None else None

    # Build matrix parameters metadata (compact, no huge arrays in summary)
    matrix_parameters_meta = {}
    for matrix_name in param_specs.keys():
        param_spec = param_specs[matrix_name]
        if isinstance(param_spec, MatrixParameterSpec) and matrix_name in best_parameters:
            try:
                W_values = best_parameters[matrix_name]
                W_array = np.asarray(W_values, dtype=np.float32)
                matrix_parameters_meta[matrix_name] = {
                    "shape": list(W_array.shape),
                    "mask": param_specs[matrix_name].mask,
                    "bounds": list(param_specs[matrix_name].bounds),
                    "finite": bool(np.isfinite(W_array).all()),
                    "min": float(np.min(W_array)) if np.isfinite(W_array).all() else None,
                    "max": float(np.max(W_array)) if np.isfinite(W_array).all() else None,
                    "mean": float(np.mean(W_array)) if np.isfinite(W_array).all() else None,
                    "nonzero": int(np.count_nonzero(W_array)),
                }
            except Exception:
                pass

    report = {
        **base_report,
        "same_model_unchanged": False,
        "tuning_status": "matrix_agsdr_optax_v0.0.1",
        "acceptance_decision": "ACCEPT_CANDIDATE" if _math.isfinite(best_score) else "REVISE",
        "best_score": best_score if _math.isfinite(best_score) else None,
        "generation_records": generation_records,
        "all_scores": all_scores,
        "n_candidates_evaluated": len(all_scores),
        "tuning_path": "matrix_two_level_agsdr_adam",
        "inner_optimizer": inner_meta,
        "inner_steps": int(inner_steps),
        "matrix_parameters": matrix_parameters_meta,
        "warnings": [
            "two_level_agsdr_adam_is_computational_scaffold_only",
            "inner_soft_rate_surrogate_is_not_biological_truth",
        ],
    }

    return TuneResult(
        best_parameters=best_parameters,
        best_score=float(best_score) if _math.isfinite(best_score) else float("inf"),
        history=generation_records,
        summary=json_safe(report),
        model=best_model,
    )
