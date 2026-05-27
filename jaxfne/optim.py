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
from typing import Any, Optional

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
