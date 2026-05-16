"""Optimization placeholders for jaxfne.

Differentiable paths should use Optax. Black-box/non-smooth paths can use GSDR/AGSDR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class AGSDR:
    """Adaptive Genetic Stochastic Delta Rule placeholder.

    This is a public API placeholder, not the final algorithm.
    """

    alpha: float = 0.7
    exploration: float = 0.05
    deselect_factor: float = 2.0

    def status(self) -> Dict[str, Any]:
        return {
            "optimizer_class": "blackbox",
            "optimizer": "AGSDR",
            "status": "prototype_api",
            "alpha": self.alpha,
            "exploration": self.exploration,
            "deselect_factor": self.deselect_factor,
        }


def require_optax():
    try:
        import optax  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This feature requires optional dependency 'optax'. Install with: pip install -e '.[opt]'"
        ) from exc
    return optax
