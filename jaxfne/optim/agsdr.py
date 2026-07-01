"""Adaptive Genetic Stochastic Delta Rule (AGSDR) optimization kernel for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp

from .base import BaseSDRState


@dataclass(frozen=True)
class AGSDRState(BaseSDRState):
    """Adaptive Genetic Stochastic Delta Rule optimizer state.

    Combines genetic deselection with adaptive alpha for two-phase search.
    """
    deselection_counter: int = 0
    alpha_adaptive: float = 0.7  # Adaptive alpha, updated via variance ratio


def step_agsdr_transform(
    u_t: jnp.ndarray,
    grad_l: jnp.ndarray,
    state: Any,
    hyperparams: dict,
) -> tuple[jnp.ndarray, Any]:
    """Plain gradient-descent step: ``u_t - eta * grad_l``.

    Despite the ``AGSDRState``/``AGSDR`` naming and ``deselection_counter``/
    ``alpha_adaptive`` fields, this does NOT implement genetic deselection or
    adaptive two-phase search -- ``state`` is passed through unchanged, none
    of its extra fields are read or updated. The real Adaptive Genetic SDR
    optimizer is :func:`jaxfne.optim.core.agsdr_transform`, which production
    code (``Model.tune(optimizer="AGSDR", ...)``) actually uses -- this
    function is not on that call path. Kept for backward compatibility with
    any caller using the lower-level ``step(u_t, grad_l, state,
    hyperparams)`` signature.
    """
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
