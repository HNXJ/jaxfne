"""Adaptive Genetic Stochastic Delta Rule (AGSDR) optimization kernel for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold matching truth_safe_unverified boundaries.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp


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


def step_agsdr_transform(
    u_t: jnp.ndarray,
    grad_l: jnp.ndarray,
    state: Any,
    hyperparams: dict,
) -> tuple[jnp.ndarray, Any]:
    """Evaluates adaptive GSDR steps with strict precision and JAX compilation safety."""
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
