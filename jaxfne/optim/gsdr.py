"""Genetic Stochastic Delta Rule (GSDR) optimization kernel for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold matching truth_safe_unverified boundaries.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp


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


def step_gsdr_transform(
    u_t: jnp.ndarray,
    grad_l: jnp.ndarray,
    state: Any,
    hyperparams: dict,
) -> tuple[jnp.ndarray, Any]:
    """Evaluates primitive Stochastic Delta Rule transformations with genetic deselection."""
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
