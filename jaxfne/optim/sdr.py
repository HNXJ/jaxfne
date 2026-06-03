"""Stochastic Delta Rule (SDR) optimization kernel for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold matching truth_safe_unverified boundaries.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp

from .base import BaseSDRState


@dataclass(frozen=True)
class SDRState(BaseSDRState):
    """Stochastic Delta Rule optimizer state.

    Holds best-loss tracking, reset counter, and EMA variance estimates
    for adaptive alpha computation.
    """


def step_sdr_transform(u_t: jnp.ndarray, grad_l: jnp.ndarray, state: Any, hyperparams: dict) -> tuple[jnp.ndarray, Any]:
    """Evaluates primitive Stochastic Delta Rule transformations with strict precision locks."""
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
