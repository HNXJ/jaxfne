"""Genetic Stochastic Delta Rule (GSDR) optimization kernel for jaxfne.optim.

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
class GSDRState(BaseSDRState):
    """Genetic Stochastic Delta Rule optimizer state.

    Similar to SDRState, with additional tracking for genetic deselection.
    """
    deselection_counter: int = 0


def step_gsdr_transform(
    u_t: jnp.ndarray,
    grad_l: jnp.ndarray,
    state: Any,
    hyperparams: dict,
) -> tuple[jnp.ndarray, Any]:
    """Plain gradient-descent step: ``u_t - eta * grad_l``.

    Despite the ``GSDRState``/``GSDR`` naming and ``deselection_counter``
    field, this does NOT implement genetic deselection or any behavior
    beyond ``state``/``hyperparams`` plumbing -- ``state``
    (including ``deselection_counter``) is passed through unchanged, never
    read or updated. The real Genetic SDR optimizer is
    :func:`jaxfne.optim.core.gsdr_transform`, which production code
    (``Model.tune(optimizer="GSDR", ...)``) actually uses -- this function
    is not on that call path. Kept for backward compatibility with any
    caller using the lower-level ``step(u_t, grad_l, state, hyperparams)``
    signature.
    """
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
