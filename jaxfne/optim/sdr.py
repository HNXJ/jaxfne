"""Stochastic Delta Rule (SDR) optimization kernel for jaxfne.optim.

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
class SDRState(BaseSDRState):
    """Stochastic Delta Rule optimizer state.

    Holds best-loss tracking, reset counter, and EMA variance estimates
    for adaptive alpha computation.
    """


def step_sdr_transform(u_t: jnp.ndarray, grad_l: jnp.ndarray, state: Any, hyperparams: dict) -> tuple[jnp.ndarray, Any]:
    """Plain gradient-descent step: ``u_t - eta * grad_l``.

    Despite the ``SDRState``/``SDR`` naming, this does NOT implement a
    stochastic delta term, EMA variance tracking, or any other behavior
    beyond ``state``/``hyperparams`` plumbing -- ``state`` is passed through
    unchanged. The real Stochastic Delta Rule optimizer (stochastic delta
    term, adaptive alpha, Optax-integrated) is :func:`jaxfne.optim.core.sdr_transform`,
    which production code (``Model.tune(optimizer="SDR", ...)``) actually
    uses -- this function is not on that call path. Kept for backward
    compatibility with any caller using the lower-level
    ``step(u_t, grad_l, state, hyperparams)`` signature.
    """
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
