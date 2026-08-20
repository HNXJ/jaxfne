"""Stochastic Delta Rule (SDR) optimization kernel for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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
    unchanged. A separate, more elaborate Stochastic Delta Rule implementation
    (stochastic delta term, adaptive alpha, Optax-integrated) exists at
    :func:`jaxfne.optim.core.sdr_transform` -- but it is NOT what
    ``Model.tune(optimizer="SDR", ...)`` calls either (confirmed 2026-07-18:
    that string dispatches through ``_resolve_optimizer`` to a blackbox
    candidate-proposal path, ``propose_blackbox_candidates``, which never
    references ``sdr_transform``; passing the ``sdr_transform()`` object
    itself as ``optimizer=`` also does not work, since ``_resolve_optimizer``
    has no branch for a raw ``GradientTransformation`` and silently falls
    through to the same generic blackbox path instead of calling its
    ``init``/``update``). ``sdr_transform`` is exercised only by
    ``tests/test_optim_tune.py`` directly, not by any confirmed production
    call path. Kept for backward compatibility
    with any caller using the lower-level
    ``step(u_t, grad_l, state, hyperparams)`` signature.
    """
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
