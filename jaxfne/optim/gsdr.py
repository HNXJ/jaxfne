"""Genetic Stochastic Delta Rule (GSDR) optimization kernel for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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
    read or updated. A separate, more elaborate Genetic SDR implementation
    exists at :func:`jaxfne.optim.core.gsdr_transform` -- but it is NOT what
    ``Model.tune(optimizer="GSDR", ...)`` calls either (confirmed
    2026-07-18: that string dispatches through ``_resolve_optimizer`` to a
    blackbox candidate-proposal path, ``propose_blackbox_candidates``, which
    never references ``gsdr_transform``; passing the ``gsdr_transform()``
    object itself as ``optimizer=`` also does not work, since
    ``_resolve_optimizer`` has no branch for a raw ``GradientTransformation``
    and silently falls through to the same generic blackbox path instead of
    calling its ``init``/``update``). ``gsdr_transform`` is exercised only
    by ``tests/test_optim_tune.py`` directly, not by any confirmed
    production call path. Kept for backward
    compatibility with any caller using the lower-level
    ``step(u_t, grad_l, state, hyperparams)`` signature.
    """
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
