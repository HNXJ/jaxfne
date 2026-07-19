"""Adaptive Genetic Stochastic Delta Rule (AGSDR) optimization kernel for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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
    of its extra fields are read or updated. A separate, more elaborate
    Adaptive Genetic SDR implementation exists at
    :func:`jaxfne.optim.core.agsdr_transform` -- but it is NOT what
    ``Model.tune(optimizer="AGSDR", ...)`` calls either (confirmed
    2026-07-18: that string dispatches through ``_resolve_optimizer`` to a
    blackbox candidate-proposal path, ``propose_blackbox_candidates``/
    ``_run_agsdr_optimization_loop``, each of which implements its own
    independent inline logic and never references ``agsdr_transform``;
    passing the ``agsdr_transform()`` object itself as ``optimizer=`` also
    does not work, since ``_resolve_optimizer`` has no branch for a raw
    ``GradientTransformation`` and silently falls through to the same
    generic blackbox path instead of calling its ``init``/``update``).
    ``agsdr_transform`` is exercised only by ``tests/test_optim_tune.py``
    directly, not by any confirmed production call path -- see
    ``jaxfne/optim/core.py::agsdr_transform-production-path-unconfirmed``
    in ``progress.json``. Kept for backward compatibility with any caller
    using the lower-level ``step(u_t, grad_l, state, hyperparams)``
    signature.
    """
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
