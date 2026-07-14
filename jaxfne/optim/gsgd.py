"""Generalized Stochastic Gradient Descent (GSGD) optimization kernel for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from typing import Any, NamedTuple
import jax.numpy as jnp


class GSGDState(NamedTuple):
    """State for GSGD optimization step."""
    count: jnp.ndarray
    step_size: jnp.ndarray


def step_gsgd_transform(
    u_t: jnp.ndarray,
    grad_l: jnp.ndarray,
    state: Any,
    hyperparams: dict,
) -> tuple[jnp.ndarray, Any]:
    """Plain gradient-descent step: ``u_t - eta * grad_l``.

    Despite the "adaptive step scaling" claimed in earlier revisions of this
    docstring, ``state`` (including ``GSGDState.step_size``) is passed
    through unchanged, never read or updated -- there is no adaptive
    behavior here. Also NOT currently on ``Model.tune()``'s real call path:
    the differentiable-optimizer branch of ``tune()`` (``optax_guarded_path``)
    is a metadata-only guard that returns without ever entering a gradient
    loop, so this function is not invoked by production tuning as of this
    writing even though ``jaxfne.optim.core.gsgd()`` declares "GSGD" as a
    differentiable ``OptimizerSpec`` and its docstring names this function.
    Kept for direct/manual use via the lower-level ``step(u_t, grad_l,
    state, hyperparams)`` signature.
    """
    target_dtype = u_t.dtype
    eta = jnp.array(hyperparams.get("eta", 0.01), dtype=target_dtype)
    u_next = u_t - eta * grad_l
    return u_next, state
