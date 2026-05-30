"""Parameter bounds and clipping operators for jaxfne.optim.

Enforces trace-safe parameter space constraints via element-wise matrix broadcasting
to eliminate python looping overhead inside hot paths.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def enforce_parameter_bounds(proposals: jnp.ndarray, lows: jnp.ndarray, highs: jnp.ndarray) -> jnp.ndarray:
    """Applies JAX-native primitive broadcasting to clip parameter updates
    without generating intermediate tracer overhead.
    """
    target_dtype = proposals.dtype
    l_bound = jnp.array(lows, dtype=target_dtype)
    h_bound = jnp.array(highs, dtype=target_dtype)
    return jnp.clip(proposals, l_bound[None, :], h_bound[None, :])


def apply_parameter_constraints(proposals: jnp.ndarray, lows: jnp.ndarray, highs: jnp.ndarray) -> jnp.ndarray:
    """Clips optimization candidates using JAX-native primitive broadcasting rules.

    Guarantees type matching against active population matrices.
    """
    target_dtype = proposals.dtype
    l_bound = jnp.array(lows, dtype=target_dtype)
    h_bound = jnp.array(highs, dtype=target_dtype)
    return jnp.clip(proposals, l_bound[None, :], h_bound[None, :])
