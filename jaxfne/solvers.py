"""State integrators for ODE solving.

This module provides numerical state integrators for time evolution. Solvers here
operate on state vectors y and dynamics functions dy_dt = f(y, t), NOT on field
PDEs or spatial structures. Use for neural state evolution only.

Truth gates:
- These are general-purpose numerical integrators, not field solvers.
- No Poisson/Maxwell/spatial PDE claims.
- Do not use for field computation; see fields.py for proxy-based field methods.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Any, Tuple


def euler_step(
    y: jax.Array,
    t: float,
    dt: float,
    dydt_fn: Callable[[jax.Array, float], jax.Array]
) -> jax.Array:
    """Single forward Euler step for ODE integration.

    Computes y_{n+1} = y_n + dt * f(y_n, t_n) where f is the dynamics function.

    Args:
        y: Current state vector [shape arbitrary, dtype float32/float64].
        t: Current time (scalar, ms).
        dt: Time step (scalar, ms).
        dydt_fn: Callable(y, t) → dy/dt (same shape as y).

    Returns:
        Next state y_{n+1} [same shape as y].
    """
    dy = dydt_fn(y, t)
    return y + dt * dy


def euler_scan(
    y_init: jax.Array,
    t_start: float,
    dt: float,
    n_steps: int,
    dydt_fn: Callable[[jax.Array, float], jax.Array]
) -> Tuple[jax.Array, jax.Array]:
    """Forward Euler integration over multiple steps using jax.lax.scan.

    Integrates the ODE from t_start for n_steps timesteps of size dt.
    Returns full trajectory (all intermediate y values).

    Args:
        y_init: Initial state [shape (...,), dtype float32/float64].
        t_start: Initial time (scalar, ms).
        dt: Time step (scalar, ms).
        n_steps: Number of steps (scalar int).
        dydt_fn: Callable(y, t) → dy/dt [same shape as y].

    Returns:
        (final_state, trajectory) where:
        - final_state: y at t = t_start + n_steps*dt [same shape as y_init]
        - trajectory: all y values [shape (n_steps, ...)] stacked along axis 0
    """
    def step_fn(carry, i):
        y, t = carry
        y_next = euler_step(y, t, dt, dydt_fn)
        return (y_next, t + dt), y_next

    (y_final, _), trajectory = lax.scan(
        step_fn,
        (y_init, t_start),
        jnp.arange(n_steps)
    )

    return y_final, trajectory


# Placeholder for future methods (bwd_euler, rk4, etc.)
__all__ = ["euler_step", "euler_scan"]
