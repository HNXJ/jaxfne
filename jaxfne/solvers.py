"""State integrators for ODE solving.

This module provides numerical state integrators for time evolution. Solvers here
operate on state vectors y and dynamics functions dy_dt = f(y, t), NOT on field
PDEs or spatial structures. Use for neural state evolution only.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SolverConfig:
    """Configuration class for ODE solvers."""
    method: str = "euler"
    dt: float = 0.1
    rtol: float = 1e-3
    atol: float = 1e-6
    solver_type: Optional[str] = None


class EulerSolver:
    """Forward Euler integrator using JAX and lax.scan."""
    def __init__(self, dt: float):
        self.dt = dt

    def solve(
        self,
        dydt_fn: Callable[[jax.Array, float], jax.Array],
        y_init: jax.Array,
        t_start: float,
        t_end: float,
    ) -> Tuple[jax.Array, jax.Array]:
        """Integrates dydt_fn from t_start to t_end using forward Euler stepping."""
        n_steps = int(round((t_end - t_start) / self.dt))
        
        def step_fn(carry, i):
            y, t = carry
            dy = dydt_fn(y, t)
            y_next = y + self.dt * dy
            return (y_next, t + self.dt), y_next

        (y_final, _), trajectory = lax.scan(
            step_fn,
            (y_init, t_start),
            jnp.arange(n_steps)
        )
        return y_final, trajectory


class DiffraxSolver:
    """Optional Runge-Kutta solver using diffrax (lazily imported)."""
    def __init__(
        self,
        dt: float,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        solver_type: Optional[str] = None
    ):
        self.dt = dt
        self.rtol = rtol
        self.atol = atol
        self.solver_type = solver_type

    def solve(
        self,
        dydt_fn: Callable[[jax.Array, float], jax.Array],
        y_init: jax.Array,
        t_start: float,
        t_end: float,
    ) -> Tuple[jax.Array, jax.Array]:
        """Integrates dydt_fn from t_start to t_end using diffrax."""
        try:
            import diffrax
        except ImportError as e:
            raise ImportError(
                "diffrax is required for DiffraxSolver. "
                "Install it using `pip install diffrax`."
            ) from e

        term = diffrax.ODETerm(lambda t, y, args: dydt_fn(y, t))
        
        if self.solver_type == "dopri5":
            solver = diffrax.Dopri5()
        elif self.solver_type == "tsit5":
            solver = diffrax.Tsit5()
        else:
            solver = diffrax.Tsit5()

        stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)
        n_save = int(round((t_end - t_start) / self.dt)) + 1
        ts_save = jnp.linspace(t_start, t_end, n_save)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_start,
            t1=t_end,
            dt0=self.dt,
            y0=y_init,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts_save)
        )
        return sol.ys[-1], sol.ys[1:]


def solve_ode(
    config: SolverConfig,
    dydt_fn: Callable[[jax.Array, float], jax.Array],
    y_init: jax.Array,
    t_start: float,
    t_end: float,
) -> Tuple[jax.Array, jax.Array]:
    """Public ODE solver entrypoint routing to appropriate solver backend."""
    if config.method == "euler":
        solver = EulerSolver(dt=config.dt)
        return solver.solve(dydt_fn, y_init, t_start, t_end)
    elif config.method == "diffrax":
        solver = DiffraxSolver(
            dt=config.dt,
            rtol=config.rtol,
            atol=config.atol,
            solver_type=config.solver_type
        )
        return solver.solve(dydt_fn, y_init, t_start, t_end)
    else:
        raise ValueError(f"Unknown solver method: {config.method}")


# Maintain original euler helpers for backward compatibility
def euler_step(
    y: jax.Array,
    t: float,
    dt: float,
    dydt_fn: Callable[[jax.Array, float], jax.Array]
) -> jax.Array:
    """Single forward Euler step (backward compatibility)."""
    return y + dt * dydt_fn(y, t)


def euler_scan(
    y_init: jax.Array,
    t_start: float,
    dt: float,
    n_steps: int,
    dydt_fn: Callable[[jax.Array, float], jax.Array]
) -> Tuple[jax.Array, jax.Array]:
    """Forward Euler integration scan (backward compatibility)."""
    solver = EulerSolver(dt=dt)
    return solver.solve(dydt_fn, y_init, t_start, t_start + n_steps * dt)


def solve_volume_conductor_experimental(*args, **kwargs):
    """Experimental volume conductor solver skeleton.

    Raises
    ------
    NotImplementedError
        Always raises NotImplementedError. Requires boundary/gauge/residual/convergence validation.
    """
    raise NotImplementedError(
        "experimental solver skeleton: requires boundary/gauge/residual/convergence validation"
    )


__all__ = [
    "SolverConfig",
    "EulerSolver",
    "DiffraxSolver",
    "solve_ode",
    "euler_step",
    "euler_scan",
    "solve_volume_conductor_experimental",
]
