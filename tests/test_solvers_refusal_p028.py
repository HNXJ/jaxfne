"""P-028: solver selection, tolerances, and time grids refuse instead of silently defaulting."""

import jax.numpy as jnp
import pytest

from jaxfne.solvers import DiffraxSolver, EulerSolver, SolverConfig, solve_ode


def _decay(y, t):
    return -y


def test_unknown_solver_type_refuses_p028():
    """P-028: an unrecognized solver_type refuses (no silent Tsit5 fallback)."""
    with pytest.raises(ValueError, match="P-028"):
        DiffraxSolver(dt=0.1, solver_type="bogus").solve(_decay, jnp.ones(3), 0.0, 1.0)


def test_euler_nondefault_tolerances_refuse_p028():
    """P-028: rtol/atol are not consumed by Euler; non-defaults refuse."""
    with pytest.raises(ValueError, match="P-028"):
        solve_ode(SolverConfig(method="euler", dt=0.1, rtol=1e-9), _decay, jnp.ones(3), 0.0, 1.0)
    with pytest.raises(ValueError, match="P-028"):
        solve_ode(SolverConfig(method="euler", dt=0.1, atol=1e-9), _decay, jnp.ones(3), 0.0, 1.0)


def test_nonintegral_grids_refuse_p028():
    """P-028: spans that are not a whole number of dt refuse (no silent rounding)."""
    with pytest.raises(ValueError, match="P-028"):
        EulerSolver(dt=0.3).solve(_decay, jnp.ones(3), 0.0, 1.0)
    with pytest.raises(ValueError, match="P-028"):
        solve_ode(SolverConfig(method="euler", dt=0.3), _decay, jnp.ones(3), 0.0, 1.0)


def test_euler_whole_grid_passes_p028():
    """P-028: integral grids and default tolerances still run."""
    y_final, traj = EulerSolver(dt=0.1).solve(_decay, jnp.ones(2), 0.0, 1.0)
    assert traj.shape == (10, 2)
    assert bool(jnp.all((y_final > 0.0) & (y_final < 1.0)))
