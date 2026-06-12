"""Tests for ODE solver APIs."""

import pytest
import jax
import jax.numpy as jnp
import jaxfne as jtfne


def test_euler_exponential_decay():
    """Test Euler solver on simple exponential decay."""
    # dy/dt = -y
    def dydt_fn(y, t):
        return -y

    y_init = jnp.array([1.0, 2.0])
    t_start = 0.0
    t_end = 2.0
    
    # Using euler solver with dt = 0.1
    config = jtfne.SolverConfig(method="euler", dt=0.1)
    y_final, trajectory = jtfne.solve_ode(config, dydt_fn, y_init, t_start, t_end)
    
    # 2.0 / 0.1 = 20 steps
    assert trajectory.shape == (20, 2)
    assert y_final.shape == (2,)
    assert jnp.all(jnp.isfinite(y_final))
    assert jnp.all(jnp.isfinite(trajectory))
    # y(t) = y_init * e^(-t) -> y(2) approx 1.0 * e^-2 = 0.1353, 2.0 * e^-2 = 0.2707
    # For forward Euler with dt=0.1, (1 - 0.1)^20 = 0.9^20 approx 0.1215
    assert jnp.allclose(y_final, y_init * (0.9 ** 20), atol=1e-5)


def test_diffrax_solver_missing_guard():
    """Test that DiffraxSolver raises ImportError or works if diffrax is present."""
    def dydt_fn(y, t):
        return -y
        
    config = jtfne.SolverConfig(method="diffrax", dt=0.1)
    
    try:
        import diffrax
        # If diffrax is installed, it should solve successfully
        y_final, trajectory = jtfne.solve_ode(config, dydt_fn, jnp.array([1.0]), 0.0, 1.0)
        assert trajectory.shape == (10, 1)
        assert jnp.all(jnp.isfinite(y_final))
    except ImportError:
        # If diffrax is not installed, it should fail with clear ImportError
        with pytest.raises(ImportError) as excinfo:
            jtfne.solve_ode(config, dydt_fn, jnp.array([1.0]), 0.0, 1.0)
        assert "diffrax is required for DiffraxSolver" in str(excinfo.value)
