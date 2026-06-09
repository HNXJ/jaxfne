"""v0.3.31 Euler state integrator tests.

Tests for forward Euler ODE solver (step and scan).
"""

import jax.numpy as jnp
import pytest

import jaxfne as jtfne


class TestEulerSolver:
    """Test Euler state integrator functions."""

    def test_euler_step_basic(self):
        """Basic euler_step: dy/dt = -y, y(0) = 1."""
        def dydt(y, t):
            return -y

        y0 = jnp.array(1.0)
        y1 = jtfne.euler_step(y0, t=0.0, dt=0.01, dydt_fn=dydt)

        # Expected: y1 = 1 + 0.01 * (-1) = 0.99
        assert jnp.allclose(y1, 0.99)

    def test_euler_scan_exponential_decay(self):
        """Integrate dy/dt = -y from y(0)=1 over 100 steps dt=0.01.

        Exact solution: y(t) = exp(-t), at t=1.0: exp(-1) ≈ 0.3679
        Euler approx at t=1.0 with dt=0.01, n=100: (1 - 0.01)^100 ≈ 0.366
        """
        def dydt(y, t):
            return -y

        y_init = jnp.array(1.0)
        n_steps = 100
        dt = 0.01

        y_final, trajectory = jtfne.euler_scan(
            y_init, t_start=0.0, dt=dt, n_steps=n_steps, dydt_fn=dydt
        )

        # Check final state
        expected_final = jnp.exp(-1.0)  # e^-1 ≈ 0.3679
        assert jnp.allclose(y_final, expected_final, atol=2e-3)

        # Check trajectory shape
        assert trajectory.shape == (n_steps,)

        # Check trajectory is decreasing
        assert jnp.all(jnp.diff(trajectory) < 0)

    def test_euler_scan_multidimensional(self):
        """Test Euler scan with multi-dimensional state."""
        def dydt(y, t):
            # dy/dt = -y (component-wise)
            return -y

        y_init = jnp.array([1.0, 2.0, 3.0])
        y_final, trajectory = jtfne.euler_scan(
            y_init, t_start=0.0, dt=0.01, n_steps=100, dydt_fn=dydt
        )

        # All components should decay
        assert jnp.all(y_final < y_init)
        assert trajectory.shape == (100, 3)
        assert jnp.allclose(trajectory[-1], y_final)

    def test_euler_step_linear_growth(self):
        """Test dy/dt = k*y growth (k=0.1)."""
        def dydt(y, t):
            return 0.1 * y

        y0 = jnp.array(1.0)
        y1 = jtfne.euler_step(y0, t=0.0, dt=0.1, dydt_fn=dydt)

        # Expected: y1 = 1 + 0.1 * 0.1 = 1.01
        assert jnp.allclose(y1, 1.01)

    def test_euler_scan_constant_rhs(self):
        """Test dy/dt = c (constant) → linear trajectory."""
        def dydt(y, t):
            return jnp.array(1.0)

        y_init = jnp.array(0.0)
        y_final, trajectory = jtfne.euler_scan(
            y_init, t_start=0.0, dt=0.1, n_steps=10, dydt_fn=dydt
        )

        # Expected: y(t) = 0 + t = 1.0 at t=1.0
        assert jnp.allclose(y_final, 1.0)

        # Trajectory contains values after each step: [0.1, 0.2, ..., 1.0]
        expected = jnp.arange(1, 11) * 0.1
        assert jnp.allclose(trajectory, expected)

    def test_euler_step_time_dependent(self):
        """Test dy/dt = t (time-dependent)."""
        def dydt(y, t):
            return t

        y0 = jnp.array(0.0)
        y1 = jtfne.euler_step(y0, t=1.0, dt=0.1, dydt_fn=dydt)

        # Expected: y1 = 0 + 0.1 * 1.0 = 0.1
        assert jnp.allclose(y1, 0.1)

    def test_euler_scan_trajectory_length(self):
        """Trajectory length must equal n_steps."""
        def dydt(y, t):
            return -y

        y_init = jnp.array(1.0)

        for n_steps in [1, 10, 100]:
            _, trajectory = jtfne.euler_scan(
                y_init, t_start=0.0, dt=0.01, n_steps=n_steps, dydt_fn=dydt
            )
            assert trajectory.shape[0] == n_steps

    def test_euler_step_dtype_preservation(self):
        """dtype should be preserved (float32 by default)."""
        def dydt(y, t):
            return -y

        y0 = jnp.array(1.0, dtype=jnp.float32)
        y1 = jtfne.euler_step(y0, t=0.0, dt=0.01, dydt_fn=dydt)
        assert y1.dtype == jnp.float32

    def test_euler_scan_is_finite(self):
        """Output must be finite (no NaN/Inf)."""
        def dydt(y, t):
            return -y

        y_init = jnp.array(1.0)
        y_final, trajectory = jtfne.euler_scan(
            y_init, t_start=0.0, dt=0.01, n_steps=100, dydt_fn=dydt
        )

        assert jnp.all(jnp.isfinite(y_final))
        assert jnp.all(jnp.isfinite(trajectory))
