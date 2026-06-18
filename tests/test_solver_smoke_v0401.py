"""Smoke test for v0.4.1 experimental 1D Poisson solver."""

import pytest
import json
import jax
import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne.fields import experimental_poisson_1d

def test_experimental_poisson_1d_smoke():
    # Grid of size 10, spacing dx=0.1, conductivity=1.5
    N = 10
    dx = 0.1
    conductivity = 1.5

    # Create balanced charge sources (sum to 0.0)
    sources = jnp.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 0.3, -0.3], dtype=jnp.float32)

    phi, residual, manifest = experimental_poisson_1d(
        sources=sources,
        conductivity=conductivity,
        dx=dx,
    )

    # 1. Assert finite outputs
    assert phi.shape == (N,)
    assert jnp.all(jnp.isfinite(phi))
    assert residual.shape == (N,)
    assert jnp.all(jnp.isfinite(residual))

    # 2. Assert residual tolerance
    residual_norm = manifest["residual_norm"]
    assert residual_norm < 1e-3, f"Residual norm {residual_norm} exceeds tolerance"
    assert manifest["convergence_status"] == "converged"

    # 3. Assert boundary and gauge metadata
    assert "boundary_condition" in manifest
    assert "gauge_choice" in manifest
    assert "residual_norm" in manifest
    assert "convergence_status" in manifest

    assert manifest["boundary_condition"] == "mean_zero_neumann"
    assert manifest["gauge_choice"] == "mean_zero"
    assert jnp.isfinite(manifest["residual_norm"])
    assert manifest["convergence_status"] == "converged"
    
    assert manifest["claim_level"] == "computational_scaffold"
    assert manifest["field_solver_status"] == "experimental_pde_solver"
    assert manifest["physical_amplitude_calibrated"] is False

    # 4. Assert strict JSON serialization
    try:
        manifest_json = json.dumps(manifest, allow_nan=False)
        assert manifest_json is not None
    except ValueError as e:
        pytest.fail(f"Manifest is not strict JSON serializable: {e}")
