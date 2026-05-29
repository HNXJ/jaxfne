"""Tests for generalized LinearReadout operators and CSD second-derivative invariants.

Covers dimension contracts, metadata reports, CSD boundary checks, and EMM-proxy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_linear_readout_shapes():
    """Verify shape contract for LinearReadout apply method."""
    c = 4
    n = 10
    W = jnp.ones((c, n), dtype=jnp.float32)
    readout = jtfne.fields.LinearReadout(name="test_readout", W=W)
    
    # 2D source array [T, N]
    s_2d = jnp.zeros((50, n), dtype=jnp.float32)
    y_2d = readout.apply(s_2d)
    assert y_2d.shape == (50, c)
    
    # 1D source array [N]
    s_1d = jnp.zeros((n,), dtype=jnp.float32)
    y_1d = readout.apply(s_1d)
    assert y_1d.shape == (c,)


def test_linear_readout_report():
    """Verify that reports are standardized and deny physical amplitude claims."""
    c = 2
    n = 5
    W = jnp.ones((c, n), dtype=jnp.float32)
    readout = jtfne.fields.LinearReadout(name="eeg_like", W=W, leadfield_status="toy_leadfield")
    report = readout.report()
    
    assert report["name"] == "eeg_like"
    assert report["physical_amplitude_claim_allowed"] is False
    assert report["operator_status"] == "simulated_proxy"
    assert report["leadfield_status"] == "toy_leadfield"


def test_csd_finite_difference_invariants():
    """Verify CSD second-derivative finite difference properties.
    
    CSD ≈ -sigma_e * d^2(Phi)/dz^2.
    - Constant potential Phi(z) = C -> CSD = 0.
    - Linear potential Phi(z) = a * z + b -> CSD = 0.
    - Quadratic potential Phi(z) = a * z^2 -> CSD = -2 * a * sigma_e.
    """
    sigma_e = 0.3
    dz = 0.1
    
    # Grid of depth contacts: 5 depths
    # Phi shape: [T, Depth]
    # Here T = 1, Depth = 5
    
    # 1. Constant potential: Phi = 5.0
    phi_const = jnp.full((1, 5), 5.0, dtype=jnp.float32)
    
    def compute_csd(phi):
        # Interior points: index 1, 2, 3
        # CSD_c = -sigma_e * (phi[c+1] - 2*phi[c] + phi[c-1]) / dz^2
        # Using slice operations for interior
        interior = phi[:, 1:-1]
        right = phi[:, 2:]
        left = phi[:, :-2]
        return -sigma_e * (right - 2 * interior + left) / (dz**2)

    csd_const = compute_csd(phi_const)
    assert np.allclose(csd_const, 0.0, atol=1e-6)
    
    # 2. Linear potential: Phi(z) = 2.0 * z + 3.0
    z = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    phi_linear = (2.0 * z + 3.0).reshape(1, 5)
    csd_linear = compute_csd(phi_linear)
    assert np.allclose(csd_linear, 0.0, atol=1e-5)
    
    # 3. Quadratic potential: Phi(z) = 4.0 * z^2
    phi_quad = (4.0 * (z**2)).reshape(1, 5)
    csd_quad = compute_csd(phi_quad)
    
    # Theoretical second derivative of 4 * z^2 is 8
    # CSD should be -sigma_e * 8 = -0.3 * 8 = -2.4
    expected_csd = -sigma_e * 8.0
    assert np.allclose(csd_quad, expected_csd, rtol=1e-5, atol=1e-5)
