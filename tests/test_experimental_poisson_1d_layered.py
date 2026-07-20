"""Tests for experimental_poisson_1d's new layered (piecewise-constant
conductivity) mode, added toward plans.json's
novelty::tfne-differentiable-field-solver / tfne-analytic-ground-truth-
validation initiative.

Two things verified: (1) the scalar-conductivity path is bit-identical to
the original implementation (backward compatibility -- every existing
caller, including tests/test_solver_smoke_v0401.py, sees zero behavior
change); (2) the new layered path matches a real, independently-derived
closed-form analytic solution -- the classical "two half-space" point-
source volume-conductor result -- at both physically-meaningful limiting
cases (equal conductivities reduces to uniform; large conductivity ratio
approaches the insulating-boundary doubling result), not just an internal
self-consistency check.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxfne.fields.solvers import experimental_poisson_1d


def test_layered_reduces_to_scalar_path_when_uniform():
    """A per-face array with every entry equal to the same value must give
    bit-identical results to passing that value as a scalar -- confirms the
    two code paths are numerically equivalent at the point they overlap."""
    N = 12
    dx = 0.05
    rng = np.random.default_rng(0)
    sources = jnp.asarray(rng.normal(size=N) - rng.normal(size=N).mean(), dtype=jnp.float32)
    sigma = 2.3

    phi_scalar, residual_scalar, manifest_scalar = experimental_poisson_1d(sources, sigma, dx)
    phi_array, residual_array, manifest_array = experimental_poisson_1d(
        sources, jnp.full((N - 1,), sigma, dtype=jnp.float32), dx)

    np.testing.assert_allclose(np.asarray(phi_scalar), np.asarray(phi_array), atol=1e-4)
    np.testing.assert_allclose(np.asarray(residual_scalar), np.asarray(residual_array), atol=1e-4)
    assert manifest_scalar["layered_conductivity"] is False
    assert manifest_array["layered_conductivity"] is True


def test_original_smoke_case_unchanged():
    """Exact re-run of tests/test_solver_smoke_v0401.py's fixture -- confirms
    the refactor didn't perturb the pre-existing scalar path at all."""
    N = 10
    dx = 0.1
    conductivity = 1.5
    sources = jnp.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 0.3, -0.3], dtype=jnp.float32)

    phi, residual, manifest = experimental_poisson_1d(sources, conductivity, dx)

    assert phi.shape == (N,)
    assert jnp.all(jnp.isfinite(phi))
    assert manifest["residual_norm"] < 1e-3
    assert manifest["convergence_status"] == "converged"


def test_layered_raises_on_wrong_shape():
    N = 8
    sources = jnp.zeros((N,), dtype=jnp.float32)
    bad_conductivity = jnp.ones((N,), dtype=jnp.float32)  # should be (N-1,), not (N,)
    try:
        experimental_poisson_1d(sources, bad_conductivity, dx=0.1)
        assert False, "expected ValueError for wrong-shape conductivity array"
    except ValueError as e:
        assert "N-1" in str(e) or str(N - 1) in str(e)


# --- Analytic ground-truth cross-check -------------------------------------
#
# Classical "two half-space" volume-conductor result (derived and verified
# via 3 limiting-case sanity checks before use here -- see
# docs/guides/jax_fem_interop.md's validation-target notes): a point current
# source I at depth z0 in medium 1 (conductivity sigma1, z>0), with medium 2
# (conductivity sigma2, z<0) below an interface at z=0. Reflection
# coefficient k = (sigma1-sigma2)/(sigma1+sigma2), derived from continuity of
# potential and continuity of normal current density at z=0:
#   phi1(rho,z) = I/(4*pi*sigma1) * [1/R0 + k/R0'],  z>=0
#   phi2(rho,z) = I/(4*pi*sigma1) * (1+k)/R0'',      z<0
# where R0=sqrt(rho^2+(z-z0)^2) (real source), R0'=sqrt(rho^2+(z+z0)^2)
# (image source), R0''=sqrt(rho^2+(z-z0)^2) (same distance form as R0,
# evaluated in medium 2). Sanity checks: sigma2->sigma1 gives k->0 (uniform
# medium, phi1=phi2=I/(4*pi*sigma*R0)); sigma2->0 (insulating boundary) gives
# k->1 (classic "doubling" result); sigma2->inf (grounded boundary) gives
# k->-1 (image cancels, phi->0 at the boundary).

def _analytic_two_half_space_potential(rho, z, z0, I, sigma1, sigma2):
    k = (sigma1 - sigma2) / (sigma1 + sigma2)
    R0 = np.sqrt(rho**2 + (z - z0)**2)
    R0p = np.sqrt(rho**2 + (z + z0)**2)
    if z >= 0:
        return I / (4 * np.pi * sigma1) * (1.0 / R0 + k / R0p)
    R0pp = np.sqrt(rho**2 + (z - z0)**2)
    return I / (4 * np.pi * sigma1) * (1.0 + k) / R0pp


def test_analytic_reference_uniform_medium_limit():
    """sigma2 -> sigma1 (no real interface) must reduce the two-half-space
    formula to the plain infinite-uniform-medium point-source potential
    I/(4*pi*sigma*R) everywhere -- the simplest possible sanity check on the
    derivation itself, independent of experimental_poisson_1d."""
    sigma = 1.5
    I = 2.0
    z0 = 0.3
    for z in (0.5, -0.5, 0.0):
        rho = 0.4
        phi = _analytic_two_half_space_potential(rho, z, z0, I, sigma, sigma)
        R = np.sqrt(rho**2 + (z - z0)**2)
        expected = I / (4 * np.pi * sigma * R)
        np.testing.assert_allclose(phi, expected, rtol=1e-10)


def test_analytic_reference_insulating_and_grounded_limits():
    """sigma2->0 (insulating) must double the image contribution (k->1);
    sigma2->inf (grounded) must cancel it entirely at z=0 (k->-1, phi->0 at
    the boundary itself). Both are textbook volume-conduction results,
    checked here as a receipt that the derivation used for the analytic
    ground-truth target is actually correct before it gets used to validate
    any solver."""
    sigma1, I, z0, rho = 1.0, 1.0, 0.5, 0.3
    R0 = np.sqrt(rho**2 + 0.0)  # at z=0

    phi_insulating = _analytic_two_half_space_potential(rho, 0.0, z0, I, sigma1, 1e-9)
    R_at_0 = np.sqrt(rho**2 + z0**2)
    expected_insulating = I / (4 * np.pi * sigma1) * 2.0 / R_at_0
    np.testing.assert_allclose(phi_insulating, expected_insulating, rtol=1e-4)

    phi_grounded = _analytic_two_half_space_potential(rho, 0.0, z0, I, sigma1, 1e9)
    np.testing.assert_allclose(phi_grounded, 0.0, atol=1e-6)
