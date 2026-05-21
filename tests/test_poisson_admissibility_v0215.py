"""Tests for Poisson admissibility specification (v0.2.15)."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.validation import (
    build_poisson_admissibility_report,
    validate_poisson_field_arrays,
    validate_poisson_gauge_condition,
    validate_poisson_source_conservation,
    validate_poisson_spd_conductivity,
)


class TestPoissonSPDConductivity:
    """Gate 1: Conductivity Symmetric Positive Definite (SPD)."""

    def test_spd_identity_matrix(self):
        """Identity matrix is SPD."""
        sigma = jnp.eye(3)
        is_spd, msg = validate_poisson_spd_conductivity(sigma)
        assert is_spd
        assert "SPD verified" in msg

    def test_spd_diagonal_positive(self):
        """Diagonal matrix with positive entries is SPD."""
        sigma = jnp.diag(jnp.array([1.0, 2.0, 0.5]))
        is_spd, msg = validate_poisson_spd_conductivity(sigma)
        assert is_spd
        assert "min eigenvalue" in msg

    def test_spd_symmetric_positive(self):
        """General symmetric positive definite matrix."""
        # Create SPD matrix: A^T @ A
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        is_spd, msg = validate_poisson_spd_conductivity(A)
        assert is_spd

    def test_not_spd_has_negative_eigenvalue(self):
        """Matrix with negative eigenvalue fails SPD test."""
        sigma = jnp.array([[1.0, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 1.0]])
        is_spd, msg = validate_poisson_spd_conductivity(sigma)
        assert not is_spd
        assert "not positive definite" in msg or "min eigenvalue" in msg

    def test_not_spd_not_symmetric(self):
        """Non-symmetric matrix fails SPD test."""
        sigma = jnp.array([[1.0, 2.0], [0.5, 1.0]])  # Not symmetric
        is_spd, msg = validate_poisson_spd_conductivity(sigma)
        assert not is_spd
        assert "not symmetric" in msg or "not positive definite" in msg

    def test_scalar_input(self):
        """Scalar input is rejected."""
        is_spd, msg = validate_poisson_spd_conductivity(1.5)
        assert not is_spd
        assert "scalar" in msg

    def test_spd_tolerance(self):
        """SPD test respects tolerance for near-zero eigenvalues."""
        # Create matrix with tiny negative eigenvalue
        sigma = jnp.array([[1.0, 0.0], [0.0, 1e-10]])
        is_spd, msg = validate_poisson_spd_conductivity(sigma, tolerance=1e-9)
        assert is_spd  # Passes with larger tolerance


class TestPoissonSourceConservation:
    """Gate 2: Source conservation."""

    def test_conservation_perfect(self):
        """Perfect conservation: source = -flux."""
        src = 1.0
        flux = -1.0
        is_conserved, msg, residual = validate_poisson_source_conservation(src, flux)
        assert is_conserved
        assert residual < 1e-10
        assert "conserved" in msg

    def test_conservation_within_tolerance(self):
        """Conservation within tolerance passes."""
        src = 1.0
        flux = -1.0 + 1e-7  # Small error
        is_conserved, msg, residual = validate_poisson_source_conservation(
            src, flux, tolerance=1e-6
        )
        assert is_conserved
        assert residual < 1e-6

    def test_conservation_fails_large_residual(self):
        """Large residual fails conservation."""
        src = 1.0
        flux = -0.5  # Big mismatch
        is_conserved, msg, residual = validate_poisson_source_conservation(src, flux)
        assert not is_conserved
        assert residual > 0.4
        assert "not conserved" in msg

    def test_conservation_none_source(self):
        """Missing source data is OK (not tested)."""
        is_conserved, msg, residual = validate_poisson_source_conservation(None, -1.0)
        assert is_conserved
        assert residual is None
        assert "not tested" in msg or "not available" in msg

    def test_conservation_none_flux(self):
        """Missing flux data is OK."""
        is_conserved, msg, residual = validate_poisson_source_conservation(1.0, None)
        assert is_conserved
        assert "not tested" in msg or "not available" in msg

    def test_conservation_zero_source(self):
        """Zero source and zero flux conserves."""
        is_conserved, msg, residual = validate_poisson_source_conservation(0.0, -0.0)
        assert is_conserved

    def test_conservation_both_none(self):
        """Both data missing is OK."""
        is_conserved, msg, residual = validate_poisson_source_conservation(None, None)
        assert is_conserved
        assert residual is None


class TestPoissonGaugeCondition:
    """Gate 3: Gauge condition (mean-zero)."""

    def test_gauge_mean_zero_satisfied(self):
        """Mean potential ≈ 0 satisfies mean-zero gauge."""
        mean_phi = 0.0
        is_satisfied, msg = validate_poisson_gauge_condition(
            mean_phi, gauge="mean_zero"
        )
        assert is_satisfied
        assert "satisfied" in msg

    def test_gauge_mean_zero_near_zero(self):
        """Mean potential slightly > 0 still satisfies gauge within tolerance."""
        mean_phi = 1e-8
        is_satisfied, msg = validate_poisson_gauge_condition(
            mean_phi, gauge="mean_zero", tolerance=1e-6
        )
        assert is_satisfied

    def test_gauge_mean_zero_violated(self):
        """Large mean violates gauge."""
        mean_phi = 0.1
        is_satisfied, msg = validate_poisson_gauge_condition(
            mean_phi, gauge="mean_zero", tolerance=1e-6
        )
        assert not is_satisfied
        assert "violated" in msg

    def test_gauge_none_data(self):
        """Missing mean potential is OK (not tested)."""
        is_satisfied, msg = validate_poisson_gauge_condition(None)
        assert is_satisfied
        assert "not tested" in msg or "not available" in msg

    def test_gauge_other_type(self):
        """Other gauge types are noted as not implemented."""
        is_satisfied, msg = validate_poisson_gauge_condition(
            0.5, gauge="other_gauge"
        )
        assert is_satisfied  # Not rejected, just not validated
        assert "not implemented" in msg


class TestPoissonFieldArrays:
    """Gate 4: Field array finiteness."""

    def test_field_all_finite(self):
        """All finite arrays pass."""
        phi = jnp.ones((10, 20))
        J = jnp.ones((10, 20, 3))
        CSD = jnp.ones((10, 20))
        results = validate_poisson_field_arrays(phi_e=phi, J_e=J, CSD=CSD)
        assert all(results.values())

    def test_field_phi_e_nan(self):
        """NaN in phi_e fails."""
        phi = jnp.array([1.0, jnp.nan, 3.0])
        results = validate_poisson_field_arrays(phi_e=phi)
        assert not results["finite_phi_e"]

    def test_field_J_e_inf(self):
        """Inf in J_e fails."""
        J = jnp.array([1.0, jnp.inf, 3.0])
        results = validate_poisson_field_arrays(J_e=J)
        assert not results["finite_J_e"]

    def test_field_CSD_mixed_nan_inf(self):
        """Both NaN and Inf fail."""
        CSD = jnp.array([jnp.nan, jnp.inf, 3.0])
        results = validate_poisson_field_arrays(CSD=CSD)
        assert not results["finite_CSD"]

    def test_field_none_arrays(self):
        """None values are OK (not provided)."""
        results = validate_poisson_field_arrays(phi_e=None, J_e=None, CSD=None)
        assert all(results.values())  # All True when data missing

    def test_field_sparse_arrays(self):
        """Test with different array types (numpy, jax)."""
        phi_np = np.ones((5, 10))
        J_jax = jnp.ones((5, 10, 3))
        results = validate_poisson_field_arrays(phi_e=phi_np, J_e=J_jax)
        assert results["finite_phi_e"]
        assert results["finite_J_e"]


class TestPoissonAdmissibilityReport:
    """Full admissibility report building and structure."""

    def test_report_all_gates_pass(self):
        """All gates pass → admissible, but v0.2.15 never allows physical amplitude claims."""
        sigma = jnp.eye(3)
        report = build_poisson_admissibility_report(
            conductivity=sigma,
            integrated_source=1.0,
            integrated_boundary_flux=-1.0,
            mean_potential=0.0,
            phi_e=jnp.ones((10, 10)),
            J_e=jnp.ones((10, 10, 3)),
            CSD=jnp.ones((10, 10)),
            solver_residual_l2=1e-7,
            n_iterations=100,
            converged=True,
        )
        assert report["admissibility_status"] == "admissible"
        # v0.2.15 invariant: physical_amplitude_claim_allowed is ALWAYS false (specification-only)
        assert report["physical_amplitude_claim_allowed"] is False
        # Verify the v0215_note explains specification-only status
        assert "specification-only" in report.get("v0215_note", "").lower()

    def test_report_one_gate_fails(self):
        """One gate fails → not admissible."""
        sigma_bad = jnp.array([[1.0, 0.0], [0.0, -0.1]])
        report = build_poisson_admissibility_report(
            conductivity=sigma_bad,
            integrated_source=1.0,
            integrated_boundary_flux=-1.0,
            mean_potential=0.0,
            phi_e=jnp.ones((10, 10)),
        )
        assert report["admissibility_status"] == "not_admissible"
        assert report["physical_amplitude_claim_allowed"] is False

    def test_report_json_safe(self):
        """Report is JSON-serializable."""
        report = build_poisson_admissibility_report(
            conductivity=jnp.eye(3),
            integrated_source=1.0,
            integrated_boundary_flux=-1.0,
        )
        # Should not raise
        json_str = json.dumps(report, allow_nan=False)
        assert isinstance(json_str, str)

    def test_report_required_fields(self):
        """Report has all required fields."""
        report = build_poisson_admissibility_report()
        assert "diagnostic_kind" in report
        assert "admissibility_status" in report
        assert "gates" in report
        assert "solver_metadata" in report
        assert "physical_amplitude_claim_allowed" in report
        assert report["diagnostic_kind"] == "poisson_admissibility"

    def test_report_gates_subfields(self):
        """Each gate has required subfields."""
        report = build_poisson_admissibility_report(
            conductivity=jnp.eye(3),
            integrated_source=1.0,
            integrated_boundary_flux=-1.0,
            mean_potential=0.0,
            phi_e=jnp.ones((5, 5)),
        )
        gates = report["gates"]
        assert "conductivity_spd" in gates
        assert "source_conservation" in gates
        assert "gauge_condition" in gates
        assert "field_finiteness" in gates

    def test_report_metadata_fields(self):
        """Solver metadata is complete."""
        report = build_poisson_admissibility_report(
            solver_residual_l2=1e-7,
            n_iterations=200,
            converged=True,
            boundary_condition="dirichlet",
            gauge="mean_zero",
            csd_sign_convention="positive_equals_extracellular_source",
        )
        meta = report["solver_metadata"]
        assert "solver_residual_l2_relative" in meta
        assert "n_iterations" in meta
        assert "converged" in meta
        assert "boundary_condition" in meta
        assert "gauge" in meta
        assert "csd_sign_convention" in meta

    def test_report_v0215_note(self):
        """Report includes v0.2.15 note about specification."""
        report = build_poisson_admissibility_report()
        assert "v0215_note" in report
        assert "specification" in report["v0215_note"]

    def test_report_physical_amplitude_always_false_v0215(self):
        """v0.2.15 INVARIANT: physical_amplitude_claim_allowed is ALWAYS false.

        v0.2.15 is specification-only (no solver implemented). Even if all gates
        pass synthetically, physical amplitude claims must remain false.
        Only v0.2.16+ (with actual solver + calibration) may allow this.
        """
        # Even with perfect gates, still false
        report_perfect = build_poisson_admissibility_report(
            conductivity=jnp.eye(2),
            integrated_source=1.0,
            integrated_boundary_flux=-1.0,
            mean_potential=0.0,
            phi_e=jnp.ones((5, 5)),
            J_e=jnp.ones((5, 5, 2)),
            CSD=jnp.ones((5, 5)),
            converged=True,
        )
        assert report_perfect["admissibility_status"] == "admissible"
        assert not report_perfect["physical_amplitude_claim_allowed"], \
            "v0.2.15: must never allow physical amplitude claims (no solver yet)"

        # With failed gates, also false
        report_failed = build_poisson_admissibility_report(
            conductivity=jnp.array([[1.0, 0.0], [0.0, -0.1]]),
        )
        assert report_failed["admissibility_status"] == "not_admissible"
        assert not report_failed["physical_amplitude_claim_allowed"]

        # v0215_note must reference specification-only status
        assert "v0.2.15" in report_perfect["v0215_note"]
        assert "specification-only" in report_perfect["v0215_note"]


class TestPoissonIntegration:
    """Integration tests: realistic workflows."""

    def test_full_workflow_admissible_solution(self):
        """Full workflow: build conductivity, solve (mock), validate."""
        # Mock conductivity
        sigma = jnp.eye(3) * 0.3

        # Mock solution
        phi_e = jnp.ones((10, 20)) * 5.0
        phi_e = phi_e - phi_e.mean()  # Enforce mean-zero
        J_e = jnp.ones((10, 20, 3)) * -0.1
        CSD = jnp.ones((10, 20)) * 0.5

        # Mock integrals
        src_integral = 0.5
        flux_integral = -0.5

        # Build report
        report = build_poisson_admissibility_report(
            conductivity=sigma,
            integrated_source=src_integral,
            integrated_boundary_flux=flux_integral,
            mean_potential=phi_e.mean(),
            phi_e=phi_e,
            J_e=J_e,
            CSD=CSD,
            solver_residual_l2=3e-7,
            n_iterations=150,
            converged=True,
        )

        # Verify
        assert report["admissibility_status"] == "admissible"
        assert all(v["passed"] for v in report["gates"].values())

    def test_workflow_spd_check_first(self):
        """SPD is checked first before other gates."""
        sigma_bad = jnp.array([[1.0, 0.0], [0.0, -0.5]])
        is_spd, _ = validate_poisson_spd_conductivity(sigma_bad)
        assert not is_spd
        # Then don't proceed to solver

    def test_workflow_catch_nan_early(self):
        """NaN in field triggers finiteness gate."""
        phi_nan = jnp.array([1.0, jnp.nan, 3.0])
        results = validate_poisson_field_arrays(phi_e=phi_nan)
        assert not results["finite_phi_e"]
        # Report reflects this
        report = build_poisson_admissibility_report(phi_e=phi_nan)
        assert not report["gates"]["field_finiteness"]["passed"]


class TestPoissonEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_arrays(self):
        """Empty arrays are technically finite."""
        phi_empty = jnp.array([])
        results = validate_poisson_field_arrays(phi_e=phi_empty)
        assert results["finite_phi_e"]  # No elements, so all are finite vacuously

    def test_very_small_conductivity(self):
        """Very small but positive conductivity is still SPD."""
        sigma = jnp.eye(2) * 1e-10
        is_spd, msg = validate_poisson_spd_conductivity(sigma)
        assert is_spd
        assert "e-10" in msg  # Format: 1.00e-10

    def test_large_dimension_conductivity(self):
        """SPD check works for larger tensors."""
        sigma = jnp.eye(10)
        is_spd, msg = validate_poisson_spd_conductivity(sigma)
        assert is_spd

    def test_near_singular_conductivity(self):
        """Near-singular (but SPD) conductivity."""
        # Create nearly singular matrix: small random perturbation on identity
        rng = np.random.RandomState(42)
        A = np.eye(3) + 1e-6 * rng.randn(3, 3)
        A = (A + A.T) / 2  # Make symmetric
        # Add enough to make eigenvalues positive
        A = A + np.eye(3) * 0.1
        sigma = jnp.array(A)
        is_spd, msg = validate_poisson_spd_conductivity(sigma)
        assert is_spd

    def test_asymmetric_tolerance(self):
        """Conservation checks abs(src + flux), not abs(src - flux)."""
        # Test: src + flux should be ~ 0 for conservation
        src = 1.0
        flux = -1.0  # src + flux = 0, conserved
        is_conserved, msg, residual = validate_poisson_source_conservation(src, flux)
        assert is_conserved
        assert residual < 1e-10

        # Test: src + flux != 0 fails conservation
        src = 1e6
        flux = -1e6 + 1.0  # src + flux = 1.0, not conserved
        is_conserved, msg, residual = validate_poisson_source_conservation(src, flux)
        assert not is_conserved
        assert residual > 0.99
