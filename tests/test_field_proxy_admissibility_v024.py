"""Field proxy admissibility diagnostics for v0.2.4.

Tests that:
1. Row sums of proxy kernel are row-normalized (stochastic).
2. Column/source conservation is not claimed in proxy mode.
3. Reports include `source_current_conservation_status`.
4. Boundary/gauge statuses are declared metadata only in proxy mode.
5. Physical amplitude claims remain false.
6. JSON reports remain strict.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxfne.fields import project_laminar_sources, validate_source_field_status


class TestFieldProxyAdmissibilityV024:
    """Field proxy admissibility tests for v0.2.4."""

    @pytest.fixture
    def minimal_setup(self):
        """Minimal source and position setup for testing."""
        n_neurons = 4
        t_steps = 10
        sources = jnp.ones((t_steps, n_neurons), dtype=jnp.float32)
        positions = jnp.array(
            [[0.0, 0.0, 0.25], [0.0, 0.0, 0.5], [0.0, 0.0, 0.75], [0.0, 0.0, 0.9]],
            dtype=jnp.float32,
        )
        return sources, positions

    def test_kernel_row_normalization(self, minimal_setup):
        """Test that kernel rows sum to 1.0 (row-stochastic)."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        kernel = field.kernel

        row_sums = jnp.sum(kernel, axis=1)
        assert jnp.allclose(
            row_sums, 1.0, atol=1e-6
        ), f"Kernel rows not normalized: max error = {jnp.max(jnp.abs(row_sums - 1.0))}"

    def test_kernel_row_normalization_in_diagnostics(self, minimal_setup):
        """Test that kernel row normalization is reported correctly."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        diag = field.diagnostics

        # Check that field_admissibility contains row normalization info
        assert "field_admissibility" in diag
        assert "kernel_row_stochastic_valid" in diag["field_admissibility"]
        assert (
            diag["field_admissibility"]["kernel_row_stochastic_valid"] is True
        ), "Kernel row-stochastic validity not reported correctly"

    def test_source_conservation_not_claimed(self, minimal_setup):
        """Test that source conservation is not claimed in proxy mode."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        diag = field.diagnostics

        # Explicitly check that source conservation is marked as not applicable
        assert "field_admissibility" in diag
        assert "source_current_conservation_status" in diag["field_admissibility"]
        assert (
            diag["field_admissibility"]["source_current_conservation_status"]
            == "not_applicable_proxy_mode"
        ), "Source conservation status not correctly declared as not applicable"

    def test_boundary_gauge_declared_metadata_only(self, minimal_setup):
        """Test that boundary and gauge are declared metadata only."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        diag = field.diagnostics

        assert "field_admissibility" in diag
        assert diag["field_admissibility"]["boundary_condition_status"] == "declared_metadata_only"
        assert diag["field_admissibility"]["gauge_status"] == "declared_metadata_only"

    def test_physical_amplitude_claim_false(self, minimal_setup):
        """Test that physical amplitude claims are false."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        diag = field.diagnostics

        assert diag["physical_amplitude_claim_allowed"] is False

    def test_source_field_status_validation_report(self, minimal_setup):
        """Test that validate_source_field_status produces correct report."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)

        status = validate_source_field_status(
            field_output=field,
            requested_modes=["CSD", "LFP"],
        )

        # Check key status fields
        assert status["field_solver_status"] == "laminar_proxy_no_pde"
        assert status["source_projection_mode"] == "proxy_no_field_solve"
        assert status["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
        assert status["physical_amplitude_claim_allowed"] is False
        assert status["is_proxy"] is True
        assert status["is_calibrated"] is False
        assert status["validation_status"] == "proxy_status_report_only"

    def test_field_claim_level_proxy_only(self, minimal_setup):
        """Test that field claim level is proxy_readout_only."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        diag = field.diagnostics

        assert diag["field_claim_level"] == "proxy_readout_only"

    def test_kernel_normalization_definition_explicit(self, minimal_setup):
        """Test that kernel normalization definition is explicit."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        diag = field.diagnostics

        assert diag["field_admissibility"]["kernel_normalization_definition"] == "contact_rows_sum_to_one_proxy"

    def test_projection_invariants_finiteness(self, minimal_setup):
        """Test that projection invariants check for NaN/Inf."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        diag = field.diagnostics

        # All key arrays should be finite
        assert diag["finite_source_proxy"] is True
        assert diag["finite_phi_e_proxy"] is True
        assert diag["finite_csd_proxy"] is True
        assert diag["finite_lfp_proxy"] is True

    def test_kernel_never_negative(self, minimal_setup):
        """Test that kernel never contains negative values."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        kernel = field.kernel

        assert jnp.all(kernel >= 0.0), "Kernel contains negative values"

    def test_diagnostics_json_safe(self, minimal_setup):
        """Test that diagnostics dict is JSON-safe (no nested jax arrays)."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)
        diag = field.diagnostics

        # Recursively check that no JAX arrays are in the diagnostics
        def has_jax_array(obj):
            if isinstance(obj, jax.Array):
                return True
            if isinstance(obj, dict):
                return any(has_jax_array(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return any(has_jax_array(v) for v in obj)
            return False

        assert not has_jax_array(diag), "Diagnostics contain JAX arrays"

    def test_proxy_modes_diagnostic_warning(self, minimal_setup):
        """Test that proxy readout modes generate warnings in validation."""
        sources, positions = minimal_setup
        field = project_laminar_sources(sources, positions, n_contacts=16)

        status = validate_source_field_status(
            field_output=field,
            requested_modes=["CSD", "LFP", "phi_e"],
        )

        # Should have warnings about proxy modes
        assert len(status["warnings"]) > 0
        assert any("proxy_readout_modes_requested" in w for w in status["warnings"])

    def test_stencil_numerical_parity_with_gradient(self):
        """Verify parity of sliced finite-difference stencil and double-gradient on smooth quadratic field."""
        # 1D coordinate array with fewer contacts to minimize float32 noise amplification
        n_contacts = 8
        contacts = jnp.linspace(0.0, 1.0, n_contacts, dtype=jnp.float32)
        dz = contacts[1] - contacts[0]

        # Analytical quadratic potential field: phi(z) = z^2
        # Shape: [T=1, n_contacts]
        phi_e_proxy = (contacts ** 2)[None, :]

        # 1. Direct stencil evaluation on interior
        interior_stencil = (phi_e_proxy[:, 2:] - 2.0 * phi_e_proxy[:, 1:-1] + phi_e_proxy[:, :-2]) / (dz * dz)

        # 2. Double gradient evaluation on interior
        grad_1 = jnp.gradient(phi_e_proxy, dz, axis=1)
        grad_2 = jnp.gradient(grad_1, dz, axis=1)
        interior_grad = grad_2[:, 1:-1]

        # Assert strict numerical parity on the interior, bypassing boundary-contaminated points [:, 1:-1]
        assert jnp.allclose(interior_stencil[:, 1:-1], interior_grad[:, 1:-1], atol=1e-5)

    def test_finite_boundary_checks(self):
        """Assert that edge padding rules maintain array dimensions and do not inject NaN/Inf."""
        # Standard input setups
        n_neurons = 4
        t_steps = 5
        sources = jnp.ones((t_steps, n_neurons), dtype=jnp.float32)
        positions = jnp.array(
            [[0.0, 0.0, 0.2], [0.0, 0.0, 0.4], [0.0, 0.0, 0.6], [0.0, 0.0, 0.8]],
            dtype=jnp.float32,
        )

        # Evaluate project_laminar_sources with various contact counts
        for n_contacts in [3, 8, 32]:
            field = project_laminar_sources(sources, positions, n_contacts=n_contacts)
            csd = field.csd
            phi = field.phi_e

            # Invariant 1: csd shape matches phi shape
            assert csd.shape == phi.shape
            assert csd.shape[-1] == n_contacts

            # Invariant 2: csd does not contain NaN or Inf at the boundaries or interior
            assert jnp.all(jnp.isfinite(csd))

