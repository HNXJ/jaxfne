"""Field/proxy diagnostic scaffolding for v0.2.6.

Tests that:
1. Source-balance, gauge, boundary, manufactured-residual diagnostics are JSON-safe
2. Proxy mode reports appropriate status (not_applicable or declared_metadata_only)
3. Physical_candidate mode accepts residuals and reports checked/candidate status
4. Operator status distinguishes proxy vs physical paths
5. All diagnostics keep physical_amplitude_claim_allowed false
6. Invalid paths raise clear errors
7. Existing v0.2.4 and v0.2.5 reports remain unaffected
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest

from jaxfne.fields import project_laminar_sources
from jaxfne.validation import (
    CalibrationSpec,
    make_boundary_diagnostic,
    make_calibration_report,
    make_field_operator_status,
    make_gauge_diagnostic,
    make_manufactured_residual_diagnostic,
    make_source_balance_diagnostic,
)


class TestSourceBalanceDiagnosticV026:
    """Source-balance diagnostic tests for v0.2.6."""

    def test_source_balance_proxy_mode(self):
        """Test source-balance diagnostic in proxy mode."""
        diag = make_source_balance_diagnostic(operator_path="proxy")

        assert diag["diagnostic_kind"] == "source_balance"
        assert diag["operator_path"] == "proxy"
        assert diag["status"] == "not_applicable_proxy_mode"
        assert diag["residual"] is None
        assert diag["physical_amplitude_claim_allowed"] is False

    def test_source_balance_physical_candidate_with_residual(self):
        """Test source-balance in physical_candidate mode with residual."""
        residual = 1.5e-6
        diag = make_source_balance_diagnostic(
            operator_path="physical_candidate", residual=residual
        )

        assert diag["operator_path"] == "physical_candidate"
        assert diag["status"] == "checked"
        assert diag["residual"] == residual

    def test_source_balance_physical_candidate_without_residual(self):
        """Test source-balance in physical_candidate mode without residual."""
        diag = make_source_balance_diagnostic(operator_path="physical_candidate")

        assert diag["status"] == "candidate_only"
        assert diag["residual"] is None

    def test_source_balance_invalid_path_raises(self):
        """Test that invalid operator_path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator_path"):
            make_source_balance_diagnostic(operator_path="invalid_path")

    def test_source_balance_json_safe(self):
        """Test that source-balance diagnostic is JSON-serializable."""
        diag = make_source_balance_diagnostic(
            operator_path="physical_candidate", residual=1e-6
        )
        json_str = json.dumps(diag)
        assert json_str is not None


class TestGaugeDiagnosticV026:
    """Gauge diagnostic tests for v0.2.6."""

    def test_gauge_proxy_mode(self):
        """Test gauge diagnostic in proxy mode."""
        diag = make_gauge_diagnostic(operator_path="proxy")

        assert diag["diagnostic_kind"] == "gauge"
        assert diag["operator_path"] == "proxy"
        assert diag["status"] == "declared_metadata_only"
        assert diag["gauge_mode"] == "mean_zero"
        assert diag["residual"] is None

    def test_gauge_physical_candidate_without_array(self):
        """Test gauge in physical_candidate mode without array."""
        diag = make_gauge_diagnostic(operator_path="physical_candidate")

        assert diag["operator_path"] == "physical_candidate"
        assert diag["status"] == "not_tested"
        assert diag["residual"] is None

    def test_gauge_physical_candidate_with_array(self):
        """Test gauge in physical_candidate mode with array."""
        # Create a zero-mean array
        arr = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float32)
        diag = make_gauge_diagnostic(
            field_array=arr,
            operator_path="physical_candidate",
            gauge_mode="mean_zero",
        )

        assert diag["status"] == "checked"
        assert diag["residual"] is not None
        assert isinstance(diag["residual"], float)

    def test_gauge_json_safe(self):
        """Test that gauge diagnostic is JSON-serializable."""
        diag = make_gauge_diagnostic(operator_path="proxy")
        json_str = json.dumps(diag)
        assert json_str is not None


class TestBoundaryDiagnosticV026:
    """Boundary diagnostic tests for v0.2.6."""

    def test_boundary_proxy_mode(self):
        """Test boundary diagnostic in proxy mode."""
        diag = make_boundary_diagnostic(operator_path="proxy")

        assert diag["diagnostic_kind"] == "boundary"
        assert diag["operator_path"] == "proxy"
        assert diag["status"] == "declared_metadata_only"
        assert diag["boundary_condition_status"] == "declared_metadata_only"

    def test_boundary_physical_candidate_mode(self):
        """Test boundary diagnostic in physical_candidate mode."""
        diag = make_boundary_diagnostic(operator_path="physical_candidate")

        assert diag["operator_path"] == "physical_candidate"
        assert diag["status"] == "candidate_only"

    def test_boundary_invalid_path_raises(self):
        """Test that invalid operator_path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator_path"):
            make_boundary_diagnostic(operator_path="invalid")

    def test_boundary_json_safe(self):
        """Test that boundary diagnostic is JSON-serializable."""
        diag = make_boundary_diagnostic(operator_path="proxy")
        json_str = json.dumps(diag)
        assert json_str is not None


class TestManufacturedResidualDiagnosticV026:
    """Manufactured-residual diagnostic tests for v0.2.6."""

    def test_manufactured_residual_proxy_mode(self):
        """Test manufactured-residual in proxy mode."""
        diag = make_manufactured_residual_diagnostic(operator_path="proxy")

        assert diag["diagnostic_kind"] == "manufactured_residual"
        assert diag["operator_path"] == "proxy"
        assert diag["status"] == "not_applicable_proxy_mode"
        assert diag["residual_l2_relative"] is None

    def test_manufactured_residual_physical_candidate_with_value(self):
        """Test manufactured-residual in physical_candidate with residual."""
        residual = 0.001
        diag = make_manufactured_residual_diagnostic(
            operator_path="physical_candidate", residual_l2_relative=residual
        )

        assert diag["operator_path"] == "physical_candidate"
        assert diag["status"] == "checked"
        assert diag["residual_l2_relative"] == residual

    def test_manufactured_residual_physical_candidate_without_value(self):
        """Test manufactured-residual in physical_candidate without value."""
        diag = make_manufactured_residual_diagnostic(operator_path="physical_candidate")

        assert diag["status"] == "candidate_only"
        assert diag["residual_l2_relative"] is None

    def test_manufactured_residual_json_safe(self):
        """Test that manufactured-residual is JSON-serializable."""
        diag = make_manufactured_residual_diagnostic(
            operator_path="physical_candidate", residual_l2_relative=0.01
        )
        json_str = json.dumps(diag)
        assert json_str is not None


class TestFieldOperatorStatusV026:
    """Field operator status tests for v0.2.6."""

    def test_operator_status_proxy_path(self):
        """Test operator status for proxy path."""
        status = make_field_operator_status(operator_path="proxy")

        assert status["diagnostic_kind"] == "field_operator_status"
        assert status["operator_path"] == "proxy"
        assert status["field_solver_selected"] is False
        assert status["field_solver_status"] == "laminar_proxy_no_pde"
        assert status["physical_field_solver_status"] == "not_selected"
        assert status["physical_amplitude_claim_allowed"] is False

    def test_operator_status_physical_candidate_path(self):
        """Test operator status for physical_candidate path."""
        status = make_field_operator_status(operator_path="physical_candidate")

        assert status["operator_path"] == "physical_candidate"
        assert status["field_solver_selected"] is False
        assert status["field_solver_status"] == "physical_field_solver_candidate"
        assert status["physical_amplitude_claim_allowed"] is False

    def test_operator_status_invalid_path_raises(self):
        """Test that invalid operator_path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator_path"):
            make_field_operator_status(operator_path="wrong_path")

    def test_operator_status_json_safe(self):
        """Test that operator status is JSON-serializable."""
        status = make_field_operator_status(operator_path="proxy")
        json_str = json.dumps(status)
        assert json_str is not None


class TestDiagnosticIntegrationV026:
    """Integration tests for v0.2.6 diagnostics with existing v0.2.4/v0.2.5."""

    def test_v024_field_diagnostics_still_present(self):
        """Test that v0.2.4 field admissibility diagnostics remain present."""
        sources = jnp.ones((10, 4), dtype=jnp.float32)
        positions = jnp.array(
            [[0.0, 0.0, 0.25], [0.0, 0.0, 0.5], [0.0, 0.0, 0.75], [0.0, 0.0, 0.9]],
            dtype=jnp.float32,
        )
        field = project_laminar_sources(sources, positions, n_contacts=16)

        # v0.2.4 diagnostics should be present
        assert "field_admissibility" in field.diagnostics
        assert "kernel_row_stochastic_valid" in field.diagnostics["field_admissibility"]
        assert field.diagnostics["physical_amplitude_claim_allowed"] is False

    def test_v025_calibration_spec_compatibility(self):
        """Test that v0.2.5 calibration specs remain compatible."""
        spec = CalibrationSpec(
            name="test_calib",
            target="readout",
            mode="uncalibrated_native",
        )
        report = make_calibration_report(spec)

        assert report["physical_amplitude_claim_allowed"] is False
        assert report["calibration_claim_level"] == "computational_proxy_with_declared_metadata"

    def test_all_diagnostics_keep_false(self):
        """Test that all v0.2.6 diagnostics keep physical_amplitude_claim_allowed false."""
        diags = [
            make_source_balance_diagnostic(operator_path="proxy"),
            make_source_balance_diagnostic(operator_path="physical_candidate"),
            make_gauge_diagnostic(operator_path="proxy"),
            make_boundary_diagnostic(operator_path="proxy"),
            make_manufactured_residual_diagnostic(operator_path="proxy"),
            make_field_operator_status(operator_path="proxy"),
            make_field_operator_status(operator_path="physical_candidate"),
        ]

        for diag in diags:
            assert (
                diag["physical_amplitude_claim_allowed"] is False
            ), f"Diagnostic {diag.get('diagnostic_kind', 'unknown')} should keep physical claim false"

    def test_proxy_vs_physical_candidate_distinction(self):
        """Test that proxy and physical_candidate paths are clearly distinguished."""
        # Proxy path
        proxy_sb = make_source_balance_diagnostic(operator_path="proxy")
        proxy_status = make_field_operator_status(operator_path="proxy")

        # Physical candidate path
        candidate_sb = make_source_balance_diagnostic(operator_path="physical_candidate")
        candidate_status = make_field_operator_status(operator_path="physical_candidate")

        # Proxy should report not_applicable or declared_metadata_only
        assert proxy_sb["status"] == "not_applicable_proxy_mode"
        assert proxy_status["field_solver_status"] == "laminar_proxy_no_pde"

        # Physical candidate should be distinct
        assert candidate_sb["status"] == "candidate_only"
        assert candidate_status["field_solver_status"] == "physical_field_solver_candidate"

        # Both should keep physical amplitude false
        assert proxy_status["physical_amplitude_claim_allowed"] is False
        assert candidate_status["physical_amplitude_claim_allowed"] is False
