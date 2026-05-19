"""Tests for v0.2.0 field admissibility validation gates.

Covers conductivity tensor validation (scalar, diagonal, full SPD),
field finiteness checks, gauge/boundary metadata, source conservation,
and JSON-safe field admissibility reports.
"""

import json

import pytest

import jaxfne
from jaxfne.validation import (
    validate_scalar_conductivity,
    validate_diagonal_conductivity,
    validate_full_spd_conductivity,
    validate_field_arrays_finite,
    build_field_admissibility_report,
)


def _cfg(n=8):
    """Minimal configuration."""
    return (
        jaxfne.configuration()
        .network(n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=4)
    )


def _model_and_signals(n=8):
    """Construct model and run simulation."""
    model = jaxfne.construct(_cfg(n=n))
    sim = jaxfne.Simulation(duration_ms=10.0, dt_ms=0.5)
    signals = model.simulate(sim)
    return model, signals


class TestScalarConductivityValidation:
    """Scalar conductivity validation."""

    def test_scalar_positive_passes(self):
        result = validate_scalar_conductivity(1.5)
        assert result["is_valid"] is True
        assert result["is_positive"] is True
        assert result["is_finite"] is True
        assert result["status"] == "passed"

    def test_scalar_zero_fails(self):
        result = validate_scalar_conductivity(0.0)
        assert result["is_valid"] is False
        assert result["is_positive"] is False
        assert result["status"] == "non_positive"

    def test_scalar_negative_fails(self):
        result = validate_scalar_conductivity(-1.5)
        assert result["is_valid"] is False
        assert result["is_positive"] is False
        assert result["status"] == "non_positive"

    def test_scalar_nan_fails(self):
        result = validate_scalar_conductivity(float("nan"))
        assert result["is_valid"] is False
        assert result["is_finite"] is False
        assert result["status"] == "non_finite"

    def test_scalar_inf_fails(self):
        result = validate_scalar_conductivity(float("inf"))
        assert result["is_valid"] is False
        assert result["is_finite"] is False
        assert result["status"] == "non_finite"


class TestDiagonalConductivityValidation:
    """Diagonal conductivity tensor validation."""

    def test_diagonal_spd_passes(self):
        import numpy as np

        sigma = np.diag([1.0, 2.0, 3.0])
        result = validate_diagonal_conductivity(sigma)
        assert result["is_valid"] is True
        assert result["is_spd"] is True
        assert result["is_diagonal"] is True if "is_diagonal" in result else True
        assert result["status"] == "passed"

    def test_diagonal_with_zero_fails(self):
        import numpy as np

        sigma = np.diag([1.0, 0.0, 3.0])
        result = validate_diagonal_conductivity(sigma)
        assert result["is_valid"] is False
        assert result["is_spd"] is False
        assert result["status"] == "non_positive"

    def test_diagonal_with_negative_fails(self):
        import numpy as np

        sigma = np.diag([1.0, -2.0, 3.0])
        result = validate_diagonal_conductivity(sigma)
        assert result["is_valid"] is False
        assert result["status"] == "non_positive"

    def test_non_diagonal_fails(self):
        import numpy as np

        sigma = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]])
        result = validate_diagonal_conductivity(sigma)
        assert result["is_valid"] is False
        assert result["status"] == "not_diagonal"

    def test_diagonal_with_nan_fails(self):
        import numpy as np

        sigma = np.diag([1.0, float("nan"), 3.0])
        result = validate_diagonal_conductivity(sigma)
        assert result["is_valid"] is False
        assert result["is_finite"] is False


class TestFullSPDConductivityValidation:
    """Full SPD conductivity tensor validation."""

    def test_spd_matrix_passes(self):
        import numpy as np

        sigma = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
        result = validate_full_spd_conductivity(sigma)
        assert result["is_valid"] is True
        assert result["is_spd"] is True
        assert result["is_symmetric"] is True
        assert result["status"] == "passed"
        assert result["min_eigenvalue"] is not None
        assert result["condition_number"] is not None

    def test_non_symmetric_fails(self):
        import numpy as np

        sigma = np.array([[4.0, 1.0, 0.0], [2.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
        result = validate_full_spd_conductivity(sigma)
        assert result["is_valid"] is False
        assert result["is_symmetric"] is False
        assert result["status"] == "not_symmetric"

    def test_negative_eigenvalue_fails(self):
        import numpy as np

        # This matrix has a negative eigenvalue
        sigma = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = validate_full_spd_conductivity(sigma)
        assert result["is_valid"] is False
        assert result["is_spd"] is False
        assert result["status"] == "not_positive_definite"

    def test_with_nan_fails(self):
        import numpy as np

        sigma = np.array([[4.0, 1.0, float("nan")], [1.0, 3.0, 0.5], [float("nan"), 0.5, 2.0]])
        result = validate_full_spd_conductivity(sigma)
        assert result["is_valid"] is False
        assert result["is_finite"] is False


class TestFieldArraysFiniteValidation:
    """Field array finiteness checks."""

    def test_all_finite_passes(self):
        import numpy as np

        phi_e = np.random.randn(10, 16)
        J_e = np.random.randn(10, 3)
        CSD = np.random.randn(10, 16)
        result = validate_field_arrays_finite(phi_e=phi_e, J_e=J_e, CSD=CSD)
        assert result["all_finite"] is True
        assert result["phi_e_finite"] is True
        assert result["J_e_finite"] is True
        assert result["CSD_finite"] is True
        assert result["evidence"] is None

    def test_phi_e_with_nan_fails(self):
        import numpy as np

        phi_e = np.random.randn(10, 16)
        phi_e[5, 8] = float("nan")
        result = validate_field_arrays_finite(phi_e=phi_e)
        assert result["all_finite"] is False
        assert result["phi_e_finite"] is False
        assert result["evidence"] is not None

    def test_csd_with_inf_fails(self):
        import numpy as np

        CSD = np.random.randn(10, 16)
        CSD[3, 4] = float("inf")
        result = validate_field_arrays_finite(CSD=CSD)
        assert result["all_finite"] is False
        assert result["CSD_finite"] is False


class TestFieldAdmissibilityReport:
    """Field admissibility report generation."""

    def test_proxy_report_json_safe(self):
        model, signals = _model_and_signals()
        report = build_field_admissibility_report(
            field_output=signals.field,
            cfg_metadata=dict(model.cfg.metadata or {}),
        )
        # Should be JSON-safe
        json_str = json.dumps(report, allow_nan=False)
        assert isinstance(json_str, str)

    def test_report_contains_required_fields(self):
        model, signals = _model_and_signals()
        report = build_field_admissibility_report(
            field_output=signals.field,
            cfg_metadata=dict(model.cfg.metadata or {}),
        )
        required_fields = [
            "field_solver_status",
            "field_claim_level",
            "boundary_condition",
            "gauge",
            "CSD_sign_convention",
            "conductivity_status",
            "physical_amplitude_claim_allowed",
        ]
        for field in required_fields:
            assert field in report, f"Missing required field: {field}"

    def test_proxy_mode_claim_not_allowed(self):
        model, signals = _model_and_signals()
        report = build_field_admissibility_report(
            field_output=signals.field,
            cfg_metadata=dict(model.cfg.metadata or {}),
        )
        assert report["physical_amplitude_claim_allowed"] is False
        assert report["field_claim_level"] == "proxy_readout_only"

    def test_proxy_mode_status(self):
        model, signals = _model_and_signals()
        report = build_field_admissibility_report(
            field_output=signals.field,
            cfg_metadata=dict(model.cfg.metadata or {}),
        )
        assert report["field_solver_status"] == "laminar_proxy_no_pde"
        assert report["conductivity_status"] == "proxy_not_solved"


class TestManifestFieldAdmissibilityIntegration:
    """Field admissibility in manifest integration."""

    def test_manifest_field_admissibility_present(self):
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        assert "backend_metadata" in manifest
        assert "field_admissibility" in manifest["backend_metadata"]

    def test_manifest_field_admissibility_json_safe(self):
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        # Should pass strict JSON serialization
        json.dumps(manifest, allow_nan=False)

    def test_manifest_physical_amplitude_claim_false(self):
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        field_adm = manifest["backend_metadata"]["field_admissibility"]
        assert field_adm["physical_amplitude_claim_allowed"] is False

    def test_manifest_boundary_and_gauge_present(self):
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        field_adm = manifest["backend_metadata"]["field_admissibility"]
        assert field_adm["boundary_condition"] is not None
        assert field_adm["gauge"] is not None


class TestFieldDiagnosticsInSignals:
    """Field admissibility diagnostics in signals."""

    def test_field_diagnostics_contains_finiteness(self):
        model, signals = _model_and_signals()
        assert signals.field is not None
        assert "field_admissibility" in signals.field.diagnostics
        field_adm = signals.field.diagnostics["field_admissibility"]
        assert "field_arrays_finite" in field_adm
        assert field_adm["field_arrays_finite"]["phi_e_finite"] is True
        assert field_adm["field_arrays_finite"]["csd_finite"] is True

    def test_field_diagnostics_source_conservation_status(self):
        model, signals = _model_and_signals()
        assert signals.field is not None
        field_adm = signals.field.diagnostics["field_admissibility"]
        assert field_adm["source_conservation_status"] == "proxy_not_solved"

    def test_kernel_normalization_valid(self):
        model, signals = _model_and_signals()
        assert signals.field is not None
        field_adm = signals.field.diagnostics["field_admissibility"]
        assert field_adm["kernel_normalization_valid"] is True


class TestManifestTruthGatesPreserved:
    """v0.2.0: Field admissibility does not change truth gates."""

    def test_manifest_truth_gates_frozen(self):
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        assert manifest["truth_mode"] == "truth_safe_unverified"
        assert manifest["claim_level"] == "computational_scaffold"
        assert manifest["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
        assert manifest["field_claim_level"] == "proxy_readout_only"
        assert manifest["physical_amplitude_claim_allowed"] is False

    def test_manifest_empirical_validation_status_not_validated(self):
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        if "v005_claim_labels" in manifest:
            assert manifest["v005_claim_labels"]["empirical_validation_status"] == "not_empirically_validated"
