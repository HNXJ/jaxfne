"""Calibration specification and reporting contracts for v0.2.5.

Tests that:
1. CalibrationSpec is JSON-safe and serializable
2. All allowed modes are valid
3. Invalid modes raise ValueError
4. Reports keep physical amplitude claims false (v0.2.5 conservative)
5. Empirical metadata is declared in reports
6. Assumptions and warnings are present
7. Integration with probe reports
"""

from __future__ import annotations

import json

import pytest

from jaxfne.validation import CalibrationSpec, make_calibration_report


class TestCalibrationSpecV025:
    """CalibrationSpec tests for v0.2.5."""

    def test_default_calibration_spec_creation(self):
        """Test creating a default CalibrationSpec."""
        spec = CalibrationSpec(name="test", target="source")
        assert spec.name == "test"
        assert spec.target == "source"
        assert spec.mode == "uncalibrated_native"
        assert spec.scale is None
        assert spec.units is None
        assert spec.reference is None

    def test_calibration_spec_json_safe(self):
        """Test that CalibrationSpec serializes to JSON-safe dict."""
        spec = CalibrationSpec(
            name="test_spec",
            target="field",
            mode="toy_scale",
            scale=1.0,
            units="proxy_V",
            reference="toy_leadfield",
        )
        spec_dict = spec.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(spec_dict)
        assert json_str is not None
        assert "test_spec" in json_str
        assert "field" in json_str

    def test_all_allowed_modes(self):
        """Test that all allowed modes can be created without error."""
        allowed_modes = [
            "uncalibrated_native",
            "toy_scale",
            "relative_normalized",
            "empirical_gain_candidate",
            "physical_units_candidate",
            "calibrated_empirical",
        ]

        for mode in allowed_modes:
            spec = CalibrationSpec(name="test", target="probe", mode=mode)
            assert spec.mode == mode

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            CalibrationSpec(name="test", target="source", mode="invalid_mode_xyz")

    def test_empty_target_raises_error(self):
        """Test that empty target raises ValueError."""
        with pytest.raises(ValueError, match="target must be"):
            CalibrationSpec(name="test", target="")

    def test_calibration_spec_with_all_fields(self):
        """Test CalibrationSpec with all fields populated."""
        spec = CalibrationSpec(
            name="complete_spec",
            target="readout",
            mode="empirical_gain_candidate",
            scale=2.5,
            units="mV",
            reference="Mendoza-Halliday et al. 2024",
            description="EEG proxy with empirical gain candidate",
        )

        spec_dict = spec.to_dict()
        assert spec_dict["name"] == "complete_spec"
        assert spec_dict["scale"] == 2.5
        assert spec_dict["units"] == "mV"
        assert "Mendoza-Halliday" in spec_dict["reference"]


class TestCalibrationReportV025:
    """CalibrationReport function tests for v0.2.5."""

    def test_default_uncalibrated_report(self):
        """Test report for default uncalibrated spec."""
        spec = CalibrationSpec(name="default", target="source")
        report = make_calibration_report(spec)

        assert report["calibration_name"] == "default"
        assert report["target"] == "source"
        assert report["mode"] == "uncalibrated_native"
        assert report["physical_amplitude_claim_allowed"] is False
        assert report["calibration_claim_level"] == "computational_proxy_with_declared_metadata"

    def test_toy_scale_report_keeps_false(self):
        """Test that toy_scale mode keeps physical amplitude false."""
        spec = CalibrationSpec(
            name="toy", target="field", mode="toy_scale", scale=1.0, units="proxy_units"
        )
        report = make_calibration_report(spec)

        assert report["mode"] == "toy_scale"
        assert report["physical_amplitude_claim_allowed"] is False

    def test_relative_normalized_report_keeps_false(self):
        """Test that relative_normalized keeps physical amplitude false."""
        spec = CalibrationSpec(
            name="rel_norm",
            target="probe",
            mode="relative_normalized",
            units="normalized_proxy_units",
        )
        report = make_calibration_report(spec)

        assert report["mode"] == "relative_normalized"
        assert report["physical_amplitude_claim_allowed"] is False

    def test_empirical_gain_candidate_report(self):
        """Test report for empirical_gain_candidate mode."""
        spec = CalibrationSpec(
            name="egc",
            target="readout",
            mode="empirical_gain_candidate",
            scale=1.5,
            units="mV_proxy",
            reference="pilot_dataset",
        )
        report = make_calibration_report(spec)

        assert report["mode"] == "empirical_gain_candidate"
        assert report["physical_amplitude_claim_allowed"] is False
        assert report["empirical_scale_declared"] is True
        assert "candidate" in str(report["warnings"]).lower()

    def test_physical_units_candidate_report(self):
        """Test report for physical_units_candidate mode."""
        spec = CalibrationSpec(
            name="puc",
            target="field",
            mode="physical_units_candidate",
            units="mV",
            scale=10.0,
            reference="conductivity_estimate",
        )
        report = make_calibration_report(spec)

        assert report["mode"] == "physical_units_candidate"
        assert report["physical_amplitude_claim_allowed"] is False
        assert report["status"] == "metadata_declared"

    def test_calibrated_empirical_complete(self):
        """Test calibrated_empirical with complete fields."""
        spec = CalibrationSpec(
            name="calib",
            target="readout",
            mode="calibrated_empirical",
            scale=5.0,
            units="mV",
            reference="Lichtenfeld et al. 2024",
        )
        report = make_calibration_report(spec)

        assert report["mode"] == "calibrated_empirical"
        assert report["physical_amplitude_claim_allowed"] is False
        assert report["empirical_reference_declared"] is True
        assert report["empirical_units_declared"] is True
        assert report["empirical_scale_declared"] is True

    def test_calibrated_empirical_incomplete_warnings(self):
        """Test that incomplete calibrated_empirical generates warnings."""
        # Missing units
        spec = CalibrationSpec(
            name="incomplete",
            target="readout",
            mode="calibrated_empirical",
            scale=1.0,
            reference="source",
        )
        report = make_calibration_report(spec)

        assert len(report["warnings"]) > 0
        assert any("units" in w.lower() for w in report["warnings"])
        assert report["physical_amplitude_claim_allowed"] is False

    def test_report_always_has_assumptions(self):
        """Test that all reports include assumptions."""
        spec = CalibrationSpec(name="test", target="source")
        report = make_calibration_report(spec)

        assert "assumptions" in report
        assert len(report["assumptions"]) > 0
        assert any("computational" in a.lower() and "proxy" in a.lower() for a in report["assumptions"])

    def test_report_with_readout_kind(self):
        """Test report with readout_kind specified."""
        spec = CalibrationSpec(name="probe_report", target="probe")
        report = make_calibration_report(spec, readout_kind="lfp_proxy")

        assert report["readout_kind"] == "lfp_proxy"

    def test_report_json_safe(self):
        """Test that reports are JSON-safe."""
        spec = CalibrationSpec(
            name="json_test",
            target="field",
            mode="empirical_gain_candidate",
            scale=2.0,
            units="mV",
            reference="test_ref",
        )
        report = make_calibration_report(spec)

        # Should be JSON-serializable
        json_str = json.dumps(report)
        assert json_str is not None
        assert "json_test" in json_str

    def test_dict_input_to_report(self):
        """Test that make_calibration_report accepts dict input."""
        spec_dict = {
            "name": "from_dict",
            "target": "source",
            "mode": "toy_scale",
            "scale": 1.0,
        }
        report = make_calibration_report(spec_dict)

        assert report["calibration_name"] == "from_dict"
        assert report["target"] == "source"

    def test_v025_keeps_false_for_all_modes(self):
        """Test that v0.2.5 keeps physical_amplitude_claim_allowed false for all modes."""
        for mode in CalibrationSpec.ALLOWED_MODES:
            spec = CalibrationSpec(
                name=f"mode_{mode}",
                target="readout",
                mode=mode,
                scale=1.0,
                units="test_units",
                reference="test_ref",
            )
            report = make_calibration_report(spec)

            assert (
                report["physical_amplitude_claim_allowed"] is False
            ), f"Mode {mode} should keep physical amplitude false in v0.2.5"
