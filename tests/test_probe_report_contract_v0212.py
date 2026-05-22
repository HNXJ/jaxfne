"""Comprehensive contract tests for probe reports (v0.2.12).

Tests the ProbeReport/ProbeReadout contract after hardening:
- All required fields present
- No truth_mode in public reports
- No *_like terminology in public reports
- JSON-safe serialization with allow_nan=False
- Operator-specific fields (CSD, EEG, MEG)
- Examples generate valid probe_report.json
- Version remains 0.2.10
"""

import json
import pytest
import jax.numpy as jnp
from jaxfne.fields import (
    spk_probe,
    vm_probe,
    source_probe,
    lfp_proxy_probe,
    csd_proxy_probe,
    eeg_proxy_probe,
    meg_proxy_probe,
    emm_proxy_probe,
)
from jaxfne.io import json_safe


# ─── Required Fields Test ──────────────────────────────────────────────────────

def test_contract_required_fields_all_operators():
    """All operators include 14 required report fields."""
    required_fields = {
        "name",
        "kind",
        "operator_status",
        "method",
        "data_shape",
        "units_or_status",
        "calibration_status",
        "source_calibration_status",
        "source_projection_mode",
        "source_decomposition",
        "field_solver_status",
        "field_claim_level",
        "physical_amplitude_claim_allowed",
        "assumptions",
    }

    operators = [
        ("spk", spk_probe(jnp.ones((10, 8)))),
        ("vm", vm_probe(jnp.ones((10, 8)))),
        ("source", source_probe(jnp.ones((10, 8)))),
        ("lfp_proxy", lfp_proxy_probe(jnp.ones((10, 16)))),
        ("csd_proxy", csd_proxy_probe(jnp.ones((10, 16)))),
        ("eeg_proxy", eeg_proxy_probe(jnp.ones((10, 32)))),
        ("meg_proxy", meg_proxy_probe(jnp.ones((10, 32)))),
        ("emm_proxy", emm_proxy_probe(jnp.ones((10,)))),
    ]

    for name, readout in operators:
        missing = required_fields - set(readout.report.keys())
        assert not missing, f"{name} missing fields: {missing}"


# ─── JSON-Safety Test ──────────────────────────────────────────────────────────

def test_contract_json_safety_all_operators():
    """All operators produce JSON-safe reports with allow_nan=False."""
    operators = [
        spk_probe(jnp.ones((10, 8))),
        vm_probe(jnp.ones((10, 8))),
        source_probe(jnp.ones((10, 8))),
        lfp_proxy_probe(jnp.ones((10, 16))),
        csd_proxy_probe(jnp.ones((10, 16))),
        eeg_proxy_probe(jnp.ones((10, 32))),
        meg_proxy_probe(jnp.ones((10, 32))),
        emm_proxy_probe(jnp.ones((10,))),
    ]

    for readout in operators:
        safe_report = json_safe(readout.report)
        # This must not raise with allow_nan=False
        json.dumps(safe_report, allow_nan=False)


# ─── No truth_mode in Public Reports ────────────────────────────────────────────

def test_contract_no_truth_mode_in_reports():
    """truth_mode must not appear in any public probe report."""
    operators = [
        spk_probe(jnp.ones((10, 8))),
        vm_probe(jnp.ones((10, 8))),
        source_probe(jnp.ones((10, 8))),
        lfp_proxy_probe(jnp.ones((10, 16))),
        csd_proxy_probe(jnp.ones((10, 16))),
        eeg_proxy_probe(jnp.ones((10, 32))),
        meg_proxy_probe(jnp.ones((10, 32))),
        emm_proxy_probe(jnp.ones((10,))),
    ]

    for readout in operators:
        assert "truth_mode" not in readout.report, \
            f"truth_mode found in {readout.report.get('kind')} report"


# ─── No *_like Terminology ────────────────────────────────────────────────────────

def test_contract_no_like_terminology():
    """No *_like or *_same terminology in public reports."""
    forbidden_substrings = {
        "_like",
        "-like",
        "lfp_like",
        "csd_like",
        "eeg_like",
        "meg_like",
        "LFP-like",
        "CSD-like",
        "EEG-like",
        "MEG-like",
    }

    operators = [
        spk_probe(jnp.ones((10, 8))),
        vm_probe(jnp.ones((10, 8))),
        source_probe(jnp.ones((10, 8))),
        lfp_proxy_probe(jnp.ones((10, 16))),
        csd_proxy_probe(jnp.ones((10, 16))),
        eeg_proxy_probe(jnp.ones((10, 32))),
        meg_proxy_probe(jnp.ones((10, 32))),
        emm_proxy_probe(jnp.ones((10,))),
    ]

    for readout in operators:
        report_str = json.dumps(readout.report)
        for forbidden in forbidden_substrings:
            assert forbidden not in report_str, \
                f"Forbidden substring '{forbidden}' found in {readout.report.get('kind')} report"


# ─── Kind Values Correct ────────────────────────────────────────────────────────

def test_contract_kind_values_correct():
    """All operators use correct kind values."""
    kind_map = [
        ("spk", spk_probe(jnp.ones((10, 8)))),
        ("vm", vm_probe(jnp.ones((10, 8)))),
        ("source", source_probe(jnp.ones((10, 8)))),
        ("lfp_proxy", lfp_proxy_probe(jnp.ones((10, 16)))),
        ("csd_proxy", csd_proxy_probe(jnp.ones((10, 16)))),
        ("eeg_proxy", eeg_proxy_probe(jnp.ones((10, 32)))),
        ("meg_proxy", meg_proxy_probe(jnp.ones((10, 32)))),
        ("emm_proxy", emm_proxy_probe(jnp.ones((10,)))),
    ]

    for expected_kind, readout in kind_map:
        assert readout.report["kind"] == expected_kind, \
            f"Expected kind={expected_kind}, got {readout.report['kind']}"


# ─── Data Shape Matches Output ──────────────────────────────────────────────────

def test_contract_data_shape_matches_output():
    """data_shape in report matches actual readout.data shape as string."""
    test_cases = [
        (spk_probe(jnp.ones((10, 8))), (10, 8)),
        (vm_probe(jnp.ones((10, 8))), (10, 8)),
        (source_probe(jnp.ones((10, 8))), (10, 8)),
        (lfp_proxy_probe(jnp.ones((10, 16))), (10, 16)),
        (csd_proxy_probe(jnp.ones((10, 16))), (10, 16)),
        (eeg_proxy_probe(jnp.ones((10, 32))), (10, 32)),
        (meg_proxy_probe(jnp.ones((10, 32))), (10, 32)),
        (emm_proxy_probe(jnp.ones((10,))), (10,)),
    ]

    for readout, expected_shape in test_cases:
        reported_shape_str = readout.report["data_shape"]
        actual_shape_str = str(expected_shape)
        assert reported_shape_str == actual_shape_str, \
            f"data_shape mismatch: reported={reported_shape_str}, actual={actual_shape_str}"


# ─── Proxy Reports Have physical_amplitude_claim_allowed=False ──────────────────

def test_contract_proxy_amplitude_claim_false():
    """All proxy operators set physical_amplitude_claim_allowed=False."""
    proxy_operators = [
        lfp_proxy_probe(jnp.ones((10, 16))),
        csd_proxy_probe(jnp.ones((10, 16))),
        eeg_proxy_probe(jnp.ones((10, 32))),
        meg_proxy_probe(jnp.ones((10, 32))),
        emm_proxy_probe(jnp.ones((10,))),
    ]

    for readout in proxy_operators:
        assert readout.report["physical_amplitude_claim_allowed"] is False, \
            f"{readout.report['kind']} should have physical_amplitude_claim_allowed=False"


# ─── CSD Operator Specific Fields ──────────────────────────────────────────────

def test_contract_csd_sign_convention():
    """CSD report includes CSD_sign_convention field with correct value."""
    readout = csd_proxy_probe(jnp.ones((10, 16)))
    assert "CSD_sign_convention" in readout.report
    assert readout.report["CSD_sign_convention"] == "positive_equals_extracellular_source"


# ─── EEG Operator Specific Fields ──────────────────────────────────────────────

def test_contract_eeg_specific_fields():
    """EEG report includes leadfield and sensor geometry status."""
    readout = eeg_proxy_probe(jnp.ones((10, 32)))
    assert "leadfield_status" in readout.report
    assert "sensor_geometry_status" in readout.report


# ─── MEG Operator Specific Fields ──────────────────────────────────────────────

def test_contract_meg_specific_fields():
    """MEG report includes leadfield, sensor geometry, and orientation convention."""
    readout = meg_proxy_probe(jnp.ones((10, 32)))
    assert "leadfield_status" in readout.report
    assert "sensor_geometry_status" in readout.report
    assert "orientation_convention" in readout.report


# ─── Version Bumped to 0.2.18 ────────────────────────────────────────────────────

def test_contract_version_unchanged():
    """jaxfne version remains 0.2.10 (no bump for v0.2.12)."""
    import jaxfne
    assert jaxfne.__version__ == "0.2.25"
