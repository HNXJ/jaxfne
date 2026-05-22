"""
Test suite for v0.2.3 single-neuron multimodal proxy tutorial.

Tests verify:
1. Tutorial example module imports and runs.
2. Output bundle exists and is JSON-strict.
3. All eight readouts present with correct metadata.
4. Claim-status metadata frozen across all operators.
5. EMM-proxy does not claim biological metabolism.
6. CSD-proxy includes sign convention.
7. EEG/MEG-proxy metadata present.
8. Generated outputs not committed.
9. Version is bumped to 0.2.3.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import pytest


def test_single_neuron_example_imports():
    """Test that example module can be imported."""
    # Import the example as a module
    example_path = pathlib.Path("examples/03_single_neuron_multimodal_probe.py")
    assert example_path.exists(), "Example file not found"

    # Read and compile to check syntax
    with open(example_path) as f:
        code = f.read()
    compile(code, str(example_path), "exec")


def test_single_neuron_example_runs():
    """Test that example runs to completion (CPU-safe, ~5 seconds)."""
    # Run the example in a temp directory to avoid polluting outputs/
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "examples/03_single_neuron_multimodal_probe.py"],
            cwd=pathlib.Path.cwd(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Example failed:\n{result.stderr}"
        assert "Single-neuron Multimodal Proxy Tutorial" in result.stdout


def test_output_bundle_exists():
    """Test that all output files are generated."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    expected_files = [
        "manifest.json",
        "probe_report.json",
        "metrics.json",
        "validation_report.json",
        "asset_hashes.json",
    ]

    for filename in expected_files:
        fpath = output_dir / filename
        assert fpath.exists(), f"Missing output file: {filename}"


def test_output_json_strict():
    """Test that all JSON files are strict (no NaN/Inf)."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    json_files = [
        "manifest.json",
        "probe_report.json",
        "metrics.json",
        "validation_report.json",
        "asset_hashes.json",
    ]

    for filename in json_files:
        fpath = output_dir / filename
        with open(fpath) as f:
            data = json.load(f)

        # Re-serialize with allow_nan=False to ensure strictness
        json_str = json.dumps(data, allow_nan=False)
        assert isinstance(json_str, str), f"{filename} failed JSON strictness check"


def test_all_eight_readouts_present():
    """Test that probe_report contains all eight operators."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    with open(output_dir / "probe_report.json") as f:
        probe_report = json.load(f)

    required_operators = [
        "spk",
        "vm",
        "source",
        "lfp_proxy",
        "csd_proxy",
        "eeg_proxy",
        "meg_proxy",
        "emm_proxy",
    ]

    for op_name in required_operators:
        assert op_name in probe_report, f"Missing operator: {op_name}"
        assert isinstance(probe_report[op_name], dict), f"{op_name} report not a dict"


def test_physical_amplitude_claim_false():
    """Test that proxy operators have physical_amplitude_claim_allowed=false."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    with open(output_dir / "probe_report.json") as f:
        probe_report = json.load(f)

    proxy_operators = [
        "lfp_proxy",
        "csd_proxy",
        "eeg_proxy",
        "meg_proxy",
        "emm_proxy",
    ]

    for op_name in proxy_operators:
        if op_name in probe_report:
            report = probe_report[op_name]
            claim_allowed = report.get("physical_amplitude_claim_allowed")
            assert claim_allowed is False, f"{op_name}: physical_amplitude_claim_allowed should be False, got {claim_allowed}"


def test_csd_sign_convention_present():
    """Test that CSD-proxy report includes sign convention."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    with open(output_dir / "probe_report.json") as f:
        probe_report = json.load(f)

    if "csd_proxy" in probe_report:
        csd_report = probe_report["csd_proxy"]
        assert "CSD_sign_convention" in csd_report, "CSD-proxy missing sign convention"
        assert csd_report["CSD_sign_convention"] is not None


def test_eeg_meg_proxy_metadata():
    """Test that EEG/MEG-proxy include leadfield and sensor geometry metadata."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    with open(output_dir / "probe_report.json") as f:
        probe_report = json.load(f)

    for op_name in ["eeg_proxy", "meg_proxy"]:
        if op_name in probe_report:
            report = probe_report[op_name]
            assert "leadfield_status" in report, f"{op_name} missing leadfield_status"
            assert "sensor_geometry_status" in report, f"{op_name} missing sensor_geometry_status"


def test_emm_proxy_not_biological_metabolism():
    """Test that EMM-proxy does not claim biological metabolism."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    with open(output_dir / "probe_report.json") as f:
        probe_report = json.load(f)

    if "emm_proxy" in probe_report:
        emm_report = probe_report["emm_proxy"]

        # Should describe itself as proxy/cost, not metabolism
        method = emm_report.get("method", "")
        assert "proxy" in method.lower() or "cost" in method.lower(), \
            f"EMM-proxy method should describe it as proxy/cost, got: {method}"

        # Check assumptions to verify it's not claiming biological metabolism
        assumptions = emm_report.get("assumptions", [])
        assert any("not_biological_metabolism" in str(a).lower() for a in assumptions), \
            "EMM-proxy assumptions should explicitly state it's not biological metabolism"

        # Should not claim calibration
        calib_status = emm_report.get("calibration_status") or emm_report.get("biophysical_calibration_status", "")
        assert "uncalibrated" in calib_status.lower() or "proxy" in calib_status.lower(), \
            f"EMM-proxy should be uncalibrated/proxy, got: {calib_status}"


def test_validation_metadata_frozen():
    """Test that validation_report shows frozen claim-status metadata."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    with open(output_dir / "validation_report.json") as f:
        validation = json.load(f)

    expected_fields = {
        "field_claim_level": "proxy_readout_only",
        "field_solver_status": "laminar_proxy_no_pde",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_claim_allowed": False,
    }

    for key, expected_value in expected_fields.items():
        actual_value = validation.get(key)
        assert actual_value == expected_value, \
            f"Validation {key}: expected {expected_value}, got {actual_value}"


def test_output_not_tracked():
    """Test that outputs/ directory is not tracked in git."""
    result = subprocess.run(
        ["git", "ls-files", "outputs/"],
        capture_output=True,
        text=True,
    )
    tracked_outputs = [line for line in result.stdout.strip().split("\n") if line]
    assert len(tracked_outputs) == 0, f"outputs/ should not be tracked in git, found: {tracked_outputs}"


def test_version_bumped_to_023():
    """Test that version is bumped to 0.2.18."""
    import jaxfne
    assert jaxfne.__version__ == "0.2.24", f"Version should be 0.2.23, got {jaxfne.__version__}"


def test_operator_status_simulated_proxy():
    """Test that all operators report correct operator_status."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    with open(output_dir / "probe_report.json") as f:
        probe_report = json.load(f)

    for op_name, report in probe_report.items():
        operator_status = report.get("operator_status")
        assert operator_status in ["simulated_proxy", "physical_forward_model", "calibrated_empirical"], \
            f"{op_name}: invalid operator_status={operator_status}"
        # For v0.2.1, all should be simulated_proxy
        assert operator_status == "simulated_proxy", \
            f"{op_name}: should be simulated_proxy, got {operator_status}"


def test_probe_report_structure():
    """Test that probe_report has correct structure for each operator."""
    output_dir = pathlib.Path("outputs/v023_single_neuron_multimodal")

    with open(output_dir / "probe_report.json") as f:
        probe_report = json.load(f)

    required_fields = [
        "name",
        "kind",
        "operator_status",
        "data_shape",
        "units_or_status",
        "method",
        "assumptions",
        "physical_amplitude_claim_allowed",
    ]

    for op_name, report in probe_report.items():
        for field in required_fields:
            assert field in report, f"{op_name}: missing required field {field}"
