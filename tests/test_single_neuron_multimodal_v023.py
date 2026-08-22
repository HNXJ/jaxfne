"""
Test suite for v0.2.3 single-neuron multimodal proxy tutorial.

Tests verify:
1. Tutorial example module imports and runs.
2. Output bundle exists and is JSON-strict.
3. All eight readouts present with correct metadata.
4. Claim-status metadata frozen across all operators.
5. EMM-proxy scopes biological metabolism.
6. CSD-proxy includes sign convention.
7. EEG/MEG-proxy metadata present.
8. Generated outputs not committed.
9. Version matches pyproject.toml.

The example is executed inside a temporary working directory so the
repository tree is never polluted; every output assertion consumes that
temporary bundle directly.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import pytest


EXAMPLE_PATH = pathlib.Path("examples/03_single_neuron_multimodal_probe.py")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_FILES = [
    "manifest.json",
    "probe_report.json",
    "metrics.json",
    "validation_report.json",
    "asset_hashes.json",
]


def test_single_neuron_example_imports():
    """Test that example module can be imported."""
    assert EXAMPLE_PATH.exists(), "Example file not found"
    code = EXAMPLE_PATH.read_text(encoding="utf-8")
    compile(code, str(EXAMPLE_PATH), "exec")


@pytest.fixture(scope="module")
def v023_outputs(tmp_path_factory):
    """Run the example in an isolated temp cwd; return its output bundle dir."""
    tmp = tmp_path_factory.mktemp("v023_single_neuron_multimodal")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH.resolve())],
        cwd=tmp,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, f"Example failed:\n{result.stderr}"
    assert "Single-neuron Multimodal Proxy Tutorial" in result.stdout
    return tmp / "outputs" / "v023_single_neuron_multimodal"


def test_output_bundle_exists(v023_outputs):
    """Test that all output files are generated."""
    for filename in EXPECTED_FILES:
        fpath = v023_outputs / filename
        assert fpath.exists(), f"Missing output file: {filename}"


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_output_json_strict(v023_outputs):
    """Test that all JSON files are strict (no NaN/Inf)."""
    for filename in EXPECTED_FILES:
        data = _load_json(v023_outputs / filename)
        json_str = json.dumps(data, allow_nan=False)
        assert isinstance(json_str, str), f"{filename} failed JSON strictness check"


def test_all_eight_readouts_present(v023_outputs):
    """Test that probe_report contains all eight operators."""
    probe_report = _load_json(v023_outputs / "probe_report.json")

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


def test_physical_amplitude_claim_false(v023_outputs):
    """Test that proxy operators have physical_amplitude_calibrated=false."""
    probe_report = _load_json(v023_outputs / "probe_report.json")

    proxy_operators = [
        "lfp_proxy",
        "csd_proxy",
        "eeg_proxy",
        "meg_proxy",
        "emm_proxy",
    ]

    for op_name in proxy_operators:
        if op_name in probe_report:
            claim_allowed = probe_report[op_name].get("physical_amplitude_calibrated")
            assert claim_allowed is False, (
                f"{op_name}: physical_amplitude_calibrated should be False, "
                f"got {claim_allowed}"
            )


def test_csd_sign_convention_present(v023_outputs):
    """Test that CSD-proxy report includes sign convention."""
    probe_report = _load_json(v023_outputs / "probe_report.json")

    if "csd_proxy" in probe_report:
        csd_report = probe_report["csd_proxy"]
        assert "CSD_sign_convention" in csd_report, "CSD-proxy missing sign convention"
        assert csd_report["CSD_sign_convention"] is not None


def test_eeg_meg_proxy_metadata(v023_outputs):
    """Test that EEG/MEG-proxy include leadfield and sensor geometry metadata."""
    probe_report = _load_json(v023_outputs / "probe_report.json")

    for op_name in ["eeg_proxy", "meg_proxy"]:
        if op_name in probe_report:
            report = probe_report[op_name]
            assert "leadfield_status" in report, f"{op_name} missing leadfield_status"
            assert "sensor_geometry_status" in report, (
                f"{op_name} missing sensor_geometry_status"
            )


def test_emm_proxy_not_biological_metabolism(v023_outputs):
    """Test that EMM-proxy scopes biological metabolism."""
    emm_report = _load_json(v023_outputs / "probe_report.json")["emm_proxy"]

    method = emm_report.get("method", "")
    assert "proxy" in method.lower() or "cost" in method.lower(), \
        f"EMM-proxy method should describe it as proxy/cost, got: {method}"

    assumptions = emm_report.get("assumptions", [])
    assert any("not_biological_metabolism" in str(a).lower() for a in assumptions), \
        "EMM-proxy assumptions should explicitly state it's not biological metabolism"

    calib_status = emm_report.get("calibration_status") or emm_report.get(
        "biophysical_calibration_status", ""
    )
    assert "uncalibrated" in calib_status.lower() or "proxy" in calib_status.lower(), \
        f"EMM-proxy should be uncalibrated/proxy, got: {calib_status}"


def test_validation_metadata_frozen(v023_outputs):
    """Test that validation_report shows frozen claim-status metadata."""
    validation = _load_json(v023_outputs / "validation_report.json")

    expected_fields = {
        "field_claim_level": "proxy_readout",
        "field_solver_status": "linear_solver",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_calibrated": False,
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
        encoding="utf-8",
    )
    tracked_outputs = [line for line in result.stdout.strip().split("\n") if line]
    assert len(tracked_outputs) == 0, (
        f"outputs/ should not be tracked in git, found: {tracked_outputs}"
    )


def test_version_matches_pyproject():
    """Test that package version matches pyproject.toml."""
    import jaxfne

    content = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
    project_section = re.search(r"\[project\](.*?)(?:\[|\Z)", content, re.DOTALL)
    pyproject_version = re.search(
        r'version\s*=\s*"([^"]+)"', project_section.group(1)
    ).group(1)
    assert jaxfne.__version__ == pyproject_version, \
        f"Version should be {pyproject_version}, got {jaxfne.__version__}"


def test_operator_status_simulated_proxy(v023_outputs):
    """Test that all operators report correct operator_status."""
    probe_report = _load_json(v023_outputs / "probe_report.json")

    for op_name, report in probe_report.items():
        operator_status = report.get("operator_status")
        assert operator_status in ["simulated_proxy", "physical_forward_model", "calibrated_empirical"], \
            f"{op_name}: invalid operator_status={operator_status}"
        assert operator_status == "simulated_proxy", \
            f"{op_name}: should be simulated_proxy, got {operator_status}"


def test_probe_report_structure(v023_outputs):
    """Test that probe_report has correct structure for each operator."""
    probe_report = _load_json(v023_outputs / "probe_report.json")

    required_fields = [
        "name",
        "kind",
        "operator_status",
        "data_shape",
        "units_or_status",
        "method",
        "assumptions",
        "physical_amplitude_calibrated",
    ]

    for op_name, report in probe_report.items():
        for field in required_fields:
            assert field in report, f"{op_name}: missing required field {field}"
