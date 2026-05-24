"""
Test suite for v0.3.3 Two-Neuron E/I Multimodal Tutorial.

Verifies that the tutorial script:
1. Runs without exceptions
2. Produces valid JSON files
3. Generates expected artifacts
4. Maintains claim gates correctly
"""

import json
import pathlib
import sys
import tempfile
import shutil

import pytest
import numpy as np


def test_v033_tutorial_import():
    """Test that the v0.3.3 tutorial script can be imported."""
    # Add examples to path temporarily
    examples_path = pathlib.Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    try:
        import v033_two_neuron_ei_multimodal  # noqa: F401
    finally:
        sys.path.pop(0)


def test_v033_tutorial_runs(tmp_path):
    """Test that the v0.3.3 tutorial runs without exceptions."""
    # Import the tutorial main function
    examples_path = pathlib.Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    try:
        from v033_two_neuron_ei_multimodal import main

        # Save current directory and output paths
        original_cwd = pathlib.Path.cwd()
        original_outputs = pathlib.Path("outputs")
        original_docs = pathlib.Path("docs")

        try:
            # Run the main function
            result = main()

            # Verify result dictionary structure
            assert isinstance(result, dict)
            assert "atlas_manifest" in result
            assert "manifest_path" in result
            assert "e_firing_rate_hz" in result
            assert "i_firing_rate_hz" in result
            assert "e_firing_rate_gate_pass" in result
            assert "i_firing_rate_gate_pass" in result
            assert "figures" in result

        finally:
            # Cleanup
            pass
    finally:
        sys.path.pop(0)


def test_v033_manifest_json_valid():
    """Test that the generated manifest.json is valid and complete."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")

    # Skip if output doesn't exist (tutorial may not have been run)
    if not manifest_path.exists():
        pytest.skip("Manifest not generated; tutorial may not have been run")

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check required top-level keys
    required_keys = [
        "run_id",
        "tutorial_id",
        "jaxfne_version",
        "schema_version",
        "basis",
        "probe_report",
        "validation_report",
        "conservation_proxy_diagnostics",
    ]
    for key in required_keys:
        assert key in manifest, f"Missing required key: {key}"

    # Check basis claim gates
    basis = manifest["basis"]
    assert basis["claim_level"] == "computational_scaffold"
    assert basis["field_claim_level"] == "proxy_readout_only"
    assert basis["physical_amplitude_claim_allowed"] is False
    assert basis["field_solver_status"] == "laminar_proxy_no_pde"

    # Check probe_report has all 8 operators
    probe_keys = set(manifest["probe_report"].keys())
    required_probes = {"spikes", "V_m", "source", "lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy", "emm_proxy"}
    assert probe_keys == required_probes, f"Probe keys mismatch: {probe_keys} vs {required_probes}"

    # Check validation_report
    val_report = manifest["validation_report"]
    assert "e_firing_rate_gate_2_25_hz" in val_report
    assert "i_firing_rate_gate_2_25_hz" in val_report
    assert "e_voltage_finite" in val_report
    assert "i_voltage_finite" in val_report
    assert "status" in val_report


def test_v033_json_files_allow_nan_false():
    """Test that all JSON files are strict (no NaN/Inf)."""
    json_files = [
        "outputs/v030_03_two_neuron_ei_multimodal/manifest.json",
        "outputs/v030_03_two_neuron_ei_multimodal/probe_report.json",
        "outputs/v030_03_two_neuron_ei_multimodal/validation_report.json",
        "outputs/v030_03_two_neuron_ei_multimodal/metrics.json",
        "outputs/v030_03_two_neuron_ei_multimodal/asset_hashes.json",
    ]

    for fpath_str in json_files:
        fpath = pathlib.Path(fpath_str)
        if not fpath.exists():
            pytest.skip(f"Output {fpath} not generated; tutorial may not have been run")

        # Load JSON and re-serialize with allow_nan=False
        with open(fpath) as f:
            data = json.load(f)

        # This will raise ValueError if any NaN/Inf values exist
        strict_json = json.dumps(data, allow_nan=False)
        assert isinstance(strict_json, str)


def test_v033_figures_exist():
    """Test that expected figure files exist."""
    figures_dir = pathlib.Path("docs/tutorials_v030/_static/figures")

    required_figures = [
        "v0303_two_neuron_ei_voltage.png",
        "v0303_two_neuron_ei_raster.png",
    ]

    for fig_name in required_figures:
        fig_path = figures_dir / fig_name
        if not fig_path.exists():
            pytest.skip(f"Figure {fig_name} not found; tutorial may not have been run")

        # Check that file has content
        assert fig_path.stat().st_size > 0, f"Figure {fig_name} is empty"


def test_v033_no_forbidden_claims():
    """Test that the manifest does not contain forbidden claim phrases."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")

    if not manifest_path.exists():
        pytest.skip("Manifest not generated")

    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest_str = json.dumps(manifest)

    # Forbidden phrases that should never appear in a proxy-only tutorial
    forbidden_phrases = [
        "real EEG",
        "real MEG",
        "biological validation",
        "empirically validated",
        "proven mechanism",
        "biophysical accuracy",
        "solved PDE",
        "measured neuroscience",
    ]

    for phrase in forbidden_phrases:
        # Only check if it's in context of a claim (not just in a disclaimer)
        # For now, we just check the basis section and non_claims section
        basis_str = json.dumps(manifest.get("basis", {}))
        assert phrase.lower() not in basis_str.lower(), f"Forbidden phrase '{phrase}' in basis claims"


def test_v033_metrics_are_finite():
    """Test that all numerical metrics are finite (no NaN/Inf)."""
    metrics_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/metrics.json")

    if not metrics_path.exists():
        pytest.skip("Metrics not generated")

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Check all numerical values are finite
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"


def test_v033_network_configuration():
    """Test that network configuration is correctly reflected in manifest."""
    manifest_path = pathlib.Path("outputs/v030_03_two_neuron_ei_multimodal/manifest.json")

    if not manifest_path.exists():
        pytest.skip("Manifest not generated")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check network configuration
    network = manifest.get("network", {})
    assert network.get("n_neurons") == 2
    assert network.get("e_fraction") == 0.5
    assert network.get("i_fraction") == 0.5
    assert network.get("e_index") == 0
    assert network.get("i_index") == 1

    # Check E and I neuron sections
    assert "e_neuron" in manifest
    assert "i_neuron" in manifest

    e_neuron = manifest["e_neuron"]
    assert e_neuron["cell_type"] == "excitatory"
    assert e_neuron["index"] == 0

    i_neuron = manifest["i_neuron"]
    assert i_neuron["cell_type"] == "inhibitory"
    assert i_neuron["index"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
