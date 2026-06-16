"""Regression and structural validation test for Etude No. 3."""

import json
from pathlib import Path
import pytest
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "tutorials" / "etudes" / "jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb"
OUTPUT_DIR = REPO_ROOT / "tutorials" / "etudes" / "outputs" / "v1_spectrolaminar_1k"

REQUIRED_TOKENS = [
    "import jaxfne as jtfne",
    "v1_column_1k",
    "N_NEURONS = 1000",
    "lfp_proxy",
    "csd_proxy",
    "eeg_proxy",
    "meg_proxy",
    "spectrolaminar_suite_proxy.png",
]

FORBIDDEN_PHRASES = [
    "real eeg",
    "real meg",
    "calibrated amplitude",
    "physical csd",
    "mechanism proof",
    "biological validation",
]

def test_notebook_file_exists():
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"

def test_notebook_structure_and_tokens():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert "cells" in nb, "Notebook is missing cells key"
    
    # Get all code and markdown source text
    all_text = ""
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        all_text += (src if isinstance(src, str) else "".join(src)).lower() + "\n"

    # Verify required tokens
    for token in REQUIRED_TOKENS:
        assert token.lower() in all_text, f"Missing required token: {token}"

    # Verify no overclaiming phrases
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in all_text, f"Forbidden overclaiming phrase found: '{phrase}'"

def test_notebook_has_correct_cell_types_count():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # We have exactly 6 code cells: Setup, Config, Construct, Simulate, Visualize, Artifacts
    assert len(code_cells) == 6, f"Expected 6 code cells, got {len(code_cells)}"

def test_json_artifacts_exist_and_conform_to_contract():
    manifest_path = OUTPUT_DIR / "manifest.json"
    validation_path = OUTPUT_DIR / "validation.json"
    metrics_path = OUTPUT_DIR / "metrics.json"
    hashes_path = OUTPUT_DIR / "asset_hashes.json"

    assert manifest_path.exists(), f"Expected manifest file at {manifest_path}"
    assert validation_path.exists(), f"Expected validation file at {validation_path}"
    assert metrics_path.exists(), f"Expected metrics file at {metrics_path}"
    assert hashes_path.exists(), f"Expected asset hashes file at {hashes_path}"

    # Verify manifest truth gates
    manifest = json.loads(manifest_path.read_text())
    assert manifest["claim_level"] == "computational_scaffold"
    assert manifest["field_solver_status"] == "linear_solver"
    assert manifest["field_claim_level"] == "proxy_readout"
    assert manifest["physical_amplitude_calibrated"] is False
    assert manifest["n_neurons"] == 1000

    # Verify validation report gates
    validation = json.loads(validation_path.read_text())
    assert validation["notebook_execution"] == "pass"
    assert validation["finite_outputs"] is True
    assert validation["strict_json_pass"] is True
    assert validation["physical_amplitude_calibrated"] is False

    # Verify metrics are finite
    metrics = json.loads(metrics_path.read_text())
    assert jnp.isfinite(metrics["mean_firing_rate_hz"])
    assert jnp.isfinite(metrics["voltage_mean_mv"])
    assert jnp.isfinite(metrics["voltage_std_mv"])
    
    # Assert JSON-safeness with strict deserialization (no NaN/Inf allowed)
    for path in [manifest_path, validation_path, metrics_path, hashes_path]:
        data = json.loads(path.read_text())
        json.dumps(data, allow_nan=False)
