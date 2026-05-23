"""Tests for v0.2.19 tutorial output contract and figure generation.

Validates that all 4 core tutorials generate:
- manifest.json, probe_report.json, metrics.json, validation_report.json, asset_hashes.json
- figures/ directory with at least one static PNG image
- figure metadata in asset_hashes
- all data is finite/nonzero/non-flat as expected
"""

import json
import pathlib
import pytest


@pytest.mark.skip(reason="v0.2.19 legacy output artifact contract tests require large tutorials to be run locally. Fast CI excludes large examples. For manual validation, run: python scripts/run_all_tutorials.py && python scripts/validate_tutorial_outputs.py. Validation is now delegated to test_tutorial_figure_manifest_v028.py which validates the canonical docs/_static/tutorial_figures/figure_manifest.json artifact.")
class TestTutorialFigureContract:
    """Test suite for tutorial output contract with figures."""

    TUTORIAL_OUTPUTS = [
        ("v023_single_neuron_multimodal", "outputs/v023_single_neuron_multimodal"),
        ("v029_two_neuron_ei_multimodal", "outputs/v029_two_neuron_ei_multimodal"),
        ("v020_spectrolaminar_public_path", "outputs/v020_spectrolaminar_public_path"),
        ("v0210_network_100_ei_multimodal", "outputs/v0210_network_100_ei_multimodal"),
    ]

    @pytest.mark.parametrize("name,path", TUTORIAL_OUTPUTS)
    def test_tutorial_outputs_exist(self, name, path):
        """Test that all tutorial output directories exist."""
        outdir = pathlib.Path(path)
        assert outdir.exists(), f"Output directory {outdir} does not exist for {name}"

    @pytest.mark.parametrize("name,path", TUTORIAL_OUTPUTS)
    def test_json_contract_files(self, name, path):
        """Test that required JSON contract files exist."""
        outdir = pathlib.Path(path)
        required_files = [
            "manifest.json",
            "probe_report.json",
            "metrics.json",
            "validation_report.json",
            "asset_hashes.json",
        ]
        for fname in required_files:
            fpath = outdir / fname
            assert fpath.exists(), f"Missing {fname} in {name}"
            assert fpath.stat().st_size > 0, f"{fname} is empty in {name}"

    @pytest.mark.parametrize("name,path", TUTORIAL_OUTPUTS)
    def test_figures_directory_exists(self, name, path):
        """Test that figures/ directory exists."""
        outdir = pathlib.Path(path)
        figures_dir = outdir / "figures"
        assert figures_dir.exists(), f"figures/ directory missing in {name}"
        assert figures_dir.is_dir(), f"figures/ is not a directory in {name}"

    @pytest.mark.parametrize("name,path", TUTORIAL_OUTPUTS)
    def test_at_least_one_figure(self, name, path):
        """Test that at least one PNG figure exists per tutorial."""
        outdir = pathlib.Path(path)
        figures_dir = outdir / "figures"
        png_files = list(figures_dir.glob("*.png"))
        assert len(png_files) > 0, f"No PNG figures found in {name}"

    @pytest.mark.parametrize("name,path", TUTORIAL_OUTPUTS)
    def test_figure_files_nonzero(self, name, path):
        """Test that all figure files are nonzero bytes."""
        outdir = pathlib.Path(path)
        figures_dir = outdir / "figures"
        for png_file in figures_dir.glob("*.png"):
            size = png_file.stat().st_size
            assert size > 0, f"Figure {png_file.name} is zero bytes in {name}"

    @pytest.mark.skip(reason="legacy output artifact contract superseded by v0.2.28 canonical tutorial figure manifest (docs/_static/tutorial_figures/figure_manifest.json); validation moved to test_tutorial_figure_manifest_v028.py")
    @pytest.mark.parametrize("name,path", TUTORIAL_OUTPUTS)
    def test_figure_hash_in_assets(self, name, path):
        """DEPRECATED: Test that figure hashes are in asset_hashes.json.

        Legacy test for v0.2.19 output artifact contract. Superseded by canonical
        v0.2.28 tutorial figure manifest validation in test_tutorial_figure_manifest_v028.py.

        The v0.2.28+ canonical figure system is:
        - docs/_static/tutorial_figures/figure_manifest.json (source of truth)
        - scripts/generate_tutorial_figures.py (generation script)
        - test_tutorial_figure_manifest_v028.py (validation test)

        Runtime outputs/ directories contain ephemeral artifacts and are no longer
        part of the release contract.
        """
        outdir = pathlib.Path(path)
        hashes_file = outdir / "asset_hashes.json"
        with open(hashes_file) as f:
            asset_hashes = json.load(f)

        figures_dir = outdir / "figures"
        for png_file in figures_dir.glob("*.png"):
            rel_path = f"figures/{png_file.name}"
            assert rel_path in asset_hashes, f"Figure {rel_path} not in asset_hashes in {name}"

    @pytest.mark.parametrize("name,path", TUTORIAL_OUTPUTS)
    def test_claim_gates_frozen(self, name, path):
        """Test that claim gates are frozen as expected."""
        outdir = pathlib.Path(path)
        manifest_file = outdir / "manifest.json"
        with open(manifest_file) as f:
            manifest = json.load(f)

        assert manifest.get("claim_level") == "computational_scaffold", f"claim_level not frozen in {name}"
        assert manifest.get("physical_amplitude_claim_allowed") is False, f"physical_amplitude_claim_allowed not False in {name}"
        assert manifest.get("field_claim_level") == "proxy_readout_only", f"field_claim_level not frozen in {name}"

    @pytest.mark.parametrize("name,path", TUTORIAL_OUTPUTS)
    def test_metrics_nonzero(self, name, path):
        """Test that metrics contain nonzero values."""
        outdir = pathlib.Path(path)
        metrics_file = outdir / "metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)

        # At minimum, we should have some metrics (structure depends on tutorial)
        assert len(metrics) > 0, f"Metrics is empty in {name}"
        assert isinstance(metrics, dict), f"Metrics is not a dict in {name}"
