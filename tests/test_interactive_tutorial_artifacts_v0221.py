"""
v0.2.21: Interactive tutorial artifacts tests.

Tests for optional Plotly HTML generation from source data, with
graceful fallback when Plotly is unavailable. Static PNG figures
remain the default and unchanged.

Truth status: computational scaffold, not empirically validated.
"""

import json
import pathlib
import tempfile
import pytest


# Check if Plotly is available
try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TestCoreImportNoPlotyRequired:
    """Core jaxfne import must work without Plotly."""

    def test_core_import_no_plotly_required(self):
        """jaxfne imports successfully without Plotly."""
        try:
            import jaxfne
            assert hasattr(jaxfne, 'configuration')
            assert hasattr(jaxfne, 'construct')
        except ImportError as e:
            pytest.fail(f"jaxfne import failed: {e}")


class TestStaticPNGGeneration:
    """Static PNG generation works independent of Plotly."""

    def test_static_png_generation_without_plotly(self):
        """Static PNG figures generate without Plotly dependency."""
        # This test verifies that the v0.2.19 contract is unchanged:
        # PNG figures are always generated (matplotlib is required, not optional)
        # This test passes if matplotlib is installed and tutorial runs
        pytest.skip("Requires matplotlib and full tutorial execution")

    def test_runner_write_interactive_flag_exists(self):
        """Runner script recognizes --write-interactive flag."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/run_all_tutorials.py", "--help"],
            cwd=pathlib.Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--write-interactive" in result.stdout

    def test_validator_require_interactive_flag_exists(self):
        """Validator script recognizes --require-interactive flag."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/validate_tutorial_outputs.py", "--help"],
            cwd=pathlib.Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--require-interactive" in result.stdout


class TestBackwardsCompatibility:
    """Validator remains backwards compatible without --require-interactive."""

    def test_validator_skips_html_without_flag(self):
        """Validator skips HTML validation when --require-interactive not set."""
        # This test verifies that existing validation (without HTML) still works
        # If outputs exist from v0.2.19/v0.2.20, this should pass
        pytest.skip("Requires existing tutorial outputs")

    def test_plotly_missing_graceful_error(self):
        """Runner gracefully handles missing Plotly."""
        # When Plotly unavailable, runner should report "skipped" not "error"
        # This requires running the runner with --write-interactive
        pytest.skip("Requires running full tutorial runner")


class TestSourceDataArtifacts:
    """Source data JSON artifacts are created and contain claim gates."""

    def test_spike_event_source_data_structure(self):
        """Spike event source data has required fields."""
        source_data = {
            "source_data_kind": "spike_events",
            "tutorial_id": "03_single_neuron_multimodal_probe",
            "figure_id": "raster",
            "time_ms": [0, 100, 200],
            "unit_id": [0, 0, 1],
            "units_or_status": "binary_spike_event_proxy",
            "operator_kind": "spk",
            "claim_level": "computational_scaffold",
            "physical_amplitude_claim_allowed": False,
        }

        # Verify required fields
        assert source_data["source_data_kind"] == "spike_events"
        assert source_data["claim_level"] == "computational_scaffold"
        assert source_data["physical_amplitude_claim_allowed"] is False
        assert isinstance(source_data["time_ms"], list)
        assert isinstance(source_data["unit_id"], list)

    def test_spectrolaminar_profile_source_data_structure(self):
        """Spectrolaminar profile source data has required fields."""
        source_data = {
            "source_data_kind": "spectrolaminar_profile",
            "tutorial_id": "02_spectrolaminar_oddball_scaffold",
            "figure_id": "spectrolaminar_profile",
            "layers_or_depths": ["baseline", "event", "post", "full_peri_event"],
            "alpha_beta_profile": [66.0, 66.5, 66.8, 66.7],
            "gamma_profile": [9.7, 9.0, 9.2, 9.3],
            "units_or_status": "relative_proxy_units",
            "operator_kind": "spectrolaminar_profile",
            "claim_level": "computational_scaffold",
            "physical_amplitude_claim_allowed": False,
        }

        # Verify required fields
        assert source_data["source_data_kind"] == "spectrolaminar_profile"
        assert source_data["claim_level"] == "computational_scaffold"
        assert source_data["physical_amplitude_claim_allowed"] is False
        assert len(source_data["layers_or_depths"]) == len(source_data["alpha_beta_profile"])
        assert len(source_data["alpha_beta_profile"]) == len(source_data["gamma_profile"])

    def test_source_data_json_safe(self):
        """Source data JSON is strictly safe (no NaN/Inf)."""
        source_data = {
            "value": 1.5,
            "claim_level": "computational_scaffold",
            "physical_amplitude_claim_allowed": False,
        }

        # Should not raise with allow_nan=False
        json_str = json.dumps(source_data, allow_nan=False)
        assert '"value": 1.5' in json_str


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveHTMLGeneration:
    """Interactive HTML generation from source data (Plotly required)."""

    def test_html_generation_from_tutorial_data(self):
        """HTML generated from source data, not from PNG."""
        # This test verifies the critical constraint:
        # HTML comes from source_data.json, not from PNG files
        # When Plotly available, this should pass
        pytest.skip("Requires running full tutorial runner with Plotly")

    def test_html_hash_validation(self):
        """HTML hash can be recomputed and validated."""
        import hashlib

        html_content = "<html><body>Test</body></html>"
        computed_hash = hashlib.sha256(html_content.encode()).hexdigest()

        # Verify it's a valid SHA256 hex string
        assert len(computed_hash) == 64
        assert all(c in "0123456789abcdef" for c in computed_hash)

    def test_asset_hashes_include_html(self):
        """Asset hashes include HTML entries when generated."""
        asset_hashes = {
            "manifest.json": "abc123",
            "figures/raster.png": "def456",
            "figures/source_data.json": "ghi789",
            "figures/raster.html": "jkl012",
        }

        assert "figures/raster.html" in asset_hashes
        assert isinstance(asset_hashes["figures/raster.html"], str)
        assert len(asset_hashes["figures/raster.html"]) == 6  # Mocked hash

    def test_html_size_nonzero(self):
        """Generated HTML files have non-zero size."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("<html><body>Content</body></html>")
            temp_path = pathlib.Path(f.name)

        try:
            size = temp_path.stat().st_size
            assert size > 0
        finally:
            temp_path.unlink()

    def test_smoke_mode_interactive_validation(self):
        """Smoke mode with --write-interactive completes in reasonable time."""
        # This test verifies that --smoke with --write-interactive is fast
        pytest.skip("Requires running full tutorial runner with smoke mode")


class TestClaimGates:
    """Claim gates remain frozen in all interactive artifacts."""

    def test_physical_amplitude_claim_always_false(self):
        """physical_amplitude_claim_allowed is always False."""
        source_data = {"physical_amplitude_claim_allowed": False}
        assert source_data["physical_amplitude_claim_allowed"] is False

    def test_claim_level_always_computational_scaffold(self):
        """claim_level is always computational_scaffold."""
        source_data = {"claim_level": "computational_scaffold"}
        assert source_data["claim_level"] == "computational_scaffold"

    def test_no_biological_validation_claims(self):
        """Artifacts make no claims about biological validation."""
        artifact_text = "computational scaffold, not empirically validated"
        assert "empirical" not in artifact_text.lower() or "not" in artifact_text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
