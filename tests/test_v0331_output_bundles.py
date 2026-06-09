"""v0.3.31 output bundle helper tests.

Tests for JSON-safe output bundle creation functions.
"""

import json
import tempfile
from pathlib import Path

import pytest

import jaxfne as jtfne


class TestOutputBundles:
    """Test output bundle helper functions."""

    def test_validation_report_basic(self):
        """Create a basic validation report."""
        report = jtfne.validation_report(config_valid=True)
        assert isinstance(report, dict)
        assert report["valid"] is True
        assert isinstance(report["issues"], list)

    def test_validation_report_with_issues(self):
        """Validation report with issues."""
        issues = ["Config missing 'emitters'", "Invalid schema version"]
        report = jtfne.validation_report(
            config_valid=False,
            issues=issues,
        )
        assert report["valid"] is False
        assert report["issues"] == issues

    def test_validation_report_json_safe(self):
        """Validation report should be JSON-safe."""
        report = jtfne.validation_report(
            config_valid=True,
            metadata={"test": 1.0, "nested": {"key": "value"}},
        )
        json_str = json.dumps(report, allow_nan=False)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["valid"] is True

    def test_probe_report_basic(self):
        """Create a basic probe report."""
        report = jtfne.probe_report(n_probes=5)
        assert isinstance(report, dict)
        assert report["n_probes"] == 5
        assert isinstance(report["probe_types"], dict)

    def test_probe_report_with_types(self):
        """Probe report with type breakdown."""
        probe_types = {"V_m": 10, "spikes": 5, "field": 1}
        report = jtfne.probe_report(
            n_probes=16,
            probe_types=probe_types,
        )
        assert report["n_probes"] == 16
        assert report["probe_types"] == probe_types

    def test_probe_report_json_safe(self):
        """Probe report should be JSON-safe."""
        report = jtfne.probe_report(
            n_probes=10,
            probe_types={"V_m": 10},
            metadata={"description": "test probes"},
        )
        json_str = json.dumps(report, allow_nan=False)
        assert isinstance(json_str, str)

    def test_asset_hashes_basic(self):
        """Create asset hashes for files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            f1 = Path(tmpdir) / "file1.txt"
            f2 = Path(tmpdir) / "file2.txt"
            f1.write_text("content1")
            f2.write_text("content2")

            hashes = jtfne.asset_hashes({"file1": f1, "file2": f2})

            assert isinstance(hashes, dict)
            assert "file1" in hashes
            assert "file2" in hashes
            # SHA256 hashes are 64-char hex strings
            assert len(hashes["file1"]) == 64
            assert len(hashes["file2"]) == 64
            # Different content should have different hashes
            assert hashes["file1"] != hashes["file2"]

    def test_asset_hashes_json_safe(self):
        """Asset hashes should be JSON-safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "test.txt"
            f1.write_text("test")
            hashes = jtfne.asset_hashes({"test": f1})
            json_str = json.dumps(hashes, allow_nan=False)
            assert isinstance(json_str, str)

    def test_output_bundles_in_api(self):
        """All output bundle functions should be in public API."""
        assert hasattr(jtfne, "validation_report")
        assert hasattr(jtfne, "probe_report")
        assert hasattr(jtfne, "asset_hashes")

    def test_validation_report_metadata(self):
        """Validation report metadata preservation."""
        meta = {
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
        }
        report = jtfne.validation_report(
            config_valid=True,
            metadata=meta,
        )
        assert report["metadata"] == meta

    def test_probe_report_metadata(self):
        """Probe report metadata preservation."""
        meta = {"operator": "simulated_proxy", "status": "ok"}
        report = jtfne.probe_report(
            n_probes=5,
            metadata=meta,
        )
        assert report["metadata"] == meta

    def test_manifest_is_json_safe(self):
        """Existing manifest() function returns JSON-safe output."""
        cfg = jtfne.suite2_single_neuron_config(duration_ms=10, dt_ms=0.1)
        m = jtfne.manifest(cfg)
        json_str = json.dumps(m, allow_nan=False)
        assert isinstance(json_str, str)

    def test_manifest_contains_truth_gates(self):
        """Manifest should include truth gates from config."""
        cfg = jtfne.suite2_single_neuron_config()
        m = jtfne.manifest(cfg)
        assert "truth_mode" in m
        assert "claim_level" in m
        assert "physical_amplitude_claim_allowed" in m
        assert m["physical_amplitude_claim_allowed"] is False
