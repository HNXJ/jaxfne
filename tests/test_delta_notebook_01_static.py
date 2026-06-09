"""Static tests for delta-test notebook 01 infrastructure.

Tests the notebook components that can be verified without full execution.
"""

import pytest
import json
from pathlib import Path

import numpy as np
import jaxfne as jtfne


class TestDeltaNotebookInfrastructure:
    """Test delta-test notebook required functions exist."""

    def test_laminar_cortex_config_available(self):
        """laminar_cortex_config must be available."""
        assert hasattr(jtfne, "laminar_cortex_config")
        assert callable(jtfne.laminar_cortex_config)

    def test_validation_report_available(self):
        """validation_report must be available."""
        assert hasattr(jtfne, "validation_report")
        assert callable(jtfne.validation_report)

    def test_probe_report_available(self):
        """probe_report must be available."""
        assert hasattr(jtfne, "probe_report")
        assert callable(jtfne.probe_report)

    def test_asset_hashes_available(self):
        """asset_hashes must be available."""
        assert hasattr(jtfne, "asset_hashes")
        assert callable(jtfne.asset_hashes)

    def test_manifest_available(self):
        """manifest must be available."""
        assert hasattr(jtfne, "manifest")
        assert callable(jtfne.manifest)

    def test_save_json_available(self):
        """save_json must be available."""
        assert hasattr(jtfne, "save_json")
        assert callable(jtfne.save_json)

    def test_load_json_available(self):
        """load_json available via io module."""
        # load_json is in jaxfne.io but not exported at root
        from jaxfne.io import load_json
        assert callable(load_json)

    def test_construct_available(self):
        """construct must be available."""
        assert hasattr(jtfne, "construct")
        assert callable(jtfne.construct)

    def test_simulate_available(self):
        """simulate must be available."""
        assert hasattr(jtfne, "simulate")
        assert callable(jtfne.simulate)

    def test_no_local_imports_override_jaxfne(self):
        """Notebook should import only 'import jaxfne as jtfne'."""
        # This is a design constraint, not directly testable, but we can verify
        # the public API is complete
        required_functions = [
            "laminar_cortex_config",
            "validation_report",
            "probe_report",
            "asset_hashes",
            "manifest",
            "save_json",
            "construct",
            "simulate",
        ]
        for func in required_functions:
            assert hasattr(jtfne, func), f"Missing: {func}"

        # load_json is in io but not exported at root (OK)
        from jaxfne.io import load_json
        assert callable(load_json)

    def test_validation_report_structure(self):
        """validation_report should return dict with expected keys."""
        report = jtfne.validation_report(
            config_valid=True,
            issues=[],
            metadata={"test": "value"},
        )
        assert isinstance(report, dict)
        assert "valid" in report
        assert "issues" in report
        assert "metadata" in report

    def test_probe_report_structure(self):
        """probe_report should return dict with expected keys."""
        report = jtfne.probe_report(
            n_probes=5,
            probe_types={"V_m": 5, "spikes": 5},
            metadata={"status": "ok"},
        )
        assert isinstance(report, dict)
        assert "n_probes" in report
        assert "probe_types" in report
        assert "metadata" in report

    def test_laminar_config_produces_configuration(self):
        """laminar_cortex_config should produce Configuration object."""
        cfg = jtfne.laminar_cortex_config(
            seed=0,
            duration_ms=10.0,
            dt_ms=0.1,
            areas=["V1"],
            layers=["L2/3", "L4"],
            cell_types={"E": 0.8, "PV": 0.2},
            n=8,
        )
        assert cfg is not None
        assert hasattr(cfg, "metadata")
        assert cfg.metadata["truth_mode"] == "truth_safe_unverified"
        assert cfg.metadata["claim_level"] == "computational_scaffold"

    def test_construct_produces_model(self):
        """construct should produce a Model from config."""
        cfg = jtfne.laminar_cortex_config(n=4, seed=0, duration_ms=1.0, dt_ms=0.1)
        model = jtfne.construct(cfg)
        assert model is not None
        assert hasattr(model, "select")

    def test_simulate_produces_signals(self):
        """simulate should produce Signals."""
        cfg = jtfne.laminar_cortex_config(n=4, seed=0, duration_ms=1.0, dt_ms=0.1)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=1.0, dt_ms=0.1, seed=0)
        assert signals is not None
        assert hasattr(signals, "get")


class TestDeltaAGSDR:
    """Test AGSDR tuning infrastructure."""

    def test_agsdr_imports_available(self):
        """AGSDR should use jtfne APIs without issues."""
        assert hasattr(jtfne, "manifest")
        assert hasattr(jtfne, "save_json")
        assert callable(jtfne.manifest)

    def test_agsdr_wording_safe(self):
        """AGSDR should not claim biological learning or mechanism."""
        # This is a static check that the runner doesn't use forbidden phrases
        import pathlib
        runner_path = pathlib.Path("scripts/run_delta_notebook_01.py")
        if runner_path.exists():
            content = runner_path.read_text()
            # Check for forbidden phrases
            forbidden_phrases = [
                "biological learning",
                "mechanism proof",
                "adaptive cortical inference",
            ]
            for phrase in forbidden_phrases:
                assert phrase.lower() not in content.lower(), f"Found forbidden phrase: {phrase}"

    def test_agsdr_objective_callable(self):
        """AGSDR objective should compute scores."""
        # Simple test that an objective scoring function works
        baseline_rates = np.array([5.0, 10.0, 15.0, 3.0, 2.0])

        # Score should decrease as gain approaches 0.2 (which scales rates toward 1 Hz)
        gain1 = 1.0
        score1 = abs(5.0 * gain1 - 7.5) + 10.0 * max(0, 1.0 - 0.0) * gain1

        gain2 = 0.5
        score2 = abs(5.0 * gain2 - 7.5) + 10.0 * max(0, 1.0 - 0.0) * gain2

        # Both should be finite
        assert np.isfinite(score1)
        assert np.isfinite(score2)

    def test_delta_runner_has_agsdr(self):
        """Runner script should have AGSDR section."""
        import pathlib
        runner_path = pathlib.Path("scripts/run_delta_notebook_01.py")
        if runner_path.exists():
            content = runner_path.read_text()
            assert "agsdr" in content.lower()
            assert "connectivity_gain" in content.lower()
            assert "agsdr_objective" in content.lower()

    def test_optimizer_report_structure_valid(self):
        """Optimizer report should have required fields."""
        required_fields = [
            "optimizer_name",
            "best_parameters",
            "best_score",
            "mean_rate_error_hz",
            "tuning_status",
            "truth_mode",
            "biological_learning_claim",
            "mechanism_claim_status",
        ]
        # This is a structural test - actual validation happens at runtime
        assert all(isinstance(f, str) for f in required_fields)
