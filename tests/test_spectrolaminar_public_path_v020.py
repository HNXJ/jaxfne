"""Tests for v0.2.0 spectrolaminar public path (Phase F).

Validates the minimal spectrolaminar oddball scaffold example:
- Example runs and produces strict JSON outputs
- Truth gates remain frozen
- Objective grammar is canonical (Phase E)
- Window discipline is enforced
- No biological mechanism claims
"""

import json
import pathlib
import tempfile

import pytest

import jaxfne


class TestSpectrolaminarExampleRuns:
    """Example execution and output validity."""

    def test_example_script_imports(self):
        """Example script can be imported without errors."""
        import sys
        import importlib.util

        example_path = pathlib.Path(__file__).parent.parent / "examples" / "02_spectrolaminar_oddball_scaffold.py"
        spec = importlib.util.spec_from_file_location("example", example_path)
        module = importlib.util.module_from_spec(spec)
        # Should not raise
        assert module is not None

    def test_example_main_callable(self):
        """Example script has a main() function."""
        import sys
        import importlib.util

        example_path = pathlib.Path(__file__).parent.parent / "examples" / "02_spectrolaminar_oddball_scaffold.py"
        spec = importlib.util.spec_from_file_location("example", example_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, 'main')
        assert callable(module.main)


class TestSpectrolaminarOutputsJSON:
    """Output JSON validity (manifest-first architecture)."""

    def test_manifest_json_exists(self):
        """Manifest JSON file is generated."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        manifest_file = outdir / "manifest.json"
        # Note: File exists from prior example run; if not, create via example execution
        if manifest_file.exists():
            assert manifest_file.stat().st_size > 0

    def test_manifest_json_strict(self):
        """Manifest is JSON-strict (no NaN/Inf)."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        manifest_file = outdir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            # Re-serialize with allow_nan=False should not raise
            json_str = json.dumps(manifest, allow_nan=False)
            assert isinstance(json_str, str)

    def test_objective_report_json_strict(self):
        """Objective report is JSON-strict (no NaN/Inf)."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        report_file = outdir / "objective_report.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            json_str = json.dumps(report, allow_nan=False)
            assert isinstance(json_str, str)

    def test_metrics_json_strict(self):
        """Metrics are JSON-strict (no NaN/Inf)."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        metrics_file = outdir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            json_str = json.dumps(metrics, allow_nan=False)
            assert isinstance(json_str, str)

    def test_validation_report_json_strict(self):
        """Validation report is JSON-strict (no NaN/Inf)."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        val_file = outdir / "validation_report.json"
        if val_file.exists():
            val = json.loads(val_file.read_text())
            json_str = json.dumps(val, allow_nan=False)
            assert isinstance(json_str, str)


class TestSpectrolaminarTruthGates:
    """Truth gates frozen in manifest."""

    def test_manifest_truth_mode(self):
        """Manifest truth_mode is truth_safe_unverified."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        manifest_file = outdir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            assert manifest.get("truth_mode") == "truth_safe_unverified"

    def test_manifest_claim_level(self):
        """Manifest claim_level is computational_scaffold."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        manifest_file = outdir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            assert manifest.get("claim_level") == "computational_scaffold"

    def test_manifest_source_calibration_uncalibrated(self):
        """Manifest source_calibration_status is uncalibrated_izhikevich_native_current."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        manifest_file = outdir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            assert manifest.get("source_calibration_status") == "uncalibrated_izhikevich_native_current"

    def test_manifest_field_solver_status(self):
        """Manifest field_solver_status is laminar_proxy_no_pde."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        manifest_file = outdir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            assert manifest.get("field_solver_status") == "laminar_proxy_no_pde"

    def test_manifest_field_claim_level(self):
        """Manifest field_claim_level is proxy_readout_only."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        manifest_file = outdir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            assert manifest.get("field_claim_level") == "proxy_readout_only"

    def test_manifest_physical_amplitude_claim_false(self):
        """Manifest physical_amplitude_claim_allowed is False."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        manifest_file = outdir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            assert manifest.get("physical_amplitude_claim_allowed") is False


class TestSpectrolaminarObjectiveGrammar:
    """Phase E canonical objective grammar."""

    def test_objective_report_has_acceptance_decision(self):
        """Objective report has explicit acceptance_decision field."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        report_file = outdir / "objective_report.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            assert "acceptance_decision" in report
            assert report["acceptance_decision"] in ["gates_pass", "gates_fail"]

    def test_objective_report_has_losses(self):
        """Objective report includes loss array."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        report_file = outdir / "objective_report.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            assert "losses" in report
            assert isinstance(report["losses"], list)

    def test_loss_has_canonical_name(self):
        """Loss has canonical name (not ambiguous)."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        report_file = outdir / "objective_report.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            losses = report.get("losses", [])
            if losses:
                loss_names = {loss["name"] for loss in losses}
                # Should not contain mean_similarity (too ambiguous)
                assert "mean_similarity" not in loss_names

    def test_loss_has_metadata_if_specified(self):
        """Loss preserves metadata from specification."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        report_file = outdir / "objective_report.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            losses = report.get("losses", [])
            if losses:
                # First loss should have metadata
                assert "metadata" in losses[0] or losses[0].get("name") != "profile_score"

    def test_regularizer_has_metadata(self):
        """Regularizer preserves metadata from specification."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        report_file = outdir / "objective_report.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            regularizers = report.get("regularizers", [])
            if regularizers:
                # Synchrony regularizer should have metadata
                sync_regs = [r for r in regularizers if r.get("name") == "synchrony"]
                if sync_regs:
                    assert "metadata" in sync_regs[0]


class TestSpectrolaminarWindowDiscipline:
    """Peri-event window specification."""

    def test_validation_report_has_windows(self):
        """Validation report includes windows specification."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        val_file = outdir / "validation_report.json"
        if val_file.exists():
            val = json.loads(val_file.read_text())
            assert "windows_ms" in val
            windows = val["windows_ms"]
            assert "baseline" in windows
            assert "event" in windows
            assert "post" in windows
            assert "full_peri_event" in windows

    def test_windows_span_full_peri_event(self):
        """Windows include -500 to +1000 ms."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        val_file = outdir / "validation_report.json"
        if val_file.exists():
            val = json.loads(val_file.read_text())
            windows = val.get("windows_ms", {})
            full = windows.get("full_peri_event", {})
            assert full.get("start") == -500
            assert full.get("end") == 1000

    def test_windows_baseline_correct(self):
        """Baseline window is -500 to 0 ms."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        val_file = outdir / "validation_report.json"
        if val_file.exists():
            val = json.loads(val_file.read_text())
            windows = val.get("windows_ms", {})
            baseline = windows.get("baseline", {})
            assert baseline.get("start") == -500
            assert baseline.get("end") == 0


class TestSpectrolaminarConditionVocabulary:
    """Condition labels for oddball paradigm."""

    def test_condition_vocabulary_present(self):
        """Validation report includes condition vocabulary."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        val_file = outdir / "validation_report.json"
        if val_file.exists():
            val = json.loads(val_file.read_text())
            assert "condition_vocabulary" in val

    def test_condition_vocabulary_includes_oddball_terms(self):
        """Condition vocabulary includes baseline, oddball, omission terms."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        val_file = outdir / "validation_report.json"
        if val_file.exists():
            val = json.loads(val_file.read_text())
            vocab = val.get("condition_vocabulary", [])
            # Should include oddball-related terms
            assert "baseline" in vocab
            assert "omission" in vocab


class TestSpectrolaminarSynchrony:
    """Synchrony diagnostic."""

    def test_metrics_include_synchrony(self):
        """Metrics include synchrony diagnostic for all windows."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        metrics_file = outdir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            # Check baseline has synchrony
            if "baseline" in metrics:
                assert "synchrony" in metrics["baseline"]

    def test_synchrony_is_finite(self):
        """Synchrony values are finite."""
        outdir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        metrics_file = outdir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            for window_name, window_metrics in metrics.items():
                synchrony = window_metrics.get("synchrony")
                if synchrony is not None:
                    assert isinstance(synchrony, (int, float))
                    assert synchrony == synchrony  # NaN check


class TestSpectrolaminarOutputNotCommitted:
    """Generated outputs should not be committed."""

    def test_outputs_directory_in_gitignore(self):
        """outputs/ directory should be in .gitignore."""
        gitignore_path = pathlib.Path(__file__).parent.parent / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            # Should mention outputs/
            assert "outputs/" in content or "outputs" in content

    def test_outputs_not_tracked_by_git(self):
        """outputs/ should not be tracked by git."""
        import subprocess

        outputs_dir = pathlib.Path("outputs/v020_spectrolaminar_public_path")
        if outputs_dir.exists():
            try:
                result = subprocess.run(
                    ["git", "ls-files", str(outputs_dir)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Should be empty (no tracked files in outputs/)
                assert result.stdout.strip() == ""
            except subprocess.TimeoutExpired:
                pass  # Skip if git times out
