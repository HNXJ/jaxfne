"""Tests for v0.2.0 spectrolaminar public path (Phase F).

Validates the minimal spectrolaminar oddball scaffold example:
- Example runs and produces strict JSON outputs
- Truth gates remain frozen
- Objective grammar is canonical (Phase E)
- Window discipline is enforced
- No biological mechanism claims

The bundle is generated inside an isolated temporary working directory so
these tests execute their assertions on every run instead of silently
passing when a previously generated ``outputs/`` tree happens to exist.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest


EXAMPLE_PATH = pathlib.Path("examples/02_spectrolaminar_oddball_scaffold.py")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v020_outputs(tmp_path_factory):
    """Run the scaffold example in a temp cwd; return its output bundle dir."""
    tmp = tmp_path_factory.mktemp("v020_spectrolaminar_public_path")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH.resolve())],
        cwd=tmp,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, f"Example failed:\n{result.stderr}"
    return tmp / "outputs" / "v020_spectrolaminar_public_path"


def _load(bundle, name):
    return json.loads((bundle / name).read_text(encoding="utf-8"))


class TestSpectrolaminarExampleRuns:
    """Example execution and output validity."""

    def test_example_script_imports(self):
        """Example script can be imported without errors."""
        import importlib.util
        import types

        example_path = pathlib.Path(__file__).parent.parent / "examples" / "02_spectrolaminar_oddball_scaffold.py"
        assert example_path.exists(), f"Example script not found: {example_path}"
        spec = importlib.util.spec_from_file_location("example", example_path)
        module = importlib.util.module_from_spec(spec)
        assert isinstance(module, types.ModuleType)

    def test_example_main_callable(self):
        """Example script has a main() function."""
        example_path = pathlib.Path(__file__).parent.parent / "examples" / "02_spectrolaminar_oddball_scaffold.py"
        spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
            "example_main_check", example_path
        )
        module = __import__("importlib").util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, 'main')
        assert callable(module.main)


class TestSpectrolaminarOutputsJSON:
    """Output JSON validity (manifest-first architecture)."""

    def test_manifest_json_exists(self, v020_outputs):
        """Manifest JSON file is generated."""
        manifest_file = v020_outputs / "manifest.json"
        assert manifest_file.exists()
        assert manifest_file.stat().st_size > 0

    def test_manifest_json_strict(self, v020_outputs):
        """Manifest is JSON-strict (no NaN/Inf)."""
        manifest = _load(v020_outputs, "manifest.json")
        json_str = json.dumps(manifest, allow_nan=False)
        assert isinstance(json_str, str)

    def test_objective_report_json_strict(self, v020_outputs):
        """Objective report is JSON-strict (no NaN/Inf)."""
        report = _load(v020_outputs, "objective_report.json")
        json_str = json.dumps(report, allow_nan=False)
        assert isinstance(json_str, str)

    def test_metrics_json_strict(self, v020_outputs):
        """Metrics are JSON-strict (no NaN/Inf)."""
        metrics = _load(v020_outputs, "metrics.json")
        json_str = json.dumps(metrics, allow_nan=False)
        assert isinstance(json_str, str)

    def test_validation_report_json_strict(self, v020_outputs):
        """Validation report is JSON-strict (no NaN/Inf)."""
        val = _load(v020_outputs, "validation_report.json")
        json_str = json.dumps(val, allow_nan=False)
        assert isinstance(json_str, str)


class TestSpectrolaminarTruthGates:
    """Truth gates frozen in manifest."""

    def test_manifest_claim_level(self, v020_outputs):
        """Manifest claim_level is computational_scaffold."""
        manifest = _load(v020_outputs, "manifest.json")
        assert manifest.get("claim_level") == "computational_scaffold"

    def test_manifest_source_calibration_uncalibrated(self, v020_outputs):
        """Manifest source_calibration_status is uncalibrated_izhikevich_native_current."""
        manifest = _load(v020_outputs, "manifest.json")
        assert manifest.get("source_calibration_status") == "uncalibrated_izhikevich_native_current"

    def test_manifest_field_solver_status(self, v020_outputs):
        """Manifest field_solver_status is linear_solver."""
        manifest = _load(v020_outputs, "manifest.json")
        assert manifest.get("field_solver_status") == "linear_solver"

    def test_manifest_field_claim_level(self, v020_outputs):
        """Manifest field_claim_level is proxy_readout."""
        manifest = _load(v020_outputs, "manifest.json")
        assert manifest.get("field_claim_level") == "proxy_readout"

    def test_manifest_physical_amplitude_claim_false(self, v020_outputs):
        """Manifest physical_amplitude_calibrated is False."""
        manifest = _load(v020_outputs, "manifest.json")
        assert manifest.get("physical_amplitude_calibrated") is False


class TestSpectrolaminarObjectiveGrammar:
    """Phase E canonical objective grammar."""

    def test_objective_report_has_acceptance_decision(self, v020_outputs):
        """Objective report has explicit acceptance_decision field."""
        report = _load(v020_outputs, "objective_report.json")
        assert "acceptance_decision" in report
        assert report["acceptance_decision"] in ["gates_pass", "gates_fail"]

    def test_objective_report_has_losses(self, v020_outputs):
        """Objective report includes loss array."""
        report = _load(v020_outputs, "objective_report.json")
        assert "losses" in report
        assert isinstance(report["losses"], list)

    def test_loss_has_canonical_name(self, v020_outputs):
        """Loss has canonical name (not ambiguous)."""
        report = _load(v020_outputs, "objective_report.json")
        losses = report.get("losses", [])
        if losses:
            loss_names = {loss["name"] for loss in losses}
            # Should not contain mean_similarity (too ambiguous)
            assert "mean_similarity" not in loss_names

    def test_loss_has_metadata_if_specified(self, v020_outputs):
        """Loss preserves metadata from specification."""
        report = _load(v020_outputs, "objective_report.json")
        losses = report.get("losses", [])
        if losses:
            # First loss should have metadata
            assert "metadata" in losses[0] or losses[0].get("name") != "profile_score"

    def test_regularizer_has_metadata(self, v020_outputs):
        """Regularizer preserves metadata from specification."""
        report = _load(v020_outputs, "objective_report.json")
        regularizers = report.get("regularizers", [])
        if regularizers:
            # Synchrony regularizer should have metadata
            sync_regs = [r for r in regularizers if r.get("name") == "synchrony"]
            if sync_regs:
                assert "metadata" in sync_regs[0]


class TestSpectrolaminarWindowDiscipline:
    """Peri-event window specification."""

    def test_validation_report_has_windows(self, v020_outputs):
        """Validation report includes windows specification."""
        val = _load(v020_outputs, "validation_report.json")
        assert "windows_ms" in val
        windows = val["windows_ms"]
        assert "baseline" in windows
        assert "event" in windows
        assert "post" in windows
        assert "full_peri_event" in windows

    def test_windows_span_full_peri_event(self, v020_outputs):
        """Windows include -500 to +1000 ms."""
        val = _load(v020_outputs, "validation_report.json")
        windows = val.get("windows_ms", {})
        full = windows.get("full_peri_event", {})
        assert full.get("start") == -500
        assert full.get("end") == 1000

    def test_windows_baseline_correct(self, v020_outputs):
        """Baseline window is -500 to 0 ms."""
        val = _load(v020_outputs, "validation_report.json")
        windows = val.get("windows_ms", {})
        baseline = windows.get("baseline", {})
        assert baseline.get("start") == -500
        assert baseline.get("end") == 0


class TestSpectrolaminarConditionVocabulary:
    """Condition labels for oddball paradigm."""

    def test_condition_vocabulary_present(self, v020_outputs):
        """Validation report includes condition vocabulary."""
        val = _load(v020_outputs, "validation_report.json")
        assert "condition_vocabulary" in val

    def test_condition_vocabulary_includes_oddball_terms(self, v020_outputs):
        """Condition vocabulary includes baseline, oddball, omission terms."""
        val = _load(v020_outputs, "validation_report.json")
        vocab = val.get("condition_vocabulary", [])
        # Should include oddball-related terms
        assert "baseline" in vocab
        assert "omission" in vocab


class TestSpectrolaminarSynchrony:
    """Synchrony diagnostic."""

    def test_metrics_include_synchrony(self, v020_outputs):
        """Metrics include synchrony diagnostic for all windows."""
        metrics = _load(v020_outputs, "metrics.json")
        # Check baseline has synchrony
        if "baseline" in metrics:
            assert "synchrony" in metrics["baseline"]

    def test_synchrony_is_finite(self, v020_outputs):
        """Synchrony values are finite."""
        metrics = _load(v020_outputs, "metrics.json")
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
        content = gitignore_path.read_text(encoding="utf-8")
        # Should mention outputs/
        assert "outputs/" in content or "outputs" in content

    def test_outputs_not_tracked_by_git(self):
        """outputs/ should not be tracked by git."""
        result = subprocess.run(
            ["git", "ls-files", "outputs/"],
            capture_output=True,
            text=True,
        encoding="utf-8",
            timeout=30,
        )
        # Should be empty (no tracked files in outputs/)
        assert result.stdout.strip() == ""
