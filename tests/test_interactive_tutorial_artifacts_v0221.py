"""
v0.2.21: Interactive tutorial artifacts tests.

Core import without Plotly, the runner and validator interactive flags, and the
conservative claim gates on a constructed model's manifest.

Truth status: computational scaffold, not empirically validated.
"""

import os
import pathlib
import pytest
from pathlib import Path
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]


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


class TestInteractiveFlags:
    """Runner and validator expose the interactive flags."""

    def test_runner_write_interactive_flag_exists(self):
        """Runner script recognizes --write-interactive flag."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/run_all_tutorials.py", "--help"],
            env={**os.environ, "PYTHONPATH": str(_SCRIPT_REPO_ROOT), "PYTHONIOENCODING": "utf-8"},
        cwd=pathlib.Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        encoding="utf-8",
        )
        assert result.returncode == 0
        assert "--write-interactive" in result.stdout

    def test_validator_require_interactive_flag_exists(self):
        """Validator script recognizes --require-interactive flag."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/validate_tutorial_outputs.py", "--help"],
            env={**os.environ, "PYTHONPATH": str(_SCRIPT_REPO_ROOT), "PYTHONIOENCODING": "utf-8"},
        cwd=pathlib.Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        encoding="utf-8",
        )
        assert result.returncode == 0
        assert "--require-interactive" in result.stdout


class TestClaimGates:
    """Claim gates remain conservative on real Model manifests."""

    def test_constructed_model_manifest_truth_gates(self):
        """manifest() from a real construct() carries conservative defaults."""
        import jaxfne as jtfne

        cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
        model = jtfne.construct(cfg)
        manifest = model.manifest()

        assert manifest["claim_level"] == "computational_scaffold"
        assert manifest["physical_amplitude_calibrated"] is False
        assert manifest["field_solver_status"] == "linear_solver"
        assert manifest["field_claim_level"] == "proxy_readout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
