"""Tests for tutorial smoke runner (v0.2.17)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestTutorialSmokeRunner:
    """Test the tutorial smoke runner script."""

    @pytest.fixture
    def repo_root(self):
        """Return the repository root."""
        return Path(__file__).parent.parent

    def test_script_exists(self, repo_root):
        """Script exists at expected path."""
        script = repo_root / "scripts" / "run_tutorial_smoke.py"
        assert script.exists(), "run_tutorial_smoke.py not found"

    def test_script_help_exits_zero(self, repo_root):
        """Script --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "run_tutorial_smoke.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Help failed: {result.stderr}"
        assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()

    def test_default_smoke_run_exits_zero(self, repo_root):
        """Default smoke run exits with code 0."""
        result = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "run_tutorial_smoke.py")],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0, f"Smoke run failed: {result.stderr}"
        assert "pass" in result.stdout.lower() or "PASS" in result.stdout

    def test_skip_examples_flag_exits_zero(self, repo_root):
        """--skip-examples flag works and exits 0."""
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_tutorial_smoke.py"),
                "--skip-examples",
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0
        assert "skipped" in result.stdout.lower()

    def test_skip_notebooks_flag_exits_zero(self, repo_root):
        """--skip-notebooks flag works and exits 0."""
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_tutorial_smoke.py"),
                "--skip-notebooks",
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0
        assert "skipped" in result.stdout.lower()

    def test_report_json_writes_valid_json(self, repo_root, tmp_path):
        """--report-json writes a valid JSON report."""
        report_path = tmp_path / "test_report.json"
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_tutorial_smoke.py"),
                "--report-json",
                str(report_path),
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0
        assert report_path.exists()

        # Validate JSON
        with open(report_path) as f:
            report = json.load(f)

        assert "status" in report
        assert "examples" in report
        assert "notebooks" in report
        assert "docs" in report

    def test_report_json_schema(self, repo_root, tmp_path):
        """Report JSON has expected schema."""
        report_path = tmp_path / "test_report.json"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_tutorial_smoke.py"),
                "--report-json",
                str(report_path),
            ],
            capture_output=True,
            cwd=repo_root,
        )

        with open(report_path) as f:
            report = json.load(f)

        # Check sections
        for section in ["examples", "notebooks", "docs"]:
            assert section in report
            assert "checked" in report[section]
            assert "skipped" in report[section]
            assert "errors" in report[section]
            assert isinstance(report[section]["checked"], list)
            assert isinstance(report[section]["skipped"], list)
            assert isinstance(report[section]["errors"], list)

    def test_report_includes_examples_section(self, repo_root, tmp_path):
        """Report includes examples section with expected examples."""
        report_path = tmp_path / "test_report.json"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_tutorial_smoke.py"),
                "--report-json",
                str(report_path),
            ],
            capture_output=True,
            cwd=repo_root,
        )

        with open(report_path) as f:
            report = json.load(f)

        # Should mark these as checked or skipped
        example_names = ["00_minimal_column", "02_spectrolaminar_oddball_scaffold",
                         "03_single_neuron_multimodal_probe", "04_two_neuron_ei_multimodal",
                         "05_network_100_ei_multimodal"]
        checked_or_skipped = (
            report["examples"]["checked"] + report["examples"]["skipped"]
        )
        for example in example_names:
            assert any(example in name for name in checked_or_skipped), \
                f"{example} not found in checked or skipped"

    def test_report_includes_notebooks_section(self, repo_root, tmp_path):
        """Report includes notebooks section with expected notebooks."""
        report_path = tmp_path / "test_report.json"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_tutorial_smoke.py"),
                "--report-json",
                str(report_path),
            ],
            capture_output=True,
            cwd=repo_root,
        )

        with open(report_path) as f:
            report = json.load(f)

        notebook_names = ["01_single_neuron_multimodal", "02_two_neuron_ei_multimodal",
                          "03_network_100_ei_multimodal"]
        checked_or_skipped = (
            report["notebooks"]["checked"] + report["notebooks"]["skipped"]
        )
        for notebook in notebook_names:
            assert any(notebook in name for name in checked_or_skipped), \
                f"{notebook} not found in checked or skipped"

    def test_notebook_structure_validation(self, repo_root):
        """Notebooks have required structure (install, version cells, no outputs)."""
        notebooks_dir = repo_root / "notebooks"
        required_notebooks = [
            "01_single_neuron_multimodal.ipynb",
            "02_two_neuron_ei_multimodal.ipynb",
            "03_network_100_ei_multimodal.ipynb",
        ]

        for nb_file in required_notebooks:
            path = notebooks_dir / nb_file
            assert path.exists(), f"{nb_file} not found"

            with open(path) as f:
                nb = json.load(f)

            # Check structure
            assert "cells" in nb, f"{nb_file}: missing cells"
            cells = nb["cells"]
            assert len(cells) >= 2, f"{nb_file}: fewer than 2 cells"

            # Check first code cell has pip install
            code_cells = [c for c in cells if c.get("cell_type") == "code"]
            assert code_cells, f"{nb_file}: no code cells"
            first_code = code_cells[0].get("source", [])
            first_code_text = (
                "".join(first_code) if isinstance(first_code, list) else first_code
            )
            assert "!pip install jaxfne" in first_code_text, \
                f"{nb_file}: first code cell missing pip install"

            # Check second code cell has version verification
            assert len(code_cells) >= 2, f"{nb_file}: fewer than 2 code cells"
            second_code = code_cells[1].get("source", [])
            second_code_text = (
                "".join(second_code) if isinstance(second_code, list) else second_code
            )
            assert "jaxfne.__version__" in second_code_text, \
                f"{nb_file}: second code cell missing version verification"

            # Check no committed outputs
            for cell in cells:
                outputs = cell.get("outputs", [])
                assert not outputs, f"{nb_file}: has committed outputs"

            # Check no private paths
            import re
            nb_text = json.dumps(nb)
            assert not re.search(r"/Users/|/home/|~|C:\\", nb_text), \
                f"{nb_file}: contains private paths"

    def test_version_remains_0_2_10(self, repo_root):
        """jaxfne version remains 0.2.10."""
        result = subprocess.run(
            [sys.executable, "-c", "import jaxfne; print(jaxfne.__version__)"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0
        version = result.stdout.strip()
        assert version == "0.2.18", f"Version is {version}, expected 0.2.18"
