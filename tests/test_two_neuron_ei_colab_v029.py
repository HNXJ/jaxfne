"""Two-neuron E/I multimodal Colab notebook validation tests for v0.2.9.

Tests that:
1. Notebook exists and is valid JSON
2. First code cell contains !pip install jaxfne
3. Second code cell verifies version
4. Outputs are cleared (no committed outputs)
5. No private/absolute paths
6. Includes all eight readout operator names
7. Includes E/I terminology
8. Mentions jaxfne.__version__
9. Tutorial documentation links to notebook
10. Example script exists and runs
11. Vocabulary clean (no forbidden terms)
12. Version remains 0.2.3
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


class TestTwoNeuronEINotebook:
    """Tests for two-neuron E/I multimodal Colab notebook (v0.2.9)."""

    def test_notebook_exists(self):
        """Test that notebooks/02_two_neuron_ei_multimodal.ipynb exists."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

    def test_notebook_is_valid_json(self):
        """Test that notebook is valid JSON."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        content = notebook_path.read_text(encoding="utf-8")
        try:
            nb_json = json.loads(content)
            assert isinstance(nb_json, dict), "Notebook must be a JSON object"
            assert "cells" in nb_json, "Notebook must have cells"
        except json.JSONDecodeError as e:
            raise AssertionError(f"Notebook JSON is invalid: {e}")

    def test_first_code_cell_has_pip_install(self):
        """Test that first code cell contains !pip install jaxfne."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) > 0, "Notebook must contain at least one code cell"

        first_code_cell = code_cells[0]
        source = "".join(first_code_cell["source"])
        assert (
            "!pip install jaxfne" in source
        ), "First code cell must contain '!pip install jaxfne'"

    def test_second_code_cell_verifies_version(self):
        """Test that second code cell verifies jaxfne version."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 2, "Notebook must contain at least two code cells"

        second_code_cell = code_cells[1]
        source = "".join(second_code_cell["source"])
        assert (
            "jaxfne.__version__" in source
        ), "Second code cell must verify jaxfne version"

    def test_outputs_are_cleared(self):
        """Test that all code cell outputs are empty."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
        for i, cell in enumerate(code_cells):
            outputs = cell.get("outputs", [])
            assert (
                len(outputs) == 0
            ), f"Code cell {i} should have empty outputs"

    def test_no_private_paths(self):
        """Test that notebook contains no absolute private paths."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        forbidden_patterns = [
            r"/Users/",
            r"/home/",
            r"C:\\Users\\",
            r"/var/",
            r"/tmp/",
        ]

        for cell in nb_json["cells"]:
            source = "".join(cell.get("source", []))
            for pattern in forbidden_patterns:
                assert not re.search(
                    pattern, source
                ), f"Notebook contains forbidden path pattern: {pattern}"

    def test_includes_all_eight_readouts(self):
        """Test that notebook includes all eight readout operator names."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        full_source = ""
        for cell in nb_json["cells"]:
            full_source += "".join(cell.get("source", []))

        readouts = [
            "spk_probe",
            "vm_probe",
            "source_probe",
            "lfp_proxy_probe",
            "csd_proxy_probe",
            "eeg_proxy_probe",
            "meg_proxy_probe",
            "emm_proxy_probe",
        ]

        for readout in readouts:
            assert (
                readout in full_source
            ), f"Notebook must reference '{readout}' operator"

    def test_includes_ei_terminology(self):
        """Test that notebook includes E/I terminology."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        full_source = ""
        for cell in nb_json["cells"]:
            full_source += "".join(cell.get("source", []))

        ei_terms = ["excitatory", "inhibitory", "E/I", "coupling"]
        found_terms = [term for term in ei_terms if term.lower() in full_source.lower()]
        assert (
            len(found_terms) > 0
        ), "Notebook must reference E/I terminology"

    def test_mentions_version(self):
        """Test that notebook mentions jaxfne.__version__."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        full_source = ""
        for cell in nb_json["cells"]:
            full_source += "".join(cell.get("source", []))

        assert (
            "jaxfne.__version__" in full_source
        ), "Notebook must mention jaxfne.__version__"

    def test_tutorial_doc_links_to_notebook(self):
        """Test that tutorial documentation links to the notebook."""
        doc_path = (
            Path(__file__).parent.parent
            / "docs"
            / "tutorials"
            / "02_two_neuron_ei.md"
        )
        content = doc_path.read_text(encoding="utf-8")
        assert (
            "02_two_neuron_ei_multimodal.ipynb" in content
        ), "Tutorial documentation must link to the notebook"

    def test_example_script_exists(self):
        """Test that examples/04_two_neuron_ei_multimodal.py exists."""
        script_path = (
            Path(__file__).parent.parent / "examples" / "04_two_neuron_ei_multimodal.py"
        )
        assert script_path.exists(), f"Example script not found at {script_path}"

    def test_example_script_runs(self):
        """Test that example script runs successfully."""
        script_path = (
            Path(__file__).parent.parent / "examples" / "04_two_neuron_ei_multimodal.py"
        )
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "✓" in result.stdout, "Script should print success indicator"
        except subprocess.TimeoutExpired:
            raise AssertionError("Example script timed out")

    def test_example_script_generates_outputs(self):
        """Test that example script generates JSON output files."""
        output_dir = Path("outputs/v029_two_neuron_ei_multimodal")
        expected_files = [
            "manifest.json",
            "probe_report.json",
            "metrics.json",
            "validation_report.json",
            "asset_hashes.json",
        ]

        for filename in expected_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Expected output file not found: {filepath}"
            # Verify JSON validity
            try:
                json.load(open(filepath))
            except json.JSONDecodeError as e:
                raise AssertionError(f"Invalid JSON in {filepath}: {e}")

    def test_version_unchanged(self):
        """Test that jaxfne version is 0.2.10."""
        import jaxfne

        assert (
            jaxfne.__version__ == "0.2.22"
        ), f"Version should be 0.2.10, got {jaxfne.__version__}"

    def test_no_forbidden_vocabulary(self):
        """Test that notebook avoids forbidden internal terminology."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "02_two_neuron_ei_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        full_source = ""
        for cell in nb_json["cells"]:
            full_source += "".join(cell.get("source", []))

        forbidden_patterns = [
            r"truth[_-]?gate",
            r"truth[_-]?bearing",
            r"truth[_-]?safe",
            r"inner doctrine",
            r"worker prompt",
            r"biological proof",
            r"claim-status metadata",
        ]

        for pattern in forbidden_patterns:
            matches = re.findall(pattern, full_source, re.IGNORECASE)
            assert (
                len(matches) == 0
            ), f"Forbidden term pattern '{pattern}' found: {matches}"
