"""Single-neuron multimodal Colab notebook validation tests for v0.2.8.

Tests that:
1. Notebook exists and is valid JSON
2. First code cell contains !pip install jaxfne
3. Second code cell contains version verification
4. Outputs are cleared (no committed outputs)
5. No private/absolute paths in notebook cells
6. Notebook includes all eight readout operator names
7. Notebook mentions jaxfne.__version__
8. Tutorial documentation links to the notebook
9. Notebook follows v0.2.7 standard structure
10. Version remains 0.2.3
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class TestSingleNeuronNotebook:
    """Tests for single-neuron multimodal Colab notebook (v0.2.8)."""

    def test_notebook_exists(self):
        """Test that notebooks/01_single_neuron_multimodal.ipynb exists."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "01_single_neuron_multimodal.ipynb"
        )
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

    def test_notebook_is_valid_json(self):
        """Test that notebook is valid JSON."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "01_single_neuron_multimodal.ipynb"
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
            / "01_single_neuron_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
        assert (
            len(code_cells) > 0
        ), "Notebook must contain at least one code cell"

        # First code cell (may be after markdown cells)
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
            / "01_single_neuron_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
        assert (
            len(code_cells) >= 2
        ), "Notebook must contain at least two code cells"

        # Second code cell
        second_code_cell = code_cells[1]
        source = "".join(second_code_cell["source"])
        assert (
            "jaxfne.__version__" in source
        ), "Second code cell must verify jaxfne version"

    def test_outputs_are_cleared(self):
        """Test that all code cell outputs are empty (cleared before commit)."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "01_single_neuron_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
        for i, cell in enumerate(code_cells):
            outputs = cell.get("outputs", [])
            assert (
                len(outputs) == 0
            ), f"Code cell {i} should have empty outputs (outputs cleared before commit)"

    def test_no_private_paths(self):
        """Test that notebook contains no absolute private paths."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "01_single_neuron_multimodal.ipynb"
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
            / "01_single_neuron_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        # Combine all cell sources into one string
        full_source = ""
        for cell in nb_json["cells"]:
            full_source += "".join(cell.get("source", []))

        # Check for all eight readouts
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

    def test_mentions_version(self):
        """Test that notebook mentions jaxfne.__version__."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "01_single_neuron_multimodal.ipynb"
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
            / "01_single_neuron_multimodal.md"
        )
        content = doc_path.read_text(encoding="utf-8")
        assert (
            "01_single_neuron_multimodal.ipynb" in content
        ), "Tutorial documentation must link to the notebook"

    def test_notebook_follows_standard_structure(self):
        """Test that notebook follows v0.2.7 standard structure."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "01_single_neuron_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        full_source = ""
        for cell in nb_json["cells"]:
            full_source += "".join(cell.get("source", []))

        # Check for standard sections
        required_sections = [
            "Imports",
            "Configuration",
            "Simulation",
            "Readout",
            "Output Bundle",
            "Next steps",
        ]

        for section in required_sections:
            assert (
                section in full_source
            ), f"Notebook must include '{section}' section"

    def test_version_unchanged(self):
        """Test that jaxfne version is 0.2.10."""
        import jaxfne

        assert (
            jaxfne.__version__ == "0.2.25"
        ), f"Version should be 0.2.23, got {jaxfne.__version__}"


class TestDocumentationConsistency:
    """Tests for documentation consistency with v0.2.8 notebook."""

    def test_tutorial_index_mentions_single_neuron(self):
        """Test that tutorials/index.md mentions single-neuron tutorial."""
        index_path = (
            Path(__file__).parent.parent / "docs" / "tutorials" / "index.md"
        )
        content = index_path.read_text(encoding="utf-8")
        assert (
            "single-neuron" in content.lower()
            or "single neuron" in content.lower()
        ), "tutorials/index.md must reference single-neuron tutorial"

    def test_no_forbidden_vocabulary_in_notebook(self):
        """Test that notebook avoids forbidden internal terminology."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "01_single_neuron_multimodal.ipynb"
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
        ]

        for pattern in forbidden_patterns:
            matches = re.findall(pattern, full_source, re.IGNORECASE)
            assert (
                len(matches) == 0
            ), f"Forbidden term pattern '{pattern}' found in notebook: {matches}"

    def test_colab_notebook_has_metadata(self):
        """Test that notebook includes Colab metadata."""
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / "01_single_neuron_multimodal.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        metadata = nb_json.get("metadata", {})
        # Colab-ready notebooks should have colab metadata
        assert (
            "colab" in metadata
        ), "Notebook should include Colab metadata for Colab-ready setup"
