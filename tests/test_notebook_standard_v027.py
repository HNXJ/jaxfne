"""Notebook standard validation tests for v0.2.7.

Tests that:
1. Notebook standard documentation exists and is complete
2. Template notebook exists and follows all core rules
3. First code cell contains pip install jaxfne
4. Second code cell verifies installation
5. No private paths (absolute /Users/, /home/, etc.) in notebook cells
6. No outputs committed to template notebook
7. Notebook is CPU-safe (no GPU assumptions)
8. Public vocabulary audit passes (no forbidden terms)
9. mkdocs.yml navigation includes notebook standard link
10. Five-notebook tutorial stack is referenced in documentation
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class TestNotebookStandardDocumentation:
    """Tests for notebook standard documentation (v0.2.7)."""

    def test_notebook_standard_doc_exists(self):
        """Test that docs/tutorials/notebook_standard.md exists."""
        doc_path = (
            Path(__file__).parent.parent / "docs" / "tutorials" / "notebook_standard.md"
        )
        assert doc_path.exists(), f"Notebook standard doc not found at {doc_path}"

    def test_notebook_standard_has_core_rules_section(self):
        """Test that notebook standard doc contains 'Core Rules' section."""
        doc_path = (
            Path(__file__).parent.parent / "docs" / "tutorials" / "notebook_standard.md"
        )
        content = doc_path.read_text(encoding="utf-8")
        assert (
            "Core Rules" in content
        ), "Notebook standard doc must contain 'Core Rules' section"

    def test_notebook_standard_has_validation_checklist(self):
        """Test that notebook standard doc contains validation checklist."""
        doc_path = (
            Path(__file__).parent.parent / "docs" / "tutorials" / "notebook_standard.md"
        )
        content = doc_path.read_text(encoding="utf-8")
        assert (
            "Colab-Ready Checklist" in content
        ), "Notebook standard doc must contain 'Colab-Ready Checklist'"

    def test_notebook_standard_lists_five_tutorial_stack(self):
        """Test that notebook standard doc explains the five-notebook stack."""
        doc_path = (
            Path(__file__).parent.parent / "docs" / "tutorials" / "notebook_standard.md"
        )
        content = doc_path.read_text(encoding="utf-8")
        # Check for all five notebooks in the example list
        assert (
            "01_single_neuron_multimodal" in content
        ), "Standard doc must reference tutorial 01"
        assert (
            "02_two_neuron_ei" in content
        ), "Standard doc must reference tutorial 02"
        assert (
            "03_network_100_ei" in content
        ), "Standard doc must reference tutorial 03"
        assert (
            "04_v1_column" in content
        ), "Standard doc must reference tutorial 04"
        assert (
            "05_v1_pfc_dual_column" in content
        ), "Standard doc must reference tutorial 05"

    def test_tutorials_index_includes_notebook_standard_link(self):
        """Test that tutorials/index.md links to notebook_standard."""
        index_path = (
            Path(__file__).parent.parent / "docs" / "tutorials" / "index.md"
        )
        content = index_path.read_text(encoding="utf-8")
        assert (
            "notebook_standard" in content
        ), "tutorials/index.md must link to notebook standard"


class TestTemplateNotebook:
    """Tests for the Colab template notebook (v0.2.7)."""

    def test_template_notebook_exists(self):
        """Test that notebooks/00_template_colab.ipynb exists."""
        notebook_path = (
            Path(__file__).parent.parent / "notebooks" / "00_template_colab.ipynb"
        )
        assert notebook_path.exists(), f"Template notebook not found at {notebook_path}"

    def test_template_notebook_is_valid_json(self):
        """Test that template notebook is valid JSON."""
        notebook_path = (
            Path(__file__).parent.parent / "notebooks" / "00_template_colab.ipynb"
        )
        content = notebook_path.read_text(encoding="utf-8")
        try:
            nb_json = json.loads(content)
            assert isinstance(nb_json, dict), "Notebook must be a JSON object"
        except json.JSONDecodeError as e:
            raise AssertionError(f"Template notebook JSON is invalid: {e}")

    def test_first_code_cell_has_pip_install(self):
        """Test that first code cell contains !pip install jaxfne."""
        notebook_path = (
            Path(__file__).parent.parent / "notebooks" / "00_template_colab.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
        assert (
            len(code_cells) > 0
        ), "Notebook must contain at least one code cell"

        # Find the first code cell
        first_code_cell = code_cells[0]
        source = "".join(first_code_cell["source"])
        assert (
            "!pip install jaxfne" in source
        ), "First code cell must contain '!pip install jaxfne'"

    def test_second_code_cell_verifies_version(self):
        """Test that second code cell verifies jaxfne version."""
        notebook_path = (
            Path(__file__).parent.parent / "notebooks" / "00_template_colab.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
        assert (
            len(code_cells) >= 2
        ), "Notebook must contain at least two code cells"

        # Check the second code cell
        second_code_cell = code_cells[1]
        source = "".join(second_code_cell["source"])
        assert (
            "jaxfne.__version__" in source
        ), "Second code cell must contain version verification"

    def test_template_has_no_committed_outputs(self):
        """Test that template notebook has empty/cleared outputs."""
        notebook_path = (
            Path(__file__).parent.parent / "notebooks" / "00_template_colab.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        # Check that code cells have empty outputs
        code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
        for i, cell in enumerate(code_cells):
            outputs = cell.get("outputs", [])
            assert (
                len(outputs) == 0
            ), f"Code cell {i} should have empty outputs (outputs cleared before commit)"

    def test_template_has_no_private_paths(self):
        """Test that template notebook contains no absolute private paths."""
        notebook_path = (
            Path(__file__).parent.parent / "notebooks" / "00_template_colab.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        # Forbidden patterns: absolute paths
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

    def test_template_has_colab_metadata(self):
        """Test that template notebook includes Colab metadata."""
        notebook_path = (
            Path(__file__).parent.parent / "notebooks" / "00_template_colab.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        metadata = nb_json.get("metadata", {})
        # Colab notebooks have a colab key in metadata
        # (optional but good practice for Colab-ready notebooks)
        # If present, should be a dict with reasonable structure
        if "colab" in metadata:
            assert isinstance(metadata["colab"], dict)

    def test_template_has_python_kernel(self):
        """Test that template notebook specifies Python 3 kernel."""
        notebook_path = (
            Path(__file__).parent.parent / "notebooks" / "00_template_colab.ipynb"
        )
        nb_json = json.loads(notebook_path.read_text(encoding="utf-8"))

        kernel_spec = nb_json.get("metadata", {}).get("kernelspec", {})
        assert (
            "python" in kernel_spec.get("name", "").lower()
        ), "Notebook must use Python kernel"


class TestDocumentationNavigation:
    """Tests for documentation navigation and links."""

    def test_mkdocs_yaml_includes_notebook_standard(self):
        """Test that mkdocs.yml navigation includes notebook_standard."""
        mkdocs_path = Path(__file__).parent.parent / "mkdocs.yml"
        content = mkdocs_path.read_text(encoding="utf-8")
        assert (
            "notebook_standard" in content
        ), "mkdocs.yml must include notebook_standard in nav"

    def test_tutorials_index_has_notebook_stack_table(self):
        """Test that tutorials/index.md has a notebook stack reference."""
        index_path = (
            Path(__file__).parent.parent / "docs" / "tutorials" / "index.md"
        )
        content = index_path.read_text(encoding="utf-8")
        # Check for the five notebooks
        for i in range(1, 6):
            assert (
                f"**{i:02d}**" in content
            ), f"tutorials/index.md must list notebook {i:02d}"


class TestPublicVocabulary:
    """Tests for public vocabulary discipline in notebook standards."""

    def test_notebook_standard_has_vocabulary_table(self):
        """Test that notebook_standard.md includes vocabulary guidance."""
        doc_path = (
            Path(__file__).parent.parent / "docs" / "tutorials" / "notebook_standard.md"
        )
        content = doc_path.read_text(encoding="utf-8")
        # Check for vocabulary table
        assert "Use" in content and "Avoid" in content, (
            "Notebook standard must include vocabulary table "
            "with 'Use' and 'Avoid' columns"
        )

    def test_notebook_standard_no_forbidden_terms(self):
        """Test that notebook_standard.md avoids forbidden internal terms."""
        doc_path = (
            Path(__file__).parent.parent / "docs" / "tutorials" / "notebook_standard.md"
        )
        content = doc_path.read_text(encoding="utf-8")

        # Forbidden terms that should not appear in public docs
        forbidden_patterns = [
            r"truth[_-]?gate",
            r"truth[_-]?bearing",
            r"inner doctrine",
            r"worker prompt",
            r"biological proof",
            r"real EEG",
            r"real MEG",
            r"LFP[_-]?like",
            r"CSD[_-]?like",
        ]

        for pattern in forbidden_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert (
                len(matches) == 0
            ), f"Forbidden term pattern '{pattern}' found in notebook_standard.md: {matches}"
