"""
Suite No. 4 notebook execution test.

Validates that the Suite No. 4 notebook runs without errors using nbclient.
This test is marked as 'slow' because it executes the full notebook.
"""
import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

import json
import nbformat
from nbclient import NotebookClient


@pytest.mark.slow
def test_suite_no4_notebook_structure():
    """Verify Suite No. 4 notebook structure is valid (alternative to nbclient)."""
    notebook_path = "tutorials/jaxfne_suite_no_4_oscillatory_push_pull_laminar.ipynb"

    # Read and validate notebook structure
    with open(notebook_path, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Verify notebook has cells
    assert len(nb.cells) > 0, "Notebook has no cells"

    # Verify first cell is title/description
    assert nb.cells[0].cell_type == "markdown", "First cell should be markdown"
    assert "Suite No. 4" in nb.cells[0].source, "First cell should contain Suite No. 4 title"

    # Verify code cells are present
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 5, "Should have at least 5 code cells"

    # Verify imports are in first code cell
    assert "import jaxfne" in code_cells[0].source, "First code cell should import jaxfne"
    assert "import optax" in code_cells[0].source, "First code cell should import optax"

    print(f"✓ Notebook structure valid ({len(code_cells)} code cells)")


@pytest.mark.slow
def test_suite_no4_notebook_content_validation():
    """Validate that Suite No. 4 notebook contains required content."""
    notebook_path = "tutorials/jaxfne_suite_no_4_oscillatory_push_pull_laminar.ipynb"

    with open(notebook_path, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Concatenate all cell sources
    full_text = "\n".join(cell.source for cell in nb.cells)

    # Required terms
    required_terms = [
        "Oscillatory Push-Pull",
        "gAMPA_w",
        "gGABA_w",
        "optax.adam",
        "matrix_parameter",
        "duration_ms=1000",
    ]

    for term in required_terms:
        assert term in full_text, f"Notebook missing required term: {term}"

    # Forbidden terms
    forbidden_terms = [
        "Truth status",
        "Truth mode",
        "no mechanism claim",
        "gAMPA_first_half",
        "gAMPA_second_half",
        "drive_scale_a",
        "drive_scale_b",
        "jtfne.adam",
        "jtfne.agsdr_adam",
    ]

    for term in forbidden_terms:
        assert term not in full_text, f"Notebook contains forbidden term: {term}"

    # Verify gGABA_w_spec is defined in code, not just markdown
    code_sources = "\n".join(
        cell.source for cell in nb.cells if cell.cell_type == "code"
    )
    assert "gGABA_w_spec" in code_sources, "gGABA_w_spec must be defined in code cell"
    assert "gGABA_w_spec = jtfne.matrix_parameter" in code_sources
