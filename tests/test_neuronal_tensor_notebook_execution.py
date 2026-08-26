"""
NeuronalTensor-first tutorial notebook execution test.

Validates that artifacts/tutorials/jaxfne_neuronal_tensor_first.ipynb runs without
errors using nbclient. This test is marked as 'slow' because it executes the
full notebook.
"""
import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

import nbformat
from nbclient import NotebookClient


@pytest.mark.slow
def test_neuronal_tensor_notebook_structure():
    """Verify the notebook structure is valid (alternative to nbclient)."""
    notebook_path = "artifacts/tutorials/jaxfne_neuronal_tensor_first.ipynb"

    with open(notebook_path, "r") as f:
        nb = nbformat.read(f, as_version=4)

    assert len(nb.cells) > 0, "Notebook has no cells"
    assert nb.cells[0].cell_type == "markdown", "First cell should be markdown"
    assert any("NeuronalTensor" in c.source for c in nb.cells[:2]), \
        "First or second cell should mention NeuronalTensor"

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 5, "Should have at least 5 code cells"

    all_code = " ".join("".join(c.source) for c in code_cells[:3])
    assert "import jaxfne" in all_code, "Setup code cells should import jaxfne"


@pytest.mark.slow
def test_neuronal_tensor_notebook_content_validation():
    """Validate that the notebook contains the required tensor-first + HDP terms."""
    notebook_path = "artifacts/tutorials/jaxfne_neuronal_tensor_first.ipynb"

    with open(notebook_path, "r") as f:
        nb = nbformat.read(f, as_version=4)

    full_text = "\n".join(cell.source for cell in nb.cells)

    required_terms = [
        "NeuronalTensor",
        "RuntimeConfiguration",
        "RuntimeConfig(enable_hdp=True",
        "size_scale_by_cell_type",
        "last_hdp_diagnostics",
        "save_neuronal_tensor",
        "load_neuronal_tensor",
    ]
    for term in required_terms:
        assert term in full_text, f"Notebook missing required term: {term}"

    # The tensor path's runtime object must not be conflated with the legacy one.
    code_sources = "\n".join(
        cell.source for cell in nb.cells if cell.cell_type == "code"
    )
    assert "jtfne.construct(tensor, jtfne.RuntimeConfiguration(" in code_sources, \
        "Construct call must use RuntimeConfiguration on the tensor path"
    assert "vis.raster" in code_sources, "Visualization must go through jaxfne.vis"


@pytest.mark.slow
@pytest.mark.notebook
def test_neuronal_tensor_notebook_execution():
    """Execute the NeuronalTensor-first notebook using nbclient."""
    notebook_path = "artifacts/tutorials/jaxfne_neuronal_tensor_first.ipynb"
    with open(notebook_path, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Inject sys.path setup so the notebook imports the local jaxfne checkout.
    import nbformat as _nbf
    from pathlib import Path
    repo_root = Path(__file__).parent.parent.resolve()
    inject_source = f'import sys\nsys.path.insert(0, "{repo_root}")\n'
    inject_cell = _nbf.v4.new_code_cell(source=inject_source)
    inject_cell.metadata["tags"] = ["injected-by-smoke-test"]
    nb.cells.insert(1, inject_cell)

    from _notebook_exec_helpers import portable_kernel_context

    with portable_kernel_context() as kernel_name:
        client = NotebookClient(
            nb,
            timeout=300,
            kernel_name=kernel_name,
        )
        client.execute()

        assert len(nb.cells) > 0, "Notebook should have cells after execution"

        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        cells_with_output = [c for c in code_cells if len(c.get("outputs", [])) > 0]
        assert len(cells_with_output) >= 5, "Notebook should have generated output in at least 5 code cells"
