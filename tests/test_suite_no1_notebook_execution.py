"""
Notebook execution smoke test for Suite No. 1 (Computational Biophysics).

Executes the full notebook via nbclient to verify:
  - All cells run without raising exceptions
  - No cell produces an 'error' output (kernel-level errors)

This is the preferred release gate for Suite No. 1. It requires:
  - nbclient and nbformat installed
  - A live Python 3 kernel (jupyter_client)
  - Up to ~15 minutes on slow hardware (timeout=900s)

To skip on CI environments that cannot launch a kernel, mark the environment
with SKIP_NOTEBOOK_EXECUTION=1 or exclude the 'slow' mark:
  pytest -m "not slow"
"""

import os
from pathlib import Path

import pytest

# Skip entire module if nbclient/nbformat not available.
nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "tutorials"
    / "jaxfne_suite_no_1_computational_biophysics.ipynb"
)

pytestmark = [pytest.mark.slow]


@pytest.mark.slow
def test_suite_no1_notebook_executes(tmp_path):
    """
    Execute Suite No. 1 notebook end-to-end and assert no cell errors.

    Uses nbclient to launch a fresh Python 3 kernel, run all cells in order,
    and verify that no cell output is of type 'error'. Figures are written
    to tmp_path to avoid polluting the source tree.
    """
    if os.environ.get("SKIP_NOTEBOOK_EXECUTION", "0") == "1":
        pytest.skip("SKIP_NOTEBOOK_EXECUTION=1 set in environment")

    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)

    # Inject FIG_DIR override so figures go to tmp_path (avoids side effects).
    import nbformat as _nbf

    inject_source = f'FIG_DIR = "{tmp_path}"\n'
    inject_cell = _nbf.v4.new_code_cell(source=inject_source)
    inject_cell.metadata["tags"] = ["injected-by-smoke-test"]

    # Prepend after the first cell (setup cell stays first).
    nb.cells.insert(1, inject_cell)

    from nbclient import NotebookClient

    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="jaxfne-venv",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()

    # Assert no cell produced a kernel error output.
    error_cells = []
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                ename = output.get("ename", "UnknownError")
                evalue = output.get("evalue", "")
                error_cells.append((i, ename, evalue))

    assert error_cells == [], (
        f"Notebook cells produced errors:\n"
        + "\n".join(f"  cell {i}: {e}: {v}" for i, e, v in error_cells)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
