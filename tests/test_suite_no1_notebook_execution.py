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

from _notebook_exec_helpers import execute_notebook_via_nbclient, format_cell_errors

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "tutorials"
    / "jaxfne_suite_no_1_computational_biophysics.ipynb"
)

pytestmark = [pytest.mark.slow, pytest.mark.notebook]


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

    errors = execute_notebook_via_nbclient(
        NOTEBOOK_PATH, tmp_path, timeout=900, extra_inject=f'FIG_DIR = "{tmp_path}"\n'
    )
    assert errors == [], format_cell_errors(errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
