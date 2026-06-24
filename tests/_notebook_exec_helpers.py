"""Shared nbclient execution helper for notebook-execution tests.

Factored out of test_suite_no1_notebook_execution.py /
test_suite_no4_notebook_execution.py, which both duplicated this same
portable-kernel-spec + execute + collect-errors logic inline. Used by every
notebook-execution test file so there is one canonical implementation.

Not a test module itself (no test_ prefix) -- pytest will not collect it.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def portable_kernel_context(kernel_name: str = "jaxfne_test_kernel"):
    """Install a throwaway IPython kernelspec pointed at sys.executable.

    Avoids depending on a pre-registered "python3" kernel being present
    (e.g. on a fresh CI runner) by writing a minimal kernelspec to a temp
    dir and registering it for the duration of the test.
    """
    from jupyter_client.kernelspec import KernelSpecManager

    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "JAXFNE Portable Test Kernel",
        "language": "python",
    }
    temp_dir = tempfile.mkdtemp()
    try:
        spec_path = Path(temp_dir) / "kernel.json"
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        ksm = KernelSpecManager()
        ksm.install_kernel_spec(temp_dir, kernel_name, user=True, replace=True)
        yield kernel_name
    finally:
        try:
            KernelSpecManager().remove_kernel_spec(kernel_name)
        except Exception:
            pass
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def execute_notebook_via_nbclient(
    notebook_path: Path,
    tmp_path: Path,
    *,
    timeout: int = 900,
    extra_inject: str = "",
) -> list[tuple[int, str, str]]:
    """Execute a notebook end-to-end in a fresh kernel; return cell errors.

    Injects a sys.path setup cell (so the notebook imports the local jaxfne
    checkout, not an installed copy) right after the first cell, plus any
    caller-supplied ``extra_inject`` source (e.g. a FIG_DIR override).

    Returns a list of (cell_index, ename, evalue) for every cell that
    produced a kernel 'error' output. An empty list means clean execution.
    """
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(notebook_path), as_version=4)

    repo_root = Path(__file__).parent.parent.resolve()
    inject_source = f'import sys\nsys.path.insert(0, "{repo_root}")\n{extra_inject}'
    inject_cell = nbformat.v4.new_code_cell(source=inject_source)
    inject_cell.metadata["tags"] = ["injected-by-smoke-test"]
    nb.cells.insert(1, inject_cell)

    with portable_kernel_context() as kernel_name:
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(tmp_path)}},
        )
        client.execute()

    error_cells: list[tuple[int, str, str]] = []
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error_cells.append(
                    (i, output.get("ename", "UnknownError"), output.get("evalue", ""))
                )
    return error_cells


def format_cell_errors(error_cells: list[tuple[int, str, str]]) -> str:
    """Human-readable rendering of execute_notebook_via_nbclient's error list."""
    return "Notebook cells produced errors:\n" + "\n".join(
        f"  cell {i}: {e}: {v}" for i, e, v in error_cells
    )
