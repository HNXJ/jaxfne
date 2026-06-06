"""v0.3.30a notebook structure hygiene tests.

Validate that Etude No. 1 has proper cell structure and executes cleanly.
"""

import json
import pathlib

import pytest


def _load_notebook(nb_path):
    """Load a notebook from disk."""
    with open(nb_path) as f:
        return json.load(f)


class TestNotebookStructureV0330:
    """Validate notebook cell structure for v0.3.30a (Etude No. 1 focus)."""

    def test_etude_no_1_cells_have_ids(self):
        """Etude No. 1: All cells must have 'id' field."""
        nb_path = pathlib.Path(
            "tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb"
        )
        nb = _load_notebook(nb_path)
        missing_ids = []
        for i, cell in enumerate(nb["cells"]):
            if "id" not in cell:
                missing_ids.append(i)
        assert not missing_ids, f"Etude No. 1: Cells missing IDs: {missing_ids}"

    def test_etude_no_1_no_adjacent_code_cells(self):
        """Etude No. 1: Code cells must be separated by markdown cells."""
        nb_path = pathlib.Path(
            "tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb"
        )
        nb = _load_notebook(nb_path)
        adjacent_code = []
        for i in range(len(nb["cells"]) - 1):
            if (
                nb["cells"][i]["cell_type"] == "code"
                and nb["cells"][i + 1]["cell_type"] == "code"
            ):
                adjacent_code.append((i, i + 1))
        assert not adjacent_code, (
            f"Etude No. 1: Adjacent code cells at indices: {adjacent_code}"
        )

    def test_etude_no_1_execution_clean(self):
        """Etude No. 1: Execution must complete without cell errors."""
        nb_path = pathlib.Path(
            "tutorials/outputs/executed/"
            "tutorials__etudes__jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb"
        )
        if not nb_path.exists():
            pytest.skip("Executed Etude notebook not found; run nbconvert first")

        nb = _load_notebook(nb_path)
        errors = []
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                outputs = cell.get("outputs", [])
                for output in outputs:
                    if output.get("output_type") == "error":
                        errors.append((i, output.get("ename", "Unknown")))

        assert not errors, (
            f"Etude No. 1 execution had {len(errors)} errors: {errors}"
        )
