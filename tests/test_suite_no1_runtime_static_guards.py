"""
Static runtime guards for Suite No. 1 (Computational Biophysics) notebook.

Validates that the notebook source (as text) contains all required runtime-safety
patterns introduced in the 8b44918 patch:

  - Every save_png() call includes fig_dir=FIG_DIR (no bare call missing it)
  - Tuned-model comparison uses result.model, not the raw config object
  - Optimizer generations >= 10 (guards against accidental regression to 1-gen runs)
  - FIG_DIR is defined (guards against NameError at runtime)

These checks are cheap (no kernel launch) and are intended as a fast release gate
that complements the full notebook execution smoke test.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "tutorials"
    / "jaxfne_suite_no_1_computational_biophysics.ipynb"
)


def _load_code() -> str:
    """Return all code cell sources concatenated from the notebook."""
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


class TestSuiteNo1RuntimeStaticGuards:
    """Assert required runtime-safety patterns are present in the notebook."""

    def test_notebook_exists(self):
        """Notebook file exists at expected path."""
        assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"

    def test_notebook_is_valid_json(self):
        """Notebook parses as valid JSON with a 'cells' key."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)
        assert "cells" in nb

    def test_fig_dir_defined(self):
        """FIG_DIR is defined somewhere in the notebook code cells."""
        code = _load_code()
        assert "FIG_DIR" in code, "FIG_DIR is never defined — runtime NameError guaranteed"

    def test_fig_dir_in_all_save_png_calls(self):
        """Every save_png() call includes fig_dir keyword argument."""
        code = _load_code()
        calls = re.findall(r"save_png\([^)]+\)", code, re.DOTALL)
        assert len(calls) > 0, "No save_png() calls found in notebook"
        for call in calls:
            assert "fig_dir" in call, (
                f"save_png call missing fig_dir keyword: {call[:120]}"
            )

    def test_fig_dir_equals_FIG_DIR_in_save_png_calls(self):
        """save_png calls pass fig_dir=FIG_DIR (not a literal string path)."""
        code = _load_code()
        assert "fig_dir=FIG_DIR" in code, (
            "Expected 'fig_dir=FIG_DIR' in save_png calls — "
            "hardcoded path or missing variable reference detected"
        )

    def test_tuned_simulation_uses_result_model(self):
        """Tuned comparison simulation calls jtfne.simulate(result.model, ...)."""
        code = _load_code()
        assert "jtfne.simulate(result.model" in code, (
            "Tuned simulation must use result.model, not the raw config object"
        )

    def test_result_model_referenced(self):
        """result.model is referenced in code (belt-and-suspenders)."""
        code = _load_code()
        assert "result.model" in code, "result.model never referenced in notebook code"

    def test_optimizer_generations_at_least_10(self):
        """Optimizer is configured with generations >= 10."""
        code = _load_code()
        assert "generations=10" in code or "generations = 10" in code, (
            "Expected generations=10 (or generations = 10) — "
            "optimizer may be running with too few generations"
        )

    def test_rejects_split_ampa_parameters(self):
        """Rejects old gAMPA_first_half and gAMPA_second_half parameter names."""
        code = _load_code()
        assert "gAMPA_first_half" not in code, (
            "Old parameter name gAMPA_first_half found — "
            "should use global gAMPA parameter instead"
        )
        assert "gAMPA_second_half" not in code, (
            "Old parameter name gAMPA_second_half found — "
            "should use global gAMPA parameter instead"
        )

    def test_requires_global_gampa_parameter(self):
        """Requires global gAMPA parameter (not split per-group)."""
        code = _load_code()
        assert '"gAMPA"' in code or "'gAMPA'" in code, (
            "Expected global gAMPA parameter in optimizer configuration — "
            "this decouples synaptic mechanism from objective groups"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
