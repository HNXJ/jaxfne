"""
Tests for Suite No. 3 (Low-Frequency Scaling Proxy Readouts) public tutorial.

Validates:
- Notebook structure and headers
- Public API usage (Configuration, construct, simulate, probes)
- Metadata formatting (truth_mode, claim_level, proxy_projection)
- Figures saved correctly
- JSON safety (no NaN/Inf in manifests/reports)
"""

import json
from pathlib import Path
import pytest

NOTEBOOK_PATH = Path(__file__).parent.parent / "tutorials" / "jaxfne_suite_no_3_low_frequency_scaling.ipynb"


class TestNotebookStructure:
    """Validate notebook presence and basic structure."""

    def test_notebook_exists(self):
        """Suite No. 3 notebook file exists."""
        assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"

    def test_notebook_is_valid_json(self):
        """Notebook is valid JSON."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)
        assert nb is not None
        assert "cells" in nb

    def test_notebook_has_cells(self):
        """Notebook has substantial cells."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)
        cells = nb["cells"]
        assert len(cells) > 10, f"Expected > 10 cells, got {len(cells)}"

    def test_section_headers_present(self):
        """Expected learning objectives and glossary headers are present."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        full_text = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "markdown"
        )

        expected_headers = [
            "Learning objectives",
            "Biological/computational question",
            "Mathematical glossary flow",
            "Formal equation",
            "Scope boundary",
        ]

        for header in expected_headers:
            assert header.lower() in full_text.lower(), f"Expected header not found: {header}"


class TestPublicAPIUsage:
    """Validate that notebook uses only public jaxfne API."""

    def test_configuration_api_used(self):
        """Notebook uses jtfne.Configuration with method chaining."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "jtfne.Configuration()" in code, "Missing Configuration() call"
        assert ".runtime(" in code, "Missing runtime() call"
        assert ".column(" in code, "Missing column() call"
        assert ".emitter(" in code, "Missing emitter() call"
        assert ".probes(" in code, "Missing probes() call"

    def test_construct_simulate_used(self):
        """Notebook uses construct() and simulate()."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "construct(" in code, "Missing construct() call"
        assert "simulate(" in code, "Missing simulate() call"


class TestMetadataFormatting:
    """Validate metadata structure and content."""

    def test_run_metadata_structure(self):
        """Notebook defines metadata with key public keys."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        required_keys = [
            "truth_mode",
            "claim_level",
            "boundary_condition",
            "gauge",
        ]

        for key in required_keys:
            assert f'"{key}"' in code or f"'{key}'" in code, \
                f"Missing metadata key: {key}"


class TestFigureReferences:
    """Validate figure references."""

    def test_figure_filenames_declared(self):
        """Notebook saves figures with expected filenames."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        expected_figs = [
            "01_suite03_raster_by_scale.png",
            "02_suite03_absolute_power_spectrum.png",
            "03_suite03_log_log_power_law_fit.png",
            "04_suite03_alpha_vs_scale.png",
            "05_suite03_low_frequency_absolute_power_vs_scale.png",
            "06_suite03_synchrony_proxy_vs_scale.png",
        ]

        for fig in expected_figs:
            assert fig in code, f"Missing figure file reference: {fig}"


class TestJSONSafety:
    """Validate JSON output safety."""

    def test_json_serialization_safety(self):
        """Notebook uses allow_nan=False in json.dumps."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "allow_nan=False" in code, "Notebook must enforce allow_nan=False"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
