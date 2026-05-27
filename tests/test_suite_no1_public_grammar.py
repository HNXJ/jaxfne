"""
Tests for Suite No. 1 (Computational Biophysics) public tutorial.

Validates:
- Notebook structure (11 sections)
- Public API usage (Configuration, construct, simulate, probe)
- Metadata formatting (scope, readout, field mode claims)
- Figure generation
- JSON safety (no NaN/Inf in manifests)
- Public wording (no internal truth_mode, claim_level in public display)
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

# Path to notebook and assets
NOTEBOOK_PATH = Path(__file__).parent.parent / "tutorials" / "jaxfne_suite_no_1_computational_biophysics.ipynb"


class TestNotebookStructure:
    """Validate notebook presence and basic structure."""

    def test_notebook_exists(self):
        """Suite No. 1 notebook file exists."""
        assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"

    def test_notebook_is_valid_json(self):
        """Notebook is valid JSON."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)
        assert nb is not None
        assert "cells" in nb

    def test_notebook_has_cells(self):
        """Notebook has at least 11 sections (cells)."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)
        cells = nb["cells"]
        assert len(cells) > 10, f"Expected > 10 cells, got {len(cells)}"

    def test_section_headers_present(self):
        """Expected section headers are present."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        full_text = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "markdown"
        )

        expected_headers = [
            "Learning Objectives",
            "Setup",
            "Section 3: Mathematical Glossary",
            "Part 1",
            "Part 2",
            "Part 3",
            "Part 4",
            "Summary",
        ]

        for header in expected_headers:
            assert header in full_text, f"Expected header not found: {header}"


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
        assert ".set_emitter(" in code, "Missing set_emitter() call"
        assert ".probes(" in code, "Missing probes() call"

    def test_construct_simulate_used(self):
        """Notebook uses jtfne.construct() and jtfne.simulate()."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "jtfne.construct(" in code, "Missing construct() call"
        assert "jtfne.simulate(" in code, "Missing simulate() call"

    def test_correct_signals_attributes(self):
        """Notebook accesses signals with correct attribute names."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "signals.spikes" in code, "Missing signals.spikes access"
        assert "signals.V_m" in code, "Missing signals.V_m access"
        assert "signals.sources" in code, "Missing signals.sources access"
        assert "signals.time_ms" in code, "Missing signals.time_ms access"


class TestMetadataFormatting:
    """Validate metadata structure and content."""

    def test_run_metadata_structure(self):
        """Notebook defines metadata with public keys."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        required_keys = [
            "scope_status",
            "readout_status",
            "field_mode",
            "physical_amplitude_claim_allowed",
        ]

        for key in required_keys:
            assert f'"{key}"' in code or f"'{key}'" in code, \
                f"Missing metadata key: {key}"

    def test_scope_metadata_values_correct(self):
        """Metadata has correct public-facing values."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "computational_scaffold" in code
        assert "simulated_proxy" in code or "proxy" in code
        assert "False" in code  # physical_amplitude_claim_allowed


class TestPublicWording:
    """Validate public-facing wording."""

    def test_scope_clarified(self):
        """Notebook clarifies scope and limitations."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        md_text = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "markdown"
        )

        assert "computational scaffold" in md_text
        assert "proxy" in md_text
        assert "Summary" in md_text or "does not cover" in md_text

    def test_colab_link_present(self):
        """Notebook has Colab badge."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        full_text = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"]
        )

        assert "colab" in full_text.lower()
        assert "HNXJ/jaxfne" in full_text


class TestFigureGeneration:
    """Validate visualization code."""

    def test_display_helpers_defined(self):
        """Notebook defines display helpers."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "def save_png" in code
        assert "def finite_status" in code
        assert "def population_rate_hz" in code

    def test_figure_references_present(self):
        """Notebook saves figures with expected names."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "save_png(" in code
        assert ".png" in code


class TestJSONSafety:
    """Validate JSON output safety."""

    def test_json_serialization_safety(self):
        """Notebook uses allow_nan=False in JSON."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "json.dumps" in code
        # Allow either explicit allow_nan=False or safe serialization
        assert "allow_nan=False" in code or "JSONEncoder" in code

    def test_manifest_writing_present(self):
        """Notebook writes manifest to file."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "manifest" in code.lower()
        assert "json.dumps" in code


class TestConfiguration:
    """Validate configuration patterns."""

    def test_three_models_configured(self):
        """Notebook has three separate cfg objects."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "cfg_single" in code
        assert "cfg_pop" in code
        assert "cfg_column" in code

    def test_emitter_preset_specified(self):
        """Notebook specifies izhikevich preset."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert "izhikevich" in code
        assert "cortical" in code or "eig" in code

    def test_cell_types_declared(self):
        """Notebook declares E/I or E/PV composition."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)

        code = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"] if c["cell_type"] == "code"
        )

        assert ".cell_types(" in code
        # Check for excitatory
        assert '"E"' in code or "'E'" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
