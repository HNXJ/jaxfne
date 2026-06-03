"""
Tests for v0.3.1 Single-Neuron Tutorial

Validates notebook structure, public wording, API usage, and simulation behavior.
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

import jaxfne as jtfne


class TestV031TutorialFiles:
    """Verify that tutorial files exist and have correct structure."""

    def test_notebook_exists(self):
        """Check that the notebook file exists."""
        notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_v031_single_neuron.ipynb"
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

    def test_docs_page_exists(self):
        """Check that the docs markdown page exists."""
        docs_path = Path(__file__).parent.parent / "docs" / "tutorials_v030" / "031_single_neuron.md"
        assert docs_path.exists(), f"Docs page not found at {docs_path}"

    def test_notebook_is_valid_json(self):
        """Check that the notebook is valid JSON."""
        notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_v031_single_neuron.ipynb"
        with open(notebook_path) as f:
            nb = json.load(f)
        assert "cells" in nb, "Notebook missing 'cells' key"
        assert len(nb["cells"]) > 0, "Notebook has no cells"

    def test_notebook_has_required_sections(self):
        """Check that notebook contains 13 required sections."""
        notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_v031_single_neuron.ipynb"
        with open(notebook_path) as f:
            nb = json.load(f)

        # Extract markdown cell content
        markdown_text = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "markdown":
                markdown_text += "\n".join(cell.get("source", [])) + "\n"

        # Check for section headers
        required_sections = [
            "Learning Objectives",
            "Biological/Computational Question",
            "Mathematical Glossary Flow",
            "Canonical Import",
            "Configuration Block",
            "Simulation Block",
            "Probe/Readout Block",
            "Manifest and Run Metadata",
            "Figures",
            "Interpretation",
            "Failure Modes",
            "Exercises",
            "Scope Boundaries",
        ]

        for section in required_sections:
            assert section in markdown_text, f"Section '{section}' not found in notebook"


class TestV031PublicWording:
    """Verify that public-facing tutorial text follows doctrine."""

    def test_no_public_claim_language_in_notebook(self):
        """Check that notebook markdown has no public 'claim' wording (except in Scope Boundaries section)."""
        notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_v031_single_neuron.ipynb"
        with open(notebook_path) as f:
            nb = json.load(f)

        # Extract only the main body text, excluding Scope Boundaries section which documents internal metadata
        markdown_text = ""
        in_scope_section = False
        for cell in nb["cells"]:
            if cell.get("cell_type") == "markdown":
                text = "\n".join(cell.get("source", []))
                # Skip the Scope Boundaries section which documents internal metadata
                if "## 13. Scope Boundaries" in text:
                    in_scope_section = True
                if not in_scope_section:
                    markdown_text += text + "\n"

        # Check for forbidden public-facing phrases (not in internal metadata documentation)
        forbidden_phrases = [
            "What this tutorial does NOT claim",
            "claim gates",
            "claim_gate_summary",
        ]

        text_lower = markdown_text.lower()
        for phrase in forbidden_phrases:
            assert phrase.lower() not in text_lower, f"Found forbidden phrase '{phrase}' in public tutorial text"

    def test_no_public_claim_language_in_docs(self):
        """Check that docs markdown has no public 'claim' wording."""
        docs_path = Path(__file__).parent.parent / "docs" / "tutorials_v030" / "031_single_neuron.md"
        with open(docs_path) as f:
            text = f.read().lower()

        forbidden_words = [
            "what this tutorial does not claim",
            "claim gates",
            "claim_gate_summary",
        ]

        for word in forbidden_words:
            assert word.lower() not in text, f"Found forbidden phrase '{word}' in docs"

    def test_canonical_import_in_notebook(self):
        """Check that notebook uses canonical import."""
        notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_v031_single_neuron.ipynb"
        with open(notebook_path) as f:
            nb = json.load(f)

        code_text = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                code_text += "\n".join(cell.get("source", [])) + "\n"

        assert "import jaxfne as jtfne" in code_text, "Canonical import not found"

    def test_preferred_grammar_cfg_probes(self):
        """Check that notebook uses cfg.probes(...) not cfg.set_probes(...)."""
        notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_v031_single_neuron.ipynb"
        with open(notebook_path) as f:
            nb = json.load(f)

        code_text = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                code_text += "\n".join(cell.get("source", [])) + "\n"

        # Check for cfg.probes usage
        assert "cfg = cfg.probes" in code_text or "cfg.probes(" in code_text, "Preferred cfg.probes(...) grammar not found"


class TestV031SimulationSmoke:
    """Smoke test the v0.3.1 simulation."""

    def test_single_neuron_simulation(self):
        """Run a single-neuron simulation and verify basic properties."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
        cfg = cfg.cell_types({"E": 1.0})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

        # Verify signal shapes
        assert signals.V_m.shape[0] == 10000, f"Expected 10000 timesteps, got {signals.V_m.shape[0]}"
        assert signals.V_m.shape[1] == 1, f"Expected 1 neuron, got {signals.V_m.shape[1]}"
        assert signals.spikes.shape == signals.V_m.shape, "Spikes and voltage shapes mismatch"

    def test_firing_rate_in_range(self):
        """Verify that the default configuration produces spiking in 2-25 Hz range."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
        cfg = cfg.cell_types({"E": 1.0})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

        n_spikes = int(np.sum(signals.spikes))
        firing_rate_hz = n_spikes / 1.0  # 1 second duration

        assert 2 <= firing_rate_hz <= 25, f"Firing rate {firing_rate_hz:.1f} Hz out of expected 2-25 Hz range"

    def test_voltage_arrays_are_finite(self):
        """Verify that voltage arrays contain no NaN or Inf."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
        cfg = cfg.cell_types({"E": 1.0})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

        assert np.all(np.isfinite(signals.V_m)), "Voltage array contains NaN or Inf"
        assert np.all(np.isfinite(signals.spikes)), "Spike array contains NaN or Inf"

    def test_run_metadata_is_json_safe(self):
        """Verify that run metadata can be serialized as strict JSON."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
        cfg = cfg.cell_types({"E": 1.0})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

        # Create metadata
        metadata = {
            "tutorial_id": "v031_single_neuron",
            "spike_count": int(np.sum(signals.spikes)),
            "firing_rate_hz": float(np.sum(signals.spikes) / 1.0),
            "v_min": float(np.min(signals.V_m)),
            "v_max": float(np.max(signals.V_m)),
        }

        # Try to serialize with allow_nan=False (strict JSON)
        json_str = json.dumps(metadata, allow_nan=False)
        assert isinstance(json_str, str), "Metadata not JSON serializable"

        # Deserialize to verify round-trip
        reloaded = json.loads(json_str)
        assert reloaded["spike_count"] == metadata["spike_count"], "Metadata round-trip failed"


class TestV031TargetRanges:
    """Verify that simulation target ranges are met."""

    def test_duration_is_sufficient(self):
        """Check that default duration is at least 1000 ms."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        assert cfg.metadata["duration_ms"] >= 1000.0, "Duration too short"

    def test_dt_is_appropriate(self):
        """Check that time step is 0.1 ms for numerical stability."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        assert cfg.metadata["dt_ms"] == 0.1, f"Expected dt=0.1, got {cfg.metadata['dt_ms']}"

    def test_dtype_is_float32(self):
        """Check that dtype is float32 as specified."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        assert cfg.metadata["dtype"] == "float32", f"Expected float32, got {cfg.metadata['dtype']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
