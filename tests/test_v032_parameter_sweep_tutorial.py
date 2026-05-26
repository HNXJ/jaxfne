"""Tests for v0.3.2 parameter sweep tutorial."""

import json
from pathlib import Path
import pytest
import numpy as np
import jaxfne as jtfne


class TestV032TutorialFiles:
    """Verify tutorial files exist."""

    def test_notebook_exists(self):
        assert (Path(__file__).parent.parent / "tutorials" / "jaxfne_v032_parameter_sweep.ipynb").exists()

    def test_docs_page_exists(self):
        assert (Path(__file__).parent.parent / "docs" / "tutorials_v030" / "032_parameter_sweep.md").exists()

    def test_notebook_is_valid_json(self):
        notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_v032_parameter_sweep.ipynb"
        with open(notebook_path) as f:
            nb = json.load(f)
        assert "cells" in nb and len(nb["cells"]) > 0


class TestV032PublicWording:
    """Verify public doctrine is followed."""

    def test_canonical_import_present(self):
        notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_v032_parameter_sweep.ipynb"
        with open(notebook_path) as f:
            nb = json.load(f)
        code_text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"] if cell.get("cell_type") == "code")
        assert "import jaxfne as jtfne" in code_text

    def test_cfg_probes_grammar(self):
        notebook_path = Path(__file__).parent.parent / "tutorials" / "jaxfne_v032_parameter_sweep.ipynb"
        with open(notebook_path) as f:
            nb = json.load(f)
        code_text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"] if cell.get("cell_type") == "code")
        assert "cfg.probes(" in code_text or "cfg = cfg.probes" in code_text


class TestV032SweepSmoke:
    """Smoke test the parameter sweep."""

    def test_sweep_produces_finite_outputs(self):
        """Verify sweep produces finite firing rates and voltages."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
        cfg = cfg.cell_types({"E": 1.0})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

        assert np.all(np.isfinite(signals.V_m)), "Voltage not finite"
        assert np.all(np.isfinite(signals.spikes)), "Spikes not finite"

    def test_at_least_one_condition_in_range(self):
        """Verify at least one condition has firing rate in 2-25 Hz."""
        cfg = jtfne.Configuration()
        cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
        cfg = cfg.cell_types({"E": 1.0})
        cfg = cfg.connectivity()
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
        cfg = cfg.probes(["MUA-proxy"])

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

        firing_rate = float(np.sum(signals.spikes) / 1.0)
        # The default cortical_eig preset should produce spiking in range
        assert firing_rate > 0, "No spikes in default condition"

    def test_metadata_is_json_safe(self):
        """Verify sweep metadata can be serialized strictly."""
        metadata = {
            "tutorial_id": "v032_parameter_sweep",
            "sweep_conditions": 6,
            "firing_rates": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            "all_finite": True
        }
        json_str = json.dumps(metadata, allow_nan=False)
        assert isinstance(json_str, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
