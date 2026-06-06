"""v0.3.30a artifact and JSON safety regression tests.

Ensure that configuration and simulation artifacts serialize correctly.
"""

import json

import numpy as np

import jaxfne as jtfne


class TestArtifactJsonSafetyV0330:
    """Validate JSON safety of jaxfne artifact schemas."""

    def test_signals_manifest_json_safe(self):
        """Signals data must contain only finite values."""
        cfg = jtfne.suite2_four_celltype_config(seed=0)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=10, dt_ms=0.1, seed=0)

        # Verify signal arrays contain only finite values
        v_m = np.asarray(signals.V_m)
        assert np.all(np.isfinite(v_m)), "Voltage trace contains non-finite values"

        spikes = np.asarray(signals.spikes)
        assert np.all(np.isfinite(spikes)), "Spike data contains non-finite values"

        # Verify metadata serializes cleanly
        manifest_data = {
            "version": jtfne.__version__,
            "n_steps": int(v_m.shape[0]),
            "n_units": int(v_m.shape[-1]),
        }
        json_str = json.dumps(manifest_data, allow_nan=False)
        assert json_str, "Manifest serialization returned empty"

    def test_run_receipt_schema_json_safe(self):
        """provenance_receipt must produce JSON-safe output."""
        receipt = jtfne.provenance_receipt(branch="main", sha="abc123", dirty=False)

        # Must serialize with allow_nan=False
        json_str = json.dumps(receipt, allow_nan=False)
        assert json_str, "Receipt serialization returned empty"

        # Verify fields are present and finite
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict), "Parsed receipt must be dict-like"
        assert "jaxfne_version" in parsed, "Receipt must have jaxfne_version"
        assert "timestamp" in parsed, "Receipt must have timestamp"

    def test_simulation_produces_finite_traces(self):
        """Simulation must produce finite voltage and spike traces."""
        cfg = jtfne.suite2_four_celltype_config(seed=0)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=100, dt_ms=0.1, seed=0)

        # Verify simulation output is finite
        v_m = np.asarray(signals.V_m)
        assert np.all(np.isfinite(v_m)), "Voltage trace contains non-finite values"

        spikes = np.asarray(signals.spikes)
        assert np.all(np.isfinite(spikes)), "Spike data contains non-finite values"
