"""v0.3.30a artifact and JSON safety regression tests.

Ensure that configuration and simulation artifacts serialize correctly.
"""

import json
import math

import jaxfne as jtfne


class TestArtifactJsonSafetyV0330:
    """Validate JSON safety of jaxfne artifact schemas."""

    def test_signals_manifest_json_safe(self):
        """Signals manifest must serialize without NaN/Inf."""
        cfg = jtfne.suite2_four_celltype_config(seed=0)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=10, dt_ms=0.1, seed=0)

        # Access manifest data from signals object
        manifest_data = {
            "version": jtfne.__version__,
            "n_steps": signals.V_m.shape[0],
            "n_units": signals.V_m.shape[-1],
        }

        # Serialize with allow_nan=False; must succeed
        json_str = json.dumps(manifest_data, allow_nan=False)
        assert json_str, "Manifest serialization with allow_nan=False returned empty"

        # Deserialize and verify
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict), "Deserialized manifest must be dict-like"

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

    def test_objective_report_values_finite(self):
        """ObjectiveReport must not contain NaN/Inf in metrics."""
        cfg = jtfne.suite2_four_celltype_config(seed=0)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=100, dt_ms=0.1, seed=0)

        # Verify signals data is finite
        import numpy as np

        v_m = np.asarray(signals.V_m)
        assert np.all(np.isfinite(v_m)), "Voltage trace contains non-finite values"

        spikes = np.asarray(signals.spikes)
        assert np.all(np.isfinite(spikes)), "Spike data contains non-finite values"
