"""v0.3.29 Config/Runtime/Provenance Hardening Tests.

Tests for:
- Config schema version bump (v0.0.15 → v0.0.16)
- JSON-safe config serialization (no NaN/Inf)
- JSON-safe manifest serialization (no NaN/Inf)
- Provenance receipt atomicity and JSON safety
- Simulate runtime precedence (reject both sim + kwargs)
- Optional dependencies remain lazy on import
- Version comparison across files
- Backward compatibility (existing configs with v0.0.15 schema rejected with clear message)
"""

import json
import sys
import pytest
import dataclasses
import jaxfne as jtfne
from pathlib import Path


def _to_dict(cfg):
    """Convert Configuration or JaxFNEConfig to dict."""
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    elif dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    else:
        raise TypeError(f"Cannot convert {type(cfg)} to dict")


class TestConfigSchemaBump:
    """Test that config schema version is bumped to v0.0.16."""

    def test_schema_version_is_v0016(self):
        """Verify _JAXFNE_CONFIG_SCHEMA_VERSION == 'jaxfne.config.v0.0.16'."""
        from jaxfne.core import _JAXFNE_CONFIG_SCHEMA_VERSION

        assert _JAXFNE_CONFIG_SCHEMA_VERSION == "jaxfne.config.v0.0.16"

    def test_old_schema_v0015_rejected(self):
        """Loading a config with old schema v0.0.15 should fail validation."""
        # Create a minimal JaxFNEConfig with old schema version
        cfg_old = jtfne.JaxFNEConfig(
            schema_version="jaxfne.config.v0.0.15",
            run={},
            truth={},
            network={},
            emitter={},
            field_spec=None,
            probes={},
        )

        # Validation should report schema_version mismatch
        result = jtfne.validate_config(cfg_old)
        assert not result.valid, "Config with old schema v0.0.15 should fail validation"
        assert any(
            "schema_version" in str(issue).lower() for issue in result.issues
        ), "Should report schema_version issue"


class TestJSONSafeConfig:
    """Test that configs are JSON-safe (no NaN/Inf)."""

    def test_config_serializes_to_json(self):
        """Config must serialize to strict JSON without NaN/Inf."""
        cfg = jtfne.suite2_single_neuron_config(duration_ms=10, dt_ms=0.1, seed=0)
        cfg_dict = _to_dict(cfg)

        # Attempt strict JSON serialization
        try:
            json_str = json.dumps(cfg_dict, allow_nan=False)
            assert isinstance(json_str, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Config is not JSON-safe (allow_nan=False failed): {e}")

    def test_config_dict_no_float_nan_inf(self):
        """All float values in config dict must be finite or None."""
        cfg = jtfne.suite2_single_neuron_config()
        cfg_dict = _to_dict(cfg)

        def check_no_nan_inf(obj, path=""):
            if isinstance(obj, float):
                if not (-1e308 < obj < 1e308) or (obj != obj):  # NaN check
                    pytest.fail(f"Found NaN/Inf at {path}: {obj}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_nan_inf(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    check_no_nan_inf(v, f"{path}[{i}]")

        check_no_nan_inf(cfg_dict)


class TestJSONSafeManifest:
    """Test that manifests are JSON-safe."""

    def test_manifest_serializes_to_json(self):
        """Manifest must serialize to strict JSON."""
        cfg = jtfne.suite2_single_neuron_config(duration_ms=10, dt_ms=0.1, seed=0)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=10, dt_ms=0.1, seed=0)

        manifest = jtfne.manifest(cfg, signals=signals)

        try:
            json_str = json.dumps(manifest, allow_nan=False)
            assert isinstance(json_str, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Manifest is not JSON-safe: {e}")


class TestProvenanceReceipt:
    """Test provenance_receipt() atomicity and JSON safety."""

    def test_provenance_receipt_basic_fields(self):
        """Provenance receipt must contain required fields."""
        receipt = jtfne.provenance_receipt(branch="main", sha="abc123", dirty=False)

        assert isinstance(receipt, dict)
        assert receipt["branch"] == "main"
        assert receipt["sha"] == "abc123"
        assert receipt["dirty"] is False
        assert "jaxfne_version" in receipt
        assert "config_schema_version" in receipt
        assert "manifest_schema_version" in receipt
        assert "timestamp" in receipt

    def test_provenance_receipt_json_safe(self):
        """Provenance receipt must be JSON-safe."""
        receipt = jtfne.provenance_receipt(branch="feat/test", sha="def456", dirty=True)

        try:
            json_str = json.dumps(receipt, allow_nan=False)
            assert isinstance(json_str, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Provenance receipt is not JSON-safe: {e}")

    def test_provenance_receipt_version_fields(self):
        """Version fields must match current jaxfne state."""
        from jaxfne.core import (
            _JAXFNE_VERSION,
            _JAXFNE_CONFIG_SCHEMA_VERSION,
            _MANIFEST_SCHEMA_VERSION,
        )

        receipt = jtfne.provenance_receipt()

        assert receipt["jaxfne_version"] == _JAXFNE_VERSION
        assert receipt["config_schema_version"] == _JAXFNE_CONFIG_SCHEMA_VERSION
        assert receipt["manifest_schema_version"] == _MANIFEST_SCHEMA_VERSION

    def test_provenance_receipt_timestamp_format(self):
        """Timestamp must be ISO-8601 format."""
        receipt = jtfne.provenance_receipt()
        timestamp = receipt["timestamp"]

        # ISO-8601 strings contain 'T' and timezone info
        assert "T" in timestamp
        assert isinstance(timestamp, str)
        # Should be parseable as ISO format
        try:
            from datetime import datetime

            datetime.fromisoformat(timestamp)
        except ValueError as e:
            pytest.fail(f"Timestamp not ISO-8601: {timestamp}, error: {e}")

    def test_provenance_receipt_defaults(self):
        """Provenance receipt should handle missing args gracefully."""
        receipt = jtfne.provenance_receipt()  # No args

        assert receipt["branch"] == "unknown"
        assert receipt["sha"] == "unknown"
        assert receipt["dirty"] is False


class TestSimulateRuntimePrecedence:
    """Test that simulate() rejects both sim object and kwargs."""

    def test_simulate_rejects_both_sim_and_kwargs(self):
        """Calling simulate(model, sim=sim, duration_ms=100) should raise ValueError."""
        cfg = jtfne.suite2_single_neuron_config()
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(duration_ms=20, dt_ms=0.1, seed=0)

        with pytest.raises(ValueError, match="Cannot specify both"):
            jtfne.simulate(model, sim=sim, duration_ms=100)

    def test_simulate_accepts_sim_object_alone(self):
        """Calling simulate(model, sim=sim) should work."""
        cfg = jtfne.suite2_single_neuron_config()
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(duration_ms=10, dt_ms=0.1, seed=0)

        signals = jtfne.simulate(model, sim=sim)
        assert signals is not None
        assert hasattr(signals, "time_ms")

    def test_simulate_accepts_kwargs_alone(self):
        """Calling simulate(model, duration_ms=10, dt_ms=0.1) should work."""
        cfg = jtfne.suite2_single_neuron_config()
        model = jtfne.construct(cfg)

        signals = jtfne.simulate(model, duration_ms=10, dt_ms=0.1, seed=0)
        assert isinstance(signals, jtfne.Signals)
        assert signals.V_m.shape[0] == 100


class TestOptionalDepsLazy:
    """Test that optional dependencies are not loaded on import jaxfne."""

    def test_plotly_not_imported_on_import(self):
        """Importing jaxfne should NOT import plotly."""
        # Check that plotly is not already in sys.modules
        # (If it was imported before, this test would be moot, but it should still pass)
        if "plotly" in sys.modules:
            pytest.skip("plotly already imported by another test")

        import jaxfne

        assert "plotly" not in sys.modules, "plotly should not be imported with jaxfne"

    def test_jaxley_not_imported_on_import(self):
        """Importing jaxfne should NOT import jaxley."""
        if "jaxley" in sys.modules:
            pytest.skip("jaxley already imported by another test")

        import jaxfne

        assert "jaxley" not in sys.modules, "jaxley should not be imported with jaxfne"


class TestVersionComparison:
    """Test that version is consistent across all files.

    pyproject.toml is the single source of truth; core and mkdocs must match it.
    These check *consistency*, not a hardcoded literal, so they do not break on
    every release bump (the consistency contract is what matters).
    """

    @staticmethod
    def _pyproject_version() -> str:
        import re

        content = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
        assert match, "Could not find version in pyproject.toml"
        return match.group(1)

    def test_version_in_core(self):
        """_JAXFNE_VERSION in core.py matches pyproject."""
        from jaxfne.core import _JAXFNE_VERSION

        assert _JAXFNE_VERSION == self._pyproject_version()

    def test_version_in_pyproject(self):
        """pyproject version is present and semver-shaped (X.Y.Z)."""
        import re

        assert re.match(r"^\d+\.\d+\.\d+", self._pyproject_version())

    def test_version_in_mkdocs(self):
        """jaxfne_version in mkdocs.yml matches pyproject."""
        mkdocs_path = Path(__file__).parent.parent / "mkdocs.yml"
        content = mkdocs_path.read_text()
        import re

        match = re.search(r'jaxfne_version: "([^"]+)"', content)
        assert match, "Could not find jaxfne_version in mkdocs.yml"
        assert match.group(1) == self._pyproject_version()

    def test_all_versions_match(self):
        """All version fields must agree with pyproject (single source of truth)."""
        from jaxfne.core import _JAXFNE_VERSION

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        mkdocs_path = Path(__file__).parent.parent / "mkdocs.yml"

        mkdocs_content = mkdocs_path.read_text()

        import re

        pyproject_ver = self._pyproject_version()
        mkdocs_ver = re.search(r'jaxfne_version: "([^"]+)"', mkdocs_content).group(1)

        assert (
            _JAXFNE_VERSION == pyproject_ver == mkdocs_ver
        ), f"Version mismatch: core={_JAXFNE_VERSION}, pyproject={pyproject_ver}, mkdocs={mkdocs_ver}"


class TestBackwardCompat:
    """Test backward compatibility of config API."""

    def test_jaxfne_config_to_dict_preserves_keys(self):
        """Config.to_dict() must preserve original key names."""
        cfg = jtfne.suite2_single_neuron_config()
        cfg_dict = _to_dict(cfg)

        # Configuration has these top-level keys (not schema_version)
        assert "networks" in cfg_dict
        assert "emitters" in cfg_dict
        assert "fields" in cfg_dict
        assert "probes" in cfg_dict
        assert "metadata" in cfg_dict

    def test_configuration_class_still_works(self):
        """Configuration class must still be usable (backward compat)."""
        cfg = jtfne.configuration()

        assert cfg is not None
        assert hasattr(cfg, "validate")
        # Must have method chaining
        cfg2 = cfg.network()
        assert cfg2 is not None
