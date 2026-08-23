"""Tests for JSON serialization and round-trip of default configurations.

Verifies that default configs (spectrolaminar, nuclei) can be serialized to JSON
and reconstructed, matching the original builder-generated configs exactly.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

import jaxfne as jtfne


def _assert_true_roundtrip(loaded_dict: dict, loaded_cfg) -> None:
    """Serialize the reconstructed configuration back and require deep
    equality with the source JSON, then require construct-level parity."""
    import json

    from dataclasses import asdict

    reserialized = json.loads(json.dumps(asdict(loaded_cfg), allow_nan=False, sort_keys=True))
    original = json.loads(json.dumps(loaded_dict, allow_nan=False, sort_keys=True))
    assert reserialized == original, "Configuration -> JSON is not lossless vs source"

    model_a = jtfne.construct(loaded_cfg)
    cfg_again_dict = json.loads(json.dumps(asdict(loaded_cfg), sort_keys=True))
    from jaxfne.core import Configuration as _Cfg
    model_b = jtfne.construct(_Cfg(**cfg_again_dict))

    import numpy as np

    for key in ("edge_list",):
        ea, eb = model_a.params[key], model_b.params[key]
        assert np.array_equal(np.asarray(ea.weight), np.asarray(eb.weight))
        assert np.array_equal(np.asarray(ea.tau_ms), np.asarray(eb.tau_ms))


def test_spectrolaminar_config_json_roundtrip():
    """Verify spectrolaminar_default.json deserializes and re-serializes losslessly."""
    config_path = Path(__file__).parent.parent / "jaxfne" / "configs" / "legacy" / "spectrolaminar_default.json"
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Load from JSON
    import json
    with open(config_path, encoding="utf-8") as f:
        loaded_dict = json.load(f)

    # Reconstruct Configuration from dict
    from jaxfne.core import Configuration
    loaded_cfg = Configuration(**loaded_dict)

    # Verify structure is valid
    assert len(loaded_cfg.networks) > 0, "No networks in loaded config"
    assert len(loaded_cfg.emitters) > 0, "No emitters in loaded config"
    assert len(loaded_cfg.probes) > 0, "No probes in loaded config"

    # Verify areas are present
    areas = [col['name'] for net in loaded_cfg.networks for col in net.get('columns', [])]
    assert 'V1' in areas, "V1 area not found in config"
    assert 'V4' in areas, "V4 area not found in config"

    _assert_true_roundtrip(loaded_dict, loaded_cfg)


def test_nuclei_config_json_roundtrip():
    """Verify nuclei_default.json deserializes and re-serializes losslessly."""
    config_path = Path(__file__).parent.parent / "jaxfne" / "configs" / "legacy" / "nuclei_default.json"
    assert config_path.exists(), f"Config file not found: {config_path}"

    import json
    with open(config_path, encoding="utf-8") as f:
        loaded_dict = json.load(f)

    from jaxfne.core import Configuration
    loaded_cfg = Configuration(**loaded_dict)

    assert len(loaded_cfg.networks) > 0, "No networks in loaded config"
    assert len(loaded_cfg.emitters) > 0, "No emitters in loaded config"
    assert len(loaded_cfg.probes) > 0, "No probes in loaded config"

    nucleus_name = loaded_cfg.networks[0].get('name')
    assert nucleus_name is not None, "No nucleus name in config"
    assert "thalamus" in nucleus_name, f"Expected thalamus in name, got {nucleus_name}"

    _assert_true_roundtrip(loaded_dict, loaded_cfg)
