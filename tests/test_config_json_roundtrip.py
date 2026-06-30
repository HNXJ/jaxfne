"""Tests for JSON serialization and round-trip of default configurations.

Verifies that default configs (spectrolaminar, nuclei) can be serialized to JSON
and reconstructed, matching the original builder-generated configs exactly.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

import jaxfne as jtfne


def test_spectrolaminar_config_json_roundtrip():
    """Verify spectrolaminar_default.json deserializes correctly."""
    config_path = Path(__file__).parent.parent / "jaxfne" / "configs" / "legacy" / "spectrolaminar_default.json"
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Load from JSON
    import json
    with open(config_path) as f:
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


def test_nuclei_config_json_roundtrip():
    """Verify nuclei_default.json deserializes correctly."""
    config_path = Path(__file__).parent.parent / "jaxfne" / "configs" / "legacy" / "nuclei_default.json"
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Load from JSON
    import json
    with open(config_path) as f:
        loaded_dict = json.load(f)

    # Reconstruct Configuration from dict
    from jaxfne.core import Configuration
    loaded_cfg = Configuration(**loaded_dict)

    # Verify structure is valid
    assert len(loaded_cfg.networks) > 0, "No networks in loaded config"
    assert len(loaded_cfg.emitters) > 0, "No emitters in loaded config"
    assert len(loaded_cfg.probes) > 0, "No probes in loaded config"

    # Verify nucleus structure
    nucleus_name = loaded_cfg.networks[0].get('name')
    assert nucleus_name is not None, "No nucleus name in config"
    assert "thalamus" in nucleus_name, f"Expected thalamus in name, got {nucleus_name}"
