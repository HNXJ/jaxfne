"""Tests for multi-area configuration API (Suite No. 5)."""

import json

import pytest

import jaxfne as jtfne


def test_configuration_areas_basic():
    """Test basic areas() declaration."""
    cfg = jtfne.Configuration().areas(["V1", "V4", "PFC"])
    assert cfg.metadata["areas"] == ["V1", "V4", "PFC"]


def test_configuration_areas_empty_raises():
    """Test that empty areas list raises ValueError."""
    with pytest.raises(ValueError, match="area_names must not be empty"):
        jtfne.Configuration().areas([])


def test_configuration_areas_duplicate_raises():
    """Test that duplicate area names raise ValueError."""
    with pytest.raises(ValueError, match="duplicate area names"):
        jtfne.Configuration().areas(["V1", "V1", "PFC"])


def test_configuration_layer_fractions_default():
    """Test layer_fractions() with default L1-L6 structure."""
    cfg = jtfne.Configuration().layer_fractions()

    # Verify L1-L6 are present
    layer_fracs = cfg.metadata["layer_fractions"]
    assert set(layer_fracs.keys()) == {"L1", "L2", "L3", "L4", "L5", "L6"}

    # Verify fractions form a partition of [0, 1]
    layers = sorted(layer_fracs.items())
    assert layers[0][1][0] == 0.0  # L1 starts at 0
    assert layers[-1][1][1] == 1.0  # L6 ends at 1

    # Verify contiguity: each layer starts where previous ends
    for i in range(len(layers) - 1):
        assert layers[i][1][1] == layers[i + 1][1][0]


def test_configuration_layer_cell_types_default():
    """Test layer_fractions() with default cell-type distributions."""
    cfg = jtfne.Configuration().layer_fractions()

    layer_ct = cfg.metadata["layer_cell_types"]
    assert set(layer_ct.keys()) == {"L1", "L2", "L3", "L4", "L5", "L6"}

    # Each layer should have E, PV, SST, VIP
    for layer, fracs in layer_ct.items():
        assert set(fracs.keys()) >= {"E", "PV", "SST", "VIP"}
        total = sum(fracs.values())
        assert abs(total - 1.0) < 0.01, f"Layer {layer} fractions don't sum to 1.0"


def test_configuration_layer_fractions_custom():
    """Test layer_fractions() with custom structure."""
    custom_fracs = {"L1": (0.0, 0.2), "L2": (0.2, 1.0)}
    custom_cts = {"L1": {"E": 1.0}, "L2": {"E": 0.5, "PV": 0.5}}

    cfg = jtfne.Configuration().layer_fractions(
        layer_fractions=custom_fracs,
        layer_cell_types=custom_cts,
    )

    assert cfg.metadata["layer_fractions"] == {"L1": [0.0, 0.2], "L2": [0.2, 1.0]}
    assert cfg.metadata["layer_cell_types"] == custom_cts


def test_configuration_layer_fractions_invalid_raises():
    """Test that invalid layer fractions raise ValueError."""
    invalid_fracs = {"L1": (0.5, 0.3)}  # z_min > z_max
    with pytest.raises(ValueError, match="invalid fraction range"):
        jtfne.Configuration().layer_fractions(layer_fractions=invalid_fracs)


def test_configuration_chainable_areas_and_layers():
    """Test that areas() and layer_fractions() are chainable."""
    cfg = (
        jtfne.Configuration()
        .areas(["V1", "V4", "PFC"])
        .layer_fractions()
    )

    assert cfg.metadata["areas"] == ["V1", "V4", "PFC"]
    assert "L1" in cfg.metadata["layer_fractions"]


def test_configuration_json_serializable():
    """Test that multi-area configuration is JSON-safe."""
    cfg = (
        jtfne.Configuration()
        .areas(["V1", "V4", "PFC"])
        .layer_fractions()
        .cell_types({"E": 0.75, "PV": 0.15, "SST": 0.10})
    )

    # Should be serializable to JSON
    try:
        cfg_dict = {
            "areas": cfg.metadata.get("areas"),
            "layer_fractions": cfg.metadata.get("layer_fractions"),
            "layer_cell_types": cfg.metadata.get("layer_cell_types"),
            "cell_types": cfg.metadata.get("cell_types"),
        }
        json_str = json.dumps(cfg_dict)
        assert isinstance(json_str, str)

        # Should deserialize correctly
        restored = json.loads(json_str)
        assert restored["areas"] == ["V1", "V4", "PFC"]
    except (TypeError, ValueError) as e:
        pytest.fail(f"Configuration not JSON-serializable: {e}")


def test_configuration_multi_area_complete():
    """Test complete multi-area configuration pipeline."""
    cfg = (
        jtfne.Configuration()
        .runtime(seed=42, dtype="float32", dt_ms=0.1, duration_ms=1000.0)
        .areas(["V1", "V4", "PFC"])
        .layer_fractions()
        .cell_types({"E": 0.75, "PV": 0.12, "SST": 0.08, "VIP": 0.05})
        .connectivity(feedforward=("V1", "V4", "PFC"), feedback=("PFC", "V4", "V1"))
        .set_emitter("izhikevich", "cortical_eig")
        .set_probes(["spikes", "V_m"])
    )

    # Verify all components are present
    assert cfg.metadata["areas"] == ["V1", "V4", "PFC"]
    assert "layer_fractions" in cfg.metadata
    assert cfg.metadata["cell_types"]["E"] == 0.75
    assert "feedforward" in cfg.metadata.get("connectivity", {})
    assert len(cfg.emitters) > 0
    assert len(cfg.probes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
