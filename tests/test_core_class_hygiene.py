"""Test that core.py classes have correct method boundaries after contamination removal."""

import pytest
import jaxfne as jtfne


def test_dataset_spec_no_contamination():
    """DatasetSpec should not have Suite No. 2 fluent methods."""
    spec = jtfne.DatasetSpec()

    # Should NOT have these fluent methods
    forbidden = ['area_layer_cell_types', 'uniform3d', 'cell_type_drives', 'suite2_interarea']
    for method in forbidden:
        assert not hasattr(spec, method), f"DatasetSpec should not have {method}"


def test_laminar_population_no_contamination():
    """LaminarPopulation should not have Suite No. 2 fluent methods."""
    # LaminarPopulation is typically used inside Configuration, but test it directly
    forbidden = ['area_layer_cell_types', 'uniform3d', 'cell_type_drives', 'suite2_interarea']

    # Check the class itself
    for method in forbidden:
        assert not hasattr(jtfne.LaminarPopulation, method), f"LaminarPopulation class should not have {method}"


def test_laminar_source_geometry_no_contamination():
    """LaminarSourceGeometry should not have Suite No. 2 fluent methods."""
    forbidden = ['area_layer_cell_types', 'uniform3d', 'cell_type_drives', 'suite2_interarea']

    for method in forbidden:
        assert not hasattr(jtfne.LaminarSourceGeometry, method), f"LaminarSourceGeometry class should not have {method}"


def test_configuration_has_fluent_methods():
    """Configuration should KEEP all Suite No. 2 fluent methods."""
    cfg = jtfne.Configuration()

    required = ['area_layer_cell_types', 'uniform3d', 'cell_type_drives', 'suite2_interarea']
    for method in required:
        assert hasattr(cfg, method), f"Configuration should have {method}"
        assert callable(getattr(cfg, method)), f"Configuration.{method} should be callable"


def test_configuration_fluent_methods_chainable():
    """Configuration fluent methods should return a Configuration for chaining."""
    cfg = (jtfne.Configuration()
           .runtime(seed=7, dtype="float32", duration_ms=100.0, dt_ms=0.1)
           .area_layer_cell_types("V1", {"L2/3": {"E": 0.75, "PV": 0.25}})
           .uniform3d(radius_mm=0.25, height_mm=0.5)
           .cell_type_drives({"E": 10.0, "PV": 5.0})
           .suite2_interarea(enabled=True))

    assert isinstance(cfg, jtfne.Configuration)
