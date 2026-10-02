"""build_laminar_column: radius_mm / height_mm are applied on the laminar route, never ignored."""

import pytest

import jaxfne as jtfne

RUNTIME = dict(seed=1, duration_ms=1.0, dt_ms=0.5)


def _extent(model):
    rows = model.neuron_table()
    rows = list(rows.to_dict("records")) if hasattr(rows, "to_dict") else list(rows)
    return max(abs(r["x"]) for r in rows)


def _build(**kw):
    cfg = jtfne.build_laminar_column("V1", 60, geometry="laminar", **kw)
    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .runtime(**RUNTIME)
    )
    return cfg, jtfne.construct(cfg)


def test_laminar_radius_changes_placement_and_height_reaches_metadata():
    _, base = _build()
    cfg, wide = _build(radius_mm=0.5, height_mm=2.0)
    assert cfg.metadata["column_radius_mm"] == 0.5 and cfg.metadata["column_height_mm"] == 2.0
    assert _extent(wide) > 1.5 * _extent(base), "radius_mm was accepted and ignored"


def test_laminar_defaults_declare_no_geometry_keys():
    cfg, _ = _build()
    assert "column_radius_mm" not in cfg.metadata and "column_height_mm" not in cfg.metadata


def test_uniform3d_defaults_and_validation_unchanged():
    cfg = jtfne.build_laminar_column("V1", 20)
    assert cfg.metadata["column_radius_mm"] == 0.25 and cfg.metadata["column_height_mm"] == 1.6
    with pytest.raises(ValueError, match="positive"):
        jtfne.build_laminar_column("V1", 20, geometry="laminar", radius_mm=0.0)
