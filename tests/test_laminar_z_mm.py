"""P-066: laminar realized z is in mm, scaled by column_height_mm, like x and y."""

import jaxfne as jtfne

RUNTIME = dict(seed=1, duration_ms=1.0, dt_ms=0.5)


def _z_span(height_mm):
    cfg = jtfne.build_laminar_column("V1", 60, geometry="laminar", height_mm=height_mm)
    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .runtime(**RUNTIME)
    )
    model = jtfne.construct(cfg)
    rows = model.neuron_table()
    rows = list(rows.to_dict("records")) if hasattr(rows, "to_dict") else list(rows)
    zs = [float(r["z"]) for r in rows]
    return min(zs), max(zs)


def test_realized_z_spans_column_height_in_mm():
    lo, hi = _z_span(8.0)
    assert hi > 1.0, "fixture must reach past the unit interval; otherwise z is still a fraction"
    assert 0.0 <= lo and hi <= 8.0
    assert hi > 0.8 * 8.0, f"z top {hi} does not reach the 8.0 mm column height"
