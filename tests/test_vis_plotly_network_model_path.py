"""Smoke test for jaxfne.vis.plotly.network.plot_network_3d on a real Model."""
from __future__ import annotations

import pytest


def test_plot_network_3d_returns_figure_for_suite2_model() -> None:
    pytest.importorskip("plotly")

    import jaxfne as jtfne
    from jaxfne.vis.plotly.network import plot_network_3d

    cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)

    fig = plot_network_3d(model, signals, title="plotly.network smoke")
    assert fig is not None
    assert len(fig.data) >= 1
