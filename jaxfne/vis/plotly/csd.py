"""CSD-proxy depth x time heatmap visualization (Plotly)."""
from __future__ import annotations

import numpy as np

from ._common import require_plotly, field_proxy, contact_depths_mm, time_ms


def plot_csd(signals, *, title: str = "CSD proxy (depth x time)"):
    """Depth x time heatmap of the CSD proxy.

    Proxy readout only (linear-solver, ``*_proxy`` field) — not a physical
    current source density.
    """
    require_plotly()
    import plotly.graph_objects as go

    csd = field_proxy(signals, "csd")  # (n_steps, n_contacts)
    depths = contact_depths_mm(signals)
    t = time_ms(signals)
    order = np.argsort(depths)

    fig = go.Figure(data=go.Heatmap(
        z=csd[:, order].T, x=t, y=depths[order],
        colorscale="RdBu", zmid=0.0,
        colorbar=dict(title="CSD proxy"),
    ))
    fig.update_layout(title=title, xaxis_title="time (ms)", yaxis_title="depth (mm)")
    return fig
