"""Figure 0 — Background: network geometry + connectivity (Plotly)."""
from __future__ import annotations

from .._common import require_plotly
from ..network import plot_network_3d
from ..connectivity import plot_connectivity


def build(signals, model, *, title: str = "Figure 0 — Background"):
    """Network geometry + recurrent connectivity overview.

    Proxy/scaffold visualization only; no biological mechanism claim.
    """
    require_plotly()
    from plotly.subplots import make_subplots

    net = plot_network_3d(model, signals, show_edges=True, title="Network geometry")
    conn = plot_connectivity(model, title="Recurrent connectivity")

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                         subplot_titles=("Network geometry", "Recurrent connectivity"))
    for trace in net.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in conn.data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(title=title, height=600, width=1300, showlegend=False)
    return fig
