"""LFP-proxy stacked-trace visualization (Plotly)."""
from __future__ import annotations

import numpy as np

from ._common import require_plotly, field_proxy, contact_depths_mm, time_ms
from ..layout import cumulative_stack_offsets


def plot_lfp(signals, *, contacts=None, offset_scale: float = 1.5, gap_frac: float = 0.15,
              title: str = "LFP proxy (stacked by depth)"):
    """Stacked LFP-proxy traces, one per contact, ordered by depth.

    Stacking uses relative accumulative coordination
    (:func:`jaxfne.vis.layout.cumulative_stack_offsets`): each trace's offset
    is the previous trace's offset plus the previous trace's own measured
    extent plus ``gap_frac`` of it — not a fixed global step — so traces with
    unusually large amplitude never overlap their neighbor. ``offset_scale``
    additionally widens every gap uniformly if more separation is wanted.

    Proxy readout only (``field_claim_level=proxy_readout``); not a real LFP.
    """
    require_plotly()
    import plotly.graph_objects as go

    lfp = field_proxy(signals, "lfp")  # (n_steps, n_contacts)
    depths = contact_depths_mm(signals)
    t = time_ms(signals)

    order = np.argsort(depths)
    if contacts is not None:
        order = order[np.isin(order, contacts)]

    series_list = [lfp[:, c] for c in order]
    offsets = cumulative_stack_offsets(series_list, gap_frac=gap_frac)

    fig = go.Figure()
    for rank, (c, offset) in enumerate(zip(order, offsets)):
        fig.add_trace(go.Scatter(
            x=t, y=lfp[:, c] + offset * offset_scale, mode="lines",
            name=f"depth={depths[c]:.3f}mm",
            hovertemplate="t=%{x:.1f}ms<br>depth=" + f"{depths[c]:.3f}mm" + "<extra></extra>",
        ))

    fig.update_layout(title=title, xaxis_title="time (ms)", yaxis_title="LFP proxy (a.u., stacked)")
    return fig
