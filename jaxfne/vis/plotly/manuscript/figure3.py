"""Figure 3 — Cell-type perturbations: depth profile comparison across conditions."""
from __future__ import annotations

from .._common import require_plotly
from ..spectra import plot_depth_profile


def build(signals_by_condition, model=None, *, title: str = "Figure 3 — Cell-type perturbations"):
    """Overlay depth profiles across cell-type-suppression conditions.

    ``signals_by_condition`` maps a condition label (e.g. ``"baseline"``,
    ``"PV_suppressed"``, ``"SST_suppressed"``, ``"VIP_suppressed"``) to a
    :class:`jaxfne.Signals`. If only a single ``Signals`` is passed (no
    multi-condition sweep available yet), it is shown alone as ``"baseline"``.
    """
    require_plotly()
    import plotly.graph_objects as go

    if not isinstance(signals_by_condition, dict):
        signals_by_condition = {"baseline": signals_by_condition}

    fig = go.Figure()
    for label, sig in signals_by_condition.items():
        sub = plot_depth_profile(sig, title="")
        for trace in sub.data:
            trace.name = label
            fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis_title="relative power (alpha/beta)",
        yaxis_title="depth (mm)",
        yaxis_autorange="reversed",
    )
    return fig
