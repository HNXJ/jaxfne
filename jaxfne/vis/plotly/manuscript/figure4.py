"""Figure 4 — Hierarchical manipulations: band power comparison across conditions."""
from __future__ import annotations

from .._common import require_plotly
from ..spectra import plot_band_power


def build(signals_by_condition, model=None, *, title: str = "Figure 4 — Hierarchical manipulations"):
    """Overlay band-power-vs-depth across feedforward/feedback gain conditions.

    ``signals_by_condition`` maps a condition label (e.g. ``"baseline"``,
    ``"FF_boosted"``, ``"FB_boosted"``, ``"FF_FB_boosted"``) to a
    :class:`jaxfne.Signals`. If only a single ``Signals`` is passed (no
    multi-condition sweep available yet), it is shown alone as ``"baseline"``.
    """
    require_plotly()
    from plotly.subplots import make_subplots

    if not isinstance(signals_by_condition, dict):
        signals_by_condition = {"baseline": signals_by_condition}

    fig = make_subplots(rows=1, cols=len(signals_by_condition),
                         subplot_titles=list(signals_by_condition.keys()))
    for col, (label, sig) in enumerate(signals_by_condition.items(), start=1):
        sub = plot_band_power(sig, title="")
        for trace in sub.data:
            trace.legendgroup = label
            trace.showlegend = col == 1
            fig.add_trace(trace, row=1, col=col)

    fig.update_layout(title=title, height=500, width=max(500, 450 * len(signals_by_condition)))
    return fig
