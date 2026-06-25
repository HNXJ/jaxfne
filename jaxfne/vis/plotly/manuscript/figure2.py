"""Figure 2 — Spectrolaminar motif: CSD proxy, band power vs depth, depth profile."""
from __future__ import annotations

from .._common import require_plotly
from ..csd import plot_csd
from ..spectra import plot_band_power, plot_depth_profile


def build(signals, model=None, *, title: str = "Figure 2 — Spectrolaminar motif"):
    """CSD-proxy heatmap + alpha/beta-vs-gamma band power + alpha/beta depth profile.

    Proxy readout only (linear-solver field); a relative-power motif, not a
    physical current source density.
    """
    require_plotly()
    from plotly.subplots import make_subplots

    csd = plot_csd(signals, title="CSD proxy (depth x time)")
    band = plot_band_power(signals, title="Band power vs depth")
    depth = plot_depth_profile(signals, title="Alpha/beta depth profile")

    fig = make_subplots(rows=1, cols=3,
                         subplot_titles=("CSD proxy (depth x time)", "Band power vs depth",
                                         "Alpha/beta depth profile"))
    for trace in csd.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in band.data:
        fig.add_trace(trace, row=1, col=2)
    for trace in depth.data:
        fig.add_trace(trace, row=1, col=3)

    fig.update_layout(title=title, height=500, width=1500, showlegend=True)
    return fig
