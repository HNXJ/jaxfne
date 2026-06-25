"""Interactive standalone dashboard assembling the Plotly primitives."""
from __future__ import annotations

from ._common import require_plotly
from .network import plot_network_3d
from .raster import plot_raster, plot_population_rates
from .lfp import plot_lfp
from .csd import plot_csd
from .spectra import plot_psd, plot_depth_profile


def create_dashboard(signals, model, *, output_html: str | None = None, title: str = "jaxfne run dashboard"):
    """Build a synchronized multi-panel dashboard (network / raster / rates / LFP / CSD / PSD / depth profile).

    Proxy readouts only — LFP/CSD/PSD panels are linear-solver proxy fields,
    not physical recordings. Returns the assembled ``plotly.graph_objects.Figure``;
    writes standalone HTML to ``output_html`` if given.
    """
    require_plotly()
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    panels = {
        "network": plot_network_3d(model, signals),
        "raster": plot_raster(signals, model),
        "rates": plot_population_rates(signals, model),
        "lfp": plot_lfp(signals),
        "csd": plot_csd(signals),
        "psd": plot_psd(signals),
        "depth_profile": plot_depth_profile(signals),
    }

    specs = [
        [{"type": "scene", "rowspan": 2}, {"type": "xy"}, {"type": "xy"}],
        [None, {"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    ]
    titles = ("Network", "Raster", "Population rate",
              "", "LFP proxy", "CSD proxy",
              "PSD proxy", "Depth profile (proxy)", "")
    fig = make_subplots(rows=3, cols=3, specs=specs, subplot_titles=titles,
                         horizontal_spacing=0.06, vertical_spacing=0.08)

    for trace in panels["network"].data:
        fig.add_trace(trace, row=1, col=1)
    for trace in panels["raster"].data:
        fig.add_trace(trace, row=1, col=2)
    for trace in panels["rates"].data:
        fig.add_trace(trace, row=1, col=3)
    for trace in panels["lfp"].data:
        fig.add_trace(trace, row=2, col=2)
    for trace in panels["csd"].data:
        fig.add_trace(trace, row=2, col=3)
    for trace in panels["psd"].data:
        fig.add_trace(trace, row=3, col=1)
    for trace in panels["depth_profile"].data:
        fig.add_trace(trace, row=3, col=2)

    fig.update_layout(title=title, height=1100, width=1500, showlegend=False)

    if output_html is not None:
        fig.write_html(output_html, include_plotlyjs="cdn")

    return fig
