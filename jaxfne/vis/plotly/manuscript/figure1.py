"""Figure 1 — Model validation: raster, population rate, V_m, LFP proxy sanity."""
from __future__ import annotations

from .._common import require_plotly
from ..raster import plot_raster, plot_population_rates, plot_membrane_potentials
from ..lfp import plot_lfp


def build(signals, model, *, title: str = "Figure 1 — Model validation"):
    """Spike raster, population rate, V_m traces, LFP proxy — basic simulated-activity sanity checks."""
    require_plotly()
    from plotly.subplots import make_subplots

    raster = plot_raster(signals, model, title="Spike raster")
    rates = plot_population_rates(signals, model, title="Population rate")
    vm = plot_membrane_potentials(signals, model, title="Membrane potential (proxy)")
    lfp = plot_lfp(signals, title="LFP proxy")

    fig = make_subplots(rows=2, cols=2,
                         subplot_titles=("Spike raster", "Population rate",
                                         "Membrane potential (proxy)", "LFP proxy"))
    for trace in raster.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in rates.data:
        fig.add_trace(trace, row=1, col=2)
    for trace in vm.data:
        fig.add_trace(trace, row=2, col=1)
    for trace in lfp.data:
        fig.add_trace(trace, row=2, col=2)

    fig.update_layout(title=title, height=900, width=1300, showlegend=False)
    return fig
