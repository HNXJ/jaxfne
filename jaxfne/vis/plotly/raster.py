"""Raster, population-rate, and membrane-potential traces (Plotly)."""
from __future__ import annotations

import numpy as np

from ._common import require_plotly, neuron_table_arrays, time_ms, dt_ms, population_rate_hz, color_for_cell_types


def plot_raster(signals, model=None, *, max_neurons: int | None = None, title: str = "Spike raster"):
    """Scatter raster: spike time vs neuron id. Colored by cell type if a model is given."""
    require_plotly()
    import plotly.graph_objects as go

    spk = np.asarray(signals.spikes)  # (n_steps, N)
    t = time_ms(signals)
    n_units = spk.shape[1]
    keep = np.arange(n_units) if max_neurons is None else np.arange(min(max_neurons, n_units))

    steps, ids = np.nonzero(spk[:, keep])
    spike_t = t[steps]
    spike_id = keep[ids]

    fig = go.Figure()
    if model is not None:
        nt = neuron_table_arrays(model)
        ct = nt["cell_type"][spike_id]
        for group in np.unique(nt["cell_type"]):
            sel = ct == group
            fig.add_trace(go.Scattergl(
                x=spike_t[sel], y=spike_id[sel], mode="markers",
                marker=dict(size=3, color=color_for_cell_types([group])[0]),
                name=str(group),
            ))
    else:
        fig.add_trace(go.Scattergl(x=spike_t, y=spike_id, mode="markers", marker=dict(size=3)))

    fig.update_layout(title=title, xaxis_title="time (ms)", yaxis_title="neuron id")
    return fig


def plot_population_rates(signals, model=None, *, bin_ms: float = 10.0, group_by: str = "cell_type",
                            title: str = "Population firing rate"):
    """Binned population rate (Hz), optionally split by cell_type/layer/area."""
    require_plotly()
    import plotly.graph_objects as go

    spk = np.asarray(signals.spikes)
    step_ms = dt_ms(signals)

    fig = go.Figure()
    if model is None:
        centers, rate = population_rate_hz(spk, step_ms, bin_ms)
        fig.add_trace(go.Scatter(x=centers, y=rate, mode="lines", name="population"))
    else:
        nt = neuron_table_arrays(model)
        groups = nt[group_by]
        for group in np.unique(groups):
            sel = groups == group
            centers, rate = population_rate_hz(spk[:, sel], step_ms, bin_ms)
            fig.add_trace(go.Scatter(x=centers, y=rate, mode="lines", name=str(group)))

    fig.update_layout(title=title, xaxis_title="time (ms)", yaxis_title="rate (Hz)")
    return fig


def plot_membrane_potentials(signals, model=None, *, neuron_ids=None, max_traces: int = 10,
                               title: str = "Membrane potential (proxy)"):
    """Stacked V_m traces for a subset of neurons."""
    require_plotly()
    import plotly.graph_objects as go

    Vm = np.asarray(signals.V_m)  # (n_steps, N)
    t = time_ms(signals)
    n_units = Vm.shape[1]

    if neuron_ids is None:
        neuron_ids = np.linspace(0, n_units - 1, min(max_traces, n_units)).astype(int)

    labels = None
    if model is not None:
        nt = neuron_table_arrays(model)
        labels = [f"#{i} {nt['cell_type'][i]}/{nt['layer'][i]}" for i in neuron_ids]

    fig = go.Figure()
    for j, i in enumerate(neuron_ids):
        fig.add_trace(go.Scatter(
            x=t, y=Vm[:, i], mode="lines",
            name=labels[j] if labels else f"#{i}",
        ))

    fig.update_layout(title=title, xaxis_title="time (ms)", yaxis_title="V_m (mV)")
    return fig
