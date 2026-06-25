"""Connectivity-matrix visualization (Plotly)."""
from __future__ import annotations

import numpy as np

from ._common import require_plotly, neuron_table_arrays


def plot_connectivity(model, *, max_neurons: int = 300, title: str = "Recurrent connectivity (weight matrix)"):
    """Dense weight-matrix heatmap reconstructed from the model's edge list.

    Neurons are ordered by (area, layer, cell_type) so block structure is
    visible. Subsamples to ``max_neurons`` if the column is larger (dense
    NxN reconstruction is for visualization only — never used in the sim).
    """
    require_plotly()
    import plotly.graph_objects as go

    nt = neuron_table_arrays(model)
    n = nt["neuron_id"].shape[0]
    order = np.lexsort((nt["cell_type"], nt["layer"], nt["area"]))

    if n > max_neurons:
        keep = np.linspace(0, n - 1, max_neurons).astype(int)
        order = order[np.isin(order, keep)]

    rank_of = np.full(n, -1, dtype=int)
    rank_of[order] = np.arange(order.shape[0])

    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    weight = np.asarray(edges.weight)

    m = order.shape[0]
    W = np.zeros((m, m), dtype=float)
    mask = (rank_of[pre] >= 0) & (rank_of[post] >= 0)
    W[rank_of[post[mask]], rank_of[pre[mask]]] = weight[mask]

    labels = [f"{nt['area'][i]}/{nt['layer'][i]}/{nt['cell_type'][i]}" for i in order]

    fig = go.Figure(data=go.Heatmap(
        z=W, x=labels, y=labels, colorscale="RdBu", zmid=0.0,
        colorbar=dict(title="weight"),
    ))
    fig.update_layout(title=title, xaxis_title="pre-synaptic", yaxis_title="post-synaptic")
    return fig
