"""Interactive 3D network visualization (Plotly)."""
from __future__ import annotations

import numpy as np

from ._common import require_plotly, neuron_table_arrays, color_for_cell_types


def plot_network_3d(
    model,
    signals=None,
    *,
    color_by: str = "cell_type",
    show_edges: bool = False,
    max_edges: int = 2000,
    point_size: float = 4.0,
    depth_multiplier: float = 1.0,
    title: str = "Network geometry",
):
    """3D scatter of neuron positions, colored by cell type/layer/area.

    Supports rotation/zoom/hover natively (Plotly 3D scatter). If
    ``signals`` is given, point opacity encodes mean firing rate
    (a lightweight proxy for "spike activity" — not an animation).
    """
    require_plotly()
    import plotly.graph_objects as go

    nt = neuron_table_arrays(model)
    z = nt["z"] * depth_multiplier

    if color_by == "cell_type":
        colors = color_for_cell_types(nt["cell_type"])
        legend_groups = nt["cell_type"]
    elif color_by in ("layer", "area"):
        colors = nt[color_by]
        legend_groups = nt[color_by]
    else:
        raise ValueError(f"color_by must be one of cell_type/layer/area, got {color_by!r}")

    opacity = None
    if signals is not None:
        spk = np.asarray(signals.spikes)
        rate = spk.sum(axis=0)
        peak = float(rate.max()) or 1.0
        opacity = 0.35 + 0.65 * (rate / peak)

    fig = go.Figure()
    for group in np.unique(legend_groups):
        sel = legend_groups == group
        fig.add_trace(
            go.Scatter3d(
                x=nt["x"][sel], y=nt["y"][sel], z=z[sel],
                mode="markers",
                name=str(group),
                marker=dict(
                    size=point_size,
                    color=[colors[i] for i in np.where(sel)[0]] if color_by == "cell_type" else None,
                    opacity=0.85 if opacity is None else None,
                ),
                customdata=np.stack([nt["area"][sel], nt["layer"][sel], nt["cell_type"][sel]], axis=1),
                hovertemplate=(
                    "area=%{customdata[0]} layer=%{customdata[1]} cell_type=%{customdata[2]}"
                    "<br>x=%{x:.4f} y=%{y:.4f} z=%{z:.4f}<extra></extra>"
                ),
            )
        )

    if show_edges:
        edges = model.params.get("edge_list") if hasattr(model, "params") else None
        if edges is not None:
            pre = np.asarray(edges.pre)
            post = np.asarray(edges.post)
            n_show = min(max_edges, pre.shape[0])
            idx = np.linspace(0, pre.shape[0] - 1, n_show).astype(int) if pre.shape[0] else np.array([], dtype=int)
            xs, ys, zs = [], [], []
            for i in idx:
                p, q = int(pre[i]), int(post[i])
                xs += [nt["x"][p], nt["x"][q], None]
                ys += [nt["y"][p], nt["y"][q], None]
                zs += [z[p], z[q], None]
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color="#888888", width=1),
                    opacity=0.2, name="edges", showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="depth z (mm)"),
        legend_title_text=color_by,
    )
    return fig
