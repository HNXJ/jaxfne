"""Interactive 3D network visualization (Plotly)."""

from __future__ import annotations

import numpy as np

from ._common import require_plotly, neuron_table_arrays, color_for_cell_types


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Hex color + alpha -> ``rgba(...)`` string (Scatter3d marker.opacity is scalar-only)."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        r, g, b = 136, 136, 136
    a = min(1.0, max(0.0, float(alpha)))
    return f"rgba({r},{g},{b},{a:.3f})"


def _rate_alpha(n_units: int, signals) -> np.ndarray | None:
    """Per-neuron alpha from mean firing rate, or None without signals."""
    if signals is None or not hasattr(signals, "spikes"):
        return None
    spk = np.asarray(signals.spikes)
    if spk.ndim != 2 or spk.shape[1] != n_units:
        return None
    rate = spk.sum(axis=0)
    peak = float(rate.max()) or 1.0
    return 0.35 + 0.65 * (rate / peak)


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
    edge_color_by: str | None = None,
):
    """3D scatter of neuron positions, colored by cell type/layer/area.

    Supports rotation/zoom/hover natively (Plotly 3D scatter). If
    ``signals`` is given, per-point alpha encodes mean firing rate
    (a lightweight proxy for "spike activity" — not an animation) for
    ``color_by="cell_type"``; Scatter3d marker.opacity is scalar-only, so the
    encoding rides on per-point ``rgba`` colors (P2).

    ``edge_color_by="source_class"`` splits the sampled edges into two
    legend-toggleable traces by the presynaptic cell-type name: types starting
    with ``E`` vs the rest. This is a naming rule, not a resolved synaptic sign;
    ``None`` keeps one grey edge trace.
    """
    if edge_color_by not in (None, "source_class"):
        raise ValueError(f"edge_color_by must be None or 'source_class', got {edge_color_by!r}")
    if edge_color_by is not None and not show_edges:
        raise ValueError("edge_color_by requires show_edges=True")
    require_plotly()
    import plotly.graph_objects as go

    nt = neuron_table_arrays(model)
    z = nt["z"] * depth_multiplier
    n_units = int(nt["x"].shape[0])

    if color_by == "cell_type":
        colors = color_for_cell_types(nt["cell_type"])
        legend_groups = nt["cell_type"]
    elif color_by in ("layer", "area"):
        colors = nt[color_by]
        legend_groups = nt[color_by]
    else:
        raise ValueError(f"color_by must be one of cell_type/layer/area, got {color_by!r}")

    alpha = _rate_alpha(n_units, signals)

    fig = go.Figure()
    for group in np.unique(legend_groups):
        sel = legend_groups == group
        idx = np.where(sel)[0]
        if color_by == "cell_type":
            base = [colors[i] for i in idx]
            point_colors = (
                [_hex_to_rgba(c, float(alpha[i])) for i, c in zip(idx, base)]
                if alpha is not None
                else base
            )
        else:
            point_colors = None
        fig.add_trace(
            go.Scatter3d(
                x=nt["x"][sel],
                y=nt["y"][sel],
                z=z[sel],
                mode="markers",
                name=str(group),
                marker=dict(
                    size=point_size,
                    color=point_colors,
                    opacity=0.85,
                ),
                customdata=np.stack(
                    [nt["area"][sel], nt["layer"][sel], nt["cell_type"][sel]], axis=1
                ),
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
            idx = (
                np.linspace(0, pre.shape[0] - 1, n_show).astype(int)
                if pre.shape[0]
                else np.array([], dtype=int)
            )
            if edge_color_by is None:
                groups = [("edges", idx, "#888888", 0.2, False)]
            else:
                exc = np.char.startswith(nt["cell_type"][pre[idx]].astype(str), "E")
                groups = [
                    ("edges from E* types", idx[exc], "#6baed6", 0.35, True),
                    ("edges from other types", idx[~exc], "#f28e8e", 0.45, True),
                ]
            for name, sel_idx, color, opacity, showlegend in groups:
                xs, ys, zs = [], [], []
                for i in sel_idx:
                    p, q = int(pre[i]), int(post[i])
                    xs += [nt["x"][p], nt["x"][q], None]
                    ys += [nt["y"][p], nt["y"][q], None]
                    zs += [z[p], z[q], None]
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=opacity,
                        name=name,
                        showlegend=showlegend,
                        hoverinfo="skip" if edge_color_by else None,
                    )
                )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="depth z (mm)"),
        legend_title_text=color_by,
    )
    return fig


def plot_area_graph(model, *, n_width_classes: int = 4, title: str = "Area graph"):
    """Multi-area model as a graph: one node per area, one chord per connected area pair.

    Areas sit on a circle in declared order (clockwise from the top), sized by
    neuron count and colored by that order. A chord carries the realized
    inter-area edge count (both directions summed), binned into
    ``n_width_classes`` quantile classes that set its width and brightness;
    hovering a chord's midpoint shows both directed counts. Within-area edges
    are not drawn. The layout is schematic: positions encode order, not
    geometry.
    """
    if isinstance(n_width_classes, bool) or not isinstance(n_width_classes, int) or n_width_classes < 1:
        raise ValueError(f"n_width_classes must be a positive int, got {n_width_classes!r}")
    require_plotly()
    import plotly.graph_objects as go

    nt = neuron_table_arrays(model)
    areas = list(dict.fromkeys(nt["area"].tolist()))
    if len(areas) < 2:
        raise ValueError("plot_area_graph needs a model with at least two areas")
    edges = model.params.get("edge_list") if hasattr(model, "params") else None
    if edges is None:
        raise ValueError("plot_area_graph needs model.params['edge_list']")
    k = len(areas)
    index = {a: i for i, a in enumerate(areas)}
    area_id = np.array([index[a] for a in nt["area"]])
    size = np.bincount(area_id, minlength=k)
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (area_id[np.asarray(edges.pre)], area_id[np.asarray(edges.post)]), 1)
    np.fill_diagonal(counts, 0)
    total = counts + counts.T
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k) if total[i, j] > 0]

    angle = np.pi / 2 - 2 * np.pi * np.arange(k) / k
    pos = np.stack([np.cos(angle), np.sin(angle)], axis=1)

    fig = go.Figure()
    if pairs:
        values = np.array([total[i, j] for i, j in pairs], dtype=float)
        bounds = np.quantile(values, np.linspace(0, 1, n_width_classes + 1))
        cls = np.clip(np.searchsorted(bounds[1:-1], values, side="right"), 0, n_width_classes - 1)
        shades = np.linspace(0.25, 0.85, n_width_classes)
        for c in range(n_width_classes):
            members = [p for p, cc in zip(pairs, cls) if cc == c]
            if not members:
                continue
            xs, ys = [], []
            for i, j in members:
                xs += [pos[i, 0], pos[j, 0], None]
                ys += [pos[i, 1], pos[j, 1], None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", hoverinfo="skip",
                line=dict(color=f"rgba(154,164,178,{shades[c]:.2f})", width=0.8 + 1.8 * c),
                name=f"{int(bounds[c])}–{int(bounds[c + 1])} edges",
            ))
        mid = np.array([(pos[i] + pos[j]) / 2 for i, j in pairs])
        fig.add_trace(go.Scatter(
            x=mid[:, 0], y=mid[:, 1], mode="markers", showlegend=False,
            marker=dict(size=6, color="rgba(154,164,178,0.01)"),
            text=[f"{areas[i]}→{areas[j]}: {counts[i, j]}<br>{areas[j]}→{areas[i]}: {counts[j, i]}"
                  for i, j in pairs],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=areas, name="areas",
        textposition=["top center" if y >= 0 else "bottom center" for y in pos[:, 1]],
        marker=dict(size=14 + 14 * size / size.max(), color=np.arange(k), colorscale="Viridis",
                    line=dict(color="#e6e6e6", width=1)),
        customdata=size, hovertemplate="%{text}: %{customdata} neurons<extra></extra>",
    ))
    axis = dict(visible=False, range=[-1.3, 1.3])
    fig.update_layout(title=title, xaxis=axis, yaxis=dict(axis, scaleanchor="x"),
                      legend_title_text="inter-area edges")
    return fig
