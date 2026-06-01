"""3D circuit and geometry plotting submodules for jaxfne/vis.

NumPy-isolated graphics for 3D neuron circuit geometries.

Public API (stable):
  circuit3d(...)        – matplotlib 3D circuit layout
  geometry3d(...)       – matplotlib 3D scatter (stable fallback)
  column_geometry(...)  – matplotlib alias for geometry3d
  visualize_network_3d(...) – interactive Plotly 3D (lazy Plotly import)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .core import prepare_static_plot_matrix, require_matplotlib


def _neuron_rows(signals: Any) -> list[dict[str, Any]]:
    meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
    rows = meta.get("neuron_metadata") if isinstance(meta, dict) else None
    return [dict(row) for row in rows] if rows else []


def _geometry3d_from_config(cfg: Any, *, areas=None, cell_types=None, figsize=(9, 7)) -> Any:
    """Internal: synthesise 3D geometry scatter from Configuration metadata."""
    import matplotlib.pyplot as plt
    meta = cfg.metadata
    columns = meta.get("columns", [])
    ct_fracs = meta.get("cell_types", {})
    if isinstance(ct_fracs, dict) and not ct_fracs:
        ct_fracs = {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07}
    all_cell_types = list(ct_fracs.keys()) if isinstance(ct_fracs, dict) else ["E", "PV", "SST", "VIP"]
    if cell_types is not None:
        all_cell_types = [c for c in all_cell_types if c in cell_types]

    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    colors_map = {ct: plt.cm.Set1(i / max(len(all_cell_types), 1)) for i, ct in enumerate(all_cell_types)}

    for col_idx, col in enumerate(columns):
        if areas is not None and col.get("name") not in areas:
            continue
        n = int(col.get("n", 50))
        x_off = col_idx * 0.6  # offset columns laterally
        for ct in all_cell_types:
            frac = float((ct_fracs if isinstance(ct_fracs, dict) else {}).get(ct, 0.25))
            n_ct = max(1, int(n * frac))
            x = rng.uniform(0, 0.5, n_ct) + x_off
            y = rng.uniform(0, 0.5, n_ct)
            z = rng.uniform(0, 1.6, n_ct)
            ax.scatter(x, y, z, s=6, alpha=0.55, label=ct if col_idx == 0 else "",
                       color=colors_map.get(ct, "gray"))

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=colors_map.get(ct, "gray"), markersize=8, label=ct)
               for ct in all_cell_types]
    ax.legend(handles=handles, title="Cell type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("Declared 3D geometry (Configuration proxy)", fontsize=11)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z — laminar depth")
    fig.tight_layout()
    return fig


def circuit3d(signals: Any, **kwargs: Any) -> Any:
    """Plot declared 3D neuron layout from signal metadata."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    rows = _neuron_rows(signals)
    if not rows:
        raise ValueError("neuron_metadata is required for circuit3d")
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111, projection="3d")
    x = np.asarray([float(row.get("x", 0.0)) for row in rows])
    y = np.asarray([float(row.get("y", 0.0)) for row in rows])
    z = np.asarray([float(row.get("z", 0.0)) for row in rows])
    ax.scatter(x, y, z, s=6)
    ax.set_title("Declared 3D circuit layout")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm or relative depth)")
    return fig


def geometry3d(
    signals_or_cfg: Any,
    *,
    areas: list[str] | None = None,
    cell_types: list[str] | None = None,
    figsize: tuple[float, float] = (9, 7),
    **kwargs: Any,
) -> Any:
    """Plot declared 3D neuron geometry (x, y, z positions by cell type)."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    rows = _neuron_rows(signals_or_cfg)

    if not rows:
        if hasattr(signals_or_cfg, "metadata"):
            meta = signals_or_cfg.metadata
            columns = meta.get("columns", [])
            if columns:
                fig = _geometry3d_from_config(signals_or_cfg, areas=areas, cell_types=cell_types,
                                               figsize=figsize)
                return fig
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
        ax.text(0.5, 0.5, 0.5, "neuron_metadata required\nfor geometry3d",
                ha="center", va="center")
        ax.set_title("Declared 3D geometry (proxy)")
        return fig

    x_vals = np.asarray([float(row.get("x", 0.0)) for row in rows])
    y_vals = np.asarray([float(row.get("y", 0.0)) for row in rows])
    z_vals = np.asarray([float(row.get("z", 0.0)) for row in rows])
    ct_vals = [str(row.get("cell_type", "unknown")) for row in rows]
    area_vals = [str(row.get("area", "")) for row in rows]

    if cell_types is None:
        cell_types = sorted(set(ct_vals))
    if areas is not None:
        keep = np.asarray([a in areas for a in area_vals])
        x_vals, y_vals, z_vals = x_vals[keep], y_vals[keep], z_vals[keep]
        ct_vals = [ct for ct, k in zip(ct_vals, keep) if k]

    colors_map = {ct: plt.cm.Set1(i / max(len(cell_types), 1))
                  for i, ct in enumerate(cell_types)}

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    for ct in cell_types:
        mask = np.asarray([v == ct for v in ct_vals])
        if not mask.any():
            continue
        ax.scatter(x_vals[mask], y_vals[mask], z_vals[mask],
                   s=8, alpha=0.6, label=ct, color=colors_map.get(ct, "gray"))

    ax.set_title("Declared 3D circuit geometry (proxy)", fontsize=11)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z — laminar depth (mm or relative)")
    ax.legend(title="Cell type", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig


def column_geometry(signals_or_cfg: Any, **kwargs: Any) -> Any:
    """Plot declared 3D column geometry (alias)."""
    return geometry3d(signals_or_cfg, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Plotly 3D visualization (additive; does NOT replace matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

_UNIT_SCALE: dict[str, float] = {
    "m": 1.0,
    "mm": 1.0e3,
    "um": 1.0e6,
    "nm": 1.0e9,
}

_DEFAULT_CELL_COLORS: dict[str, str] = {
    "E":   "#e69500",
    "PV":  "#0072ce",
    "SST": "#ffbf00",
    "VIP": "#7b3294",
}

_FALLBACK_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
]


def _require_plotly():
    """Lazy-import Plotly; raise with a clear install message if missing."""
    try:
        import plotly  # noqa: F401
        import plotly.graph_objects as go  # noqa: F401
        return go
    except ImportError:
        raise ImportError(
            "Plotly is required for interactive 3D visualization.\n"
            'Install visualization extras with: pip install "jaxfne[viz]"'
        ) from None


def _coerce_to_rows(
    obj: Any,
    *,
    seed: int = 0,
    jitter_threshold: float = 1e-9,
) -> list[dict[str, Any]]:
    """Normalize all supported input types to a list of row dicts.

    Each row is guaranteed to have:
        node_id, x_m, y_m, z_m, cell_type, layer, area, jittered

    Supports:
    - dict with ``positions_m`` key (shape N×1, N×2, or N×3)
    - pandas DataFrame with ``x_m``, ``y_m``, ``z_m`` columns
    - list of dicts (records)
    - jaxfne Model via ``model.neuron_table()``
    - jaxfne Signals via ``signals.metadata["neuron_metadata"]``
    - networkx-like graph via ``nodes(data=True)`` + ``edges()``
    """
    rows: list[dict[str, Any]] = []

    # ── jaxfne Model ──────────────────────────────────────────────────────────
    if hasattr(obj, "neuron_table"):
        raw = obj.neuron_table()
        for i, r in enumerate(raw):
            rows.append({
                "node_id":   r.get("neuron_id", i),
                "x_m":       float(r.get("x", 0.0)),
                "y_m":       float(r.get("y", 0.0)),
                "z_m":       float(r.get("z", 0.0)),
                "cell_type": str(r.get("cell_type", "unknown")),
                "layer":     str(r.get("layer", "")),
                "area":      str(r.get("area", "")),
                "jittered":  False,
            })
        return _apply_jitter(rows, seed=seed, threshold=jitter_threshold)

    # ── jaxfne Signals ─────────────────────────────────────────────────────────
    if hasattr(obj, "metadata") and isinstance(getattr(obj, "metadata", None), dict):
        meta = obj.metadata
        if "neuron_metadata" in meta:
            for i, r in enumerate(meta["neuron_metadata"]):
                rows.append({
                    "node_id":   r.get("neuron_id", i),
                    "x_m":       float(r.get("x", 0.0)),
                    "y_m":       float(r.get("y", 0.0)),
                    "z_m":       float(r.get("z", 0.0)),
                    "cell_type": str(r.get("cell_type", "unknown")),
                    "layer":     str(r.get("layer", "")),
                    "area":      str(r.get("area", "")),
                    "jittered":  False,
                })
            return _apply_jitter(rows, seed=seed, threshold=jitter_threshold)

    # ── pandas DataFrame ───────────────────────────────────────────────────────
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            for i, row in obj.iterrows():
                rows.append(_row_from_flat(row.to_dict(), i))
            return _apply_jitter(rows, seed=seed, threshold=jitter_threshold)
    except ImportError:
        pass

    # ── dict with positions_m ─────────────────────────────────────────────────
    if isinstance(obj, dict) and "positions_m" in obj:
        pos = np.asarray(obj["positions_m"], dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(-1, 1)
        ids      = obj.get("node_ids", list(range(len(pos))))
        ctypes   = obj.get("cell_types", ["unknown"] * len(pos))
        layers   = obj.get("layers", [""] * len(pos))
        areas    = obj.get("areas", [""] * len(pos))
        for i in range(len(pos)):
            xyz = _promote_to_3d(pos[i])
            rows.append({
                "node_id":   ids[i] if i < len(ids) else i,
                "x_m":       float(xyz[0]),
                "y_m":       float(xyz[1]),
                "z_m":       float(xyz[2]),
                "cell_type": str(ctypes[i] if i < len(ctypes) else "unknown"),
                "layer":     str(layers[i] if i < len(layers) else ""),
                "area":      str(areas[i] if i < len(areas) else ""),
                "jittered":  False,
            })
        return _apply_jitter(rows, seed=seed, threshold=jitter_threshold)

    # ── dict with flat x_m/y_m/z_m ────────────────────────────────────────────
    if isinstance(obj, dict) and any(k in obj for k in ("x_m", "x", "positions")):
        rows.append(_row_from_flat(obj, 0))
        return _apply_jitter(rows, seed=seed, threshold=jitter_threshold)

    # ── list of dicts (records) ───────────────────────────────────────────────
    if isinstance(obj, (list, tuple)) and all(isinstance(r, dict) for r in obj):
        for i, r in enumerate(obj):
            rows.append(_row_from_flat(r, i))
        return _apply_jitter(rows, seed=seed, threshold=jitter_threshold)

    # ── networkx-like graph ───────────────────────────────────────────────────
    if hasattr(obj, "nodes") and hasattr(obj, "edges") and callable(obj.nodes):
        node_data = dict(obj.nodes(data=True))
        for nid, attrs in node_data.items():
            flat = dict(attrs or {})
            flat.setdefault("node_id", nid)
            rows.append(_row_from_flat(flat, nid))
        return _apply_jitter(rows, seed=seed, threshold=jitter_threshold)

    raise TypeError(
        f"visualize_network_3d: unsupported input type {type(obj).__name__}. "
        "Pass a jaxfne Model, Signals, DataFrame, list-of-dicts, or dict with "
        "'positions_m' key."
    )


def _promote_to_3d(coords: np.ndarray) -> np.ndarray:
    """Promote 1D or 2D coordinate array to 3D."""
    coords = np.asarray(coords, dtype=float).ravel()
    if len(coords) >= 3:
        return coords[:3]
    if len(coords) == 2:
        return np.array([coords[0], coords[1], 0.0])
    return np.array([coords[0], 0.0, 0.0])


def _row_from_flat(d: dict, fallback_id: Any) -> dict[str, Any]:
    """Build a normalized row from a flat dict with flexible key names."""
    nid = d.get("node_id", d.get("neuron_id", d.get("id", fallback_id)))
    x = float(d.get("x_m", d.get("x", 0.0)))
    y = float(d.get("y_m", d.get("y", 0.0)))
    z = float(d.get("z_m", d.get("z", 0.0)))
    # Handle positions_m or pos keys inside flat dict
    if "positions_m" in d:
        xyz = _promote_to_3d(np.asarray(d["positions_m"], dtype=float))
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return {
        "node_id":   nid,
        "x_m":       x,
        "y_m":       y,
        "z_m":       z,
        "cell_type": str(d.get("cell_type", "unknown")),
        "layer":     str(d.get("layer", "")),
        "area":      str(d.get("area", "")),
        "jittered":  False,
    }


def _apply_jitter(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Deterministically jitter near-duplicate positions."""
    if not rows:
        return rows
    positions = np.array([[r["x_m"], r["y_m"], r["z_m"]] for r in rows])
    rng = np.random.default_rng(seed)
    for i in range(len(positions)):
        diffs = np.linalg.norm(positions[:i] - positions[i], axis=1) if i > 0 else np.array([])
        if i > 0 and np.any(diffs < threshold):
            j = rng.uniform(-threshold * 10, threshold * 10, 3)
            positions[i] += j
            rows[i]["x_m"] += j[0]
            rows[i]["y_m"] += j[1]
            rows[i]["z_m"] += j[2]
            rows[i]["jittered"] = True
    return rows


def visualize_network_3d(
    data: Any,
    *,
    title: str = "Network 3D visualization",
    coordinate_unit: str = "m",
    display_unit: str = "um",
    show_layers: bool = False,
    show_column_shells: bool = False,
    show_edges: bool = False,
    max_edges: int = 500,
    seed: int = 0,
    output_html: str | Path | None = None,
    return_node_table: bool = False,
    cell_type_colors: dict[str, str] | None = None,
) -> Any:
    """Interactive Plotly 3D network visualization.

    Parameters
    ----------
    data : Model | Signals | DataFrame | list[dict] | dict
        Network data. Supported types: jaxfne Model (via ``neuron_table()``),
        jaxfne Signals (via ``metadata["neuron_metadata"]``), pandas DataFrame
        with ``x_m``/``y_m``/``z_m`` columns, list of dicts, or dict with
        ``positions_m`` key. 1D/2D coordinates are promoted to 3D.
    title : str
        Figure title. Default: "Network 3D visualization".
    coordinate_unit : str
        Unit of stored coordinates (``"m"``, ``"mm"``, ``"um"``, ``"nm"``).
    display_unit : str
        Unit for display axes (``"m"``, ``"mm"``, ``"um"``, ``"nm"``).
    show_layers : bool
        Draw translucent horizontal planes at inferred layer z-boundaries.
    show_column_shells : bool
        Draw bounding-box wireframes per area/column.
    show_edges : bool
        Draw edges if data provides them (networkx graph only).
    max_edges : int
        Maximum number of edges to render (capped for performance).
    seed : int
        PRNG seed for deterministic duplicate-position jitter.
    output_html : str | Path | None
        If provided, write an interactive HTML file to this path.
    return_node_table : bool
        If True, return ``(fig, rows)`` instead of just ``fig``.
    cell_type_colors : dict | None
        Override default cell-type color map.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    rows : list[dict]  (only when ``return_node_table=True``)

    Notes
    -----
    - Requires Plotly: ``pip install "jaxfne[viz]"``
    - Geometry is proxy/declared only. Not calibrated anatomical coordinates.
    - truth_mode: truth_safe_unverified; physical_amplitude_claim_allowed: false
    """
    go = _require_plotly()

    # ── Normalize input ────────────────────────────────────────────────────────
    rows = _coerce_to_rows(data, seed=seed)
    if not rows:
        raise ValueError("visualize_network_3d: no node data found in input.")

    # ── Coordinate scaling ─────────────────────────────────────────────────────
    from_scale = _UNIT_SCALE.get(coordinate_unit, 1.0)
    to_scale   = _UNIT_SCALE.get(display_unit, 1.0)
    scale      = to_scale / from_scale  # convert stored unit → display unit

    x_disp = np.array([r["x_m"] for r in rows]) * scale
    y_disp = np.array([r["y_m"] for r in rows]) * scale
    z_disp = np.array([r["z_m"] for r in rows]) * scale

    cell_types_in_data = sorted({r["cell_type"] for r in rows})
    areas_in_data      = sorted({r["area"] for r in rows})

    colors = dict(_DEFAULT_CELL_COLORS)
    if cell_type_colors:
        colors.update(cell_type_colors)
    # Assign fallback colors for unknown cell types
    for i, ct in enumerate([c for c in cell_types_in_data if c not in colors]):
        colors[ct] = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]

    # ── Build traces ───────────────────────────────────────────────────────────
    traces: list[Any] = []

    ct_arr   = np.array([r["cell_type"] for r in rows])
    area_arr = np.array([r["area"]      for r in rows])
    id_arr   = [r["node_id"] for r in rows]
    layer_arr = [r["layer"]  for r in rows]

    for ct in cell_types_in_data:
        mask = ct_arr == ct
        hover = [
            f"id={id_arr[i]}<br>area={area_arr[i]}<br>"
            f"layer={layer_arr[i]}<br>"
            f"x={x_disp[i]:.2f} {display_unit}<br>"
            f"y={y_disp[i]:.2f} {display_unit}<br>"
            f"z={z_disp[i]:.2f} {display_unit}"
            for i in np.where(mask)[0]
        ]
        traces.append(go.Scatter3d(
            x=x_disp[mask], y=y_disp[mask], z=z_disp[mask],
            mode="markers",
            marker=dict(size=3, color=colors.get(ct, "#888888"), opacity=0.75),
            name=ct,
            text=hover,
            hoverinfo="text",
        ))

    # ── Optional: edges (networkx only) ───────────────────────────────────────
    if show_edges and hasattr(data, "edges"):
        node_pos_map = {r["node_id"]: (x_disp[i], y_disp[i], z_disp[i])
                        for i, r in enumerate(rows)}
        edge_list = list(data.edges())[:max_edges]
        ex, ey, ez = [], [], []
        for u, v in edge_list:
            if u in node_pos_map and v in node_pos_map:
                ex += [node_pos_map[u][0], node_pos_map[v][0], None]
                ey += [node_pos_map[u][1], node_pos_map[v][1], None]
                ez += [node_pos_map[u][2], node_pos_map[v][2], None]
        if ex:
            traces.append(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(color="#888888", width=1),
                opacity=0.3,
                name="edges",
                showlegend=True,
            ))

    # ── Optional: layer planes ─────────────────────────────────────────────────
    if show_layers and len(rows) > 1:
        unique_layers = sorted({r["layer"] for r in rows if r["layer"]})
        for lyr in unique_layers:
            lyr_mask = np.array([r["layer"] for r in rows]) == lyr
            if not lyr_mask.any():
                continue
            z_mid = float(z_disp[lyr_mask].mean())
            x_range = [float(x_disp.min()), float(x_disp.max())]
            y_range = [float(y_disp.min()), float(y_disp.max())]
            traces.append(go.Mesh3d(
                x=[x_range[0], x_range[1], x_range[1], x_range[0]],
                y=[y_range[0], y_range[0], y_range[1], y_range[1]],
                z=[z_mid, z_mid, z_mid, z_mid],
                opacity=0.08,
                color="lightblue",
                name=f"layer {lyr}",
                showlegend=False,
                hoverinfo="skip",
            ))

    # ── Optional: column shells (bounding box wireframes) ─────────────────────
    if show_column_shells:
        for area in areas_in_data:
            if not area:
                continue
            amask = area_arr == area
            if not amask.any():
                continue
            xlo, xhi = float(x_disp[amask].min()), float(x_disp[amask].max())
            ylo, yhi = float(y_disp[amask].min()), float(y_disp[amask].max())
            zlo, zhi = float(z_disp[amask].min()), float(z_disp[amask].max())
            # 12-edge wireframe
            vx = [xlo, xhi, xhi, xlo, xlo, None,
                  xlo, xhi, None, xhi, xhi, None,
                  xlo, xlo, None, xhi, xhi, None,
                  xlo, xhi, None, xlo, xhi, None]
            vy = [ylo, ylo, yhi, yhi, ylo, None,
                  ylo, ylo, None, ylo, yhi, None,
                  ylo, yhi, None, yhi, yhi, None,
                  yhi, yhi, None, yhi, yhi, None]
            vz = [zlo, zlo, zlo, zlo, zlo, None,
                  zhi, zhi, None, zlo, zlo, None,
                  zlo, zlo, None, zlo, zlo, None,
                  zlo, zlo, None, zhi, zhi, None]
            traces.append(go.Scatter3d(
                x=vx, y=vy, z=vz,
                mode="lines",
                line=dict(color="#aaaaaa", width=1.5),
                opacity=0.4,
                name=f"{area} shell",
                showlegend=True,
            ))

    # ── Layout ─────────────────────────────────────────────────────────────────
    ax_label = f"{display_unit}"
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis=dict(title=f"x ({ax_label})"),
                yaxis=dict(title=f"y ({ax_label})"),
                zaxis=dict(title=f"z — depth ({ax_label})"),
            ),
            legend=dict(title="Cell type", itemsizing="constant"),
            margin=dict(l=0, r=0, b=0, t=40),
        ),
    )

    # ── HTML output ────────────────────────────────────────────────────────────
    if output_html is not None:
        out = Path(output_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out))

    if return_node_table:
        return fig, rows
    return fig
