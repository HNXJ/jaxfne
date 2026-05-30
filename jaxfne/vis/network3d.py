"""3D circuit and geometry plotting submodules for jaxfne/vis.

NumPy-isolated graphics for 3D neuron circuit geometries.
"""
from __future__ import annotations

from typing import Any

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
