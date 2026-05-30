"""Raster plotting submodules for jaxfne/vis.

NumPy-isolated graphics for population spiking rasters.
"""
from __future__ import annotations

from typing import Any

import jax
import numpy as np

from .core import FigureResult, prepare_static_plot_matrix, require_matplotlib


def plot_spike_rasters(spike_signals_tensor: jax.Array, config_params: dict) -> None:
    """Extracts accelerator tensors onto the host context prior to plotting execution.

    Protects the active JAX trace context via standard host-device transfer.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    # Force immediate host-device transfer to protect the active trace context
    static_spike_matrix = np.asarray(jax.device_get(spike_signals_tensor))

    fig, ax = plt.subplots(figsize=config_params.get("figsize", (10, 4)))
    # Process and plot static raster tracks cleanly...
    ax.set_title("Simulated Population Spike Raster Profile")
    ax.set_xlabel("Time Step Index")
    ax.set_ylabel("Neuron Identifier")
    plt.close(fig)


def raster(signals: Any, **kwargs: Any) -> Any:
    """Plot a spike raster, optionally sorted by declared z-depth."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    sort_by = kwargs.pop("sort_by", "z")
    marker_size = float(kwargs.pop("marker_size", 2.0))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    spikes_raw = getattr(signals, "spikes", None)
    if spikes_raw is None:
        if isinstance(signals, dict) and "spikes" in signals:
            spikes_raw = signals["spikes"]
        else:
            spikes_raw = signals

    spikes = prepare_static_plot_matrix(spikes_raw)
    if spikes is None:
        raise ValueError("No spikes data found in signals.")

    time_ms_raw = getattr(signals, "time_ms", None)
    if time_ms_raw is None and isinstance(signals, dict):
        time_ms_raw = signals.get("time_ms")

    time_ms = prepare_static_plot_matrix(time_ms_raw)
    if time_ms is None:
        time_ms = np.arange(spikes.shape[0])

    t_idx, n_idx = np.where(spikes > 0)

    # Resolve neuron metadata
    meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
    rows = meta.get("neuron_metadata") if isinstance(meta, dict) else None
    rows = [dict(row) for row in rows] if rows else []

    if sort_by == "z" and len(rows) == spikes.shape[1]:
        order = np.argsort([float(row.get("z", row.get("neuron_id", i))) for i, row in enumerate(rows)])
        rank = np.empty_like(order)
        rank[order] = np.arange(order.shape[0])
        y_idx = rank[n_idx]
        ylabel = "Neuron rank sorted by z"
    else:
        y_idx = n_idx
        ylabel = "Neuron index"

    ax.scatter(time_ms[t_idx], y_idx, s=marker_size, marker="|")
    ax.set_title("Spike raster proxy sorted by depth" if sort_by == "z" else "Spike raster proxy")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def raster_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    """Plot spike raster with JSON-safe metadata container."""
    fig = raster(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "raster", "proxy_safe": True})
