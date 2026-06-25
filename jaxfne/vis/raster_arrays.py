"""Spike-raster plotting from raw arrays (no Signals/Model object needed).

Moved from ``jaxfne.export`` — for the Signals/Model-driven raster use
``jaxfne.vis.raster`` (matplotlib) or ``jaxfne.vis.plot_raster`` (canonical,
plotly by default).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def raster_from_arrays(spike_times: np.ndarray, spike_ids: np.ndarray,
                        time_ms: np.ndarray, figsize: Tuple[float, float] = (10, 4),
                        title: str = "Spike Raster") -> object:
    """Plot a spike raster directly from spike-time/spike-id arrays.

    Parameters
    ----------
    spike_times : ndarray
        Spike times in milliseconds
    spike_ids : ndarray
        Neuron IDs for each spike
    time_ms : ndarray
        Time axis for axis limits

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .core import require_matplotlib
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(spike_times, spike_ids, s=2, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron ID")
    ax.set_title(title)
    ax.set_xlim(time_ms.min(), time_ms.max())
    return fig
