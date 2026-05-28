"""Tutorial utility functions for Suite No. 1 and other examples.

This module provides reusable helper functions for notebook-based tutorials,
including simulation summary displays, visualization utilities, and data
processing helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class ConfigSummary:
    """Summary of configured model."""
    name: str
    n_neurons: int
    layers: List[str]
    cell_types: Dict[str, float]
    connectivity: str
    emitter_family: str
    emitter_preset: str
    probes: List[str]


def save_png(fig, name: str, fig_dir: Path, show: bool = False) -> str:
    """Save figure to PNG and return path."""
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return str(path)


def finite_status(*arrays) -> bool:
    """Check if all arrays contain only finite values."""
    return all(bool(np.all(np.isfinite(np.asarray(a)))) for a in arrays if a is not None)


def population_rate_hz(spikes: np.ndarray, dt_ms: float) -> float:
    """Compute mean population firing rate in Hz."""
    spikes = np.asarray(spikes)
    return float(spikes.mean() * (1000.0 / dt_ms)) if spikes.size > 0 else 0.0


def display_run_summary(label: str, spikes: np.ndarray, V_m: np.ndarray,
                       dt_ms: float, finite: bool) -> None:
    """Display simulation summary as a clean table."""
    rate_hz = population_rate_hz(spikes, dt_ms)
    print(f"\n{label}:")
    print(f"  Spikes: {int(spikes.sum())} | Shape: {spikes.shape} | Rate: {rate_hz:.2f} Hz")
    print(f"  Voltage: [{V_m.min():.1f}, {V_m.max():.1f}] mV | Finite: {finite}")


def plot_raster(spike_times_list, spike_neuron_ids_list, t, figsize=(10, 4),
               title="Population Raster", show: bool = True):
    """Plot spike raster from list of spike times per neuron."""
    fig, ax = plt.subplots(figsize=figsize)
    for neuron_id, (times, ids) in enumerate(zip(spike_times_list, spike_neuron_ids_list)):
        ax.scatter(times, ids, s=2, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_xlim(t.min(), t.max())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_population_rate(t, spikes, bin_ms=25.0, dt_ms=0.1, figsize=(10, 3),
                        title="Population Rate", show: bool = True) -> Tuple:
    """Plot time-binned population firing rate."""
    bin_edges = np.arange(0, t.max() + bin_ms, bin_ms)
    rates = [float(spikes[(t >= lo) & (t < hi)].mean() * (1000.0 / dt_ms))
             for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), rates, lw=1.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_title(title)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, rates


def plot_voltage_samples(t, V_m, title="Voltage trajectory", figsize=(10, 3),
                        max_neurons=10, show: bool = True):
    """Plot voltage time series from first N neurons."""
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(min(max_neurons, V_m.shape[1])):
        ax.plot(t, V_m[:, i], lw=0.8, alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V-like state (mV)")
    ax.set_title(title)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_connectivity_matrix(W, title="Connectivity matrix", figsize=(5, 5), show: bool = True):
    """Plot connectivity weight matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(W, aspect="auto", cmap="RdBu", vmin=-W.max(), vmax=W.max())
    ax.set_xlabel("Sending neuron")
    ax.set_ylabel("Receiving neuron")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_laminar_readout(t, lfp_proxy, csd_proxy=None, figsize=(12, 4),
                        title="Laminar Readout", show: bool = True):
    """Plot LFP-proxy and optionally CSD-proxy."""
    if csd_proxy is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(t, lfp_proxy[:, :4], lw=0.8)
        axes[0].set_title("LFP-proxy (first 4 contacts)")
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Proxy units")
        axes[1].imshow(csd_proxy.T, aspect="auto", origin="upper", cmap="RdBu")
        axes[1].set_title("CSD-proxy heatmap")
        axes[1].set_xlabel("Time index")
        axes[1].set_ylabel("Contact")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, lfp_proxy[:, :4], lw=0.8)
        ax.set_title("LFP-proxy")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Proxy units")
    fig.suptitle(title)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
