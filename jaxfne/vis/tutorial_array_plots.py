"""Raw-array tutorial plot helpers, cloned from ``jaxfne.tutorial_utils``.

These operate directly on numpy arrays (no Signals/Model object required) —
distinct from the Signals/Model-driven canonical functions in
:mod:`jaxfne.vis.canonical`. ``jaxfne.tutorial_utils`` now forwards to these;
this module is the source of truth going forward.
"""
from __future__ import annotations

import numpy as np

from .layout import cumulative_stack_offsets


def _finish_figure(fig, show: bool):
    import matplotlib.pyplot as plt
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_population_raster(spike_times_list, spike_neuron_ids_list, t, figsize=(10, 4),
                            title="Population Raster", show: bool = True):
    """Spike raster from a list of (spike_times, neuron_ids) per group."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    for times, ids in zip(spike_times_list, spike_neuron_ids_list):
        ax.scatter(times, ids, s=2, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_xlim(t.min(), t.max())
    return _finish_figure(fig, show)


def plot_population_rate_array(t, spikes, bin_ms=25.0, dt_ms=0.1, figsize=(10, 3),
                                title="Population Rate", show: bool = True):
    """Time-binned population firing rate from raw spike/time arrays."""
    import matplotlib.pyplot as plt
    bin_edges = np.arange(0, t.max() + bin_ms, bin_ms)
    rates = [float(spikes[(t >= lo) & (t < hi)].mean() * (1000.0 / dt_ms))
             for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), rates, lw=1.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_title(title)
    _finish_figure(fig, show)
    return fig, rates


def plot_voltage_samples_array(t, V_m, title="Voltage trajectory", figsize=(10, 3),
                                max_neurons=10, stacked: bool = False, gap_frac: float = 0.15,
                                show: bool = True):
    """Voltage time series from the first N neurons of a raw V_m array.

    ``stacked=True`` offsets each neuron's trace using relative accumulative
    coordination (:func:`jaxfne.vis.layout.cumulative_stack_offsets`): each
    trace's vertical offset is the previous trace's offset plus the previous
    trace's own measured extent plus ``gap_frac`` — never a fixed step — so
    traces never overlap regardless of per-neuron amplitude. Default
    ``stacked=False`` overlays traces unchanged (back-compatible).
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    n = min(max_neurons, V_m.shape[1])
    if stacked:
        series_list = [V_m[:, i] for i in range(n)]
        offsets = cumulative_stack_offsets(series_list, gap_frac=gap_frac)
    else:
        offsets = [0.0] * n
    for i, offset in zip(range(n), offsets):
        ax.plot(t, V_m[:, i] + offset, lw=0.8, alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V-like state (mV, stacked)" if stacked else "V-like state (mV)")
    ax.set_title(title)
    return _finish_figure(fig, show)


def plot_connectivity_matrix_array(W, title="Connectivity matrix", figsize=(5, 5), show: bool = True):
    """Connectivity weight-matrix heatmap from a raw dense W array."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    W = np.asarray(W)
    scale = float(np.max(np.abs(W))) if W.size else 1.0
    scale = scale if scale > 0 else 1.0
    im = ax.imshow(W, aspect="auto", cmap="RdBu", vmin=-scale, vmax=scale)
    ax.set_xlabel("Sending neuron")
    ax.set_ylabel("Receiving neuron")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _finish_figure(fig, show)


def plot_laminar_readout_array(t, lfp_proxy, csd_proxy=None, figsize=(12, 4),
                                title="Laminar Readout", show: bool = True):
    """LFP-proxy (and optionally CSD-proxy) from raw arrays."""
    import matplotlib.pyplot as plt
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
    return _finish_figure(fig, show)


def plot_spectrolaminar_power_array(
    t: np.ndarray,
    signal: np.ndarray,
    freq_min: float = 1.0,
    freq_max: float = 120.0,
    n_freqs: int = 96,
    title: str = "Spectrolaminar Power",
    figsize: tuple = (10, 5),
    show: bool = True,
) -> object:
    """Spectrolaminar PSD heatmap (depth x freq) from a raw signal array."""
    import matplotlib.pyplot as plt
    n_freqs = max(64, int(n_freqs))
    freqs = np.linspace(float(freq_min), float(freq_max), n_freqs)
    dt_ms = float(t[1] - t[0]) if len(t) > 1 else 0.1
    fs = 1000.0 / dt_ms
    sig = np.asarray(signal)
    if sig.ndim == 1:
        sig = sig[:, None]
    n_contacts = sig.shape[1]
    psd = np.zeros((n_freqs, n_contacts))
    for ci in range(n_contacts):
        x = sig[:, ci]
        for fi, freq in enumerate(freqs):
            n = len(x)
            k = freq / fs
            phase = 2.0 * np.pi * k * np.arange(n)
            psd[fi, ci] = np.abs(np.dot(x, np.exp(-1j * phase))) / max(n, 1)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        psd, aspect="auto", origin="lower", cmap="viridis",
        extent=[0, n_contacts, float(freq_min), float(freq_max)],
    )
    ax.set_xlabel("Contact index")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Power proxy")
    return _finish_figure(fig, show)
