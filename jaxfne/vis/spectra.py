"""Spectral plotting submodules for jaxfne/vis.

NumPy-isolated graphics for power spectral density (PSD) and spectrograms.
"""
from __future__ import annotations

from typing import Any

import jax
import numpy as np
from scipy import signal

from .core import FigureResult, prepare_static_plot_matrix, require_matplotlib


def plot_spectrogram_profiles(lfp_signals_tensor: jax.Array, time_steps: np.ndarray) -> None:
    """Extracts accelerator tensors onto the host context prior to plotting execution.

    Protects the active JAX trace context via standard host-device transfer.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    # Force immediate host-device transfer to protect the active trace context
    static_lfp_matrix = np.asarray(jax.device_get(lfp_signals_tensor))

    fig, ax = plt.subplots(figsize=(10, 4))
    # Process and plot spectrogram profiles cleanly...
    ax.set_title("Simulated LFP Spectrogram Profiles")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    plt.close(fig)


def _get_time_ms(signals: Any, default_len: int) -> np.ndarray:
    time_raw = getattr(signals, "time_ms", None)
    if time_raw is None and isinstance(signals, dict):
        time_raw = signals.get("time_ms")
    time_ms = prepare_static_plot_matrix(time_raw)
    if time_ms is None:
        return np.arange(default_len)
    return time_ms


def psd(signals: Any, **kwargs: Any) -> Any:
    """Plot Power Spectral Density."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    if hasattr(signals, "field") and signals.field is not None:
        data = prepare_static_plot_matrix(signals.field.lfp_proxy)
        if dt_ms is None:
            dt_ms = float(signals.metadata.get("dt_ms", 0.1))
    elif isinstance(signals, dict) and "lfp_proxy" in signals:
        data = prepare_static_plot_matrix(signals["lfp_proxy"])
        if dt_ms is None:
            meta = signals.get("metadata", {})
            dt_ms = float(meta.get("dt_ms", 0.1)) if isinstance(meta, dict) else 0.1
    elif hasattr(signals, "lfp_proxy"):
        data = prepare_static_plot_matrix(signals.lfp_proxy)
        if dt_ms is None:
            dt_ms = 0.1
    else:
        data = prepare_static_plot_matrix(signals)
        if dt_ms is None:
            dt_ms = 0.1

    fs = 1000.0 / dt_ms
    nperseg = min(256, data.shape[0])
    freqs, psds = signal.welch(data, fs=fs, axis=0, nperseg=nperseg)

    if psds.ndim > 1:
        mean_psd = np.mean(psds, axis=1)
    else:
        mean_psd = psds

    ax.loglog(freqs, mean_psd, c="#7048e8", lw=1.5)
    ax.set_title("Simulated LFP Power Spectral Density Proxy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB-proxy)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    return fig


def psd_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    """Plot PSD with JSON-safe metadata container."""
    fig = psd(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "psd", "proxy_safe": True})


def spectrogram(signals: Any, **kwargs: Any) -> Any:
    """Plot Spectrogram heatmap."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    if hasattr(signals, "field") and signals.field is not None:
        data = prepare_static_plot_matrix(signals.field.lfp_proxy)
        if dt_ms is None:
            dt_ms = float(signals.metadata.get("dt_ms", 0.1))
    elif isinstance(signals, dict) and "lfp_proxy" in signals:
        data = prepare_static_plot_matrix(signals["lfp_proxy"])
        if dt_ms is None:
            meta = signals.get("metadata", {})
            dt_ms = float(meta.get("dt_ms", 0.1)) if isinstance(meta, dict) else 0.1
    elif hasattr(signals, "lfp_proxy"):
        data = prepare_static_plot_matrix(signals.lfp_proxy)
        if dt_ms is None:
            dt_ms = 0.1
    else:
        data = prepare_static_plot_matrix(signals)
        if dt_ms is None:
            dt_ms = 0.1

    fs = 1000.0 / dt_ms
    x = data[:, 0] if data.ndim > 1 else data
    nperseg = min(128, x.shape[0])
    f, t_spec, Sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg)

    im = ax.pcolormesh(t_spec * 1000.0, f, 10 * np.log10(Sxx + 1e-10), cmap="magma", shading="gouraud")
    ax.set_title("Simulated Spectrogram Time-Frequency Proxy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    return fig


def spectrogram_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    """Plot spectrogram with JSON-safe metadata container."""
    fig = spectrogram(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "spectrogram", "proxy_safe": True})


def windowed_band_power(
    window_labels: Any,
    band_values: dict,
    **kwargs: Any,
) -> Any:
    """Plot a grouped bar chart of named proxy band-power values across labeled windows.

    Parameters
    ----------
    window_labels:
        Sequence of window names (e.g. ``["baseline", "event", "post"]``).
    band_values:
        Mapping of band name -> sequence of proxy power values aligned with
        ``window_labels`` (e.g. ``{"alpha_beta_proxy_power": [...], "gamma_proxy_power": [...]}``).
        One subplot is generated per band.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    figsize = kwargs.pop("figsize", (6 * max(len(band_values), 1), 5))
    colors = kwargs.pop("colors", None)

    labels = list(window_labels)
    band_names = list(band_values.keys())
    n_bands = max(len(band_names), 1)

    fig, axes = plt.subplots(1, n_bands, figsize=figsize)
    if n_bands == 1:
        axes = [axes]

    default_colors = ["steelblue", "coral", "seagreen", "purple"]
    for i, band_name in enumerate(band_names):
        ax = axes[i]
        values = prepare_static_plot_matrix(np.asarray(band_values[band_name], dtype=float))
        color = colors[i] if colors else default_colors[i % len(default_colors)]
        ax.bar(labels, values, color=color, alpha=0.7)
        ax.set_ylabel(band_name)
        ax.set_title(f"Windowed proxy power: {band_name}")
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig
