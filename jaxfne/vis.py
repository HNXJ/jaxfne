"""Visualization submodule for jaxfne.

Provides standard, high-fidelity spectrolaminar and laminar proxy visualization
functions for neurophysiology models.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from typing import Any, Optional

from .core import Signals


def require_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    try:
        import matplotlib
    except ImportError:
        raise ImportError(
            "The visualization features require the optional dependency 'matplotlib'. "
            "Please install it via `pip install matplotlib` or `pip install jaxfne[viz]`."
        )


def spectrolaminar(signals: Signals, **kwargs: Any) -> Any:
    """Generate a publication-ready, 3-panel spectrolaminar profile figure.

    Computes multi-contact Power Spectral Density (PSD) using `scipy.signal.welch`
    and renders space-time and frequency-depth laminar heatmaps.

    Parameters
    ----------
    signals : Signals
        The simulation output containing laminar field/readout arrays.
    **kwargs : Any
        Keyword arguments forwarded to `plt.figure()`.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered matplotlib Figure object.
    """
    if signals.field is None:
        raise ValueError(
            "Cannot generate spectrolaminar profile: signals.field is None. "
            "Ensure that record_fields was set to True in the Simulation config."
        )

    # Enforce/check matplotlib availability
    require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Extract signals and metadata
    time_ms = np.asarray(signals.time_ms)
    lfp = np.asarray(signals.field.lfp_proxy)
    csd = np.asarray(signals.field.csd_proxy)
    depths = np.asarray(signals.field.contact_depths)
    dt_ms = float(signals.metadata.get("dt_ms", 0.05))
    fs = 1000.0 / dt_ms  # Sampling frequency in Hz

    # Compute Power Spectral Density (PSD) via SciPy Welch for each laminar contact
    nperseg = min(256, lfp.shape[0])
    freqs, psds = signal.welch(lfp, fs=fs, axis=0, nperseg=nperseg)

    # Slice frequency axis to neurophysiologically relevant band (0 to 150 Hz)
    freq_mask = freqs <= 150.0
    freqs_masked = freqs[freq_mask]
    psd_masked = psds[freq_mask, :]  # Shape: [F_masked, n_contacts]

    # Create figure
    fig_kwargs = {"figsize": (16, 5), "facecolor": "#f8f9fa"}
    fig_kwargs.update(kwargs)
    fig = plt.figure(**fig_kwargs)

    gs = GridSpec(1, 3, wspace=0.35)

    # Common extent formatting [xmin, xmax, ymin, ymax]
    # We plot with Y-axis descending from surface (0.0) at top to deep (1.0) at bottom
    extent_time = [time_ms[0], time_ms[-1], depths[-1], depths[0]]
    extent_freq = [freqs_masked[0], freqs_masked[-1], depths[-1], depths[0]]

    # 1. Extracellular Potential Heatmap (LFP-proxy)
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(
        lfp.T,
        cmap="viridis",
        aspect="auto",
        extent=extent_time,
        origin="upper",
    )
    ax0.set_title("Extracellular Potential (LFP-proxy)", fontsize=12, fontweight="bold", pad=12)
    ax0.set_xlabel("Time (ms)", fontsize=10)
    ax0.set_ylabel("Laminar Depth (relative)", fontsize=10)
    ax0.grid(True, linestyle="--", alpha=0.3)
    cbar0 = fig.colorbar(im0, ax=ax0)
    cbar0.set_label("Potential (proxy units)", fontsize=9)

    # 2. Current Source Density Heatmap (CSD-proxy)
    ax1 = fig.add_subplot(gs[1])
    # Compute maximum absolute deviation to center the diverging colormap symmetric around 0
    csd_max = float(np.max(np.abs(csd)))
    vmin = -csd_max if csd_max > 0 else -1.0
    vmax = csd_max if csd_max > 0 else 1.0
    im1 = ax1.imshow(
        csd.T,
        cmap="RdBu_r",
        aspect="auto",
        extent=extent_time,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("Current Source Density (CSD-proxy)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Time (ms)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("CSD (proxy units)", fontsize=9)

    # 3. LFP Power Spectral Density Heatmap (Spectrolaminar Profile)
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(
        psd_masked.T,
        cmap="magma",
        aspect="auto",
        extent=extent_freq,
        origin="upper",
    )
    ax2.set_title("Spectrolaminar Power Profile (PSD)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Frequency (Hz)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("Power (dB-proxy)", fontsize=9)

    # Visual branding and positive-scope disclaimer
    fig.suptitle(
        "jaxfne Spectrolaminar Profile  |  "
        "Status: Simulated Laminar Proxy Readout",
        fontsize=11,
        color="#495057",
        fontstyle="italic",
        y=1.02,
    )

    return fig
