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


from dataclasses import dataclass

@dataclass(frozen=True)
class FigureResult:
    """Rich container holding a matplotlib figure and JSON-safe metadata."""
    fig: Any
    metadata: dict[str, Any]


def raster(signals: Any, **kwargs: Any) -> Any:
    """Plot spike raster from signals."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    
    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    
    # Extract spikes array [T, N] or similar
    if hasattr(signals, "spikes"):
        spikes = np.asarray(signals.spikes)
        time_ms = np.asarray(signals.time_ms)
    elif isinstance(signals, dict) and "spikes" in signals:
        spikes = np.asarray(signals["spikes"])
        time_ms = np.asarray(signals.get("time_ms", np.arange(spikes.shape[0])))
    else:
        spikes = np.asarray(signals)
        time_ms = np.arange(spikes.shape[0])
        
    t_idx, n_idx = np.where(spikes > 0)
    ax.scatter(time_ms[t_idx], n_idx, s=2, c="#228be6", marker="|")
    ax.set_title("Simulated Spike Raster Proxy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron Index")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def raster_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    fig = raster(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "raster", "proxy_safe": True})


def vm(signals: Any, **kwargs: Any) -> Any:
    """Plot voltage traces over time."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    
    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    
    if hasattr(signals, "V_m"):
        v = np.asarray(signals.V_m)
        time_ms = np.asarray(signals.time_ms)
    elif isinstance(signals, dict) and "V_m" in signals:
        v = np.asarray(signals["V_m"])
        time_ms = np.asarray(signals.get("time_ms", np.arange(v.shape[0])))
    else:
        v = np.asarray(signals)
        time_ms = np.arange(v.shape[0])
        
    # Plot up to 5 neurons
    n_plot = min(5, v.shape[1] if v.ndim > 1 else 1)
    if v.ndim > 1:
        for i in range(n_plot):
            ax.plot(time_ms, v[:, i], label=f"Neuron {i}", alpha=0.8)
        ax.legend()
    else:
        ax.plot(time_ms, v)
        
    ax.set_title("Simulated Membrane Potential (Vm) Proxy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Potential (proxy mV)")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def vm_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    fig = vm(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "vm", "proxy_safe": True})


def rate(signals: Any, **kwargs: Any) -> Any:
    """Plot population firing rate."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    
    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    
    if hasattr(signals, "spikes"):
        spikes = np.asarray(signals.spikes)
        time_ms = np.asarray(signals.time_ms)
        if dt_ms is None:
            dt_ms = float(signals.metadata.get("dt_ms", 0.1))
    elif isinstance(signals, dict) and "spikes" in signals:
        spikes = np.asarray(signals["spikes"])
        time_ms = np.asarray(signals.get("time_ms", np.arange(spikes.shape[0])))
        if dt_ms is None:
            dt_ms = float(signals.get("metadata", {}).get("dt_ms", 0.1)) if isinstance(signals.get("metadata"), dict) else 0.1
    else:
        spikes = np.asarray(signals)
        time_ms = np.arange(spikes.shape[0])
        if dt_ms is None:
            dt_ms = 0.1
        
    # Population mean rate in Hz
    mean_rate = np.mean(spikes, axis=1) * (1000.0 / dt_ms)
    ax.plot(time_ms, mean_rate, c="#f03e3e", lw=1.5)
    ax.set_title("Simulated Population Mean Rate Proxy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def rate_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    fig = rate(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "rate", "proxy_safe": True})


def source(signals: Any, **kwargs: Any) -> Any:
    """Plot source current traces."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    
    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    
    if hasattr(signals, "sources"):
        s = np.asarray(signals.sources)
        time_ms = np.asarray(signals.time_ms)
    elif isinstance(signals, dict) and "sources" in signals:
        s = np.asarray(signals["sources"])
        time_ms = np.asarray(signals.get("time_ms", np.arange(s.shape[0])))
    else:
        s = np.asarray(signals)
        time_ms = np.arange(s.shape[0])
        
    n_plot = min(5, s.shape[1] if s.ndim > 1 else 1)
    if s.ndim > 1:
        for i in range(n_plot):
            ax.plot(time_ms, s[:, i], label=f"Source {i}", alpha=0.8)
        ax.legend()
    else:
        ax.plot(time_ms, s)
        
    ax.set_title("Simulated Source Current Proxy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (proxy units)")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def source_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    fig = source(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "source", "proxy_safe": True})


def lfp(signals: Any, **kwargs: Any) -> Any:
    """Plot LFP heatmap."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    
    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    
    if hasattr(signals, "field") and signals.field is not None:
        lfp_data = np.asarray(signals.field.lfp_proxy)
        time_ms = np.asarray(signals.time_ms)
    elif isinstance(signals, dict) and "lfp_proxy" in signals:
        lfp_data = np.asarray(signals["lfp_proxy"])
        time_ms = np.asarray(signals.get("time_ms", np.arange(lfp_data.shape[0])))
    elif hasattr(signals, "lfp_proxy"):
        lfp_data = np.asarray(signals.lfp_proxy)
        time_ms = np.arange(lfp_data.shape[0])
    else:
        lfp_data = np.asarray(signals)
        time_ms = np.arange(lfp_data.shape[0])
        
    im = ax.imshow(lfp_data.T, cmap="viridis", aspect="auto", extent=[time_ms[0], time_ms[-1], lfp_data.shape[1], 0])
    ax.set_title("Simulated Extracellular Potential (LFP-like) Heatmap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Contact Index")
    fig.colorbar(im, ax=ax, label="Potential (proxy units)")
    return fig


def lfp_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    fig = lfp(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "lfp", "proxy_safe": True})


def csd(signals: Any, **kwargs: Any) -> Any:
    """Plot CSD heatmap."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    
    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    
    if hasattr(signals, "field") and signals.field is not None:
        csd_data = np.asarray(signals.field.csd_proxy)
        time_ms = np.asarray(signals.time_ms)
    elif isinstance(signals, dict) and "csd_proxy" in signals:
        csd_data = np.asarray(signals["csd_proxy"])
        time_ms = np.asarray(signals.get("time_ms", np.arange(csd_data.shape[0])))
    elif hasattr(signals, "csd_proxy"):
        csd_data = np.asarray(signals.csd_proxy)
        time_ms = np.arange(csd_data.shape[0])
    else:
        csd_data = np.asarray(signals)
        time_ms = np.arange(csd_data.shape[0])
        
    csd_max = float(np.max(np.abs(csd_data)))
    vmin = -csd_max if csd_max > 0 else -1.0
    vmax = csd_max if csd_max > 0 else 1.0
    
    im = ax.imshow(csd_data.T, cmap="RdBu_r", aspect="auto", extent=[time_ms[0], time_ms[-1], csd_data.shape[1], 0], vmin=vmin, vmax=vmax)
    ax.set_title("Simulated Current Source Density (CSD-like) Heatmap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Contact Index")
    fig.colorbar(im, ax=ax, label="CSD (proxy units)")
    return fig


def csd_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    fig = csd(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "csd", "proxy_safe": True})


def psd(signals: Any, **kwargs: Any) -> Any:
    """Plot Power Spectral Density."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    
    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    
    if hasattr(signals, "field") and signals.field is not None:
        data = np.asarray(signals.field.lfp_proxy)
        if dt_ms is None:
            dt_ms = float(signals.metadata.get("dt_ms", 0.1))
    elif isinstance(signals, dict) and "lfp_proxy" in signals:
        data = np.asarray(signals["lfp_proxy"])
        if dt_ms is None:
            dt_ms = float(signals.get("metadata", {}).get("dt_ms", 0.1)) if isinstance(signals.get("metadata"), dict) else 0.1
    elif hasattr(signals, "lfp_proxy"):
        data = np.asarray(signals.lfp_proxy)
        if dt_ms is None:
            dt_ms = 0.1
    else:
        data = np.asarray(signals)
        if dt_ms is None:
            dt_ms = 0.1
        
    fs = 1000.0 / dt_ms
    nperseg = min(256, data.shape[0])
    freqs, psds = signal.welch(data, fs=fs, axis=0, nperseg=nperseg)
    
    # Average across contacts if multidimensional
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
        data = np.asarray(signals.field.lfp_proxy)
        if dt_ms is None:
            dt_ms = float(signals.metadata.get("dt_ms", 0.1))
    elif isinstance(signals, dict) and "lfp_proxy" in signals:
        data = np.asarray(signals["lfp_proxy"])
        if dt_ms is None:
            dt_ms = float(signals.get("metadata", {}).get("dt_ms", 0.1)) if isinstance(signals.get("metadata"), dict) else 0.1
    elif hasattr(signals, "lfp_proxy"):
        data = np.asarray(signals.lfp_proxy)
        if dt_ms is None:
            dt_ms = 0.1
    else:
        data = np.asarray(signals)
        if dt_ms is None:
            dt_ms = 0.1
        
    fs = 1000.0 / dt_ms
    # Pick first channel
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
    fig = spectrogram(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "spectrogram", "proxy_safe": True})


def summary(signals: Any, **kwargs: Any) -> Any:
    """Generate a multi-panel summary figure."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    
    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(figsize=(12, 8), **kwargs)
    
    # 3 Panels
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    # Plot raster
    if hasattr(signals, "spikes"):
        spikes = np.asarray(signals.spikes)
        time_ms = np.asarray(signals.time_ms)
        if dt_ms is None:
            dt_ms = float(signals.metadata.get("dt_ms", 0.1))
    elif isinstance(signals, dict) and "spikes" in signals:
        spikes = np.asarray(signals["spikes"])
        time_ms = np.asarray(signals.get("time_ms", np.arange(spikes.shape[0])))
        if dt_ms is None:
            dt_ms = float(signals.get("metadata", {}).get("dt_ms", 0.1)) if isinstance(signals.get("metadata"), dict) else 0.1
    else:
        spikes = np.asarray(signals)
        time_ms = np.arange(spikes.shape[0])
        if dt_ms is None:
            dt_ms = 0.1
        
    t_idx, n_idx = np.where(spikes > 0)
    ax1.scatter(time_ms[t_idx], n_idx, s=1, c="#228be6", marker="|")
    ax1.set_title("Raster")
    ax1.grid(True, alpha=0.3)
    
    # Plot LFP-like
    if hasattr(signals, "field") and signals.field is not None:
        lfp_data = np.asarray(signals.field.lfp_proxy)
        ax2.imshow(lfp_data.T, cmap="viridis", aspect="auto", extent=[time_ms[0], time_ms[-1], lfp_data.shape[1], 0])
        ax2.set_title("LFP Heatmap")
    elif isinstance(signals, dict) and "lfp_proxy" in signals:
        lfp_data = np.asarray(signals["lfp_proxy"])
        ax2.imshow(lfp_data.T, cmap="viridis", aspect="auto", extent=[time_ms[0], time_ms[-1], lfp_data.shape[1], 0])
        ax2.set_title("LFP Heatmap")
    else:
        ax2.text(0.5, 0.5, "LFP field not available", ha="center")
        
    # Plot population mean rate
    mean_rate = np.mean(spikes, axis=1) * (1000.0 / dt_ms)
    ax3.plot(time_ms, mean_rate, c="#f03e3e")
    ax3.set_title("Population Mean Rate (Hz)")
    ax3.grid(True, alpha=0.3)
        
    fig.suptitle("Simulated Proxy Summary Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def summary_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    fig = summary(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "summary", "proxy_safe": True})


def bandpower(signals: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO: implement jtfne.vis.bandpower")


def laminar_profile(signals: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO: implement jtfne.vis.laminar_profile")


def connectivity(signals: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO: implement jtfne.vis.connectivity")


def geometry3d(signals: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO: implement jtfne.vis.geometry3d")


def eeg(signals: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO: implement jtfne.vis.eeg")


def meg(signals: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO: implement jtfne.vis.meg")


def emm(signals: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO: implement jtfne.vis.emm")

