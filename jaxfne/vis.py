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


def bandpower(
    signals: Any,
    *,
    band_definitions: dict[str, tuple[float, float]] | None = None,
    figsize: tuple[float, float] = (10, 5),
    **kwargs: Any,
) -> Any:
    """Plot mean spectral band power per contact (alpha/beta and gamma bands).

    Parameters
    ----------
    signals : Signals
        Output from simulate() or suite2_run_bundle().
    band_definitions : dict, optional
        Band name → (low_hz, high_hz). Defaults to alpha_beta (8-25 Hz) and
        gamma (40-150 Hz).
    figsize : tuple, default (10, 5)
        Figure size in inches.

    Returns
    -------
    matplotlib Figure — proxy relative units, no amplitude calibration.

    Notes
    -----
    Proxy relative units only. Amplitude calibration is outside scope.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    if band_definitions is None:
        band_definitions = {
            "alpha/beta\n(8-25 Hz)": (8.0, 25.0),
            "gamma\n(40-150 Hz)": (40.0, 150.0),
        }

    lfp_arr = _signals_lfp(signals)
    if not np.all(np.isfinite(lfp_arr)):
        lfp_arr = np.nan_to_num(lfp_arr, nan=0.0, posinf=0.0, neginf=0.0)

    dt_ms = float(getattr(signals, "metadata", {}).get("dt_ms", 0.1)) if hasattr(signals, "metadata") else 0.1
    fs = 1000.0 / dt_ms
    freqs, pxx = signal.welch(lfp_arr, fs=fs, axis=0, nperseg=min(512, lfp_arr.shape[0]))
    # pxx shape: (n_freqs, n_contacts)

    n_bands = len(band_definitions)
    fig, axes = plt.subplots(1, n_bands, figsize=figsize, squeeze=False)
    n_contacts = lfp_arr.shape[1] if lfp_arr.ndim > 1 else 1

    for col, (band_name, (lo, hi)) in enumerate(band_definitions.items()):
        ax = axes[0, col]
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, f"Band {band_name}\nnot covered by\nsampling rate",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(band_name)
            continue
        if lfp_arr.ndim > 1:
            band_power = np.mean(pxx[mask, :], axis=0)   # shape: (n_contacts,)
        else:
            band_power = np.array([np.mean(pxx[mask])])

        contact_labels = [str(c) for c in range(len(band_power))]
        ax.barh(contact_labels, band_power, color="steelblue", alpha=0.8)
        ax.set_title(f"Band power\n{band_name}", fontsize=10)
        ax.set_xlabel("Power (proxy a.u.)")
        ax.set_ylabel("Contact")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Spectrolaminar Band Power (proxy relative units)", fontsize=11)
    fig.tight_layout()
    return fig


def laminar_profile(
    signals: Any,
    *,
    cell_types: list[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    **kwargs: Any,
) -> Any:
    """Plot laminar neuron count distribution by cell type.

    Shows the fraction of neurons assigned to each depth bin (declared metadata).
    Works from neuron_metadata z-coordinates stored in signals.

    Parameters
    ----------
    signals : Signals
        Output from simulate(). Neuron z-positions read from signals.neuron_metadata.
    cell_types : list of str, optional
        Cell-type labels to plot. Inferred from metadata if available.
    figsize : tuple, default (8, 5)
        Figure size.

    Returns
    -------
    matplotlib Figure — declared geometry only.

    Notes
    -----
    Declared proxy geometry only. Physical anatomy is outside scope.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    rows = _neuron_rows(signals)
    if not rows:
        # Fallback: draw a placeholder with explanation
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "neuron_metadata not available\nin this signals object.\nRun with n_contacts > 0.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Laminar profile (declared geometry proxy)")
        return fig

    z_vals = np.asarray([float(row.get("z", 0.0)) for row in rows])
    ct_vals = [str(row.get("cell_type", "unknown")) for row in rows]

    if cell_types is None:
        cell_types = sorted(set(ct_vals))

    # Bin z into 10 depth bins
    z_min, z_max = float(z_vals.min()), float(z_vals.max())
    if z_max == z_min:
        z_max = z_min + 1.0
    n_bins = 10
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    colors = plt.cm.Set2(np.linspace(0, 1, len(cell_types)))
    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(n_bins)
    for ct, color in zip(cell_types, colors):
        ct_mask = np.asarray([v == ct for v in ct_vals])
        ct_z = z_vals[ct_mask]
        counts, _ = np.histogram(ct_z, bins=bin_edges)
        ax.barh(bin_centers, counts, height=(z_max - z_min) / n_bins * 0.8,
                left=bottom, color=color, alpha=0.85, label=ct)
        bottom += counts

    ax.set_title("Laminar neuron profile (declared geometry proxy)", fontsize=11)
    ax.set_xlabel("Neuron count")
    ax.set_ylabel("Depth z (mm or relative)")
    ax.invert_yaxis()
    ax.legend(title="Cell type", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# Alias: layer_celltype_counts → laminar_profile (THETA Phase 5 name)
def layer_celltype_counts(signals: Any, **kwargs: Any) -> Any:
    """Plot laminar neuron count distribution by cell type.

    Alias for :func:`laminar_profile`. See that function for full documentation.
    """
    return laminar_profile(signals, **kwargs)


def connectivity(
    model_or_W: Any,
    *,
    cell_type_labels: list[str] | None = None,
    colormap: str = "RdBu_r",
    figsize: tuple[float, float] = (8, 7),
    **kwargs: Any,
) -> Any:
    """Plot the recurrent connectivity weight matrix as a heatmap.

    Parameters
    ----------
    model_or_W : Model or array-like
        A jaxfne Model object (weight matrix extracted from model.params) or a
        raw weight matrix of shape (N, N).
    cell_type_labels : list of str, optional
        Labels for each row/column (neuron cell types). Shown as tick labels.
    colormap : str, default "RdBu_r"
        Matplotlib colormap. Red = excitatory (+), blue = inhibitory (−).
    figsize : tuple, default (8, 7)
        Figure size.

    Returns
    -------
    matplotlib Figure — proxy weight matrix, declared metadata.

    Notes
    -----
    Shows the declared weight structure, not a physically calibrated connectivity.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    # Extract weight matrix
    W = None
    if hasattr(model_or_W, "params"):
        # Model object — try to find the weight matrix
        params = model_or_W.params
        if hasattr(params, "get"):
            W = params.get("W", params.get("weights", None))
        elif hasattr(params, "W"):
            W = params.W
        elif hasattr(params, "weights"):
            W = params.weights
    if W is None and hasattr(model_or_W, "__array__"):
        W = np.asarray(model_or_W)
    if W is None and hasattr(model_or_W, "shape"):
        W = np.asarray(model_or_W)

    if W is None:
        # No accessible weight matrix — return informative placeholder
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5,
                "Weight matrix W not accessible\nfrom this model object.\n"
                "Pass W directly: jtfne.vis.connectivity(model.params['W'])",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Recurrent connectivity (proxy weight structure)")
        return fig

    W_np = np.asarray(W)
    if not np.all(np.isfinite(W_np)):
        W_np = np.nan_to_num(W_np, nan=0.0, posinf=0.0, neginf=0.0)

    vmax = float(np.nanmax(np.abs(W_np))) or 1.0
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(W_np, cmap=colormap, vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="Weight (proxy a.u.)")

    if cell_type_labels is not None and len(cell_type_labels) == W_np.shape[0]:
        # Show gridlines between cell types
        ax.set_xticks(range(len(cell_type_labels)))
        ax.set_yticks(range(len(cell_type_labels)))
        ax.set_xticklabels(cell_type_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cell_type_labels, fontsize=7)

    ax.set_title("Recurrent connectivity W (proxy weight structure)", fontsize=11)
    ax.set_xlabel("Target neuron")
    ax.set_ylabel("Source neuron")
    fig.tight_layout()
    return fig


# Alias: connectivity_matrix → connectivity (THETA Phase 5 name)
def connectivity_matrix(model_or_W: Any, **kwargs: Any) -> Any:
    """Plot the recurrent connectivity weight matrix as a heatmap.

    Alias for :func:`connectivity`. See that function for full documentation.
    """
    return connectivity(model_or_W, **kwargs)


def geometry3d(
    signals_or_cfg: Any,
    *,
    areas: list[str] | None = None,
    cell_types: list[str] | None = None,
    figsize: tuple[float, float] = (9, 7),
    **kwargs: Any,
) -> Any:
    """Plot declared 3D neuron geometry (x, y, z positions by cell type).

    Parameters
    ----------
    signals_or_cfg : Signals or Configuration
        If Signals, reads neuron_metadata for x/y/z positions and cell_type labels.
        If Configuration, reads columns and geometry metadata.
    areas : list of str, optional
        Area names to include. All areas shown if None.
    cell_types : list of str, optional
        Cell types to color. Defaults to all detected types.
    figsize : tuple, default (9, 7)
        Figure size.

    Returns
    -------
    matplotlib Figure — declared proxy geometry only.

    Notes
    -----
    Declared proxy geometry only. Physical anatomy and 3D PDE grids are outside scope.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    rows = _neuron_rows(signals_or_cfg)

    if not rows:
        # Try reading from Configuration metadata
        if hasattr(signals_or_cfg, "metadata"):
            meta = signals_or_cfg.metadata
            columns = meta.get("columns", [])
            if columns:
                # Synthesize positions from declared geometry
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


# Alias: column_geometry → geometry3d (THETA Phase 5 name)
def column_geometry(signals_or_cfg: Any, **kwargs: Any) -> Any:
    """Plot declared 3D column geometry (x, y, z positions by cell type).

    Alias for :func:`geometry3d`. See that function for full documentation.
    """
    return geometry3d(signals_or_cfg, **kwargs)


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


def multi_area_layout(
    signals_or_cfg: Any,
    *,
    areas: list[str] | None = None,
    figsize: tuple[float, float] = (10, 5),
    **kwargs: Any,
) -> Any:
    """Plot a 2D top-down layout showing multiple cortical areas side by side.

    Each area is shown as a rectangle with neuron count annotations.
    Inter-area connectivity (if declared in metadata) is shown as arrows.

    Parameters
    ----------
    signals_or_cfg : Signals or Configuration
        If Configuration, reads columns and inter_column_connectivity metadata.
        If Signals, reads area labels from neuron_metadata.
    areas : list of str, optional
        Which areas to include. All areas shown if None.
    figsize : tuple, default (10, 5)
        Figure size.

    Returns
    -------
    matplotlib Figure — declared metadata only.

    Notes
    -----
    Declared connectivity metadata only. Fitted weights are outside scope.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe

    # Extract column info from configuration or signals
    columns = []
    inter_conn = {}
    if hasattr(signals_or_cfg, "metadata"):
        meta = signals_or_cfg.metadata
        columns = meta.get("columns", [])
        inter_conn = meta.get("inter_column_connectivity", {})
    elif hasattr(signals_or_cfg, "neuron_metadata"):
        rows = _neuron_rows(signals_or_cfg)
        area_counts = {}
        for row in rows:
            area = str(row.get("area", "unknown"))
            area_counts[area] = area_counts.get(area, 0) + 1
        columns = [{"name": a, "n": n} for a, n in area_counts.items()]

    if not columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Column metadata required\nfor multi_area_layout.\n"
                "Use a Configuration or signals with neuron_metadata.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Multi-area layout (declared metadata proxy)")
        return fig

    if areas is not None:
        columns = [c for c in columns if c.get("name") in areas]

    fig, ax = plt.subplots(figsize=figsize)
    n_cols = len(columns)
    col_width = 0.8 / max(n_cols, 1)
    spacing = 1.0 / max(n_cols, 1)

    col_centers = {}
    for i, col in enumerate(columns):
        cx = spacing * i + spacing / 2
        cy = 0.5
        col_centers[col.get("name", str(i))] = (cx, cy)
        rect = mpatches.FancyBboxPatch(
            (cx - col_width / 2, cy - 0.25), col_width, 0.5,
            boxstyle="round,pad=0.02", linewidth=2,
            edgecolor="steelblue", facecolor="lightsteelblue", alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + 0.1, col.get("name", f"Area {i}"), ha="center", va="center",
                fontsize=12, fontweight="bold", color="navy")
        n_neurons = col.get("n", "?")
        layers = col.get("layers", [])
        layer_str = f"{len(layers)} layers" if layers else ""
        ax.text(cx, cy - 0.1, f"N={n_neurons}\n{layer_str}", ha="center", va="center",
                fontsize=9, color="darkblue")

    # Draw inter-area connectivity arrows
    if inter_conn:
        src_area = inter_conn.get("source_area", "")
        tgt_area = inter_conn.get("target_area", "")
        if src_area in col_centers and tgt_area in col_centers:
            sx, sy = col_centers[src_area]
            tx, ty = col_centers[tgt_area]
            # Feedforward arrow (top)
            ax.annotate("", xy=(tx - col_width / 2, ty + 0.15),
                         xytext=(sx + col_width / 2, sy + 0.15),
                         arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))
            ax.text((sx + tx) / 2, sy + 0.22, "FF", ha="center", fontsize=9, color="darkorange")
            # Feedback arrow (bottom)
            ax.annotate("", xy=(sx + col_width / 2, sy - 0.15),
                         xytext=(tx - col_width / 2, ty - 0.15),
                         arrowprops=dict(arrowstyle="->", color="mediumpurple", lw=2))
            ax.text((sx + tx) / 2, sy - 0.22, "FB", ha="center", fontsize=9, color="mediumpurple")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Multi-area layout (declared connectivity metadata proxy)", fontsize=11)
    fig.tight_layout()
    return fig


def objective_report(
    tune_result_or_history: Any,
    *,
    figsize: tuple[float, float] = (9, 4),
    **kwargs: Any,
) -> Any:
    """Plot optimization objective value history.

    Parameters
    ----------
    tune_result_or_history : TuneResult, dict, or array-like
        - TuneResult from model.tune(): reads summary.score_history or similar.
        - dict with 'score_history' or 'objective_history' key.
        - array-like of objective values (one per iteration).
    figsize : tuple, default (9, 4)
        Figure size.

    Returns
    -------
    matplotlib Figure — optimization proxy only, no biological claims.

    Notes
    -----
    Shows surrogate-objective convergence only. Surrogate paths are for inner-loop
    optimization; biological claim gates are independent of optimizer convergence.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    # Extract history
    history = None
    if hasattr(tune_result_or_history, "summary"):
        summary = tune_result_or_history.summary
        if isinstance(summary, dict):
            history = (summary.get("score_history")
                       or summary.get("objective_history")
                       or summary.get("loss_history"))
    elif isinstance(tune_result_or_history, dict):
        history = (tune_result_or_history.get("score_history")
                   or tune_result_or_history.get("objective_history")
                   or tune_result_or_history.get("loss_history"))
    if history is None:
        # Try treating as direct array
        try:
            history = list(tune_result_or_history)
        except (TypeError, ValueError):
            history = None

    fig, axes = plt.subplots(1, 2 if history is not None else 1, figsize=figsize, squeeze=False)

    if history is not None:
        history_arr = np.asarray(history, dtype=float)
        if not np.all(np.isfinite(history_arr)):
            history_arr = np.nan_to_num(history_arr, nan=np.nan)
        iters = np.arange(len(history_arr))
        ax = axes[0, 0]
        ax.plot(iters, history_arr, "o-", color="steelblue", ms=4, lw=1.5)
        ax.set_title("Objective history (surrogate proxy)", fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective value (proxy a.u.)")
        ax.grid(True, linestyle="--", alpha=0.4)

        if len(history_arr) > 1:
            ax2 = axes[0, 1] if axes.shape[1] > 1 else axes[0, 0]
            delta = np.diff(history_arr)
            ax2.bar(iters[1:], delta, color=np.where(delta < 0, "green", "red"), alpha=0.7)
            ax2.axhline(0, color="black", lw=0.8)
            ax2.set_title("Δ Objective per step", fontsize=11)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Δ value")
            ax2.grid(True, linestyle="--", alpha=0.4)
    else:
        ax = axes[0, 0]
        ax.text(0.5, 0.5,
                "Objective history not found.\nPass TuneResult, dict with\n"
                "'score_history' key, or list of scores.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Objective report (proxy)")

    fig.suptitle("Optimization objective — surrogate proxy only; not a biological claim gate",
                 fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Suite No. 2 generalized visualization facade.
# These functions intentionally operate on the public Signals container and its
# metadata so tutorials do not need to plot from raw arrays.
# -----------------------------------------------------------------------------

def _signals_time_ms(signals: Any, n: int | None = None) -> np.ndarray:
    if hasattr(signals, "time_ms"):
        return np.asarray(signals.time_ms)
    if isinstance(signals, dict) and "time_ms" in signals:
        return np.asarray(signals["time_ms"])
    return np.arange(int(n or 0))


def _signals_spikes(signals: Any) -> np.ndarray:
    if hasattr(signals, "spikes"):
        return np.asarray(signals.spikes)
    if isinstance(signals, dict) and "spikes" in signals:
        return np.asarray(signals["spikes"])
    return np.asarray(signals)


def _signals_sources(signals: Any) -> np.ndarray:
    if hasattr(signals, "sources") and signals.sources is not None:
        return np.asarray(signals.sources)
    if isinstance(signals, dict) and "sources" in signals:
        return np.asarray(signals["sources"])
    raise ValueError("signals.sources is required for this proxy readout")


def _signals_lfp(signals: Any) -> np.ndarray:
    if hasattr(signals, "field") and signals.field is not None:
        return np.asarray(signals.field.lfp_proxy)
    if isinstance(signals, dict) and "lfp_proxy" in signals:
        return np.asarray(signals["lfp_proxy"])
    raise ValueError("signals.field.lfp_proxy is required for this figure")


def _signals_csd(signals: Any) -> np.ndarray:
    if hasattr(signals, "field") and signals.field is not None:
        return np.asarray(signals.field.csd_proxy)
    if isinstance(signals, dict) and "csd_proxy" in signals:
        return np.asarray(signals["csd_proxy"])
    raise ValueError("signals.field.csd_proxy is required for this figure")


def _neuron_rows(signals: Any) -> list[dict[str, Any]]:
    meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
    rows = meta.get("neuron_metadata") if isinstance(meta, dict) else None
    return [dict(row) for row in rows] if rows else []


def raster(signals: Any, **kwargs: Any) -> Any:  # type: ignore[override]
    """Plot a spike raster, optionally sorted by declared z-depth."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    sort_by = kwargs.pop("sort_by", "z")
    marker_size = float(kwargs.pop("marker_size", 2.0))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    spikes = _signals_spikes(signals)
    time_ms = _signals_time_ms(signals, spikes.shape[0])
    t_idx, n_idx = np.where(spikes > 0)

    rows = _neuron_rows(signals)
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


def lfp_traces(signals: Any, **kwargs: Any) -> Any:
    """Plot selected LFP-like contact traces."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    max_contacts = int(kwargs.pop("max_contacts", 6))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    data = _signals_lfp(signals)
    time_ms = _signals_time_ms(signals, data.shape[0])
    n_contacts = min(max_contacts, data.shape[1] if data.ndim > 1 else 1)
    arr = data if data.ndim > 1 else data[:, None]
    scale = np.nanstd(arr[:, :n_contacts]) or 1.0
    for contact in range(n_contacts):
        ax.plot(time_ms, arr[:, contact] / scale + contact, label=f"contact {contact}")
    ax.set_title("LFP-like contact traces")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Contact offset")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def csd_traces(signals: Any, **kwargs: Any) -> Any:
    """Plot selected CSD-like contact traces."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    max_contacts = int(kwargs.pop("max_contacts", 6))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    data = _signals_csd(signals)
    time_ms = _signals_time_ms(signals, data.shape[0])
    n_contacts = min(max_contacts, data.shape[1] if data.ndim > 1 else 1)
    arr = data if data.ndim > 1 else data[:, None]
    scale = np.nanstd(arr[:, :n_contacts]) or 1.0
    for contact in range(n_contacts):
        ax.plot(time_ms, arr[:, contact] / scale + contact, label=f"contact {contact}")
    ax.set_title("CSD-like contact traces")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Contact offset")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def _linear_proxy_from_sources(signals: Any, *, n_channels: int, phase: float = 0.0) -> np.ndarray:
    src = _signals_sources(signals)
    n_src = src.shape[1]
    idx = np.arange(n_src, dtype=float)
    channels = []
    for c in range(int(n_channels)):
        channels.append(np.cos((c + 1.0) * (idx + 1.0) / max(n_src, 1) + phase))
    lead = np.asarray(channels, dtype=float)
    lead = lead / np.maximum(np.linalg.norm(lead, axis=1, keepdims=True), 1e-12)
    return src @ lead.T


def eeg(signals: Any, **kwargs: Any) -> Any:  # type: ignore[override]
    """Plot EEG-proxy traces from deterministic linear source projections."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    n_channels = int(kwargs.pop("n_channels", 4))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    y = _linear_proxy_from_sources(signals, n_channels=n_channels, phase=0.0)
    time_ms = _signals_time_ms(signals, y.shape[0])
    scale = np.nanstd(y) or 1.0
    for ch in range(y.shape[1]):
        ax.plot(time_ms, y[:, ch] / scale + ch, label=f"EEG-proxy {ch}")
    ax.set_title("EEG-proxy linear readout")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel offset")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def meg(signals: Any, **kwargs: Any) -> Any:  # type: ignore[override]
    """Plot MEG-proxy traces from deterministic oriented source projections."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    n_channels = int(kwargs.pop("n_channels", 4))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    y = _linear_proxy_from_sources(signals, n_channels=n_channels, phase=np.pi / 4.0)
    time_ms = _signals_time_ms(signals, y.shape[0])
    scale = np.nanstd(y) or 1.0
    for ch in range(y.shape[1]):
        ax.plot(time_ms, y[:, ch] / scale + ch, label=f"MEG-proxy {ch}")
    ax.set_title("MEG-proxy linear readout")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel offset")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def emm(signals: Any, **kwargs: Any) -> Any:  # type: ignore[override]
    """Plot normalized EMM-proxy activity/source/field cost."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    spikes = _signals_spikes(signals)
    src = _signals_sources(signals)
    lfp_arr = _signals_lfp(signals)
    rate = np.mean(spikes, axis=1)
    cost = rate + np.mean(np.abs(src), axis=1) + np.mean(lfp_arr * lfp_arr, axis=1)
    cost = cost / max(float(np.nanmax(np.abs(cost))), 1e-12)
    time_ms = _signals_time_ms(signals, cost.shape[0])
    ax.plot(time_ms, cost)
    ax.set_title("EMM-proxy normalized activity cost")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized proxy units")
    ax.grid(True, linestyle="--", alpha=0.3)
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


def spectrolaminar_suite(signals: Any, **kwargs: Any) -> Any:
    """Render the core Suite No. 2 readout panel in one figure.

    Supports custom visualization parameters:
    - freq_min_hz: minimum frequency for PSD plot (default 0 Hz)
    - freq_max_hz: maximum frequency for PSD plot (default 80 Hz)
    - freq_count: number of frequency bins (currently unused, for future expansion)
    - psd_nperseg: Welch window length in samples (default auto)
    - dpi: dots per inch for figure (handled by plt.figure)
    - title: overall figure title
    - figsize: figure size tuple (default (12, 10))
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    # Pop custom kwargs before plt.figure to avoid passing unsupported args
    freq_min_hz = float(kwargs.pop("freq_min_hz", 0.0))
    freq_max_hz = float(kwargs.pop("freq_max_hz", 80.0))
    freq_count = int(kwargs.pop("freq_count", 128))  # For future expansion
    psd_nperseg = kwargs.pop("psd_nperseg", None)  # None means auto
    figure_title = kwargs.pop("title", None)  # Figure-level title
    figsize = kwargs.pop("figsize", (12, 10))
    dpi = kwargs.pop("dpi", None)  # Will be passed to plt.figure

    # Create figure with remaining kwargs (dpi, etc.)
    fig_kwargs = {"figsize": figsize}
    if dpi is not None:
        fig_kwargs["dpi"] = int(dpi)
    # Pass remaining kwargs (should be empty now, but be safe)
    fig = plt.figure(**fig_kwargs)

    axes = [fig.add_subplot(3, 2, i + 1) for i in range(6)]
    spikes = _signals_spikes(signals)
    time_ms = _signals_time_ms(signals, spikes.shape[0])
    rows = _neuron_rows(signals)
    t_idx, n_idx = np.where(spikes > 0)
    if len(rows) == spikes.shape[1]:
        order = np.argsort([float(row.get("z", i)) for i, row in enumerate(rows)])
        rank = np.empty_like(order)
        rank[order] = np.arange(order.shape[0])
        n_plot = rank[n_idx]
    else:
        n_plot = n_idx
    axes[0].scatter(time_ms[t_idx], n_plot, s=1, marker="|")
    axes[0].set_title("Raster proxy sorted by depth")
    axes[0].set_xlabel("Time (ms)")

    lfp_arr = _signals_lfp(signals)
    if lfp_arr is not None and np.all(np.isfinite(lfp_arr)):
        im1 = axes[1].imshow(lfp_arr.T, aspect="auto", origin="upper", extent=[time_ms[0], time_ms[-1], lfp_arr.shape[1], 0])
        axes[1].set_title("LFP-like contacts")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
    else:
        axes[1].text(0.5, 0.5, "LFP data unavailable", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("LFP-like contacts")

    csd_arr = _signals_csd(signals)
    if csd_arr is not None and np.all(np.isfinite(csd_arr)):
        vmax = float(np.nanmax(np.abs(csd_arr))) or 1.0
        im2 = axes[2].imshow(csd_arr.T, aspect="auto", origin="upper", extent=[time_ms[0], time_ms[-1], csd_arr.shape[1], 0], vmin=-vmax, vmax=vmax)
        axes[2].set_title("CSD-like contacts")
        fig.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        axes[2].text(0.5, 0.5, "CSD data unavailable", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("CSD-like contacts")

    # PSD with custom frequency range and nperseg
    if lfp_arr is not None and np.all(np.isfinite(lfp_arr)):
        dt_ms = float(getattr(signals, "metadata", {}).get("dt_ms", 0.1)) if hasattr(signals, "metadata") else 0.1
        fs = 1000.0 / dt_ms
        nperseg = psd_nperseg if psd_nperseg is not None else min(512, lfp_arr.shape[0])
        freqs, pxx = signal.welch(lfp_arr, fs=fs, axis=0, nperseg=int(nperseg))
        mask = (freqs >= freq_min_hz) & (freqs <= freq_max_hz)
        axes[3].plot(freqs[mask], np.mean(pxx[mask], axis=1))
        axes[3].set_title("Mean PSD")
        axes[3].set_xlabel("Frequency (Hz)")
        axes[3].set_xlim(freq_min_hz, freq_max_hz)
    else:
        axes[3].text(0.5, 0.5, "PSD unavailable", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_title("Mean PSD")

    eeg_y = _linear_proxy_from_sources(signals, n_channels=3, phase=0.0)
    if eeg_y is not None and np.all(np.isfinite(eeg_y)):
        scale = np.nanstd(eeg_y) or 1.0
        for ch in range(eeg_y.shape[1]):
            axes[4].plot(time_ms, eeg_y[:, ch] / scale + ch)
        axes[4].set_title("EEG-proxy")
    else:
        axes[4].text(0.5, 0.5, "EEG proxy unavailable", ha="center", va="center", transform=axes[4].transAxes)
        axes[4].set_title("EEG-proxy")

    spikes_mean = np.mean(spikes, axis=1)
    src = _signals_sources(signals)
    if src is not None and np.all(np.isfinite(src)):
        emm_cost = spikes_mean + np.mean(np.abs(src), axis=1) + np.mean(lfp_arr * lfp_arr, axis=1) if lfp_arr is not None else spikes_mean
        emm_cost = emm_cost / max(float(np.nanmax(np.abs(emm_cost))), 1e-12)
        axes[5].plot(time_ms, emm_cost)
        axes[5].set_title("EMM-proxy")
    else:
        axes[5].plot(time_ms, spikes_mean / max(float(np.nanmax(spikes_mean)), 1e-12))
        axes[5].set_title("EMM-proxy (fallback: spike rate)")
    axes[5].set_xlabel("Time (ms)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()

    # Add figure-level title if provided
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=14, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.99])

    return fig
