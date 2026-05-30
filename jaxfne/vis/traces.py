"""Traces and time-series plotting submodules for jaxfne/vis.

NumPy-isolated graphics for membrane potential (Vm), firing rate, source current,
LFP-like, CSD-like, EEG-proxy, MEG-proxy, and EMM-proxy traces.
"""
from __future__ import annotations

from typing import Any

import jax
import numpy as np

from .core import FigureResult, prepare_static_plot_matrix, require_matplotlib


def plot_continuous_traces(traces_tensor: jax.Array, config_params: dict) -> None:
    """Extracts accelerator tensors onto the host context prior to plotting execution.

    Protects the active JAX trace context via standard host-device transfer.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    # Force immediate host-device transfer to protect the active trace context
    static_traces = np.asarray(jax.device_get(traces_tensor))

    fig, ax = plt.subplots(figsize=config_params.get("figsize", (10, 4)))
    # Process and plot static trace tracks cleanly...
    ax.set_title("Simulated Continuous Signal Profile")
    ax.set_xlabel("Time Step Index")
    ax.set_ylabel("Signal Amplitude")
    plt.close(fig)


def _get_time_ms(signals: Any, default_len: int) -> np.ndarray:
    time_raw = getattr(signals, "time_ms", None)
    if time_raw is None and isinstance(signals, dict):
        time_raw = signals.get("time_ms")
    time_ms = prepare_static_plot_matrix(time_raw)
    if time_ms is None:
        return np.arange(default_len)
    return time_ms


def vm(signals: Any, **kwargs: Any) -> Any:
    """Plot voltage traces over time."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    if hasattr(signals, "V_m"):
        v = prepare_static_plot_matrix(signals.V_m)
        time_ms = _get_time_ms(signals, v.shape[0])
    elif isinstance(signals, dict) and "V_m" in signals:
        v = prepare_static_plot_matrix(signals["V_m"])
        time_ms = prepare_static_plot_matrix(signals.get("time_ms"))
        if time_ms is None:
            time_ms = np.arange(v.shape[0])
    else:
        v = prepare_static_plot_matrix(signals)
        time_ms = np.arange(v.shape[0])

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
    """Plot membrane potential with JSON-safe metadata container."""
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
        spikes = prepare_static_plot_matrix(signals.spikes)
        time_ms = _get_time_ms(signals, spikes.shape[0])
        if dt_ms is None:
            dt_ms = float(signals.metadata.get("dt_ms", 0.1))
    elif isinstance(signals, dict) and "spikes" in signals:
        spikes = prepare_static_plot_matrix(signals["spikes"])
        time_ms = prepare_static_plot_matrix(signals.get("time_ms"))
        if time_ms is None:
            time_ms = np.arange(spikes.shape[0])
        if dt_ms is None:
            meta = signals.get("metadata", {})
            dt_ms = float(meta.get("dt_ms", 0.1)) if isinstance(meta, dict) else 0.1
    else:
        spikes = prepare_static_plot_matrix(signals)
        time_ms = np.arange(spikes.shape[0])
        if dt_ms is None:
            dt_ms = 0.1

    mean_rate = np.mean(spikes, axis=1) * (1000.0 / dt_ms)
    ax.plot(time_ms, mean_rate, c="#f03e3e", lw=1.5)
    ax.set_title("Simulated Population Mean Rate Proxy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def rate_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    """Plot population rate with JSON-safe metadata container."""
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
        s = prepare_static_plot_matrix(signals.sources)
        time_ms = _get_time_ms(signals, s.shape[0])
    elif isinstance(signals, dict) and "sources" in signals:
        s = prepare_static_plot_matrix(signals["sources"])
        time_ms = prepare_static_plot_matrix(signals.get("time_ms"))
        if time_ms is None:
            time_ms = np.arange(s.shape[0])
    else:
        s = prepare_static_plot_matrix(signals)
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
    """Plot source current with JSON-safe metadata container."""
    fig = source(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "source", "proxy_safe": True})


def lfp(signals: Any, **kwargs: Any) -> Any:
    """Plot LFP heatmap."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    if hasattr(signals, "field") and signals.field is not None:
        lfp_data = prepare_static_plot_matrix(signals.field.lfp_proxy)
        time_ms = _get_time_ms(signals, lfp_data.shape[0])
    elif isinstance(signals, dict) and "lfp_proxy" in signals:
        lfp_data = prepare_static_plot_matrix(signals["lfp_proxy"])
        time_ms = prepare_static_plot_matrix(signals.get("time_ms"))
        if time_ms is None:
            time_ms = np.arange(lfp_data.shape[0])
    elif hasattr(signals, "lfp_proxy"):
        lfp_data = prepare_static_plot_matrix(signals.lfp_proxy)
        time_ms = np.arange(lfp_data.shape[0])
    else:
        lfp_data = prepare_static_plot_matrix(signals)
        time_ms = np.arange(lfp_data.shape[0])

    im = ax.imshow(lfp_data.T, cmap="viridis", aspect="auto", extent=[time_ms[0], time_ms[-1], lfp_data.shape[1], 0])
    ax.set_title("Simulated Extracellular Potential (LFP-like) Heatmap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Contact Index")
    fig.colorbar(im, ax=ax, label="Potential (proxy units)")
    return fig


def lfp_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    """Plot LFP with JSON-safe metadata container."""
    fig = lfp(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "lfp", "proxy_safe": True})


def csd(signals: Any, **kwargs: Any) -> Any:
    """Plot CSD heatmap."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    if hasattr(signals, "field") and signals.field is not None:
        csd_data = prepare_static_plot_matrix(signals.field.csd_proxy)
        time_ms = _get_time_ms(signals, csd_data.shape[0])
    elif isinstance(signals, dict) and "csd_proxy" in signals:
        csd_data = prepare_static_plot_matrix(signals["csd_proxy"])
        time_ms = prepare_static_plot_matrix(signals.get("time_ms"))
        if time_ms is None:
            time_ms = np.arange(csd_data.shape[0])
    elif hasattr(signals, "csd_proxy"):
        csd_data = prepare_static_plot_matrix(signals.csd_proxy)
        time_ms = np.arange(csd_data.shape[0])
    else:
        csd_data = prepare_static_plot_matrix(signals)
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
    """Plot CSD with JSON-safe metadata container."""
    fig = csd(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "csd", "proxy_safe": True})


def lfp_traces(signals: Any, **kwargs: Any) -> Any:
    """Plot selected LFP-like contact traces."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    max_contacts = int(kwargs.pop("max_contacts", 6))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    if hasattr(signals, "field") and signals.field is not None:
        data = prepare_static_plot_matrix(signals.field.lfp_proxy)
        time_ms = _get_time_ms(signals, data.shape[0])
    elif isinstance(signals, dict) and "lfp_proxy" in signals:
        data = prepare_static_plot_matrix(signals["lfp_proxy"])
        time_ms = prepare_static_plot_matrix(signals.get("time_ms"))
        if time_ms is None:
            time_ms = np.arange(data.shape[0])
    else:
        data = prepare_static_plot_matrix(signals)
        time_ms = np.arange(data.shape[0])

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

    if hasattr(signals, "field") and signals.field is not None:
        data = prepare_static_plot_matrix(signals.field.csd_proxy)
        time_ms = _get_time_ms(signals, data.shape[0])
    elif isinstance(signals, dict) and "csd_proxy" in signals:
        data = prepare_static_plot_matrix(signals["csd_proxy"])
        time_ms = prepare_static_plot_matrix(signals.get("time_ms"))
        if time_ms is None:
            time_ms = np.arange(data.shape[0])
    else:
        data = prepare_static_plot_matrix(signals)
        time_ms = np.arange(data.shape[0])

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
    src_raw = getattr(signals, "sources", None)
    if src_raw is None and isinstance(signals, dict):
        src_raw = signals.get("sources")
    src = prepare_static_plot_matrix(src_raw)
    if src is None:
        raise ValueError("signals.sources is required for this proxy readout")
    n_src = src.shape[1]
    idx = np.arange(n_src, dtype=float)
    channels = []
    for c in range(int(n_channels)):
        channels.append(np.cos((c + 1.0) * (idx + 1.0) / max(n_src, 1) + phase))
    lead = np.asarray(channels, dtype=float)
    lead = lead / np.maximum(np.linalg.norm(lead, axis=1, keepdims=True), 1e-12)
    return src @ lead.T


def eeg(signals: Any, **kwargs: Any) -> Any:
    """Plot EEG-proxy traces from deterministic linear source projections."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    n_channels = int(kwargs.pop("n_channels", 4))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    y = _linear_proxy_from_sources(signals, n_channels=n_channels, phase=0.0)
    time_ms = _get_time_ms(signals, y.shape[0])
    scale = np.nanstd(y) or 1.0
    for ch in range(y.shape[1]):
        ax.plot(time_ms, y[:, ch] / scale + ch, label=f"EEG-proxy {ch}")
    ax.set_title("EEG-proxy linear readout")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel offset")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def meg(signals: Any, **kwargs: Any) -> Any:
    """Plot MEG-proxy traces from deterministic oriented source projections."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    n_channels = int(kwargs.pop("n_channels", 4))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    y = _linear_proxy_from_sources(signals, n_channels=n_channels, phase=np.pi / 4.0)
    time_ms = _get_time_ms(signals, y.shape[0])
    scale = np.nanstd(y) or 1.0
    for ch in range(y.shape[1]):
        ax.plot(time_ms, y[:, ch] / scale + ch, label=f"MEG-proxy {ch}")
    ax.set_title("MEG-proxy linear readout")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel offset")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def emm(signals: Any, **kwargs: Any) -> Any:
    """Plot normalized EMM-proxy activity/source/field cost."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    spikes_raw = getattr(signals, "spikes", None)
    if spikes_raw is None and isinstance(signals, dict):
        spikes_raw = signals.get("spikes")
    spikes = prepare_static_plot_matrix(spikes_raw)
    if spikes is None:
        raise ValueError("No spikes data found for EMM proxy calculation.")

    src_raw = getattr(signals, "sources", None)
    if src_raw is None and isinstance(signals, dict):
        src_raw = signals.get("sources")
    src = prepare_static_plot_matrix(src_raw)
    if src is None:
        raise ValueError("No sources data found for EMM proxy calculation.")

    try:
        lfp_raw = getattr(signals.field, "lfp_proxy", None)
    except AttributeError:
        lfp_raw = signals.get("lfp_proxy") if isinstance(signals, dict) else None
    lfp_arr = prepare_static_plot_matrix(lfp_raw)

    rate_val = np.mean(spikes, axis=1)
    cost = rate_val + np.mean(np.abs(src), axis=1)
    if lfp_arr is not None:
        cost = cost + np.mean(lfp_arr * lfp_arr, axis=1)

    cost = cost / max(float(np.nanmax(np.abs(cost))), 1e-12)
    time_ms = _get_time_ms(signals, cost.shape[0])

    ax.plot(time_ms, cost)
    ax.set_title("EMM-proxy normalized activity cost")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized proxy units")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def summary(signals: Any, **kwargs: Any) -> Any:
    """Generate a multi-panel summary figure.

    Evaluated as an uncalibrated computational scaffold matching truth_safe_unverified boundaries.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    dt_ms = kwargs.pop("dt_ms", None)
    fig = plt.figure(figsize=(12, 8), **kwargs)

    # 3 Panels
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    spikes_raw = getattr(signals, "spikes", None)
    if spikes_raw is None:
        if isinstance(signals, dict) and "spikes" in signals:
            spikes_raw = signals["spikes"]
        else:
            spikes_raw = signals

    spikes = prepare_static_plot_matrix(spikes_raw)

    time_ms_raw = getattr(signals, "time_ms", None)
    if time_ms_raw is None and isinstance(signals, dict):
        time_ms_raw = signals.get("time_ms")
    time_ms = prepare_static_plot_matrix(time_ms_raw)

    if time_ms is None:
        time_ms = np.arange(spikes.shape[0])

    if dt_ms is None:
        meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
        dt_ms = float(meta.get("dt_ms", 0.1)) if isinstance(meta, dict) else 0.1

    t_idx, n_idx = np.where(spikes > 0)
    ax1.scatter(time_ms[t_idx], n_idx, s=1, c="#228be6", marker="|")
    ax1.set_title("Raster")
    ax1.grid(True, alpha=0.3)

    field = getattr(signals, "field", None)
    if field is None and isinstance(signals, dict):
        field = signals.get("field")

    lfp_raw = None
    if field is not None:
        lfp_raw = getattr(field, "lfp_proxy", None)
        if lfp_raw is None and isinstance(field, dict):
            lfp_raw = field.get("lfp_proxy")
    elif isinstance(signals, dict):
        lfp_raw = signals.get("lfp_proxy")

    lfp_data = prepare_static_plot_matrix(lfp_raw)

    if lfp_data is not None and np.all(np.isfinite(lfp_data)):
        ax2.imshow(lfp_data.T, cmap="viridis", aspect="auto", extent=[time_ms[0], time_ms[-1], lfp_data.shape[1], 0])
        ax2.set_title("LFP Heatmap")
    else:
        ax2.text(0.5, 0.5, "LFP field not available", ha="center")

    mean_rate = np.mean(spikes, axis=1) * (1000.0 / dt_ms)
    ax3.plot(time_ms, mean_rate, c="#f03e3e")
    ax3.set_title("Population Mean Rate (Hz)")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Simulated Proxy Summary Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def summary_with_meta(signals: Any, **kwargs: Any) -> FigureResult:
    """Generate a multi-panel summary figure with metadata.

    Evaluated as an uncalibrated computational scaffold matching truth_safe_unverified boundaries.
    """
    fig = summary(signals, **kwargs)
    return FigureResult(fig, {"plot_type": "summary", "proxy_safe": True})

