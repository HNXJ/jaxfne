"""Small plotting and summary helpers for tutorial notebooks."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Mapping, Tuple


@dataclass(frozen=True)
class ConfigSummary:
    """Summary of configured model."""
    name: str
    n_neurons: int
    layers: Sequence[str]
    cell_types: Mapping[str, float]
    connectivity: str
    emitter_family: str
    emitter_preset: str
    probes: Sequence[str]


def _finish_figure(fig, show: bool):
    """Display or close a Matplotlib figure."""
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def save_png(fig, name: str, fig_dir: Path, show: bool = False) -> str:
    """Save a figure and optionally display it."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    _finish_figure(fig, show)
    print(f"saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return str(path)


def finite_status(*arrays) -> bool:
    """Check if all arrays contain only finite values."""
    return all(bool(np.all(np.isfinite(np.asarray(a)))) for a in arrays if a is not None)


def population_rate_hz(spikes: np.ndarray, dt_ms: float) -> float:
    """Compute mean population firing rate in Hz."""
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    spikes = np.asarray(spikes)
    return float(spikes.mean() * (1000.0 / dt_ms)) if spikes.size else 0.0


def display_run_summary(label: str, spikes: np.ndarray, V_m: np.ndarray,
                       dt_ms: float, finite: bool) -> None:
    """Print a compact simulation summary."""
    rate_hz = population_rate_hz(spikes, dt_ms)
    voltage = np.asarray(V_m)
    v_min = float(voltage.min()) if voltage.size else float("nan")
    v_max = float(voltage.max()) if voltage.size else float("nan")
    print(f"\n{label}:")
    spike_array = np.asarray(spikes)
    print(
        f"  Spikes: {int(spike_array.sum())} | "
        f"Shape: {spike_array.shape} | Rate: {rate_hz:.2f} Hz"
    )
    print(f"  Voltage: [{v_min:.1f}, {v_max:.1f}] mV | Finite: {finite}")


def plot_raster(spike_times_list, spike_neuron_ids_list, t, figsize=(10, 4),
               title="Population Raster", show: bool = True):
    """Plot spike raster from list of spike times per neuron."""
    fig, ax = plt.subplots(figsize=figsize)
    for times, ids in zip(spike_times_list, spike_neuron_ids_list):
        ax.scatter(times, ids, s=2, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_xlim(t.min(), t.max())
    return _finish_figure(fig, show)


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
    _finish_figure(fig, show)
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
    return _finish_figure(fig, show)


def plot_connectivity_matrix(W, title="Connectivity matrix", figsize=(5, 5), show: bool = True):
    """Plot connectivity weight matrix."""
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
    return _finish_figure(fig, show)

def plot_spectrolaminar_power(
    t: np.ndarray,
    signal: np.ndarray,
    freq_min: float = 1.0,
    freq_max: float = 120.0,
    n_freqs: int = 96,
    title: str = "Spectrolaminar Power",
    figsize: tuple = (10, 5),
    show: bool = True,
) -> object:
    """Plot spectrolaminar PSD heatmap with configurable frequency resolution.

    Parameters
    ----------
    t : array of shape (n_steps,)
        Time axis in ms.
    signal : array of shape (n_steps, n_contacts) or (n_steps,)
        Signal to analyze.
    freq_min, freq_max : float
        Frequency axis bounds in Hz.
    n_freqs : int
        Number of frequency bins (minimum 64).
    show : bool
        If True, call plt.show(); if False, close figure after save.
    """
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


def hh_reference_trace(duration_ms: float = 500.0, dt_ms: float = 0.1,
                       current_amplitude: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hodgkin-Huxley reference trace with gating variables and ionic currents.

    Implements classical HH membrane equation with sodium, potassium, and leak currents.

    Parameters
    ----------
    duration_ms : float
        Simulation duration in milliseconds.
    dt_ms : float
        Time step in milliseconds.
    current_amplitude : float
        Injected current amplitude in μA/cm².

    Returns
    -------
    t : ndarray (n_steps,)
        Time in ms.
    V : ndarray (n_steps,)
        Membrane potential in mV.
    I_inj : ndarray (n_steps,)
        Injected current in μA/cm².
    """
    # HH parameters (squid axon, Hodgkin & Huxley 1952)
    C_m = 1.0  # membrane capacitance μF/cm²
    g_Na = 120.0  # sodium conductance mS/cm²
    g_K = 36.0  # potassium conductance mS/cm²
    g_L = 0.3  # leak conductance mS/cm²
    E_Na = 50.0  # sodium reversal mV
    E_K = -77.0  # potassium reversal mV
    E_L = -54.4  # leak reversal mV

    n_steps = int(duration_ms / dt_ms) + 1
    t = np.linspace(0, duration_ms, n_steps, dtype=np.float32)
    V = np.zeros(n_steps, dtype=np.float32)
    m = np.zeros(n_steps, dtype=np.float32)
    h = np.zeros(n_steps, dtype=np.float32)
    n = np.zeros(n_steps, dtype=np.float32)

    # Initial values (rest)
    V[0] = -65.0
    m[0] = 0.05
    h[0] = 0.6
    n[0] = 0.32

    dt = dt_ms / 1000.0  # convert to seconds

    for i in range(1, n_steps):
        # Rate constants (voltage-dependent)
        alpha_m = 0.1 * (V[i-1] + 40.0) / (1.0 - np.exp(-(V[i-1] + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(V[i-1] + 65.0) / 18.0)
        alpha_h = 0.07 * np.exp(-(V[i-1] + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(V[i-1] + 35.0) / 10.0))
        alpha_n = 0.01 * (V[i-1] + 55.0) / (1.0 - np.exp(-(V[i-1] + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(V[i-1] + 65.0) / 80.0)

        # Update gating variables
        m[i] = m[i-1] + (alpha_m * (1.0 - m[i-1]) - beta_m * m[i-1]) * dt
        h[i] = h[i-1] + (alpha_h * (1.0 - h[i-1]) - beta_h * h[i-1]) * dt
        n[i] = n[i-1] + (alpha_n * (1.0 - n[i-1]) - beta_n * n[i-1]) * dt

        # Clamp gating variables to [0, 1]
        m[i] = np.clip(m[i], 0.0, 1.0)
        h[i] = np.clip(h[i], 0.0, 1.0)
        n[i] = np.clip(n[i], 0.0, 1.0)

        # Ionic currents
        I_Na = g_Na * (m[i] ** 3) * h[i] * (V[i-1] - E_Na)
        I_K = g_K * (n[i] ** 4) * (V[i-1] - E_K)
        I_L = g_L * (V[i-1] - E_L)

        # Membrane voltage equation: C_m dV/dt = I_inj - I_Na - I_K - I_L
        dV = (current_amplitude - I_Na - I_K - I_L) / C_m
        V[i] = V[i-1] + dV * dt

    I_inj = np.full(n_steps, current_amplitude, dtype=np.float32)
    return t, V, I_inj
