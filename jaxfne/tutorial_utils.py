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


def hh_reference_trace_jaxley(duration_ms: float = 500.0, dt_ms: float = 0.1,
                              current_amplitude: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hodgkin-Huxley reference trace via optional Jaxley bridge.

    **Scope:** HH emitter reference generated through the optional Jaxley bridge.
    **Evidence:** Simulated voltage trace for tutorial comparison.
    **Interpretation:** Emitter-level reference before TFNE source/readout projection.

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

    Raises
    ------
    NotImplementedError
        If Jaxley HH bridge is not available. Install jaxley to use this path.
    """
    from .bridges import hh_jaxley_reference_trace
    return hh_jaxley_reference_trace(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        current_amplitude=current_amplitude,
    )


def hh_numpy_reference_trace(duration_ms: float = 500.0, dt_ms: float = 0.1,
                             current_amplitude: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standalone tutorial/reference Hodgkin-Huxley single-compartment trace.

    **Scope:** Standalone NumPy Hodgkin-Huxley trace, NOT Jaxley bridge validation,
    and NOT evidence that JaxleyBridge executed.
    """
    from .bridges import hh_numpy_reference_trace
    return hh_numpy_reference_trace(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        current_amplitude=current_amplitude,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Suite No. 2 analysis helpers (Phase 5b: AGSDR tuning)
# ─────────────────────────────────────────────────────────────────────────────


def select_neurons(model, area: str | None = None, layer: str | None = None,
                  cell_type: str | None = None) -> np.ndarray:
    """
    Select neuron indices matching given criteria (area, layer, cell_type).

    Args:
        model: Model object with neuron_table()
        area: area name (e.g., "V1", "V4"); if None, all areas included
        layer: layer name (e.g., "L4", "L5"); if None, all layers included
        cell_type: cell type label (e.g., "E", "PV"); if None, all types included

    Returns:
        ndarray of neuron indices matching the criteria

    Example:
        >>> v1_l4_e_indices = select_neurons(model, area="V1", layer="L4", cell_type="E")
        >>> len(v1_l4_e_indices)
        75
    """
    # Access neuron table from model
    if not hasattr(model, 'neuron_table') or not callable(model.neuron_table):
        return np.array([], dtype=int)

    try:
        rows = model.neuron_table()
    except Exception:
        return np.array([], dtype=int)

    if not rows:
        return np.array([], dtype=int)

    matches = []
    for row in rows:
        if area is not None and str(row.get("area", "")) != area:
            continue
        if layer is not None and str(row.get("layer", "")) != layer:
            continue
        if cell_type is not None and str(row.get("cell_type", "")) != cell_type:
            continue
        matches.append(int(row.get("neuron_id", 0)))

    return np.array(matches, dtype=int)


def kappa_synchrony(spikes: np.ndarray, dt_ms: float = 0.1) -> float:
    """
    Compute spike synchrony measure (kappa statistic) across neurons.

    Args:
        spikes: spike matrix [n_timesteps, n_neurons] with boolean values
        dt_ms: timestep in milliseconds (default 0.1)

    Returns:
        float: kappa synchrony value in [-1, 1].
        0 indicates asynchronous spiking.
        1 indicates perfect synchrony.
        -1 indicates perfect anti-synchrony.

    Notes:
        - Computes pairwise correlation of spike trains across neurons (columns)
        - Returns mean pairwise spike-time correlation
        - Proxy metric; not a biological invariant
    """
    spikes = np.asarray(spikes, dtype=float)
    if spikes.size == 0:
        return 0.0

    # Ensure shape is [T, N]
    if spikes.ndim == 1:
        return 0.0  # Single neuron or single timestep

    if spikes.ndim != 2:
        return 0.0

    # Shape is [T, N] — n_neurons is the second dimension
    n_timesteps, n_neurons = spikes.shape

    if n_neurons < 2 or n_timesteps < 1:
        return 0.0

    # Compute average pairwise spike-time correlation across neurons (columns)
    correlations = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            spike_i = spikes[:, i]  # Column i (neuron i across all timesteps)
            spike_j = spikes[:, j]  # Column j (neuron j across all timesteps)
            if np.sum(spike_i) == 0 or np.sum(spike_j) == 0:
                continue
            # Pearson correlation of spike trains
            mean_i = np.mean(spike_i)
            mean_j = np.mean(spike_j)
            std_i = np.std(spike_i)
            std_j = np.std(spike_j)
            if std_i > 0 and std_j > 0:
                corr = np.mean((spike_i - mean_i) * (spike_j - mean_j)) / (std_i * std_j)
                correlations.append(corr)

    if not correlations:
        return 0.0
    return float(np.mean(correlations))


def rate_synchrony_targets(
    target_rate_hz: float,
    target_kappa_synchrony: float,
    rate_weight: float = 1.0,
    synchrony_weight: float = 0.25,
):
    """
    Create an objective specification for AGSDR tuning toward rate and synchrony targets.

    Args:
        target_rate_hz: target population firing rate (Hz)
        target_kappa_synchrony: target kappa synchrony value (typically 0.0 for asynchronous)
        rate_weight: weight for rate term in objective
        synchrony_weight: weight for synchrony term in objective

    Returns:
        Objective instance with rate and synchrony loss terms

    Example:
        >>> objective = rate_synchrony_targets(target_rate_hz=5.0, target_kappa_synchrony=0.0)
        >>> print(objective.name)
        rate_synchrony_targets

    Notes:
        - Returned Objective can be used with Model.evaluate() and Model.tune()
        - Surrogate objective for inner-loop optimization only; not a biological claim gate
        - Weights control relative importance of rate vs. synchrony targets
        - Truth gates preserved: truth_safe_unverified, computational_scaffold
    """
    # Import Objective here to avoid circular imports
    import jaxfne as jtfne

    obj = jtfne.Objective(
        name="rate_synchrony_targets",
        kind="rate_synchrony_targets",
    )

    # Add rate loss term
    obj = obj.loss(
        name="population_firing_rate",
        target=float(target_rate_hz),
        weight=float(rate_weight),
        metric="mean_firing_rate_hz",
        metadata={
            "target_rate_hz": float(target_rate_hz),
        }
    )

    # Add synchrony loss term
    obj = obj.loss(
        name="kappa_synchrony",
        target=float(target_kappa_synchrony),
        weight=float(synchrony_weight),
        metric="kappa_synchrony",
        metadata={
            "target_kappa_synchrony": float(target_kappa_synchrony),
        }
    )

    # Add truth gates
    obj = obj.gate(
        name="truth_mode",
        threshold="truth_safe_unverified",
        criterion="exact",
        metric="truth_mode"
    )

    obj = obj.gate(
        name="claim_level",
        threshold="computational_scaffold",
        criterion="exact",
        metric="claim_level"
    )

    return obj
