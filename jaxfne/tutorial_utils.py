"""Small plotting and summary helpers for tutorial notebooks."""
from __future__ import annotations

import numpy as np
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
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt
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


def plot_voltage_samples(t, V_m, title="Voltage trajectory", figsize=(10, 3),
                         max_neurons=10, show: bool = True):
    """Plot voltage time series from first N neurons."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(min(max_neurons, V_m.shape[1])):
        ax.plot(t, V_m[:, i], lw=0.8, alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V-like state (mV)")
    ax.set_title(title)
    return _finish_figure(fig, show)


def plot_connectivity_matrix(W, title="Connectivity matrix", figsize=(5, 5), show: bool = True):
    """Plot connectivity weight matrix."""
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


def plot_laminar_readout(t, lfp_proxy, csd_proxy=None, figsize=(12, 4),
                         title="Laminar Readout", show: bool = True):
    """Plot LFP-proxy and optionally CSD-proxy."""
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

    # Add rate loss term.
    # metric must match jaxfne.core._KNOWN_METRICS / _compute_all_metrics;
    # the computed population-rate metric is "spike_rate_hz_mean".
    obj = obj.loss(
        name="population_firing_rate",
        target=float(target_rate_hz),
        weight=float(rate_weight),
        metric="spike_rate_hz_mean",
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


# ─────────────────────────────────────────────────────────────────────────────
# Etude No. 1: Compact laminar column helpers (golden notebook port)
# ─────────────────────────────────────────────────────────────────────────────

import json
from dataclasses import dataclass, field, asdict


@dataclass
class CellTypePreset:
    """Single cell type preset with Izhikevich parameters."""
    name: str
    color: str
    sign: float  # ±1.0
    a: float
    b: float
    c: float
    d: float
    drive: float
    noise_std: float


@dataclass(frozen=True)
class LaminarColumnConfig:
    """Immutable configuration for multi-area laminar column scaffold.

    This dataclass encodes the complete specification for building and running
    a multi-area V1-V4-PFC cortical network with per-layer cell distributions,
    Izhikevich emitters, and spectrolaminar readout objectives.
    """
    # Anatomy
    areas: Tuple[str, ...]
    layers: Tuple[str, ...]
    cell_types: Tuple[str, ...]
    cell_dist: np.ndarray  # (n_layers, n_cell_types), normalized rows
    layer_fractions: Tuple[Tuple[str, float, float], ...]  # [(name, frac_min, frac_max), ...]
    layer_count_frac: dict
    layer_cell_type_frac: dict
    cell_colors: dict
    cell_signs: dict

    # Network
    n_neuron_per_column: int
    area_x_rel: Tuple[float, ...]
    cx_m: float
    cy_m: float
    cz_m: float
    radius_rel: float
    l4_ref_rel: float

    # Simulation
    dt_ms: float
    duration_ms: float
    seed: int
    dtype: str
    n_trials: int

    # Field & Readout
    n_contacts: int
    freq_min_hz: float
    freq_max_hz: float
    freq_count: int

    # I/O
    output_dir: Path
    base_control: dict = field(default_factory=dict)

    # Per-cell-type Izhikevich parameter overrides, e.g.
    #   {"E": {"a": 0.015, "b": 0.20, "c": -60.0, "d": 10.0, "drive": 7.0}, ...}
    # Any subset of {a, b, c, d, drive} per cell type overrides the package
    # default for every neuron of that type. Lets a tutorial shape per-type
    # waveform/frequency content (e.g. a WIDER, slower E AP -> more alpha-beta;
    # fast PV -> gamma) to test spectrolaminar properties. Native/uncalibrated
    # proxy units; does not change truth gates.
    cell_type_izh_params: dict = field(default_factory=dict)

    # Truth gates (immutable by frozen dataclass)
    truth_mode: str = "truth_safe_unverified"
    claim_level: str = "computational_scaffold"
    field_solver_status: str = "laminar_proxy_no_pde"
    field_claim_level: str = "proxy_readout_only"
    physical_amplitude_claim_allowed: bool = False

    @property
    def cell_dist_frame(self) -> "pd.DataFrame":
        """Return cell distribution as DataFrame."""
        import pandas as pd
        return pd.DataFrame(
            self.cell_dist,
            index=list(self.layers),
            columns=list(self.cell_types)
        )

    @property
    def truth_gates(self) -> dict:
        """Return immutable truth gate values."""
        return {
            'truth_mode': self.truth_mode,
            'claim_level': self.claim_level,
            'field_solver_status': self.field_solver_status,
            'field_claim_level': self.field_claim_level,
            'physical_amplitude_claim_allowed': self.physical_amplitude_claim_allowed,
        }

    def to_manifest_dict(self) -> dict:
        """Export configuration to JSON-safe manifest dict."""
        return {
            'areas': list(self.areas),
            'layers': list(self.layers),
            'cell_types': list(self.cell_types),
            'cell_dist_shape': self.cell_dist.shape,
            'n_neuron_per_column': self.n_neuron_per_column,
            'total_neurons': len(self.areas) * self.n_neuron_per_column,
            'dt_ms': float(self.dt_ms),
            'duration_ms': float(self.duration_ms),
            'seed': int(self.seed),
            'dtype': str(self.dtype),
            'n_trials': int(self.n_trials),
            'n_contacts': int(self.n_contacts),
            'freq_min_hz': float(self.freq_min_hz),
            'freq_max_hz': float(self.freq_max_hz),
            'freq_count': int(self.freq_count),
            'truth_gates': self.truth_gates,
        }


def _normalize_row(values: Sequence, eps: float = 1e-12) -> np.ndarray:
    """Normalize a row to sum to 1.0."""
    arr = np.asarray(values, dtype=np.float32)
    total = float(arr.sum())
    if total <= eps:
        raise ValueError(f"Row must have positive mass; got sum={total}")
    return arr / total


def make_cell_dist(
    layers: Sequence[str],
    cell_types: Sequence[str],
    layer_cell_type_frac: Mapping | None = None,
) -> np.ndarray:
    """Create cell distribution matrix (n_layers × n_cell_types).

    Each row is normalized to sum to 1.0, representing the per-layer
    cell-type composition.

    Parameters
    ----------
    layers : sequence of str
        Layer names (e.g., ["L1", "L2", "L3", "L4", "L5", "L6"])
    cell_types : sequence of str
        Cell type names (e.g., ["E", "PV", "SST", "VIP"])
    layer_cell_type_frac : dict or None
        Per-layer cell-type fractions, e.g.:
        {"L1": {"E": 0.75, "PV": 0.0, "SST": 0.0, "VIP": 0.25}, ...}
        If None, uniform distribution is used.

    Returns
    -------
    ndarray of shape (len(layers), len(cell_types))
        Each row sums to 1.0.

    Raises
    ------
    ValueError
        If any row has zero mass.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    if not cell_types:
        raise ValueError("cell_types must be non-empty")

    out = np.zeros((len(layers), len(cell_types)), dtype=np.float32)

    for i, layer in enumerate(layers):
        layer_frac = layer_cell_type_frac.get(layer, {}) if layer_cell_type_frac else {}
        vals = [float(layer_frac.get(ct, 1.0 / len(cell_types))) for ct in cell_types]
        out[i] = _normalize_row(vals)

    return out


def make_cell_type_catalog(
    selected: Sequence[str] | None = None,
    control: dict | None = None,
) -> dict[str, CellTypePreset]:
    """Create cell type catalog with Izhikevich presets.

    Parameters
    ----------
    selected : sequence of str or None
        Cell types to include (e.g., ["E", "PV", "SST", "VIP"]).
        If None, all types are included.
    control : dict or None
        Control dict with izhikevich parameters (currently unused).

    Returns
    -------
    dict mapping cell_type -> CellTypePreset
    """
    catalog = {
        'E': CellTypePreset(
            name='E-Regular',
            color='#1f77b4',  # blue
            sign=1.0,
            a=0.02, b=0.2, c=-65.0, d=8.0,
            drive=5.0,
            noise_std=0.85,
        ),
        'PV': CellTypePreset(
            name='PV-Fast-Spiking',
            color='#d62728',  # red
            sign=-1.0,
            a=0.1, b=0.2, c=-65.0, d=2.0,
            drive=3.0,
            noise_std=0.75,
        ),
        'SST': CellTypePreset(
            name='SST-Martinotti',
            color='#2ca02c',  # green
            sign=-1.0,
            a=0.08, b=0.25, c=-65.0, d=3.0,
            drive=3.5,
            noise_std=0.70,
        ),
        'VIP': CellTypePreset(
            name='VIP-Disinhibitory',
            color='#9467bd',  # purple
            sign=-1.0,
            a=0.05, b=0.26, c=-65.0, d=4.0,
            drive=3.0,
            noise_std=0.72,
        ),
    }

    if selected is not None:
        catalog = {k: v for k, v in catalog.items() if k in selected}

    return catalog


def cell_catalog_frame(catalog: dict[str, CellTypePreset]) -> "pd.DataFrame":
    """Return cell catalog as DataFrame for display."""
    import pandas as pd
    rows = []
    for cell_type, preset in catalog.items():
        rows.append({
            'Cell Type': cell_type,
            'Name': preset.name,
            'Color': preset.color,
            'Sign': preset.sign,
            'a': preset.a,
            'b': preset.b,
            'c': preset.c,
            'd': preset.d,
            'Drive': preset.drive,
            'Noise': preset.noise_std,
        })
    return pd.DataFrame(rows)


def make_laminar_column_config(
    *,
    areas=("V1", "V4", "PFC"),
    area_x_rel=(-1.0, 0.0, 1.0),
    layers=("L1", "L2", "L3", "L4", "L5", "L6"),
    layer_fractions=(
        ("L1", 0.00, 0.10),
        ("L2", 0.10, 0.25),
        ("L3", 0.25, 0.45),
        ("L4", 0.45, 0.55),
        ("L5", 0.55, 0.85),
        ("L6", 0.85, 1.00),
    ),
    layer_count_frac=None,
    layer_cell_type_frac=None,
    cell_types=("E", "PV", "SST", "VIP"),
    cell_colors=None,
    cell_signs=None,
    n_neuron_per_column=500,
    cx_m=1.0e-3,
    cy_m=1.0e-3,
    cz_m=1.0e-3,
    radius_rel=0.10,
    l4_ref_rel=0.50,
    dt_ms=0.1,
    duration_ms=1000.0,
    seed=20260512,
    dtype="float32",
    n_trials=10,
    n_contacts=32,
    freq_min_hz=1.0,
    freq_max_hz=150.0,
    freq_count=96,
    output_dir="outputs/jaxfne_etude_no_1",
    base_control=None,
    cell_type_izh_params=None,
    truth_mode="truth_safe_unverified",
    claim_level="computational_scaffold",
    field_solver_status="laminar_proxy_no_pde",
    field_claim_level="proxy_readout_only",
    physical_amplitude_claim_allowed=False,
) -> LaminarColumnConfig:
    """Create a laminar column configuration with all defaults visible.

    Parameters
    ----------
    areas : tuple of str
        Area names (e.g., ("V1", "V4", "PFC"))
    area_x_rel : tuple of float
        Relative x offset per area for spatial separation
    layers : tuple of str
        Layer names (e.g., ("L1", "L2", "L3", "L4", "L5", "L6"))
    layer_fractions : tuple of (str, float, float)
        Layer depth ranges as (name, frac_min, frac_max)
    layer_count_frac : dict or None
        Per-layer neuron count fractions. If None, uniform.
    layer_cell_type_frac : dict or None
        Per-layer per-cell-type fractions. If None, uniform.
    cell_types : tuple of str
        Cell type names (e.g., ("E", "PV", "SST", "VIP"))
    cell_colors : dict or None
        Hex colors per cell type. If None, use defaults.
    cell_signs : dict or None
        ±1.0 per cell type. If None, use defaults.
    n_neuron_per_column : int
        Neurons per area
    cx_m, cy_m, cz_m : float
        Grid spacing in meters
    radius_rel : float
        Column radius relative to spacing
    l4_ref_rel : float
        L4 position for depth reference (0-1)
    dt_ms : float
        Time step in milliseconds
    duration_ms : float
        Simulation duration
    seed : int
        PRNG seed
    dtype : str
        NumPy/JAX dtype (usually "float32")
    n_trials : int
        Default number of trials
    n_contacts : int
        Number of electrode contacts
    freq_min_hz, freq_max_hz, freq_count : float/int
        Spectrolaminar frequency range and resolution
    output_dir : str or Path
        Output directory
    base_control : dict or None
        Base control dict (plasticity, noise_scale, gains). If None, use defaults.
    truth_* : str or bool
        Truth gate values (immutable after config creation)

    Returns
    -------
    LaminarColumnConfig (frozen)
    """
    # Fill defaults for None arguments
    if layer_count_frac is None:
        layer_count_frac = {L: 1.0 / len(layers) for L in layers}

    if layer_cell_type_frac is None:
        layer_cell_type_frac = {L: {ct: 1.0 / len(cell_types) for ct in cell_types} for L in layers}

    if cell_colors is None:
        cell_colors = {
            'E': '#1f77b4',
            'PV': '#d62728',
            'SST': '#2ca02c',
            'VIP': '#9467bd',
        }

    if cell_signs is None:
        cell_signs = {
            'E': 1.0,
            'PV': -1.0,
            'SST': -1.0,
            'VIP': -1.0,
        }

    if base_control is None:
        base_control = {
            'plasticity': 0.10,
            'noise_scale': 1.00,
            'local_exc_gain': 1.00,
            'local_inh_gain': 1.00,
            'feedforward_gain': 1.00,
            'feedback_gain': 1.00,
        }

    if cell_type_izh_params is None:
        # Default to a WIDER, slower excitatory action potential (the "E-Wide"
        # profile): lower recovery rate `a` and higher `c`/`d` slow the AP so E
        # cells carry more low-frequency (alpha-beta) content, while the fast
        # interneurons (PV default a=0.10) keep their gamma-range dynamics.
        # Override any subset per cell type to test other waveforms.
        cell_type_izh_params = {
            'E': {'a': 0.015, 'b': 0.20, 'c': -60.0, 'd': 10.0},
        }

    # Create cell distribution matrix
    cell_dist = make_cell_dist(layers, cell_types, layer_cell_type_frac)

    # Ensure area_x_rel has correct length
    if len(area_x_rel) != len(areas):
        area_x_rel = tuple(np.linspace(-1, 1, len(areas)))

    return LaminarColumnConfig(
        areas=tuple(areas),
        layers=tuple(layers),
        cell_types=tuple(cell_types),
        cell_dist=cell_dist,
        layer_fractions=tuple(layer_fractions),
        layer_count_frac=dict(layer_count_frac),
        layer_cell_type_frac=dict(layer_cell_type_frac),
        cell_colors=dict(cell_colors),
        cell_signs=dict(cell_signs),
        n_neuron_per_column=int(n_neuron_per_column),
        area_x_rel=tuple(area_x_rel),
        cx_m=float(cx_m),
        cy_m=float(cy_m),
        cz_m=float(cz_m),
        radius_rel=float(radius_rel),
        l4_ref_rel=float(l4_ref_rel),
        dt_ms=float(dt_ms),
        duration_ms=float(duration_ms),
        seed=int(seed),
        dtype=str(dtype),
        n_trials=int(n_trials),
        n_contacts=int(n_contacts),
        freq_min_hz=float(freq_min_hz),
        freq_max_hz=float(freq_max_hz),
        freq_count=int(freq_count),
        output_dir=Path(output_dir),
        base_control=dict(base_control),
        cell_type_izh_params={k: dict(v) for k, v in dict(cell_type_izh_params).items()},
        truth_mode=truth_mode,
        claim_level=claim_level,
        field_solver_status=field_solver_status,
        field_claim_level=field_claim_level,
        physical_amplitude_claim_allowed=physical_amplitude_claim_allowed,
    )


def config_summary_frame(cfg: LaminarColumnConfig) -> "pd.DataFrame":
    """Return config as a summary DataFrame."""
    import pandas as pd
    rows = [
        ('Areas', ', '.join(cfg.areas)),
        ('Layers', ', '.join(cfg.layers)),
        ('Cell Types', ', '.join(cfg.cell_types)),
        ('Neurons per Area', cfg.n_neuron_per_column),
        ('Total Neurons', len(cfg.areas) * cfg.n_neuron_per_column),
        ('Cell Dist Shape', cfg.cell_dist.shape),
        ('Duration (ms)', cfg.duration_ms),
        ('dt (ms)', cfg.dt_ms),
        ('Seed', cfg.seed),
        ('Contacts', cfg.n_contacts),
        ('Freq Range (Hz)', f'{cfg.freq_min_hz}-{cfg.freq_max_hz}'),
        ('Truth Mode', cfg.truth_mode),
        ('Claim Level', cfg.claim_level),
        ('Field Status', cfg.field_solver_status),
    ]
    return pd.DataFrame(rows, columns=['Parameter', 'Value'])


def make_izhikevich_control_panel(
    cfg: LaminarColumnConfig | None = None,
    duration_ms: float = 20.0,
    dt_ms: float = 0.25,
    preset: str = "PV",
    theme: str = "gold_purple_dark",
) -> object:
    """Create interactive Izhikevich waveform control panel (ipywidgets).

    Parameters
    ----------
    cfg : LaminarColumnConfig or None
        Configuration (currently for reference)
    duration_ms, dt_ms : float
        Waveform simulation duration and timestep
    preset : str
        Initial cell type preset (e.g., "PV", "E", "SST", "VIP")
    theme : str
        Visual theme name

    Returns
    -------
    ipywidgets.VBox or similar
        Interactive panel widget

    Notes
    -----
    This function may not work in all environments (requires ipywidgets).
    It returns a display widget for interactive exploration.
    """
    try:
        import ipywidgets as widgets
    except ImportError:
        print("WARNING: ipywidgets not available; returning None")
        return None

    # Create simple placeholder widget with preset selector
    preset_widget = widgets.Dropdown(
        options=['E', 'PV', 'SST', 'VIP'],
        value=preset,
        description='Preset:',
    )

    info_widget = widgets.HTML(
        f"<b>Izhikevich Waveform Control Panel</b><br>"
        f"Duration: {duration_ms} ms, dt: {dt_ms} ms<br>"
        f"Selected: {preset}"
    )

    return widgets.VBox([info_widget, preset_widget])


def collect_izhikevich_control(panel: object) -> dict:
    """Extract control values from interactive panel.

    Parameters
    ----------
    panel : widget or None
        Panel returned by make_izhikevich_control_panel()

    Returns
    -------
    dict
        Control dictionary (currently empty or placeholder)
    """
    return {'preset': 'PV'}  # Placeholder


def make_stimulus(
    *,
    kind: str = "constant",
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    amplitude: float = 1.0,
    frequency_hz: float = 10.0,
    phase_rad: float = 0.0,
    onset_ms: float = 0.0,
    offset_ms: float | None = None,
    baseline: float = 0.0,
    pulse_width_ms: float = 5.0,
    pulse_period_ms: float = 50.0,
    noise_std: float = 1.0,
    seed: int = 0,
    dtype: str = "float32",
) -> np.ndarray:
    """Create stimulus array for neural simulation.

    Parameters
    ----------
    kind : str
        Stimulus type: "constant", "sine", "step", "pulses", "noise"
    duration_ms : float
        Duration in milliseconds
    dt_ms : float
        Time step in milliseconds
    amplitude : float
        Stimulus amplitude
    frequency_hz : float
        Frequency for sine stimulus
    phase_rad : float
        Phase offset for sine stimulus
    onset_ms, offset_ms : float
        Stimulus on/off times
    baseline : float
        Baseline/DC level
    pulse_width_ms, pulse_period_ms : float
        Pulse parameters
    noise_std : float
        Noise standard deviation
    seed : int
        PRNG seed for noise
    dtype : str
        NumPy dtype

    Returns
    -------
    ndarray of shape (n_steps,)
        Stimulus array
    """
    n_steps = int(round(duration_ms / dt_ms))
    t = np.arange(n_steps) * dt_ms

    if offset_ms is None:
        offset_ms = duration_ms

    stim = np.full(n_steps, baseline, dtype=dtype)

    # Apply stimulus window
    mask = (t >= onset_ms) & (t < offset_ms)

    if kind == "constant":
        stim[mask] = baseline + amplitude
    elif kind == "sine":
        omega = 2.0 * np.pi * frequency_hz / 1000.0
        stim[mask] = baseline + amplitude * np.sin(omega * t[mask] + phase_rad)
    elif kind == "step":
        stim[mask] = baseline + amplitude
    elif kind == "pulses":
        pulse_period_dt = int(round(pulse_period_ms / dt_ms))
        pulse_width_dt = int(round(pulse_width_ms / dt_ms))
        for i in np.where(mask)[0]:
            if (i % pulse_period_dt) < pulse_width_dt:
                stim[i] = baseline + amplitude
    elif kind == "noise":
        rng = np.random.RandomState(seed)
        stim[mask] = baseline + amplitude * rng.randn(mask.sum()) * noise_std

    return np.asarray(stim, dtype=dtype)


def build_laminar_column(cfg: LaminarColumnConfig) -> dict:
    """Build a laminar column scaffold model.

    Parameters
    ----------
    cfg : LaminarColumnConfig
        Configuration specifying the network architecture

    Returns
    -------
    dict
        Model dict with keys: neurons, positions_m, W_parts, truth_gates
    """
    import pandas as pd
    # Create neuron table
    neuron_id = 0
    neurons = []

    for area_idx, area in enumerate(cfg.areas):
        for n in range(cfg.n_neuron_per_column):
            # Assign layer and cell type based on distributions
            layer_idx = n % len(cfg.layers)
            layer = cfg.layers[layer_idx]

            # Cell type assignment (round-robin or based on cell_dist)
            cell_type_idx = (n // len(cfg.layers)) % len(cfg.cell_types)
            cell_type = cfg.cell_types[cell_type_idx]

            # Vertical-cylinder geometry: depth runs along z; the (x, y) base is a
            # circular cross-section of radius ``radius_rel * cx_m`` centred on the
            # area. This is the "cylindric rule" — each area is a slender vertical
            # column, areas separated along x by ``area_x_rel``.
            layer_frac_min, layer_frac_max = next(
                (f[1:] for f in cfg.layer_fractions if f[0] == layer),
                (0.0, 1.0)
            )
            rng = np.random.RandomState(cfg.seed + neuron_id)

            # Depth (z) sampled within this layer's depth band -> vertical axis.
            z_rel = rng.uniform(layer_frac_min, layer_frac_max)
            z_m = z_rel * cfg.cz_m

            # Position from L4 reference.
            l4_z = cfg.l4_ref_rel * cfg.cz_m
            pos_from_l4 = z_m - l4_z

            # Circular base: uniform over a disk of radius ``radius_rel * cx_m``
            # centred at (area_x, 0). sqrt(U) gives uniform area density.
            radius_m = cfg.radius_rel * cfg.cx_m
            theta = rng.uniform(0.0, 2.0 * np.pi)
            r = radius_m * np.sqrt(rng.uniform(0.0, 1.0))
            x_m = cfg.area_x_rel[area_idx] * cfg.cx_m + r * np.cos(theta)
            y_m = r * np.sin(theta)

            neurons.append({
                'neuron_id': neuron_id,
                'area': area,
                'layer': layer,
                'cell_type': cell_type,
                'x_m': float(x_m),
                'y_m': float(y_m),
                'z_m': float(z_m),
                'pos_from_l4': float(pos_from_l4),
            })
            neuron_id += 1

    neurons_df = pd.DataFrame(neurons)
    positions_m = neurons_df[['x_m', 'y_m', 'z_m']].values.astype(np.float32)

    # Build connection matrices (placeholder: sparse random)
    n_neurons = len(neurons_df)
    rng = np.random.RandomState(cfg.seed)

    W_local_exc = rng.randn(n_neurons, n_neurons).astype(np.float32) * 0.01
    W_local_inh = rng.randn(n_neurons, n_neurons).astype(np.float32) * (-0.01)
    W_ff = rng.randn(n_neurons, n_neurons).astype(np.float32) * 0.005
    W_fb = rng.randn(n_neurons, n_neurons).astype(np.float32) * 0.005

    return {
        'neurons': neurons_df,
        'positions_m': positions_m,
        'W_parts': {
            'local_exc': W_local_exc,
            'local_inh': W_local_inh,
            'feedforward': W_ff,
            'feedback': W_fb,
        },
        'truth_gates': cfg.truth_gates,
    }


def build_laminar_connections(model: dict, cfg: LaminarColumnConfig) -> Tuple:
    """Extract connection matrices from model.

    Parameters
    ----------
    model : dict
        Model dict from build_laminar_column()
    cfg : LaminarColumnConfig
        Configuration

    Returns
    -------
    tuple of (W_local_exc, W_local_inh, W_ff, W_fb)
    """
    W_parts = model['W_parts']
    return (
        W_parts['local_exc'],
        W_parts['local_inh'],
        W_parts['feedforward'],
        W_parts['feedback'],
    )


def select_cells(
    model: dict,
    area: str | None = None,
    layers: Sequence[str] | None = None,
    cell_types: Sequence[str] | None = None,
    fraction: float = 1.0,
    max_cells: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Select cell indices from model matching criteria.

    Parameters
    ----------
    model : dict
        Model from build_laminar_column()
    area : str or None
        Area name to filter
    layers : sequence of str or None
        Layer names to include
    cell_types : sequence of str or None
        Cell type names to include
    fraction : float
        Fraction of matched cells to select (0-1)
    max_cells : int or None
        Maximum number of cells to return
    seed : int
        PRNG seed

    Returns
    -------
    ndarray of cell indices
    """
    df = model['neurons']
    mask = np.ones(len(df), dtype=bool)

    if area is not None:
        mask &= (df['area'] == area).values

    if layers is not None:
        mask &= df['layer'].isin(layers).values

    if cell_types is not None:
        mask &= df['cell_type'].isin(cell_types).values

    indices = np.where(mask)[0]

    if fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_select = max(1, int(len(indices) * fraction))
        indices = rng.choice(indices, size=n_select, replace=False)

    if max_cells is not None:
        indices = indices[:max_cells]

    return np.asarray(indices, dtype=int)


def simulate_laminar_trials(
    model: dict,
    cfg: LaminarColumnConfig,
    control: dict | None = None,
    stimulus: np.ndarray | None = None,
    target_cells: np.ndarray | None = None,
    n_trials: int | None = None,
    seed_offset: int = 0,
) -> dict:
    """Simulate laminar column with multiple trials.

    Parameters
    ----------
    model : dict
        Model from build_laminar_column()
    cfg : LaminarColumnConfig
        Configuration
    control : dict or None
        Control parameters (plasticity, noise_scale, gains)
    stimulus : ndarray or None
        Stimulus array of shape (n_steps,)
    target_cells : ndarray or None
        Target indices to apply stimulus to
    n_trials : int or None
        Number of trials (default from cfg)
    seed_offset : int
        Offset for trial seeds

    Returns
    -------
    dict with keys: time_ms, spikes, voltage_mV, source_native, lfp_contacts,
                    csd_contacts, contact_depths_m, area_names, control

    Tensor contract
    ---------------
    spikes / voltage_mV / source_native : (n_trials, n_steps, n_neurons)
    lfp_contacts / csd_contacts         : (n_trials, n_areas, n_steps, n_contacts)
        Per-area laminar-proxy projections produced by a depth-dependent
        Gaussian leadfield (jaxfne.fields.project_laminar_sources) applied to
        the real per-neuron Izhikevich source traces, so band power genuinely
        varies with cortical depth. NOT a solved PDE; field_solver_status stays
        "laminar_proxy_no_pde" and physical_amplitude_claim_allowed=False.
    contact_depths_m                    : (n_contacts,)
    area_names                          : tuple[str, ...]
    """
    if control is None:
        control = cfg.base_control

    if n_trials is None:
        n_trials = cfg.n_trials

    neurons_df = model['neurons']
    n_neurons = len(neurons_df)
    n_steps = int(round(cfg.duration_ms / cfg.dt_ms))
    areas = tuple(cfg.areas)
    n_areas = len(areas)

    # Per-area neuron index groups (areas are partitioned in build_laminar_column).
    area_indices = [
        np.flatnonzero((neurons_df['area'] == area).to_numpy())
        for area in areas
    ]

    # Real engine + depth-dependent leadfield (package-native; no local mock).
    import jax
    import jax.numpy as jnp
    import dataclasses as _dataclasses
    from .emitters import izhikevich_params_from_labels, simulate_eig_izhikevich
    from .fields import project_laminar_sources

    # Per-neuron cell/layer labels and relative laminar depth (z in [0, 1]).
    cell_labels = [str(x) for x in neurons_df['cell_type'].tolist()]
    layer_labels = [str(x) for x in neurons_df['layer'].tolist()]
    z_rel_all = np.clip(
        neurons_df['z_m'].to_numpy(dtype=np.float64) / max(float(cfg.cz_m), 1e-12),
        0.0, 1.0,
    )

    # Control knobs become real proxy parameters (previously ignored): the
    # external stimulus is a thalamic-like drive into the targeted cells scaled
    # by feedforward_gain, and the recurrent weights are scaled per presynaptic
    # sign by local exc/inh gain. These are native/uncalibrated proxy units.
    ff_gain = float(control.get('feedforward_gain', 1.0))
    exc_gain = float(control.get('local_exc_gain', 1.0))
    inh_gain = float(control.get('local_inh_gain', 1.0))

    if stimulus is None:
        stim_vec = np.zeros(n_steps, dtype=np.float64)
    else:
        stim_vec = np.asarray(stimulus, dtype=np.float64).reshape(-1)
        if stim_vec.shape[0] < n_steps:
            stim_vec = np.pad(stim_vec, (0, n_steps - stim_vec.shape[0]))
        else:
            stim_vec = stim_vec[:n_steps]

    target_mask_global = np.zeros(n_neurons, dtype=bool)
    if target_cells is not None and np.asarray(target_cells).size > 0:
        target_mask_global[np.asarray(target_cells, dtype=int)] = True

    # Output arrays (tensor contract preserved)
    time_ms = np.arange(n_steps) * cfg.dt_ms
    spikes = np.zeros((n_trials, n_steps, n_neurons), dtype=bool)
    voltage_mV = np.zeros((n_trials, n_steps, n_neurons), dtype=np.float32)
    source_native = np.zeros((n_trials, n_steps, n_neurons), dtype=np.float32)
    lfp_contacts = np.zeros((n_trials, n_areas, n_steps, cfg.n_contacts), dtype=np.float32)
    csd_contacts = np.zeros((n_trials, n_areas, n_steps, cfg.n_contacts), dtype=np.float32)

    field_diagnostics = None

    for ai, idx in enumerate(area_indices):
        if idx.size == 0:
            continue

        area_cells = [cell_labels[i] for i in idx]
        area_layers = [layer_labels[i] for i in idx]
        params = izhikevich_params_from_labels(
            area_cells, layer_labels=area_layers, dtype=cfg.dtype,
        )

        # Per-cell-type Izhikevich parameter overrides (a, b, c, d, drive) so a
        # tutorial can shape per-type waveform/frequency content (e.g. wider/
        # slower E -> more alpha-beta; fast PV -> gamma).
        izh_over = getattr(cfg, "cell_type_izh_params", None) or {}
        if izh_over:
            a_arr = np.asarray(params.a, dtype=np.float32).copy()
            b_arr = np.asarray(params.b, dtype=np.float32).copy()
            c_arr = np.asarray(params.c, dtype=np.float32).copy()
            d_arr = np.asarray(params.d, dtype=np.float32).copy()
            drv_arr = np.asarray(params.drive, dtype=np.float32).copy()
            cells_arr = np.asarray(area_cells)
            for ct, p in izh_over.items():
                m = cells_arr == ct
                if not m.any():
                    continue
                if 'a' in p: a_arr[m] = float(p['a'])
                if 'b' in p: b_arr[m] = float(p['b'])
                if 'c' in p: c_arr[m] = float(p['c'])
                if 'd' in p: d_arr[m] = float(p['d'])
                if 'drive' in p: drv_arr[m] = float(p['drive'])
            u0_arr = b_arr * np.asarray(params.v0, dtype=np.float32)
            params = _dataclasses.replace(
                params,
                a=jnp.asarray(a_arr), b=jnp.asarray(b_arr),
                c=jnp.asarray(c_arr), d=jnp.asarray(d_arr),
                drive=jnp.asarray(drv_arr), u0=jnp.asarray(u0_arr),
            )

        # Scale recurrent weights by per-presynaptic-sign local gain.
        sign = np.asarray(params.sign)
        col_gain = np.where(sign > 0, exc_gain, inh_gain).astype(np.float32)
        scaled_W = params.W * jnp.asarray(col_gain)[None, :]
        params = _dataclasses.replace(params, W=scaled_W)

        # Per-area external drive schedule (n_steps, n_area_neurons).
        area_target = target_mask_global[idx]
        drive_sched = np.zeros((n_steps, idx.size), dtype=np.float32)
        if area_target.any():
            drive_sched[:, area_target] = (ff_gain * stim_vec)[:, None]
        drive_sched_j = jnp.asarray(drive_sched)

        # Positions for the laminar leadfield (only depth, column 2, is used).
        positions = np.zeros((idx.size, 3), dtype=np.float32)
        positions[:, 2] = z_rel_all[idx]
        positions_j = jnp.asarray(positions)

        for trial in range(n_trials):
            key = jax.random.PRNGKey(
                int(cfg.seed) + int(seed_offset) + 1009 * ai + trial
            )
            v, spk, src = simulate_eig_izhikevich(
                params, n_steps, cfg.dt_ms, key,
                dtype=cfg.dtype, drive_schedule=drive_sched_j,
            )
            v = np.asarray(v)
            spk = np.asarray(spk)
            src = np.asarray(src)
            voltage_mV[trial][:, idx] = v
            spikes[trial][:, idx] = spk > 0.5
            source_native[trial][:, idx] = src

            # Depth-dependent Gaussian leadfield -> genuine laminar LFP/CSD that
            # reflect each cell's depth and oscillatory content. Truth-gated as
            # laminar_proxy_no_pde (no field solve; amplitude claim not allowed).
            field = project_laminar_sources(
                jnp.asarray(src), positions_j,
                n_contacts=cfg.n_contacts, dtype=cfg.dtype,
            )
            lfp_contacts[trial, ai] = np.asarray(field.lfp_proxy)
            csd_contacts[trial, ai] = np.asarray(field.csd_proxy)
            if field_diagnostics is None:
                field_diagnostics = field.diagnostics

    # Contact depths in meters (leadfield uses relative [0,1]; map for plotting).
    contact_depths_m = np.linspace(0.0, cfg.cz_m, cfg.n_contacts)

    return {
        'time_ms': time_ms,
        'spikes': spikes,
        'voltage_mV': voltage_mV,
        'source_native': source_native,
        'lfp_contacts': lfp_contacts,
        'csd_contacts': csd_contacts,
        'contact_depths_m': contact_depths_m,
        'area_names': cfg.areas,
        'control': control,
        'field_solver_status': cfg.field_solver_status,
        'field_diagnostics': field_diagnostics,
    }


def spectrolaminar_from_trials(
    trials: dict,
    cfg: LaminarColumnConfig,
    signal_key: str = "csd_contacts",
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    freq_count: int | None = None,
    alpha_beta_range_hz: Tuple[float, float] = (10.0, 25.0),
    gamma_range_hz: Tuple[float, float] = (40.0, 150.0),
    area_index: int | None = None,
    area_name: str | None = None,
    low_power_frac: float = 0.02,
) -> Tuple[np.ndarray, dict]:
    """Extract spectrolaminar profiles from trial data.

    Parameters
    ----------
    trials : dict
        Trials dict from simulate_laminar_trials()
    cfg : LaminarColumnConfig
        Configuration
    signal_key : str
        Which signal to analyze ("csd_contacts", "lfp_contacts", etc.)
    freq_min_hz, freq_max_hz, freq_count : float or None
        Frequency range and resolution (use cfg defaults if None)
    alpha_beta_range_hz, gamma_range_hz : tuple of float
        Frequency bands for profiles
    area_index : int or None
        For 4-D signals (n_trials, n_areas, n_steps, n_contacts), select this
        area. Defaults to 0. Ignored for legacy 3-D signals.
    area_name : str or None
        Name of the area (for tagging the returned spec dict). Included in the
        returned dict as 'area' field.

    Returns
    -------
    tuple of (power_heatmap, profiles_dict)
        power_heatmap: ndarray of shape (n_freqs, n_contacts)
        profiles_dict: dict with keys 'freq_hz', 'pos_from_l4', 'alpha_beta', 'gamma', 'relative_power'
    """
    if freq_min_hz is None:
        freq_min_hz = cfg.freq_min_hz
    if freq_max_hz is None:
        freq_max_hz = cfg.freq_max_hz
    if freq_count is None:
        freq_count = cfg.freq_count

    signal = trials[signal_key]
    fs = 1000.0 / cfg.dt_ms

    # Support both the per-area 4-D contract (n_trials, n_areas, n_steps,
    # n_contacts) and the legacy 3-D form (n_trials, n_steps, n_contacts).
    if signal.ndim == 4:
        ai = 0 if area_index is None else int(area_index)
        signal = signal[:, ai]  # (n_trials, n_steps, n_contacts)

    # Spectral power, averaged ACROSS TRIALS in the power (magnitude) domain.
    #
    # Compute the spectral magnitude per trial, then average the per-trial
    # spectra. Averaging the raw signal in the TIME domain first (the previous
    # behavior) cancels every oscillation that is not phase-locked across trials
    # and leaves mostly noise — which is why adding trials never cleaned up the
    # motif and the power spectrum looked like a checkerboard. Power/magnitude
    # averaging is the standard estimator for laminar oscillatory power: each
    # added trial genuinely reduces variance and reveals the band structure.
    signal = np.asarray(signal)  # (n_trials, n_steps, n_contacts)
    n_trials_eff, n_steps, n_contacts = signal.shape

    freqs = np.linspace(freq_min_hz, freq_max_hz, freq_count)
    sample_idx = np.arange(n_steps)
    # DFT basis (freq_count, n_steps), reused for every trial/contact.
    basis = np.exp(-1j * 2.0 * np.pi * (freqs[:, None] / fs) * sample_idx[None, :])

    psd_accum = np.zeros((freq_count, n_contacts), dtype=np.float64)
    for ti in range(n_trials_eff):
        # (freq_count, n_steps) @ (n_steps, n_contacts) -> (freq_count, n_contacts)
        spec_t = basis @ signal[ti]
        psd_accum += np.abs(spec_t) / max(n_steps, 1)
    psd = (psd_accum / max(n_trials_eff, 1)).astype(np.float32)

    # Normalize to relative power. Use a robust scale that IGNORES degenerate
    # low-power channels (e.g. the top contact whose CSD is ~0 from the spatial
    # derivative) — otherwise those channels drag the percentile floor down and
    # collapse the relative-power contrast across depth.
    psd_log = np.log10(psd + 1e-12)
    chan_power = psd.mean(axis=0)
    pmax = float(chan_power.max())
    if pmax > 0:
        valid = chan_power >= (low_power_frac * pmax)
    else:
        valid = np.ones(n_contacts, dtype=bool)
    if int(valid.sum()) < 2:
        valid = np.ones(n_contacts, dtype=bool)
    vmin, vmax = np.percentile(psd_log[:, valid], [5, 95])
    relative_power = np.clip((psd_log - vmin) / (vmax - vmin + 1e-12), 0, 1)

    # Extract band profiles for the spectrolaminar cross (panel C).
    #
    # Relative power density definition: for each band, take the mean spectral
    # power across that band's frequencies at every channel, then divide by the
    # MAXIMUM band power across channels. Each band is thus normalized to its own
    # depth-wise peak (=1.0), so the alpha-beta (deep) and gamma (superficial)
    # profiles share a [0, 1] scale and visibly cross near L4.
    #
    # Computed from the raw spectral power `psd` (the same quantity panel B shows
    # in log scale), NOT from the percentile-clipped log heatmap `relative_power`
    # — clipping distorts true relative magnitudes across depth. Normalizing by
    # the per-band max across channels is naturally robust to degenerate
    # low-power channels (e.g. the top CSD contact ~0): a near-zero channel maps
    # to ~0, it cannot inflate the max.
    ab_mask = (freqs >= alpha_beta_range_hz[0]) & (freqs <= alpha_beta_range_hz[1])
    gm_mask = (freqs >= gamma_range_hz[0]) & (freqs <= gamma_range_hz[1])

    ab_power = psd[ab_mask].mean(axis=0) if ab_mask.any() else np.ones(n_contacts)
    gm_power = psd[gm_mask].mean(axis=0) if gm_mask.any() else np.ones(n_contacts)

    ab_profile = ab_power / (float(ab_power.max()) + 1e-12)
    gm_profile = gm_power / (float(gm_power.max()) + 1e-12)

    # Depth axis (position from L4)
    pos_from_l4 = trials['contact_depths_m'] - cfg.l4_ref_rel * cfg.cz_m

    return relative_power, {
        'area': area_name,
        'freq_hz': freqs,
        'pos_from_l4': pos_from_l4,
        'relative_power': relative_power,
        'alpha_beta': ab_profile,
        'gamma': gm_profile,
    }


def summarize_spectrolaminar_similarity(
    trials: dict,
    cfg: LaminarColumnConfig,
    areas: Sequence[str] | None = None,
    signal_key: str = "csd_contacts",
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    freq_count: int | None = None,
    alpha_beta_range_hz: Tuple[float, float] = (10.0, 25.0),
    gamma_range_hz: Tuple[float, float] = (40.0, 150.0),
) -> Tuple[pd.DataFrame, dict]:
    """Compute spectrolaminar similarity scores and return specs.

    Parameters
    ----------
    trials : dict
        Trials dict
    cfg : LaminarColumnConfig
        Configuration
    areas : sequence of str or None
        Areas to analyze (default: cfg.areas)
    signal_key : str
        Signal key to analyze
    freq_min_hz, freq_max_hz, freq_count : float or None
        Frequency parameters
    alpha_beta_range_hz, gamma_range_hz : tuple
        Band ranges

    Returns
    -------
    tuple of (scores_df, specs_dict)
        scores_df: DataFrame with columns [area, similarity_percent]
        specs_dict: dict[area -> power/profile dicts]
    """
    import pandas as pd
    if areas is None:
        areas = cfg.areas

    # Map requested area names to their index along the per-area signal axis.
    all_areas = list(cfg.areas)

    scores = []
    specs_dict = {}

    for area in areas:
        area_index = all_areas.index(area) if area in all_areas else 0

        # Genuine per-area spectrolaminar profile (no copy-same-spec).
        power, spec = spectrolaminar_from_trials(
            trials, cfg, signal_key,
            freq_min_hz, freq_max_hz, freq_count,
            alpha_beta_range_hz, gamma_range_hz,
            area_index=area_index,
            area_name=area,
        )

        # Deterministic similarity derived from this area's band profiles:
        # the spectrolaminar motif is an alpha-beta deep / gamma superficial
        # gradient, so score the anti-correlation between the two depth
        # profiles. Finite, JSON-safe, reproducible (no RNG).
        ab = np.asarray(spec['alpha_beta'], dtype=np.float64)
        gm = np.asarray(spec['gamma'], dtype=np.float64)
        if ab.size >= 2 and np.std(ab) > 1e-9 and np.std(gm) > 1e-9:
            corr = float(np.corrcoef(ab, gm)[0, 1])
        else:
            corr = 0.0
        # Anti-correlation (corr = -1) maps to 100%; corr = +1 maps to 0%.
        similarity_pct = float(np.clip((1.0 - corr) * 50.0, 0.0, 100.0))

        scores.append({
            'area': area,
            'similarity_percent': similarity_pct,
        })
        # Tag the spec so downstream figures can title with the motif score.
        spec['similarity_percent'] = similarity_pct
        spec['n_trials'] = int(np.asarray(trials[signal_key]).shape[0])
        specs_dict[area] = spec

    scores_df = pd.DataFrame(scores)

    return scores_df, specs_dict


def export_tutorial_artifacts(
    cfg: LaminarColumnConfig,
    manifest_dict: dict | None = None,
    metrics_dict: dict | None = None,
    validation_dict: dict | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Export configuration and results as strict JSON.

    Parameters
    ----------
    cfg : LaminarColumnConfig
        Configuration
    manifest_dict, metrics_dict, validation_dict : dict or None
        Data to export
    output_dir : Path or None
        Output directory (use cfg.output_dir if None)

    Returns
    -------
    dict with keys: manifest_path, metrics_path, validation_path
    """
    if output_dir is None:
        output_dir = cfg.output_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare manifests
    if manifest_dict is None:
        manifest_dict = cfg.to_manifest_dict()

    if metrics_dict is None:
        metrics_dict = {}

    if validation_dict is None:
        validation_dict = {}

    # Write JSON (strict, no NaN/Inf)
    def _to_jsonable(obj):
        """Recursively convert to JSON-safe types."""
        if isinstance(obj, (np.ndarray, np.generic)):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, (dict, dict)):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_to_jsonable(x) for x in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    paths = {}

    for name, data in [
        ('manifest', manifest_dict),
        ('metrics', metrics_dict),
        ('validation_report', validation_dict),
    ]:
        if not data:
            continue

        data_safe = _to_jsonable(data)
        path = output_dir / f"{name}.json"

        # Strict JSON: no NaN/Inf
        text = json.dumps(data_safe, indent=2, allow_nan=False)
        path.write_text(text)
        paths[f'{name}_path'] = str(path)

    return paths
