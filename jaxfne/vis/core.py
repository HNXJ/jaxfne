"""Core visualization utilities and metadata containers for jaxfne/vis.

Evaluated as an uncalibrated computational scaffold.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import jax
import numpy as np


@dataclass(frozen=True)
class FigureResult:
    """Rich container holding a matplotlib figure and JSON-safe metadata.

    Evaluated as an uncalibrated computational scaffold.
    """

    fig: Any
    metadata: dict[str, Any]


def require_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError(
            "The visualization features require the optional dependency 'matplotlib'. "
            "Please install it via `pip install matplotlib` or `pip install jaxfne[viz]`."
        )


def close_all() -> None:
    """Close all open matplotlib figures (``plt.close("all")``).

    The one matplotlib call every caller needs that isn't "render a figure" --
    exists so scripts/examples never need a raw ``import matplotlib.pyplot``
    just for cleanup. Lazily imports matplotlib; no-op if it's not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.close("all")


def prepare_static_plot_matrix(arr: Any) -> Any:
    """Extracts JAX or NumPy arrays safely to a static NumPy array on the host.

    Protects trace compilation context by forcing immediate device-to-host transfer.
    """
    if arr is None:
        return None
    try:
        # Check if the array is an accelerator array (has a device method or attribute)
        if hasattr(arr, "device") or hasattr(arr, "device_buffer"):
            return np.asarray(jax.device_get(arr))
    except Exception:
        pass
    return np.asarray(arr)


def is_E_cell_type(cell_type: str) -> bool:
    """True for excitatory class labels (shared viewer helper)."""
    return str(cell_type) == "E"


def get_time_ms(signals: Any, default_len: int) -> np.ndarray:
    """Time axis for a Signals-like or dict payload (shared viewer helper).

    Falls back to ``np.arange(default_len)`` (sample indices, NOT
    milliseconds) when the payload carries no ``time_ms``. Callers must use
    :func:`time_axis` so the axis label reflects which case applied.
    """
    time_raw = getattr(signals, "time_ms", None)
    if time_raw is None and isinstance(signals, dict):
        time_raw = signals.get("time_ms")
    time_ms = prepare_static_plot_matrix(time_raw)
    if time_ms is None:
        return np.arange(default_len)
    return time_ms


_MS_LABEL = "Time (ms)"
_INDEX_LABEL = "Time step index"


def time_axis(signals: Any, default_len: int) -> tuple[np.ndarray, str]:
    """Time axis plus the honest axis label.

    Returns ``(axis, label)`` where ``label`` is ``"Time (ms)"`` when the
    payload carries ``time_ms`` and ``"Time step index"`` when falling back
    to sample indices. Use this instead of :func:`get_time_ms` + a hardcoded
    ``"Time (ms)"`` label (P3: index data labeled as ms).
    """
    time_raw = getattr(signals, "time_ms", None)
    if time_raw is None and isinstance(signals, dict):
        time_raw = signals.get("time_ms")
    time_ms = prepare_static_plot_matrix(time_raw)
    if time_ms is None:
        return np.arange(default_len), _INDEX_LABEL
    return time_ms, _MS_LABEL


def binned_population_rate_hz(
    spikes: Any, dt_ms: float, bin_ms: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Population mean firing rate (Hz) in non-overlapping bins.

    Canonical numeric contract shared by the matplotlib
    (:func:`jaxfne.vis.traces.rate`) and Plotly
    (:func:`jaxfne.vis.plotly.raster.plot_population_rates`) renderers (P5):
    ``rate = mean over units and steps in bin * (1000 / dt_ms)``. ``spikes``
    is ``(n_steps, n_units)``. Returns ``    (centers_ms, rate_hz)`` with centers
    relative to the first sample. ``bin_ms <= dt_ms`` collapses to the
    per-step instantaneous rate. Short inputs (``T < bin_steps``) yield one
    bin over the available steps (overall mean, as in the tutorials).
    Input dtype is preserved (no float32→float64 upcast): float32 simulation
    data stays float32 through the reduction.
    """
    arr = np.asarray(spikes)
    if arr.ndim != 2:
        raise ValueError(f"binned_population_rate_hz expects (n_steps, n_units), got {arr.shape}")
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms!r}")
    if not np.isfinite(bin_ms) or bin_ms <= 0:
        raise ValueError(f"bin_ms must be positive, got {bin_ms!r}")
    n_steps = arr.shape[0]
    bin_steps = max(1, int(round(bin_ms / dt_ms)))
    n_bins = n_steps // bin_steps
    if n_bins == 0:
        return np.array([0.0]), np.array([float(arr.mean()) * (1000.0 / dt_ms)])
    trimmed = arr[: n_bins * bin_steps]
    per_bin = trimmed.reshape(n_bins, bin_steps, -1).mean(axis=(1, 2))
    rate_hz = per_bin * (1000.0 / dt_ms)
    centers_ms = (np.arange(n_bins) + 0.5) * bin_steps * dt_ms
    return centers_ms, rate_hz


def welch_psd(
    x: Any, fs_hz: float, *, nperseg: int = 256, freq_max_hz: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Shared Welch PSD along axis 0 (P5 canonical spectral contract).

    ``nperseg`` is clamped to the input length (scipy degrades gracefully;
    Item-9 probe 6 refuted the tiny-T crash). Returns ``(freqs, pxx)``.
    Input dtype is preserved (scipy keeps float32→float32); no upcast.
    """
    from scipy import signal as _signal

    arr = np.asarray(x)
    if arr.shape[0] < 1:
        raise ValueError("welch_psd needs at least one sample along axis 0")
    seg = int(min(nperseg, arr.shape[0]))
    freqs, pxx = _signal.welch(arr, fs=float(fs_hz), axis=0, nperseg=seg)
    if freq_max_hz is not None:
        keep = freqs <= freq_max_hz
        freqs, pxx = freqs[keep], pxx[keep]
    return freqs, pxx


def inband_power_mean(pxx: np.ndarray, freqs: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Mean PSD over in-band freqs per trailing channel (absolute proxy units).

    Canonical band-power contract (P5): absolute, per-contact, mean over the
    band. Returns 0.0 (or zeros) when the band is uncovered by the grid.
    Input dtype is preserved (no upcast).
    """
    pxx = np.asarray(pxx)
    freqs = np.asarray(freqs)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(mask):
        return np.zeros(pxx.shape[1:]) if pxx.ndim > 1 else np.float64(0.0)
    return np.mean(pxx[mask], axis=0)


def neuron_rows(signals: Any) -> list[dict[str, Any]]:
    """Copies of the neuron-table rows carried in signal metadata (shared)."""
    meta = (
        getattr(signals, "metadata", {})
        if not isinstance(signals, dict)
        else signals.get("metadata", {})
    )
    rows = meta.get("neuron_metadata") if isinstance(meta, dict) else None
    return [dict(row) for row in rows] if rows else []


def geometry3d_from_config(cfg: Any, *, areas=None, cell_types=None, figsize=(9, 7)) -> Any:
    """Internal: synthesise 3D geometry scatter from Configuration metadata.

    SYNTHETIC PROVENANCE (P4): the plotted coordinates are uniform random
    draws (``rng(42)``), NOT declared geometry — they stand in for column
    layout only. Axes are relative/synthetic units; no calibrated-mm mapping
    is implied. Do not use for field/geometry claims.
    """
    import matplotlib.pyplot as plt

    meta = cfg.metadata
    columns = meta.get("columns", [])
    ct_fracs = meta.get("cell_types", {})
    if isinstance(ct_fracs, dict) and not ct_fracs:
        ct_fracs = {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07}
    all_cell_types = (
        list(ct_fracs.keys()) if isinstance(ct_fracs, dict) else ["E", "PV", "SST", "VIP"]
    )
    if cell_types is not None:
        all_cell_types = [c for c in all_cell_types if c in cell_types]

    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    colors_map = {
        ct: plt.cm.Set1(i / max(len(all_cell_types), 1)) for i, ct in enumerate(all_cell_types)
    }

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
            ax.scatter(
                x,
                y,
                z,
                s=6,
                alpha=0.55,
                label=ct if col_idx == 0 else "",
                color=colors_map.get(ct, "gray"),
            )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors_map.get(ct, "gray"),
            markersize=8,
            label=ct,
        )
        for ct in all_cell_types
    ]
    ax.legend(handles=handles, title="Cell type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(
        "Synthetic 3D geometry (Configuration proxy — NOT declared coordinates)", fontsize=11
    )
    ax.set_xlabel("x (synthetic relative units)")
    ax.set_ylabel("y (synthetic relative units)")
    ax.set_zlabel("z — synthetic depth (relative)")
    fig.tight_layout()
    return fig
