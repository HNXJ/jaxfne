"""Shared helpers for the Plotly visualization pipeline. Not public API."""
from __future__ import annotations


import numpy as np

CELL_TYPE_COLORS = {
    "E": "#1f77b4",
    "PV": "#d62728",
    "SST": "#2ca02c",
    "VIP": "#9467bd",
}


def require_plotly():
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "plotly is required for jaxfne.vis.plotly.* — install with `pip install plotly`"
        ) from exc


def neuron_table_arrays(model) -> dict[str, np.ndarray]:
    """Pull neuron_table() into flat numpy arrays, once."""
    nt = model.neuron_table()
    return {
        "neuron_id": np.array([r["neuron_id"] for r in nt]),
        "area": np.array([r["area"] for r in nt]),
        "layer": np.array([r["layer"] for r in nt]),
        "cell_type": np.array([r["cell_type"] for r in nt]),
        "x": np.array([r["x"] for r in nt], dtype=float),
        "y": np.array([r["y"] for r in nt], dtype=float),
        "z": np.array([r["z"] for r in nt], dtype=float),
    }


def field_proxy(signals, key: str) -> np.ndarray:
    """Fetch a laminar field proxy array by short name ('lfp', 'csd', 'phi_e', 'source')."""
    if signals.field is None:
        raise ValueError("signals.field is None — probes(['LFP','CSD',...]) was not requested")
    attr = {
        "lfp": "lfp_proxy",
        "csd": "csd_proxy",
        "phi_e": "phi_e_proxy",
        "source": "source_proxy",
    }.get(key, key)
    return np.asarray(getattr(signals.field, attr))


def contact_depths_mm(signals) -> np.ndarray:
    if signals.field is None:
        raise ValueError("signals.field is None")
    return np.asarray(signals.field.contact_depths) * 1.0e3


def time_ms(signals) -> np.ndarray:
    return np.asarray(signals.time_ms)


def dt_ms(signals) -> float:
    t = time_ms(signals)
    return float(t[1] - t[0]) if t.shape[0] > 1 else 1.0


def population_rate_hz(spikes: np.ndarray, dt_s_ms: float, bin_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Binned population mean firing rate (Hz). spikes: (n_steps, N)."""
    n_steps = spikes.shape[0]
    bin_steps = max(1, int(round(bin_ms / dt_s_ms)))
    n_bins = n_steps // bin_steps
    if n_bins == 0:
        return np.array([0.0]), np.array([float(spikes.mean()) * (1000.0 / dt_s_ms)])
    trimmed = spikes[: n_bins * bin_steps]
    per_bin = trimmed.reshape(n_bins, bin_steps, -1).sum(axis=(1, 2))
    n_units = spikes.shape[1]
    rate_hz = per_bin / (n_units * bin_steps * dt_s_ms / 1000.0)
    centers_ms = (np.arange(n_bins) + 0.5) * bin_steps * dt_s_ms
    return centers_ms, rate_hz


def color_for_cell_types(cell_types: np.ndarray) -> list[str]:
    return [CELL_TYPE_COLORS.get(ct, "#888888") for ct in cell_types]
