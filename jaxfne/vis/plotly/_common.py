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
        "neuron_id": np.array([r.get("neuron_id", i) for i, r in enumerate(nt)]),
        "area": np.array([r.get("area", "network") for r in nt]),
        "layer": np.array([r.get("layer", "unspecified") for r in nt]),
        "cell_type": np.array([r.get("cell_type", "unknown") for r in nt]),
        "x": np.array([r.get("x", 0.0) for r in nt], dtype=float),
        "y": np.array([r.get("y", 0.0) for r in nt], dtype=float),
        "z": np.array([r.get("z", 0.0) for r in nt], dtype=float),
    }


def field_proxy(signals, key: str) -> np.ndarray:
    """Fetch a laminar field proxy array by short name ('lfp', 'csd', 'phi_e', 'source').

    Route note (P8): DECLARED field access — raises when ``signals.field``
    is None (probes were not requested before ``compute_fields``). Unlike
    constructed probes (which synthesize missing contacts) or
    visualization-only proxies (which never enter ``Signals.field``), this
    path never invents data.
    """
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


def population_rate_hz(
    spikes: np.ndarray, dt_s_ms: float, bin_ms: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Binned population mean firing rate (Hz). spikes: (n_steps, N).

    Thin alias of :func:`jaxfne.vis.core.binned_population_rate_hz` — the
    P5 canonical numeric contract shared with the matplotlib renderer.
    """
    from ..core import binned_population_rate_hz as _contract

    return _contract(spikes, dt_s_ms, bin_ms)


def color_for_cell_types(cell_types: np.ndarray) -> list[str]:
    return [CELL_TYPE_COLORS.get(ct, "#888888") for ct in cell_types]
