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
    """Time axis for a Signals-like or dict payload (shared viewer helper)."""
    time_raw = getattr(signals, "time_ms", None)
    if time_raw is None and isinstance(signals, dict):
        time_raw = signals.get("time_ms")
    time_ms = prepare_static_plot_matrix(time_raw)
    if time_ms is None:
        return np.arange(default_len)
    return time_ms


def neuron_rows(signals: Any) -> list[dict[str, Any]]:
    """Copies of the neuron-table rows carried in signal metadata (shared)."""
    meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
    rows = meta.get("neuron_metadata") if isinstance(meta, dict) else None
    return [dict(row) for row in rows] if rows else []


def geometry3d_from_config(cfg: Any, *, areas=None, cell_types=None, figsize=(9, 7)) -> Any:
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
