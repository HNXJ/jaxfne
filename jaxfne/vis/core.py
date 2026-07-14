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
