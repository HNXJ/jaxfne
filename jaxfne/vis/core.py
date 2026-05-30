"""Core visualization utilities and metadata containers for jaxfne/vis.

Evaluated as an uncalibrated computational scaffold matching truth_safe_unverified boundaries.
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

    Evaluated as an uncalibrated computational scaffold matching truth_safe_unverified boundaries.
    """
    fig: Any
    metadata: dict[str, Any]


def require_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    try:
        import matplotlib
    except ImportError:
        raise ImportError(
            "The visualization features require the optional dependency 'matplotlib'. "
            "Please install it via `pip install matplotlib` or `pip install jaxfne[viz]`."
        )


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
