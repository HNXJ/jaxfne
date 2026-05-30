"""Optimization manifests and serialization helpers for jaxfne.optim.

Evaluated as an uncalibrated computational scaffold matching truth_safe_unverified boundaries.
Outputs are handled as a structured simulation proxy (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from typing import Any
import jax
import jax.numpy as jnp


def serialize_optimization_manifest(state: Any, hyperparams: dict) -> dict[str, Any]:
    """Serializes the current optimization state and hyperparams into a JSON-safe dictionary.

    Ensures that any JAX or NumPy arrays are properly converted to standard Python scalars.
    """
    manifest = {
        "hyperparams": {
            k: float(v) if isinstance(v, (int, float, jax.Array, jnp.ndarray)) else v
            for k, v in hyperparams.items()
            if not callable(v)
        },
        "state": {},
    }

    if state is not None:
        # Extract fields from NamedTuple or dataclass safely
        if hasattr(state, "_asdict"):
            state_dict = state._asdict()
        elif hasattr(state, "__dict__"):
            state_dict = state.__dict__
        else:
            state_dict = {}

        for k, v in state_dict.items():
            if isinstance(v, (jax.Array, jnp.ndarray)):
                try:
                    manifest["state"][k] = v.tolist()
                except Exception:
                    manifest["state"][k] = str(v)
            elif isinstance(v, (int, float)):
                manifest["state"][k] = v
            elif v is None:
                manifest["state"][k] = None
            else:
                manifest["state"][k] = str(v)

    return manifest
