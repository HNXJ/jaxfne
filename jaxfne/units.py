"""Dtype-keyed declarative numeric defaults (epsilon, dither scale).

Scope, deliberately narrow (see plans.json item
bf16-quantized-tfne-izhikevich-mode, sections 5 and 7): a single shared
resolution point for "how small is negligible at this storage dtype", so
callers stop hand-picking epsilon/dither constants per call site (jaxfne-
modular-grammar rule 6). This module does NOT touch dt_ms/dxyz -- those are
physical/discretization choices independent of storage dtype; silently
adjusting them based on dtype would break run reproducibility (use
``warn_dt_dtype_mismatch`` instead of an auto-adjustment).
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp

# Roughly half machine-epsilon at unit magnitude for each supported storage
# dtype (see RuntimeConfig._ALLOWED_DTYPES in jaxfne/core.py -- kept in sync
# manually since this module must not import core.py, which imports this one
# transitively via emitters.py in a future pass).
_EPSILON_BY_DTYPE: dict[str, float] = {
    "float32": 6.0e-8,
    "float64": 1.1e-16,
    "bfloat16": 4.0e-3,
}

_DITHER_SCALE_BY_DTYPE: dict[str, float] = {
    "float32": 3.0e-8,
    "float64": 5.5e-17,
    "bfloat16": 2.0e-3,
}


def _dtype_key(dtype) -> str:
    """Normalize a jnp/np dtype or dtype-name string to one of the 3 keys above."""
    name = jnp.dtype(dtype).name
    if name not in _EPSILON_BY_DTYPE:
        raise ValueError(f"No epsilon/dither default registered for dtype {name!r}; "
                          f"known dtypes: {sorted(_EPSILON_BY_DTYPE)}")
    return name


def epsilon(dtype) -> float:
    """Roughly half machine-epsilon at unit magnitude for ``dtype``.

    For denominators/floors on strictly-positive quantities (e.g. barrier
    terms, tau_i) that need "don't divide by exactly zero" protection scaled
    to the storage precision actually in use, rather than a single constant
    tuned for float32 and reused unexamined at bfloat16 (where it would be
    far too small to matter) or float64 (where it would be needlessly loose).
    """
    return _EPSILON_BY_DTYPE[_dtype_key(dtype)]


def dither_scale(dtype) -> float:
    """Magnitude of dither noise to add at the ``dtype`` cast boundary.

    ONLY for quantities that are strictly positive by construction (H,
    tau_i, barrier denominators) -- never for signed state that legitimately
    crosses zero (V_m, u, weight deltas, synaptic current). Flooring those
    would inject a permanent bias into dynamics meant to pass through zero
    cleanly, which is a correctness bug, not a stability feature. Callers
    are responsible for only applying this to the correct quantities; this
    function does not know what value it will be added to.
    """
    return _DITHER_SCALE_BY_DTYPE[_dtype_key(dtype)]


def warn_dt_dtype_mismatch(dt_ms: float, dtype, *, quantity_name: str = "dt_ms") -> None:
    """Warn (never silently adjust) if a discretization step looks too fine
    for the chosen storage dtype's precision -- i.e. ``dt_ms`` multiplied by
    a typical per-step increment could round to exactly zero relative to a
    unit-magnitude state variable, causing a silent integration stall.

    dt_ms/dxyz are physical/discretization choices independent of storage
    dtype (jaxfne-modular-grammar rule 6): this function only surfaces the
    risk, it never auto-adjusts dt_ms or the config.
    """
    key = _dtype_key(dtype)
    eps = _EPSILON_BY_DTYPE[key]
    if dt_ms > 0.0 and dt_ms < eps * 1.0e3:
        warnings.warn(
            f"{quantity_name}={dt_ms!r} is smaller than roughly {eps * 1.0e3:.2e} at "
            f"dtype={key!r} (unit-magnitude scale) -- a fixed-step Euler update "
            f"(dt * d(state)/dt) risks rounding to exactly zero at this precision, "
            f"which silently stalls integration rather than raising an error. "
            f"This is a warning only -- {quantity_name} is not being auto-adjusted.",
            stacklevel=2,
        )
