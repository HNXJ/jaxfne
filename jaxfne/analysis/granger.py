"""Spectral Granger causality and directionality (NumPy only).

Geweke (1982) spectral decomposition of a bivariate vector-autoregressive
fit: ``granger_spectral`` fits ``VAR(order)`` to two simultaneously sampled
traces by least squares and decomposes the spectral density matrix, so
``gc_xy[f]`` is the directed influence ``x -> y`` at ``freqs_hz[f]`` and
``gc_yx[f]`` the reverse. ``directionality_index`` is the paper's DAI
form (Mejias et al. 2016, Eq. 1)::

    DAI(f) = (GC_{x->y}(f) - GC_{y->x}(f)) / (GC_{x->y}(f) + GC_{y->x}(f)).

No engine import: both functions take and return plain NumPy arrays, so
lane scorers can call them on binned spike trains or rate traces without
touching the simulator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _checked_trace(name: str, values: Any, n: int | None = None) -> np.ndarray:
    """Finite 1-D float trace, refusing the rest by name."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            f"granger {name} is not realized: it takes a non-empty 1-D trace; "
            f"got shape {np.asarray(values).shape!r}"
        )
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"granger {name} must stay finite")
    if n is not None and arr.size != n:
        raise ValueError(
            f"granger traces disagree: {name} has {arr.size} samples, "
            f"expected {n}"
        )
    return arr


def granger_spectral(
    x: Any,
    y: Any,
    dt_ms: float,
    order: int,
    freqs_hz: Any,
) -> dict[str, np.ndarray]:
    """Spectral Granger causality between two traces (Geweke 1982).

    Fits a bivariate ``VAR(order)`` by least squares, forms the spectral
    transfer ``H(f) = A(f)^{-1}`` with
    ``A(f) = I - sum_k A_k exp(-2j pi f k dt)`` and the spectral density
    ``S(f) = H(f) Sigma H(f)*``, then::

        gc_{x->y}(f) = ln(S_yy / (S_yy - (Sigma_xx - Sigma_xy^2/Sigma_yy) |H_yx|^2)).

    ``dt_ms`` is finite and > 0; ``order`` is an int >= 1; the traces need
    at least ``3 * order`` samples after the lag window (fewer leave the
    least-squares fit underdetermined and are refused); every ``freqs_hz``
    entry lies in ``(0, Nyquist]`` (anything else is refused, never
    clipped). Returns ``{"freqs_hz", "gc_xy", "gc_yx"}`` float arrays
    (``gc_xy`` is ``x -> y``). Non-negative up to least-squares noise:
    tiny negative values from an exactly uncoupled fit clip at 0.0.
    """
    dt = float(dt_ms)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            f"granger dt_ms={dt_ms!r} is not realized: it takes a finite value > 0"
        )
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError(
            f"granger order={order!r} is not realized: it takes an int >= 1"
        )
    order = int(order)
    if order < 1:
        raise ValueError(
            f"granger order={order!r} is not realized: it takes an int >= 1"
        )
    xa = _checked_trace("x", x)
    ya = _checked_trace("y", y, n=xa.size)
    n = xa.size
    if n - order < 3 * order:
        raise ValueError(
            f"granger series (n={n}) too short for order={order}: the "
            f"least-squares fit needs at least {3 * order} post-lag samples; "
            "pass longer traces or a smaller order"
        )
    freqs = np.asarray(freqs_hz, dtype=float).ravel()
    if freqs.size == 0 or not bool(np.isfinite(freqs).all()):
        raise ValueError("granger freqs_hz takes a non-empty finite array")
    nyquist = 1000.0 / (2.0 * dt)
    if bool((freqs <= 0.0).any()) or bool((freqs > nyquist).any()):
        raise ValueError(
            f"granger freqs_hz={freqs!r} is not realized: every entry lies in "
            f"(0, Nyquist] with Nyquist={nyquist} Hz at dt_ms={dt}"
        )
    # VAR(order) by least squares: Y = Z @ B, Z rows are the 2*order lags.
    cols = []
    for k in range(1, order + 1):
        cols.append(xa[order - k : n - k])
        cols.append(ya[order - k : n - k])
    Z = np.column_stack(cols)
    Y = np.column_stack([xa[order:], ya[order:]])
    coef, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    resid = Y - Z @ coef
    dof = resid.shape[0] - 2 * order
    sigma = (resid.T @ resid) / dof
    # VAR coefficient matrices A_k (2x2 each): coef rows alternate x/y lags.
    Aks = np.zeros((order, 2, 2))
    for k in range(order):
        Aks[k, 0, 0] = coef[2 * k, 0]
        Aks[k, 0, 1] = coef[2 * k + 1, 0]
        Aks[k, 1, 0] = coef[2 * k, 1]
        Aks[k, 1, 1] = coef[2 * k + 1, 1]
    sxx, sxy, syy = sigma[0, 0], sigma[0, 1], sigma[1, 1]
    cond_xx = sxx - sxy * sxy / syy if syy > 0.0 else 0.0
    cond_yy = syy - sxy * sxy / sxx if sxx > 0.0 else 0.0
    gc_xy = np.zeros_like(freqs)
    gc_yx = np.zeros_like(freqs)
    eye = np.eye(2)
    for i, f in enumerate(freqs.tolist()):
        phase = -2.0 * math.pi * float(f) * (dt / 1000.0)
        Af = eye - sum(
            Aks[k] * complex(math.cos(phase * (k + 1)), math.sin(phase * (k + 1)))
            for k in range(order)
        )
        try:
            H = np.linalg.inv(Af)
        except np.linalg.LinAlgError:
            raise ValueError(
                f"granger VAR({order}) transfer is singular at f={f} Hz: "
                "the fit is degenerate there (near-unit-root dynamics); "
                "pass a smaller order"
            )
        S = H @ sigma @ H.conj().T
        s_yy, s_xx = float(S[1, 1].real), float(S[0, 0].real)
        if cond_xx > 0.0 and s_yy > 0.0:
            gc_xy[i] = math.log(s_yy / max(s_yy - cond_xx * abs(H[1, 0]) ** 2, 1e-300))
        if cond_yy > 0.0 and s_xx > 0.0:
            gc_yx[i] = math.log(s_xx / max(s_xx - cond_yy * abs(H[0, 1]) ** 2, 1e-300))
    return {
        "freqs_hz": freqs.copy(),
        "gc_xy": np.maximum(gc_xy, 0.0),
        "gc_yx": np.maximum(gc_yx, 0.0),
    }


def directionality_index(gc_xy: Any, gc_yx: Any) -> np.ndarray:
    """Directionality (DAI) index from two directed spectra (Mejias Eq. 1).

    Elementwise ``(gc_xy - gc_yx) / (gc_xy + gc_yx)`` over scalars or
    broadcastable arrays; ``0/0`` (no coupling either way) reads as 0.0,
    never NaN. Finite inputs only.
    """
    a = np.asarray(gc_xy, dtype=float)
    b = np.asarray(gc_yx, dtype=float)
    try:
        a, b = np.broadcast_arrays(a, b)
    except ValueError:
        raise ValueError(
            f"directionality shapes {a.shape} and {b.shape} do not broadcast"
        )
    if not bool(np.isfinite(a).all()) or not bool(np.isfinite(b).all()):
        raise ValueError("directionality inputs must stay finite")
    denom = a + b
    out = np.zeros_like(denom)
    nz = denom != 0.0
    out[nz] = (a[nz] - b[nz]) / denom[nz]
    return out
