"""Long-horizon diagnostics: boundedness and stability are separate measurements.

Boundedness = trajectories stay within declared bounds. Stability = a state
perturbation contracts. Neither is inferred from the other (H8, ``d_H=1``).
Moved unchanged from ``tests/test_long_diagnostics_053.py`` so scenario
scripts import one implementation.
"""

from __future__ import annotations

import numpy as np


def boundedness_report(H_trace, w_trace, *, H_bounds, w_ceiling):
    """Separate measurement 1: are trajectories within declared bounds?"""
    H = np.asarray(H_trace, dtype=float)
    W = np.asarray(w_trace, dtype=float)
    lo, hi = H_bounds
    h_bad = (H < lo) | (H > hi)
    w_bad = np.abs(W) > float(w_ceiling)
    within = bool(not h_bad.any() and not w_bad.any())
    return {
        "within_bounds": within,
        "H_violations": int(h_bad.sum()),
        "W_violations": int(w_bad.sum()),
        "H_min": float(H.min()),
        "H_max": float(H.max()),
        "W_max_abs": float(np.abs(W).max()),
        "H_bounds": [float(lo), float(hi)],
        "w_ceiling": float(w_ceiling),
    }


def stability_report(H_a, H_b, *, early_end=200, late_start=-200):
    """Separate measurement 2: does a state perturbation contract?

    Twin trajectories differing only in initial H. DIVERGING/PERSISTING/
    RETURNING from the late-vs-early peak distance ratio. A zero early
    distance is a vacuous assay and is refused, not scored.
    """
    A = np.asarray(H_a, dtype=float)
    B = np.asarray(H_b, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"twin H shapes differ: {A.shape} vs {B.shape}")
    d = np.abs(A - B).max(axis=tuple(range(1, A.ndim)))
    d_early = float(d[:early_end].max())
    d_late = float(d[late_start:].max())
    if not d_early > 0:
        raise ValueError("degenerate assay: twins identical in the early window")
    ratio = d_late / d_early
    if ratio < 0.5:
        verdict = "RETURNING"
    elif ratio <= 2.0:
        verdict = "PERSISTING"
    else:
        verdict = "DIVERGING"
    return {
        "verdict": verdict,
        "contraction_ratio": float(ratio),
        "d_early": d_early,
        "d_late": d_late,
    }
