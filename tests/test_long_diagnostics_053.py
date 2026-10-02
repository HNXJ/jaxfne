"""0.5.3 item 7 (ENGINE W2): long-horizon diagnostics.

Boundedness and stability are SEPARATE measurements; neither is inferred
from the other. Boundedness = trajectories stay within declared bounds
over long horizons. Stability = twin-trajectory perturbation assay:
RETURNING (contracts), PERSISTING (neutral offset), DIVERGING (grows).
Case B is bounded yet non-returning — the non-inference proof. d_H=1
throughout (H8).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.emitters import simulate_edge_recurrent_izhikevich_hdp

T_STEPS = 2000
H_LO, H_HI = 0.1, 10.0
W_CEIL = 50.0


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


def _fixture():
    params = IzhikevichParams(
        a=jnp.full((3,), 0.02),
        b=jnp.full((3,), 0.2),
        c=jnp.full((3,), -65.0),
        d=jnp.full((3,), 8.0),
        drive=jnp.zeros(3),
        sign=jnp.ones((3,)),
        W=jnp.zeros((3, 3)),
        v0=jnp.full((3,), -65.0),
        u0=jnp.full((3,), -13.0),
        source_scale=jnp.ones((3,)),
        labels=("E", "E", "I"),
        layer_labels=("L4", "L4", "L4"),
        source_calibration_status="x",
    )
    edges = EdgeList(
        pre=jnp.array([0, 0], dtype=jnp.int32),
        post=jnp.array([1, 2], dtype=jnp.int32),
        weight=jnp.array([0.2, 0.4], dtype=jnp.float32),
        receptor_index=jnp.array([0, 0], dtype=jnp.int32),
        tau_ms=jnp.array([2.0, 2.0], dtype=jnp.float32),
        source_calibration_status="x",
    )
    drive = jnp.zeros((T_STEPS, 3), dtype=jnp.float32).at[:, 0].set(8.0)
    return params, edges, drive


def _base_init(h):
    p, _, _ = _fixture()
    return {
        "v": p.v0,
        "u": p.u0,
        "prev_spikes": jnp.zeros(3),
        "syn_state": jnp.zeros(2),
        "H_final": h,
    }


def _twins(**kw):
    p, e, d = _fixture()
    _, _, _, da = simulate_edge_recurrent_izhikevich_hdp(
        p,
        e,
        T_STEPS,
        1.0,
        jax.random.PRNGKey(0),
        drive_schedule=d,
        init_state=_base_init(jnp.ones(3)),
        **kw,
    )
    _, _, _, db = simulate_edge_recurrent_izhikevich_hdp(
        p,
        e,
        T_STEPS,
        1.0,
        jax.random.PRNGKey(0),
        drive_schedule=d,
        init_state=_base_init(jnp.full(3, 1.05)),
        **kw,
    )
    return da, db


CASE_A = dict(K_HDP=0.0, K_w_ctrl=0.0, K_ctrl=5.0, alpha=0.05, tau_0_ms=5.0, noise_scale=0.0)
CASE_B = dict(K_HDP=0.0, K_w_ctrl=0.0, K_ctrl=0.0, alpha=0.05, tau_0_ms=5.0, noise_scale=0.0)
CASE_C_PLASTIC = dict(
    K_HDP=0.2, K_w_ctrl=0.002, K_ctrl=0.15, alpha=0.05, tau_0_ms=5.0, noise_scale=0.0
)


# --- live long-horizon measurements --------------------------------------------


def test_boundedness_case_A_within():
    da, _ = _twins(**CASE_A)
    rep = boundedness_report(da["H_trace"], da["w_trace"], H_bounds=(H_LO, H_HI), w_ceiling=W_CEIL)
    assert rep["within_bounds"] and rep["H_violations"] == 0 and rep["W_violations"] == 0


def test_boundedness_case_B_within():
    da, _ = _twins(**CASE_B)
    rep = boundedness_report(da["H_trace"], da["w_trace"], H_bounds=(H_LO, H_HI), w_ceiling=W_CEIL)
    assert rep["within_bounds"] and rep["H_violations"] == 0 and rep["W_violations"] == 0


def test_boundedness_plastic_long_run_within():
    da, _ = _twins(**CASE_C_PLASTIC)
    rep = boundedness_report(da["H_trace"], da["w_trace"], H_bounds=(H_LO, H_HI), w_ceiling=W_CEIL)
    assert rep["within_bounds"], rep
    assert rep["W_max_abs"] > 0.4, "plastic run must actually move W"


def test_stability_case_A_returning():
    da, db = _twins(**CASE_A)
    rep = stability_report(da["H_trace"], db["H_trace"])
    assert rep["verdict"] == "RETURNING", rep
    assert rep["contraction_ratio"] < 0.5


def test_stability_case_B_nonreturning():
    da, db = _twins(**CASE_B)
    rep = stability_report(da["H_trace"], db["H_trace"])
    assert rep["verdict"] == "PERSISTING", rep
    assert rep["contraction_ratio"] >= 0.5


def test_bounded_does_not_imply_stable():
    """The non-inference proof: case B is bounded AND non-returning."""
    da, db = _twins(**CASE_B)
    b = boundedness_report(da["H_trace"], da["w_trace"], H_bounds=(H_LO, H_HI), w_ceiling=W_CEIL)
    s = stability_report(da["H_trace"], db["H_trace"])
    assert b["within_bounds"] is True
    assert s["verdict"] != "RETURNING"


def test_stable_case_is_also_bounded():
    da, db = _twins(**CASE_A)
    b = boundedness_report(da["H_trace"], da["w_trace"], H_bounds=(H_LO, H_HI), w_ceiling=W_CEIL)
    s = stability_report(da["H_trace"], db["H_trace"])
    assert b["within_bounds"] is True and s["verdict"] == "RETURNING"


# --- measurement functions, adversarial -----------------------------------------


def test_boundedness_catches_violations_synthetic():
    da, _ = _twins(**CASE_A)
    H = np.asarray(da["H_trace"]).copy()
    W = np.asarray(da["w_trace"]).copy()
    H[100, 0] = H_HI + 1.0
    W[200, 1] = W_CEIL + 5.0
    rep = boundedness_report(H, W, H_bounds=(H_LO, H_HI), w_ceiling=W_CEIL)
    assert rep["within_bounds"] is False
    assert rep["H_violations"] == 1 and rep["W_violations"] == 1


def test_stability_diverging_tier_synthetic():
    da, _ = _twins(**CASE_A)
    H = np.asarray(da["H_trace"])
    drift = np.linspace(0, 5, H.shape[0])[:, None] * np.ones_like(H)
    rep = stability_report(H, H + drift)
    assert rep["verdict"] == "DIVERGING", rep


def test_stability_degenerate_assay_refused():
    da, _ = _twins(**CASE_A)
    H = np.asarray(da["H_trace"])
    with pytest.raises(ValueError, match="degenerate assay"):
        stability_report(H, H.copy())
    with pytest.raises(ValueError, match="differ"):
        stability_report(H, H[:, :2])


def test_reports_are_json_safe():
    da, db = _twins(**CASE_A)
    b = boundedness_report(da["H_trace"], da["w_trace"], H_bounds=(H_LO, H_HI), w_ceiling=W_CEIL)
    s = stability_report(da["H_trace"], db["H_trace"])
    json.dumps({"boundedness": b, "stability": s})
