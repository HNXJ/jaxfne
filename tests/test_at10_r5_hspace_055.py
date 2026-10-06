"""AT-10-R5: H-space separation in the stabilization assay (synthetic H traces).

No 20k run, no ``build_model()``: the helper is exercised on small synthetic
traces. Bounded != returning: a flat non-zero gap is bounded AND
non-returning (the non-inference case).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import jaxfne as J
from artifacts.atlas import at10_n20_055 as A

T = 500  # > early_end (200) + late window (200) of stability_report
K = 4
H_BOUNDS = (0.1, 10.0)
W_CEIL = 50.0


def _ref():
    return np.full((T, K), 1.0)


def _w():
    return np.full((10,), 1.0)


def _report(H_ref, H_arm, **kw):
    return A.h_space_report(H_ref, H_arm, H_bounds=H_BOUNDS, w_ceiling=W_CEIL,
                            w_final=kw.get("w_final", _w()))


def test_contracting_twin_is_returning():
    decay = 2.0 * np.exp(-np.arange(T) / 50.0)[:, None] * np.ones((1, K))
    rep = _report(_ref(), _ref() + decay)
    assert rep["stability"]["verdict"] == "RETURNING", rep["stability"]
    assert rep["boundedness"]["within_bounds"] is True


def test_flat_nonzero_gap_is_persisting():
    rep = _report(_ref(), _ref() + 1.0)
    assert rep["stability"]["verdict"] == "PERSISTING", rep["stability"]
    assert rep["stability"]["contraction_ratio"] == pytest.approx(1.0)


def test_growing_gap_is_diverging():
    drift = np.linspace(0, 5, T)[:, None] * np.ones((1, K))
    rep = _report(_ref(), _ref() + drift)
    assert rep["stability"]["verdict"] == "DIVERGING", rep["stability"]


def test_identical_twins_are_refused_not_scored():
    H = _ref()
    rep = _report(H, H.copy())
    assert rep["stability"]["verdict"] == "REFUSED_DEGENERATE", rep["stability"]
    assert "degenerate" in rep["stability"]["reason"]


def test_report_is_json_safe():
    drift = np.linspace(0, 5, T)[:, None] * np.ones((1, K))
    rep = _report(_ref(), _ref() + drift)
    json.dumps(rep)


def test_bounded_and_non_returning_coexist():
    """The non-inference case: bounded does not imply returning."""
    rep = _report(_ref(), _ref() + 1.0)
    assert rep["boundedness"]["within_bounds"] is True
    assert rep["stability"]["verdict"] != "RETURNING"


def test_w_enters_as_final_state_only():
    W = _w()
    W[3] = W_CEIL + 5.0
    rep = _report(_ref(), _ref() + 1.0, w_final=W)
    assert rep["boundedness"]["W_violations"] == 1
    assert rep["boundedness"]["W_scope"] == "w_final-only (no W trace recorded)"
    assert rep["boundedness"]["within_bounds"] is False


def test_helper_uses_the_public_surface():
    """Firewall: the helper must resolve through the public root, not a private import."""
    assert J.stability_report is not None
    assert J.boundedness_report is not None
    import ast
    from pathlib import Path
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("jaxfne")}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
             if a.name.startswith("jaxfne")}
    assert mods <= {"jaxfne", "jaxfne.public_surface"}, mods


def test_assay_h_bounds_match_the_kernel_contract():
    """H bounds are the HDP defaults actually in force for HP_HEBB."""
    from jaxfne._model_simulate import _hdp_kernel_kwargs
    kw = _hdp_kernel_kwargs({})
    assert tuple(A.H_BOUNDS) == (kw["H_min"], kw["H_max"]) == (0.1, 10.0)
    assert A.W_CEILING == kw["w_ceiling"] == 50.0
    assert set(A.HP_HEBB) & {"H_min", "H_max", "w_ceiling"} == set()


def test_hidden_neuron_violation_flagged_via_extremes():
    """Area means inside bounds, but one area's neurons leave them over time."""
    H_arm = np.full((T, K), 2.0)  # every area mean safely inside bounds
    lo = np.full((K,), 2.0)
    hi = np.full((K,), 2.0)
    hi[2] = H_BOUNDS[1] + 5.0  # one area's extreme leaves bounds
    rep = A.h_space_report(_ref(), H_arm, H_bounds=H_BOUNDS, w_ceiling=W_CEIL,
                           w_final=_w(), H_extremes=(lo, hi))
    assert rep["boundedness"]["within_bounds"] is False
    assert rep["boundedness"]["H_violations"] >= 1
    assert rep["stability"]["verdict"] == "PERSISTING"


def test_means_alone_hide_that_violation():
    """Without extremes the same means pass: why boundedness needs extremes."""
    rep = A.h_space_report(_ref(), np.full((T, K), 2.0), H_bounds=H_BOUNDS,
                           w_ceiling=W_CEIL, w_final=_w())
    assert rep["boundedness"]["within_bounds"] is True


def test_engaged_arm_twins_with_engaged_ref():
    ref, off = _ref(), _ref() + 5.0
    twin_H, label = A._assay_twin("engaged", ref, off)
    assert label == "engaged_ref"
    assert twin_H is ref


def test_disabled_arm_twins_with_disabled_ref():
    ref, off = _ref(), _ref() + 5.0
    twin_H, label = A._assay_twin("disabled", ref, off)
    assert label == "disabled_ref"
    assert twin_H is off


def test_h_space_records_which_twin():
    rep = A.h_space_report(_ref(), _ref() + 1.0, H_bounds=H_BOUNDS, w_ceiling=W_CEIL,
                           w_final=_w(), twin="disabled_ref")
    assert rep["twin"] == "disabled_ref"
    assert rep["stability"]["verdict"] == "PERSISTING"
    json.dumps(rep)


# --- R5 valid-twin arm (HP_NOCTRL) + window sensitivity: pure scoring ---

TWIN_T = 2500  # > max early_end (2000) + late window (200) of stability_report


def _twin_ref():
    return np.full((TWIN_T, K), 1.0)


def _twin_decay():
    return 2.0 * np.exp(-np.arange(TWIN_T) / 50.0)[:, None] * np.ones((1, K))


def test_hp_noctrl_removes_only_k_ctrl():
    assert A.HP_NOCTRL["K_ctrl"] == 0.0
    assert {k: v for k, v in A.HP_NOCTRL.items() if k != "K_ctrl"} == {
        k: v for k, v in A.HP_HEBB.items() if k != "K_ctrl"}


def test_r5_sweep_returns_three_keys_json_safe():
    scored = A.score_r5_twin_pair(_twin_ref(), _twin_ref() + _twin_decay(),
                                  H_trace=_twin_ref() + _twin_decay())
    assert set(scored["sweep"]) == {"200", "1000", "2000"}
    for entry in scored["sweep"].values():
        assert set(entry) >= {"verdict", "contraction_ratio", "d_early", "d_late"}
    json.dumps(scored)


def test_r5_clip_reached_for_float32_pinned_trace():
    # The kernel records float32: float32(H_min) != the float64 bound exactly.
    pinned = np.full((TWIN_T, K), H_BOUNDS[0], dtype=np.float32)
    assert float(pinned[0, 0]) != H_BOUNDS[0]
    scored = A.score_r5_twin_pair(_twin_ref(), pinned, H_trace=pinned)
    assert scored["clip_reached"] is True


def test_r5_clip_reached_when_pinned_at_h_min():
    pinned = np.full((TWIN_T, K), H_BOUNDS[0])  # recorded H sits exactly on H_min
    scored = A.score_r5_twin_pair(_twin_ref(), pinned, H_trace=pinned)
    assert scored["clip_reached"] is True
    # The pinned trace is inside bounds, so boundedness alone would pass: the
    # clip flag is what disqualifies it as evidence of genuine boundedness.
    assert scored["boundedness"]["within_bounds"] is True
    assert scored["boundedness"]["clip_reached"] is True
    assert scored["H_min_obs"] == pytest.approx(H_BOUNDS[0])
    json.dumps(scored)


def test_r5_clip_false_for_interior_trace():
    arm = _twin_ref() + _twin_decay()  # strictly inside (0.1, 10)
    scored = A.score_r5_twin_pair(_twin_ref(), arm, H_trace=arm)
    assert scored["clip_reached"] is False
    assert scored["H_min_obs"] > H_BOUNDS[0]
    assert scored["H_max_obs"] < H_BOUNDS[1]
    json.dumps(scored)


def test_r5_identical_twins_refused_across_sweep():
    scored = A.score_r5_twin_pair(_twin_ref(), _twin_ref().copy(),
                                  H_trace=_twin_ref().copy())
    assert set(scored["sweep"]) == {"200", "1000", "2000"}
    for entry in scored["sweep"].values():
        assert entry["verdict"] == "REFUSED_DEGENERATE"
    json.dumps(scored)
