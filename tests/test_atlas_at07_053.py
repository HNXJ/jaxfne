"""0.5.3 ATLAS item 8: AT-07 fixed-W vs Hebbian/noisy HDP.

Same realized network, matched stimulation, item-5 controls in the
item-6 intervention shape. Observation difference with declared tests;
no adaptation phenotype claimed. AT-07-R4 (B as input) stays REFUSED.
"""

from __future__ import annotations

import functools

import numpy as np

import artifacts.atlas.at01_at06_052 as A052
import artifacts.atlas.at01_at10_toy as ATOY
import artifacts.atlas.at07_at04_053 as A


@functools.lru_cache(maxsize=1)
def _out() -> dict:
    return A.run_at07()


def test_at07_same_realized_network():
    out = _out()
    assert out["scenario"] == "AT-07"
    assert out["status"] == "OK"
    assert out["realized_network"]["identical_realization"] is True
    assert len(out["realized_network"]["spec_digest"]) == 64


def test_at07_fixed_w_bit_exact():
    out = _out()
    by_name = {t["name"]: t for t in out["declared_tests"]}
    assert by_name["fixed_w_equals_w0"]["verdict"] == "PASS"
    assert out["chain"]["fixed"]["dW_max_abs"] == 0.0
    # Disabled W equals the pre-0.5.3 fixed path: no movement at all.
    assert out["chain"]["hebbian"]["dW_max_abs"] > 0.0


def test_at07_declared_differences_pass():
    out = _out()
    by_name = {t["name"]: t for t in out["declared_tests"]}
    for name in (
        "hebbian_differs_from_fixed",
        "noisy_differs_from_hebbian",
        "noisy_reproduces_same_seed",
        "clamp_subset_pinned_exact",
        "clamp_unclamped_moves",
    ):
        assert by_name[name]["verdict"] == "PASS", (name, by_name[name])
    assert len(out["interventions"]) == len(out["declared_tests"])
    for iv in out["interventions"]:
        assert iv["network"]["spec_digest"] == out["realized_network"]["spec_digest"]


def test_at07_chain_observed_no_phenotype():
    out = _out()
    for arm, row in out["chain"].items():
        for key in ("dW_max_abs", "dX_spike_count", "dQ_mean_abs", "dPhi_mean_abs"):
            assert np.isfinite(row[key]), (arm, key)
    assert out["boundedness"]["finite"] is True
    blob = repr(out)
    assert "adaptation phenotype" not in blob or "no adaptation phenotype" in blob
    assert "homeostatically stabilized" in blob  # bounded != stable stated


def test_at07_b_as_input_refused_phi_b_refused():
    out = _out()
    assert out["b_as_input"]["state"] == "REFUSED"  # AT-07-R4 candidate
    assert "AT-07-R4" in out["b_as_input"]["reason"]
    assert out["phi_b"]["state"] == "REFUSED"
    rec = A.measure_v2("AT-07", out)
    assert rec["Phi_B"]["state"] == "REFUSED"
    assert rec["Phi_B"]["value"] is None


def test_at07_budgets_and_trajectories():
    out = _out()
    traj = out["trajectories"]
    assert traj["H_trace_shape"] == [2000, 8]
    assert traj["w_trace_shape"] == [2000, 56]
    assert traj["budget"]["kept_frames_equal_full"] is True
    assert traj["budget"]["record_stride"] == A.BUDGET_STRIDE
    rec = A.measure_v2("AT-07", out)
    assert rec["H"]["state"] == "IMPLEMENTED"
    assert rec["W"]["state"] == "IMPLEMENTED"


def test_at07_proxy_never_calibrated():
    out = _out()
    assert out["level"] == "RELATIVE_PROXY"
    assert "CALIBRATED" not in repr(out)
    for arm in out["arms"].values():
        assert arm["level"] == "RELATIVE_PROXY"


def test_at07_wall_budget():
    out = _out()
    assert out["wall_s"] < A.WALL_BUDGET_S


def test_schema_v2_keeps_v1_and_v0_names_stable():
    assert A.SCHEMA_VERSION == "v2"
    assert A.Y_KEYS_V2 == A052.Y_KEYS_V1
    assert tuple(ATOY.Y_KEYS) == tuple(A052.Y_KEYS_V1)
    mat = A.gap_matrix_v2({"AT-07": _out(), "AT-04": A.run_at04r2()})
    assert set(mat) == {"AT-07", "AT-04"}
    assert set(mat["AT-07"]) == set(A.Y_KEYS_V2)
