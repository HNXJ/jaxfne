"""0.5.4 ATLAS items 6-7: AT-08 two-area adaptation arms, AT-09 plastic coupling.

Same realized two-area network per scenario; declared observation
differences with verdicts; R4 causal decomposition; R2 H/W separation.
Schema v3 keeps every v1/v2 name stable and in order (additive only).
"""

from __future__ import annotations

import functools

import numpy as np

import artifacts.atlas.at07_at04_053 as A053
import artifacts.atlas.at08_at09_054 as A


@functools.lru_cache(maxsize=1)
def _at08() -> dict:
    return A.run_at08()


@functools.lru_cache(maxsize=1)
def _at09() -> dict:
    return A.run_at09()


def test_at08_status_and_declares_pass():
    out = _at08()
    assert out["scenario"] == "AT-08" and out["status"] == "OK"
    assert out["verdict"] == "PASS"
    by_name = {t["name"]: t for t in out["declares"]}
    for name in ("stimulus-drives-A1", "adaptation-moves-W", "fixed-exposes-no-W"):
        assert by_name[name]["verdict"] == "PASS", (name, by_name[name])


def test_at08_arms_finite_and_r4_decomposition():
    out = _at08()
    for arm, row in out["arms"].items():
        for key in (
            "spike_count_A1",
            "spike_count_A2",
            "Phi_mean_abs",
            "C_12_band_mean",
            "dphi_12_band_mean",
            "kappa",
        ):
            assert np.isfinite(row[key]), (arm, key)
    r4 = out["r4_answer"]
    for key in (
        "propagated_spike_delta_A2",
        "coupling_spike_delta_A2",
        "propagated_field_delta_A2",
        "coupling_field_delta_A2",
    ):
        assert np.isfinite(r4[key]), key
    # full adaptation effect == propagated + coupling by construction:
    full_spk = out["arms"]["adapt_full"]["spike_count_A2"] - out["arms"]["fixed"]["spike_count_A2"]
    assert full_spk == r4["propagated_spike_delta_A2"] + r4["coupling_spike_delta_A2"]


def test_at08_proxy_never_calibrated_and_budget():
    out = _at08()
    for arm, row in out["arms"].items():
        assert row["level"] == "RELATIVE_PROXY", arm
        assert row["wall_s"] < 600.0, arm
    assert sum(row["wall_s"] for row in out["arms"].values()) < 600.0


def test_at09_status_and_r2_separation():
    out = _at09()
    assert out["scenario"] == "AT-09" and out["status"] == "OK"
    assert out["verdict"] == "PASS"
    by_name = {t["name"]: t for t in out["declares"]}
    assert by_name["cross-arm-moves-cross-W"]["verdict"] == "PASS"
    assert by_name["frozen-keeps-W"]["verdict"] == "PASS"
    # R2: H evolves while dW/dt = 0 on the frozen arm.
    assert out["r2_separation"]["H_moves_while_dW_zero"] is True


def test_at09_r3_local_vs_inter():
    out = _at09()
    r3 = out["r3_local_vs_inter"]
    for key in (
        "cross_plastic_W_cross_change",
        "cross_plastic_W_member_change",
        "member_plastic_W_cross_change",
        "member_plastic_W_member_change",
        "C_12_cross",
        "C_12_member",
        "C_12_frozen",
    ):
        assert np.isfinite(r3[key]), key
    # scoping works: the unmasked projection moves most.
    assert r3["cross_plastic_W_cross_change"] >= r3["cross_plastic_W_member_change"] == 0.0
    assert r3["member_plastic_W_member_change"] >= r3["member_plastic_W_cross_change"] == 0.0


def test_schema_v3_keeps_v1_and_v2_names_stable():
    assert list(A.Y_KEYS_V3[:13]) == list(A053.Y_KEYS_V2)
    assert A.SCHEMA_VERSION == "v3"
    gm = A.gap_matrix_v3({"AT-08": _at08(), "AT-09": _at09()})
    assert gm["AT-08"]["Phi_B"] == "REFUSED"
    assert gm["AT-08"]["C_12"] == "IMPLEMENTED"
    assert gm["AT-09"]["W_12"] == "IMPLEMENTED"
    assert gm["AT-09"]["W_21"] == "IMPLEMENTED"
