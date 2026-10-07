"""AT-05-R2: coupled field against the isolated twin (same geometry)."""

from __future__ import annotations

import artifacts.atlas.at01_at06_052 as A


def test_isolation_arm_structure_and_verdict_consistency(monkeypatch):
    monkeypatch.setattr(A, "AT05_ISO_ARMS", {"n4_s7": {"n": 4, "seed": 7}})
    monkeypatch.setattr(A, "AT05_ISO_GAINS", (10.0,))
    out = A.run_at05_isolation()
    arm = out["arms"]["n4_s7"]
    assert arm["isolated"]["within_gain"] == 0.0
    c = arm["coupled"]["10.0"]
    assert c["rel_dev_from_isolated"] == abs(
        c["a_phi_mid_contact"] / arm["isolated"]["a_phi_mid_contact"] - 1.0
    )
    assert c["superposition_holds"] == (c["rel_dev_from_isolated"] <= A.AT05_ISO_REL_TOL)
    assert c["spikes_identical_to_isolated"] == (c["n_spikes"] == arm["isolated"]["n_spikes"])
    assert out["tolerance_predeclared"]["rel_dev"] == 0.01
    # within_gain must reach the field: a strong gain moves it off the isolated twin
    assert c["rel_dev_from_isolated"] > 0.0
    assert arm["isolated"]["a_phi_mid_contact"] != c["a_phi_mid_contact"]


def test_gain_zero_is_the_isolated_twin_of_itself():
    a = A._iso_arm(4, 7, 0.0)
    b = A._iso_arm(4, 7, 0.0)
    assert a == {**b, "within_gain": 0.0}


def test_run_at05_is_unchanged_by_the_isolation_addition():
    assert set(A.AT05_ARMS) == {"n8_s11", "n8_s7", "n4_s7", "n2_s7"}
    assert A.AT05_RATIO_MARGIN == 1e-4
