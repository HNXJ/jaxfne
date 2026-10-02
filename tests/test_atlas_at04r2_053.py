"""0.5.3 ATLAS item 9: AT-04 H-perturbation arm (AT-04-R2).

Causal state effect on excitability vs correlation X -> Phi, in the
item-6 intervention shape. Phi -> X stays REFUSED (AT-04-R3
OUT_OF_SCOPE); H-freeze is recorded unavailable, never emulated.
"""

from __future__ import annotations

import functools

import artifacts.atlas.at07_at04_053 as A


@functools.lru_cache(maxsize=1)
def _out() -> dict:
    return A.run_at04r2()


def test_at04r2_same_realized_network_w0():
    out = _out()
    assert out["scenario"] == "AT-04R2"
    assert out["status"] == "OK"
    assert out["realized_network"]["identical_realization"] is True
    assert out["perturbation"]["H0_baseline"] == A.H0_BASELINE
    assert out["perturbation"]["H0_perturbed"] == A.H0_PERTURBED


def test_at04r2_h_perturbation_changes_excitability():
    out = _out()
    by_name = {t["name"]: t for t in out["declared_tests"]}
    t = by_name["h_perturbation_changes_excitability"]
    assert t["verdict"] == "PASS", t
    assert t["gap_max_abs"] > 0.0
    assert out["baseline"]["spike_count"] != out["perturbed"]["spike_count"]
    # H itself moved with the perturbation (state was actually perturbed).
    assert out["baseline"]["H_final_mean"] != out["perturbed"]["H_final_mean"]


def test_at04r2_disabled_control_isolates_hdp_path():
    out = _out()
    by_name = {t["name"]: t for t in out["declared_tests"]}
    assert by_name["disabled_control_x_unaffected_by_h0"]["verdict"] == "PASS"
    assert by_name["disabled_control_w_fixed"]["verdict"] == "PASS"
    assert out["perturbation"]["w_fixed_disabled_pair"] is True


def test_at04r2_x_to_phi_correlation_reported_both_arms():
    out = _out()
    for side in ("baseline", "perturbed"):
        corr = out[side]["x_to_phi"]
        for key in ("per_neuron_rate_vs_source_r", "temporal_poprate_vs_field_r"):
            v = corr[key]
            assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9, (side, key, v)
        assert "correlation != causal" in corr["note"]
    rec = A.measure_v2("AT-04", out)
    assert rec["H"]["state"] == "IMPLEMENTED"
    assert rec["W"]["state"] == "IMPLEMENTED"
    assert rec["C"]["state"] == "IMPLEMENTED"


def test_at04r2_phi_to_x_and_h_freeze_fail_closed():
    out = _out()
    assert out["phi_to_x"]["state"] == "REFUSED"  # AT-04-R3
    assert out["h_freeze"]["state"] == "REFUSED"  # no H-freeze control
    assert out["phi_b"]["state"] == "REFUSED"  # B observation-only


def test_at04r2_proxy_never_calibrated():
    out = _out()
    assert out["level"] == "RELATIVE_PROXY"
    assert "CALIBRATED" not in repr(out)
    assert " mm" not in repr(out)


def test_at04r2_wall_budget():
    out = _out()
    assert out["wall_s"] < A.WALL_BUDGET_S
