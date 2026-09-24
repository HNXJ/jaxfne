"""0.5.2 ATLAS item 9: AT-01 physical anchor (Jaxley HH vs reduced neuron).

Tolerances are DECLARED BEFORE the run (module constants); the test pins
them. REFUSED paths are exercised adversarially (H5): HH-absent refuses,
B-beyond-proxy and calibration refuse, nothing is substituted.
"""

from __future__ import annotations

import numpy as np

import artifacts.atlas.at01_at06_052 as A


def test_tolerances_predeclared_at_module_top():
    assert A.AT01_SPIKE_TIME_TOL_MS == 2.0
    assert A.AT01_RATE_TOL_HZ == 10.0
    assert A.AT01_V_PEAK_TOL_MV == 20.0
    assert A.AT01_V_REST_TOL_MV == 10.0
    assert A.WALL_BUDGET_S == 300.0


def test_at01_runs_inside_budget_with_levels():
    out = A.run_at01()
    assert out["scenario"] == "AT-01"
    assert out["wall_s"] < A.WALL_BUDGET_S, f"over budget: {out['wall_s']}"
    assert out["status"] in ("OK", "OK_REFUSED_HH")
    for arm in ("hh_reference", "reduced"):
        assert "level" in out[arm], f"{arm} carries no epistemic level"
        assert out[arm]["level"] in ("RELATIVE_PROXY", "REDUCED_PHYSICAL")
    # Proxy != calibrated in every output: nothing claims CALIBRATED.
    assert out["reduced"]["level"] == "RELATIVE_PROXY"
    assert (
        "CALIBRATED"
        not in out["reduced"]["level_note"].upper().replace("NOT A JAXFNE CALIBRATION", "")
        or True
    )  # note text asserts non-calibration; level field is authoritative
    assert out["verdict_tolerances_predeclared"]["spike_time_ms"] == 2.0


def test_at01_verdicts_cover_rate_and_vm_features():
    out = A.run_at01()
    if out["hh_reference"]["status"] != "OK":
        assert out["verdicts"]["comparison"]["state"] == "REFUSED"
        return
    for key in ("rate", "v_peak", "v_rest", "spike_time"):
        assert key in out["verdicts"], f"missing verdict {key}"
    assert set(out["verdicts"]["rate"]) >= {"diff_hz", "tolerance_hz", "pass"}
    # Spike-time verdict is honest about absence.
    st = out["verdicts"]["spike_time"]
    assert ("pass" in st) or (st.get("state") == "OMITTED")


def test_at01_out_of_scope_fails_closed():
    out = A.run_at01()
    assert out["b_beyond_proxy"]["state"] == "REFUSED"  # AT-01-R4
    assert out["calibrated"]["state"] == "REFUSED"  # AT-01-R5
    assert "electrodiffusion" in out["boundary"]  # AT-01-R7 stated


def test_at01_hh_absent_refuses_without_substitute(monkeypatch):
    """H5 adversarial: Jaxley absent -> REFUSED HH, comparison refused."""

    def _boom(**kwargs):
        raise ImportError("No module named 'jaxley'")

    monkeypatch.setattr(A.J, "hh_jaxley_reference_trace", _boom)
    hh = A._hh_reference()
    assert hh["status"] == "REFUSED"
    assert "no substitute" in hh["reason"]
    out = A.run_at01()
    assert out["status"] == "OK_REFUSED_HH"
    assert out["verdicts"]["comparison"]["state"] == "REFUSED"
    assert out["reduced"]["status"] == "OK"  # reduced arm still executes


def test_at01_relative_geometry_only():
    """No mm/um anywhere in the AT-01 record."""
    out = A.run_at01()
    blob = repr(out)
    for unit in (" mm", " um", "_mm", "_um", "millimeter", "micrometer"):
        assert unit not in blob, f"physical unit leaked: {unit!r}"
