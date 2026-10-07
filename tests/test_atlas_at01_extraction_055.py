"""AT-01-R1/R2/R3: HH current sweep, ion currents and charge-balance closure."""

from __future__ import annotations

import numpy as np
import pytest

import jaxfne as J
import artifacts.atlas.at01_at06_052 as A

pytest.importorskip("jaxley")


def test_default_trace_is_unchanged_3_tuple_and_currents_variant_agrees():
    base = J.hh_jaxley_reference_trace(duration_ms=20.0, dt_ms=0.5, current_amplitude=2.0)
    ext = J.hh_jaxley_reference_trace(
        duration_ms=20.0, dt_ms=0.5, current_amplitude=2.0, return_currents=True
    )
    assert len(base) == 3 and len(ext) == 4
    for a, b in zip(base, ext[:3]):
        np.testing.assert_array_equal(a, b)
    cur = ext[3]
    assert set(("I_Na", "I_K", "I_L", "m", "h", "n", "params", "area_cm2")) <= set(cur)
    assert "I_m" not in cur and cur["I_Na"].shape == ext[1].shape


def test_extraction_spans_regimes_and_charge_balance_closes():
    out = A.run_at01_extraction()
    assert out["status"] == "OK" and out["sweep_spans_subthreshold_to_ap"]
    assert out["I_m"]["state"] == "ABSENT"
    for arm in out["arms"].values():
        assert arm["closure_pass"] and arm["closure_rel"] <= A.AT01_CLOSURE_REL_TOL
        assert set(arm["charge_uC_cm2"]) == {"I_Na", "I_K", "I_L"}
    ap = [a for a in out["arms"].values() if a["regime"] == "action_potential"]
    assert ap and all(a["peak_abs_current_mA_cm2"]["I_Na"] > 0.1 for a in ap)


def test_phi_series_follow_the_ionic_current_and_closure_is_tight():
    out = A.run_at01_extraction()
    ap = next(a for a in out["arms"].values() if a["regime"] == "action_potential")
    ph = ap["phi_e_relative"]["2.0"]["phi_rel_t"]
    t, v, i_inj, cur = J.hh_jaxley_reference_trace(
        duration_ms=A.AT01_DURATION_MS, dt_ms=A.DT_MS,
        current_amplitude=ap["current_amplitude"], return_currents=True,
    )
    ion = 1e3 * (cur["I_Na"] + cur["I_K"] + cur["I_L"])
    np.testing.assert_allclose(ph, ion / 2.0, rtol=1e-5, atol=1e-6)
    # regression guard well inside the predeclared 0.10 tolerance
    assert max(a["closure_rel"] for a in out["arms"].values()) < 0.03
