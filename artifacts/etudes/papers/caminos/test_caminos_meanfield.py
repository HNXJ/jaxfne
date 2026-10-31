"""Linear mean-field model of the PLoS CB paper (Eqs. 3-14, Fig. 6) against its closed forms."""
import numpy as np
import pytest

import caminos_meanfield as mf


def test_base_model_is_a_pure_oscillation_with_pv_leading_som():
    """Eq. 10: purely imaginary pair near 40 Hz, PV about 4 ms ahead of SOM (Fig. 6A)."""
    m = mf.oscillatory_mode(mf.coupling_matrix())
    assert m is not None, "precondition: the base model must oscillate"
    assert abs(m["real"]) * mf.TAU_S < 1e-9
    assert 35.0 < m["freq_hz"] < 45.0
    assert 3.5 < m["lead_over_som"]["PV"]["ms"] < 4.5


def test_red_line_separates_growing_from_damped():
    """Eq. 7 line Q_SOM->PV = 0.6 Q_SOM->E - 0.35 is the pure-oscillation boundary (Fig. 6B)."""
    examined = 0
    for q_se in np.linspace(1.0, 8.0, 8):
        on = mf.red_line(q_se)
        assert mf.regime(mf.coupling_matrix(SOME=q_se, SOMPV=on)) == "pure oscillation"
        assert mf.regime(mf.coupling_matrix(SOME=q_se, SOMPV=on + 0.3)) == "damped"
        assert mf.regime(mf.coupling_matrix(SOME=q_se, SOMPV=on - 0.3)) == "growing"
        examined += 1
    assert examined == 8


@pytest.mark.parametrize("q", [0.5, 2.0, 3.0, 5.0, 9.0])
def test_pv_to_som_determinant_closed_form(q):
    """Eq. 13: det(Y) = (59 Q_PV->SOM - 220) / 5 (Fig. 6C left)."""
    assert np.linalg.det(mf.coupling_matrix(PVSOM=q)) == pytest.approx((59 * q - 220) / 5, rel=1e-12)


def test_pv_to_som_beyond_220_over_59_has_a_positive_eigenvalue():
    q0 = 220.0 / 59.0
    assert mf.regime(mf.coupling_matrix(PVSOM=q0 + 0.05)) == "positive real"
    assert mf.regime(mf.coupling_matrix(PVSOM=q0 - 0.05)) != "positive real"


def test_som_self_inhibition_damps_more_as_it_grows():
    """Eq. 14: every Q_SOM->SOM > 0 damps, and more strongly as it grows (Fig. 6C right)."""
    reals = [mf.oscillatory_mode(mf.coupling_matrix(SOMSOM=q))["real"] for q in (0.25, 0.5, 1, 2, 4)]
    assert all(r < 0 for r in reals)
    assert all(b < a for a, b in zip(reals, reals[1:]))


def test_proposition_1_e_and_pv_lead_som_for_positive_couplings():
    """Proposition 1: any oscillating mode with all Q > 0 has E and PV ahead of SOM in phase."""
    rng = np.random.default_rng(7)
    examined = 0
    for _ in range(400):
        q = dict(zip(["EE", "EPV", "PVE", "PVPV", "ESOM", "SOME", "SOMPV"], rng.uniform(0.1, 8.0, 7)))
        m = mf.oscillatory_mode(mf.coupling_matrix(**q))
        if m is None:
            continue
        assert m["lead_over_som"]["PV"]["rad"] > 0 and m["lead_over_som"]["E"]["rad"] > 0, q
        examined += 1
    assert examined > 50, "fixture must hold oscillating draws"


def test_unknown_coupling_is_refused():
    with pytest.raises(ValueError):
        mf.coupling_matrix(SOMXX=1.0)


@pytest.mark.parametrize("ctype", mf.TYPES)
def test_transfer_is_finite_nonnegative_and_monotone_in_inputs(ctype):
    """Rate grid up to 5 Hz (the network's operating range): finite, >= 0, rises with nu_E for every type
    and falls with nu_PV for E. SOM has no PV input (Q_PV->SOM = 0), so its output is exactly
    independent of nu_PV. PV is pinned by the next test. The Table 3 polynomial is not monotone in nu_PV
    for E once nu_E >= 10 Hz (85.26 Hz at nu_PV 0.5, 85.84 Hz at 2)."""
    grid = [0.5, 1.0, 2.0, 3.5, 5.0]
    out = np.array([[[mf.transfer(ctype, nE, nP, nS, 0.0) for nS in grid] for nP in grid] for nE in grid])
    assert out.size == 125
    assert np.all(np.isfinite(out)) and np.all(out >= 0)
    assert np.ptp(out) > 0, "precondition: the outputs must differ"
    assert np.all(np.diff(out, axis=0) > 0), "must rise with nu_E"
    if ctype == "SOM":
        assert np.all(np.diff(out, axis=1) == 0), "SOM has no PV input"
    elif ctype == "E":
        assert np.all(np.diff(out, axis=1) < 0), "must fall with nu_PV"


def test_pv_transfer_rises_with_pv_input_at_low_rates_as_printed():
    """Observed property of the printed Table 3 PV row: at nu_E = nu_SOM = 0.5 Hz the PV output rises
    with nu_PV from 0.5 to 2 Hz (72.75 -> 74.99 Hz), then falls. Recorded, not corrected."""
    out = [mf.transfer("PV", 0.5, nP, 0.5, 0.0) for nP in (0.5, 1.0, 2.0, 5.0)]
    assert out[2] > out[0] and out[3] < out[0], out


@pytest.mark.parametrize("ctype", mf.TYPES)
def test_mu_V_equals_E_L_with_all_rates_zero_and_no_external_drive(ctype):
    p = dict(mf.PARAMS)
    p["nu_ext"] = 0.0
    muV, _, _ = mf.moments(ctype, 0.0, 0.0, 0.0, 0.0, p)
    assert muV == pytest.approx(mf.PARAMS["E_L"][ctype], rel=1e-12)


def test_delay_buffer_passes_a_step_two_ms_later():
    """Unit check: a step in nu at t0 is read as the delayed transfer input at t0 + tau_d."""
    dt = 1e-4
    line = mf.DelayLine(round(mf.TAU_D / dt), np.zeros(3))
    k0 = 50  # t0 = 5 ms
    first = None
    for k in range(200):
        if first is None and line.read()[0] > 0.5:
            first = k
        line.write(np.ones(3) if k >= k0 else np.zeros(3))
    assert first is not None, "fixture must see the step"
    assert (first - k0) * dt == pytest.approx(mf.TAU_D)


def test_fixed_point_residual_below_1e8_from_default_start():
    x, res = mf.fixed_point(mf.PARAMS)
    assert x.size == 5 and res.size == 5
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(res))
    assert np.max(np.abs(res)) < 1e-8, res


def test_silencing_all_pv_changes_the_trajectory():
    _, nu_ctl, _ = mf.simulate_meanfield(0.5, pv_silence=0.0)
    _, nu_pv0, _ = mf.simulate_meanfield(0.5, pv_silence=1.0)
    assert np.all(np.isfinite(nu_ctl)) and np.all(np.isfinite(nu_pv0))  # precondition
    assert np.max(np.abs(nu_pv0 - nu_ctl)) > 1e-6, "silencing all PV must change the trajectory"
