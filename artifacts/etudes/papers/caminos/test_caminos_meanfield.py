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
    """Grid around the Fig. 1D operating point (E 1.6, PV 8.8, SOM 8.0 Hz): finite, >= 0, rises with
    nu_E for every type and falls with nu_PV for E. SOM has no PV input (Q_PV->SOM = 0), so its output
    is exactly independent of nu_PV. PV is pinned by the next test. With weak inhibition the fitted
    polynomial is not monotone in nu_PV for E: at nu_E = nu_SOM = 0.5 Hz, 39.71 Hz at nu_PV 0.5 and
    45.43 Hz at 2; at nu_E 10 Hz, 53.73 Hz at nu_PV 0.5 and 76.47 Hz at 2."""
    gE, gP, gS = [0.5, 1.0, 1.5, 2.0, 3.0], [6.0, 7.0, 8.0, 9.0, 11.0], [5.0, 6.0, 7.0, 8.0, 10.0]
    out = np.array([[[mf.transfer(ctype, nE, nP, nS, 0.0) for nS in gS] for nP in gP] for nE in gE])
    assert out.size == 125
    assert np.all(np.isfinite(out)) and np.all(out >= 0)
    assert np.ptp(out) > 0, "precondition: the outputs must differ"
    assert np.all(np.diff(out, axis=0) > 0), "must rise with nu_E"
    if ctype == "SOM":
        assert np.all(np.diff(out, axis=1) == 0), "SOM has no PV input"
    elif ctype == "E":
        assert np.all(np.diff(out, axis=1) < 0), "must fall with nu_PV"


def test_pv_transfer_rises_with_pv_input_at_low_rates_as_printed():
    """Observed property of the fitted PV polynomial: at nu_E = nu_SOM = 0.5 Hz the PV output rises
    with nu_PV from 0.5 to 2 Hz (72.73 -> 74.15 Hz), then falls. Recorded, not corrected."""
    out = [mf.transfer("PV", 0.5, nP, 0.5, 0.0) for nP in (0.5, 1.0, 2.0, 5.0)]
    assert out[2] > out[0] and out[3] < out[0], out


def test_fig1d_gamma_near_35_hz_with_pv_above_som_above_e():
    """Fig. 1D (text line 22): oscillation at 35 Hz, PV > SOM > E. Fails with the quadratic terms in
    the Table 3 header order (12.5 Hz, SOM above PV); passes in the order of the authors' code."""
    t, nu, _ = mf.simulate_meanfield(4.0)
    sel = t >= 2.0
    e = nu[sel, 0]
    assert np.ptp(e) > 0.5, "fixture must oscillate"
    amp = np.abs(np.fft.rfft(e - e.mean()))
    peak = np.fft.rfftfreq(e.size, t[1] - t[0])[1 + int(np.argmax(amp[1:]))]
    assert abs(peak - 35.0) / 35.0 <= 0.15, peak
    m = nu[sel].mean(axis=0)
    assert m[1] > m[2] > m[0], m


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


@pytest.mark.parametrize("coupling", ["PVSOM", "SOMSOM"])
def test_fig5a_zero_coupling_equals_reference_run(coupling):
    """At 0 nS the sweep point equals the reference run (no added coupling)."""
    out = mf.fig5a_sweep(coupling, [0.0], T_s=1.0)
    assert out["coupling"] == coupling and out["g_values_ns"] == [0.0]
    r = out["results"][0]
    t, nu, _ = mf.simulate_meanfield(1.0)
    e = nu[t >= 0.5, 0]
    assert r["min_E_hz"] == e.min() and r["max_E_hz"] == e.max()
    assert r["fixed_E_hz"] == pytest.approx(mf.fixed_point(mf.PARAMS)[0][0])


@pytest.mark.parametrize("coupling", ["PVSOM", "SOMSOM"])
def test_fig5a_nonzero_coupling_changes_e_trajectory(coupling):
    """A nonzero added coupling is consumed: the E-rate trace differs from the reference."""
    p = mf.params_with_coupling(coupling, 3.0)
    assert p["Q"][mf.FIG5A_COUPLINGS[coupling]] == pytest.approx(3e-9)
    _, nu_mod, _ = mf.simulate_meanfield(0.5, params=p)
    _, nu_ref, _ = mf.simulate_meanfield(0.5)
    assert np.all(np.isfinite(nu_mod))  # precondition
    assert np.max(np.abs(nu_mod[:, 0] - nu_ref[:, 0])) > 1e-6


@pytest.mark.parametrize("coupling, g_cycle, g_decay", [("SOMSOM", 1.1, 1.6), ("PVSOM", 0.7, 1.0)])
def test_fig5a_small_sustained_cycle_counts_as_oscillation(coupling, g_cycle, g_decay):
    """SOM->SOM at 1.1 nS and PV->SOM at 0.7 nS keep a constant small cycle (a 16 s run from the +5% perturbed
    fixed point held E peak-to-peak at 0.36 and 0.28 Hz in every 2 s window from 4 s on, under the old 0.5 Hz
    threshold); at g_decay the perturbation decays. First unstable, second stable."""
    lo = mf.fig5a_point(coupling, g_cycle)
    assert 0.0 < 2.0 * lo["amplitude_hz"] < 0.5, "fixture: cycle must sit under the old 0.5 Hz threshold"
    assert lo["stable"] is False and lo["freq_hz"] > 0.0
    hi = mf.fig5a_point(coupling, g_decay)
    assert hi["stable"] is True and hi["freq_hz"] == 0.0


def test_tv_form_code_matches_transcription_and_printed_stays_default():
    """O-93 variant: the code form of T_v gives the transcription's 35.00 Hz (RESULT_MEANFIELD Fig 1D table);
    the default stays the printed form at 33.50 Hz."""
    p = dict(mf.PARAMS, tv_form="code")
    peaks = []
    for params in (mf.PARAMS, p):
        t, nu, _ = mf.simulate_meanfield(4.0, params=params)
        e = nu[t >= 2.0, 0]
        assert np.ptp(e) > 0.5, "fixture must oscillate"
        peaks.append(mf._spectrum_peak(e, t[1] - t[0]))
    assert peaks == [pytest.approx(33.5, abs=0.25), pytest.approx(35.0, abs=0.25)], peaks  # bin 0.5 Hz


def test_tv_form_default_printed_and_unknown_refused():
    assert mf.PARAMS["tv_form"] == "printed"
    bad = dict(mf.PARAMS, tv_form="tau")
    with pytest.raises(ValueError):
        mf.moments("E", 1.0, 1.0, 1.0, 0.0, bad)
    with pytest.raises(ValueError):
        mf.simulate_meanfield(0.0, params=bad)


def test_tv_form_code_moves_som_som_transition():
    """Code form: SOM->SOM still oscillates at 1.5 nS and is stable at 1.7 nS; printed is stable at 1.5."""
    p = dict(mf.PARAMS, tv_form="code")
    assert mf.fig5a_point("SOMSOM", 1.5, params=p)["stable"] is False
    assert mf.fig5a_point("SOMSOM", 1.7, params=p)["stable"] is True
    assert mf.fig5a_point("SOMSOM", 1.5)["stable"] is True


def test_fig5a_unknown_coupling_is_refused():
    with pytest.raises(ValueError):
        mf.fig5a_sweep("SOMXX", [0.0])
    with pytest.raises(ValueError):
        mf.params_with_coupling("SOMXX", 1.0)
    with pytest.raises(ValueError):
        mf.params_with_coupling("PVSOM", -1.0)
