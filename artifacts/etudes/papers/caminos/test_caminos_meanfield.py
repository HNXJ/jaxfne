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
