"""F1 (state reduction for HDP traces): L1-L3 checks and a hand case. See artifacts/etudes/theory_f1/README.md."""
from __future__ import annotations

import sys

import numpy as np

import jaxfne  # noqa: F401

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

import artifacts.etudes.theory_f1.f1_check as F  # noqa: E402


def test_l1_factored_equals_direct_float64():
    for name, (M, b) in F.cases().items():
        for seed in F.SEEDS:
            rel = F.check1_rel(M, b, seed)
            assert rel <= 1e-12, (name, seed, rel)


def test_l2_lazy_equals_eager_float64():
    for name, (M, b) in F.cases().items():
        for seed in F.SEEDS:
            rel = F.check2_rel(M, b, seed)
            assert rel <= 1e-12, (name, seed, rel)


def test_l3_negative_control_exceeds_1e3():
    for seed in F.SEEDS:
        diff = F.check3_diff(seed)
        assert diff > 1e-3, (seed, diff)


def test_l4_float32_difference_is_float32_sized_and_within_bound():
    # A float64 run would differ by ~1e-15; a float32 run by ~1e-7 to 1e-5 (measured C 0.012-0.057).
    for a in (F.A_SLOW, F.A_FAST):
        diff = F.check4_diff(a, 0)
        C = diff / (F.U32 / (1 - a) ** 2)
        assert diff > 1e-9, (a, diff)
        assert C <= 1.0, (a, C)


def test_fixture_trains_have_spikes_and_resets():
    for seed in F.SEEDS:
        s, r = F.trains(seed)
        assert s.shape == (F.T,) and r.shape == (F.T,)
        assert int(s.sum()) >= 10, (seed, int(s.sum()))
        assert int(r.sum()) >= 10, (seed, int(r.sum()))


def test_hand_case_a_half_spikes_at_0_and_3_reset_at_2():
    # s = [1, 0, 0, 1], r = [0, 0, 1, 0], M = a = 0.5, b = 1, all states start at 0.
    # Direct: Ahat(0) = 0.5*0 + 1 = 1; Ahat(1) = 0.5*1 = 0.5; Ahat(2) = 0.5*0.5 = 0.25;
    #         Ahat(3) = 0.5*0 + 1 = 1. Reset at t=2 gives A(2) = 0.
    #         So A = [1, 0.5, 0, 1] and Ahat = [1, 0.5, 0.25, 1].
    # Factored: x = [1, 0.5, 0.25, 1.125]; S(2) = x(2) = 0.25 (reset);
    #         Ahat(3) = x(3) - 0.5*S(2) = 1.125 - 0.125 = 1; A(2) = 0.25 - 0.25 = 0.
    s = np.array([1.0, 0.0, 0.0, 1.0])
    r = np.array([0.0, 0.0, 1.0, 0.0])
    M = np.array([[0.5]])
    b = np.array([1.0])

    Ahat, A = F.direct(M, b, s, r)
    np.testing.assert_allclose(A[:, 0], [1.0, 0.5, 0.0, 1.0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(Ahat[:, 0], [1.0, 0.5, 0.25, 1.0], rtol=1e-12, atol=1e-15)

    for lazy in (False, True):
        Fh, Fa = F.factored(M, b, s, r, lazy=lazy)
        np.testing.assert_allclose(Fa[:, 0], [1.0, 0.5, 0.0, 1.0], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(Fh[:, 0], [1.0, 0.5, 0.25, 1.0], rtol=1e-12, atol=1e-15)
