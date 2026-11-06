"""Tests for the Jansen-Rit 1995 neural-mass column emitter (function level).

Covers: convergence to a fixed point with constant input (residual of the
right-hand side at the final state), the linear synaptic kernel (impulse
response against A a t e^{-a t}), refusal of non-finite or nonpositive
parameters by name, and bit-identical determinism for one drive array.
"""

import numpy as np
import pytest

from jaxfne.emitters_jansen_rit import (
    check_jansen_rit_params,
    jansen_rit_rhs,
    simulate_jansen_rit,
    simulate_linear_kernel,
)

pytestmark = pytest.mark.x64

PARAMS = {
    "A": 2.0,
    "B": 8.0,
    "a": 50.0,
    "b": 25.0,
    "C1": 10.0,
    "C2": 10.0,
    "C3": 10.0,
    "C4": 10.0,
    "e0": 1.0,
    "v0": 5.0,
    "r": 0.5,
}


def _valid_kwargs(**overrides):
    kw = dict(PARAMS)
    kw.update(overrides)
    kw.setdefault("dt", 0.001)
    kw.setdefault("n_steps", 20000)
    kw.setdefault("drive", np.full((kw["n_steps"],), 1.0))
    return kw


def test_fixed_point_rhs_zero():
    """Constant input settles where the right-hand side vanishes."""
    kw = _valid_kwargs()
    out = simulate_jansen_rit(**kw)
    y = out["y"]
    assert y.shape == (kw["n_steps"], 6)
    tail = out["eeg"][kw["n_steps"] // 2 :]
    assert float(tail.max() - tail.min()) < 1e-12
    residual = np.asarray(jansen_rit_rhs(y[-1], 1.0, **PARAMS), dtype=np.float64)
    assert float(np.linalg.norm(residual)) < 1e-8


def test_linear_kernel_impulse():
    """One-step pulse traces h(t) = A a t e^{-a t} on the grid.

    Tolerance rtol=0.01, atol=1e-6 on samples k >= 100: a single-step
    pulse of height 1/dt differs from an ideal impulse by the pulse
    width, which dominates the first steps (relative gap ~1/(2k)) and
    has decayed past the tolerance by k = 100.
    """
    gain, rate, dt, n_steps = 2.0, 10.0, 1e-4, 10000
    drive = np.zeros((n_steps,))
    drive[0] = 1.0 / dt
    y = simulate_linear_kernel(gain, rate, dt, n_steps, drive)["y"]
    t = (np.arange(n_steps) + 1) * dt
    expected = gain * rate * t * np.exp(-rate * t)
    np.testing.assert_allclose(y[100:], expected[100:], rtol=0.01, atol=1e-6)


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"A": 0.0}, "param A="),
        ({"A": float("nan")}, "param A="),
        ({"B": -2.0}, "param B="),
        ({"B": float("inf")}, "param B="),
        ({"a": 0.0}, "param a="),
        ({"a": -1.0}, "param a="),
        ({"a": float("nan")}, "param a="),
        ({"b": float("inf")}, "param b="),
        ({"e0": 0.0}, "param e0="),
        ({"r": -1.0}, "param r="),
        ({"v0": float("nan")}, "param v0="),
        ({"C1": -1.0}, "param C1="),
        ({"dt": 0.0}, "param dt="),
        ({"dt": -0.5}, "param dt="),
        ({"n_steps": 0}, "param n_steps="),
    ],
)
def test_refusals_name_parameter(override, needle):
    """Non-finite or nonpositive constants raise ValueError naming them."""
    kw = _valid_kwargs(**override)
    with pytest.raises(ValueError) as excinfo:
        simulate_jansen_rit(**kw)
    assert needle in str(excinfo.value)


def test_refusal_drive_names_drive():
    """A drive array of the wrong shape or with non-finite samples is refused."""
    kw = _valid_kwargs()
    with pytest.raises(ValueError) as excinfo:
        simulate_jansen_rit(**{**kw, "drive": np.zeros((kw["n_steps"] - 1,))})
    assert "param drive" in str(excinfo.value)
    bad = np.full((kw["n_steps"],), 1.0)
    bad[3] = float("nan")
    with pytest.raises(ValueError) as excinfo:
        check_jansen_rit_params(**{**kw, "drive": bad})
    assert "param drive" in str(excinfo.value)


def test_determinism_bit_identical():
    """One drive array gives one bit-identical trajectory."""
    n_steps = 4000
    drive = np.full((n_steps,), 0.5)
    drive[1000:1200] = 2.0
    first = simulate_jansen_rit(dt=0.001, n_steps=n_steps, drive=drive, **PARAMS)
    second = simulate_jansen_rit(dt=0.001, n_steps=n_steps, drive=drive, **PARAMS)
    np.testing.assert_array_equal(first["y"], second["y"])
    np.testing.assert_array_equal(first["eeg"], second["eeg"])
