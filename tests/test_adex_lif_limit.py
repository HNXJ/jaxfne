"""AdEx conductance-LIF limit: ``Delta_mV == 0.0`` runs exact LIF.

``jaxfne/emitters_adex.py:adex_step`` divides by ``Delta_mV``, so a LIF cell
(``Delta_mV = 0.0``) is inexpressible without a limit branch: at ``V == VT``
the quotient is ``0/0`` and the step returns NaN. The limit branch selects
the exact LIF equations per cell (exponential current zero, threshold at
``VT``); positive ``Delta_mV`` steps the reference equations bit-identically.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters_adex import AdExParams, adex_step, make_adex_params

assert "site-packages" not in jtfne.__file__

DT_MS = 0.1


def _one_cell(*, delta, v0=-65.0, current=200.0):
    f = np.float32
    params = AdExParams(
        C_pF=jnp.asarray([f(200.0)]),
        gL_nS=jnp.asarray([f(10.0)]),
        EL_mV=jnp.asarray([f(-60.0)]),
        VT_mV=jnp.asarray([f(-50.0)]),
        Delta_mV=jnp.asarray([f(delta)]),
        Vreset_mV=jnp.asarray([f(-60.0)]),
        tauw_ms=jnp.asarray([f(500.0)]),
        a_nS=jnp.asarray([f(0.0)]),
        b_pA=jnp.asarray([f(0.0)]),
        Tref_ms=jnp.asarray([f(5.0)]),
        Vpeak_mV=jnp.asarray([f(float("nan"))]),
        labels=("E",),
    )
    state = (
        jnp.asarray([f(v0)]),
        jnp.asarray([f(0.0)]),
        jnp.asarray([0], dtype=jnp.int32),
    )
    return params, state, jnp.asarray([f(current)])


def _euler_lif(v0, current):
    """Hand Euler step of the LIF limit in float32, same operation order."""
    f = np.float32
    dV = (f(10.0) * (f(-60.0) - f(v0)) + f(current)) / f(200.0) * f(DT_MS)
    return f(v0) + dV


def test_lif_limit_matches_euler_closed_form():
    """Subthreshold Delta=0 steps the exact LIF Euler update."""
    params, state, I = _one_cell(delta=0.0, v0=-65.0, current=200.0)
    (V_new, w_new, _), spk = adex_step(state, I, params, DT_MS)
    assert bool(np.asarray(spk)[0]) is False, "subthreshold cell must not spike"
    np.testing.assert_allclose(
        np.asarray(V_new), [_euler_lif(-65.0, 200.0)], rtol=1e-6,
        err_msg="Delta=0 must equal the LIF closed form",
    )
    np.testing.assert_allclose(np.asarray(w_new), [0.0], rtol=0, atol=0)


def test_lif_limit_spikes_at_vt():
    """A driven cell starting exactly at VT spikes and resets (0/0 site)."""
    params, state, I = _one_cell(delta=0.0, v0=-50.0, current=200.0)
    (V_new, _, ref_new), spk = adex_step(state, I, params, DT_MS)
    assert bool(np.asarray(spk)[0]) is True, "driven LIF cell at VT must spike"
    np.testing.assert_allclose(np.asarray(V_new), [-60.0], rtol=0, atol=0)
    assert int(np.asarray(ref_new)[0]) == 50, "ref must be round(5/0.1)"


def test_lif_limit_exact_threshold_stays_finite():
    """Undriven cell at exactly VT relaxes below it, finite, no spike."""
    params, state, I = _one_cell(delta=0.0, v0=-50.0, current=0.0)
    (V_new, _, _), spk = adex_step(state, I, params, DT_MS)
    assert bool(np.isfinite(np.asarray(V_new)).all()), "V must stay finite at V==VT"
    assert bool(np.asarray(spk)[0]) is False, "relaxing cell must not spike"
    np.testing.assert_allclose(
        np.asarray(V_new), [_euler_lif(-50.0, 0.0)], rtol=1e-6,
        err_msg="Delta=0 at VT must equal the LIF closed form",
    )


def test_positive_delta_matches_reference_formula():
    """Delta=2 keeps the exponential term (guards the branch refactor)."""
    f = np.float32
    params, state, I = _one_cell(delta=2.0, v0=-65.0, current=200.0)
    (V_new, _, _), spk = adex_step(state, I, params, DT_MS)
    arg = (f(-65.0) - f(-50.0)) / f(2.0)
    exp_current = f(10.0) * f(2.0) * np.exp(arg, dtype=np.float32)
    dV = (f(10.0) * (f(-60.0) - f(-65.0)) + exp_current + f(200.0)) / f(200.0) * f(DT_MS)
    assert bool(np.asarray(spk)[0]) is False
    np.testing.assert_allclose(np.asarray(V_new), [f(-65.0) + dV], rtol=1e-5)


def test_mixed_delta_population_steps_each_cell_under_its_model():
    """One Delta=0 cell and one Delta=2 cell step side by side exactly."""
    f = np.float32
    two = {
        name: jnp.asarray([v, v2])
        for name, v, v2 in [
            ("C_pF", 200.0, 200.0), ("gL_nS", 10.0, 10.0),
            ("EL_mV", -60.0, -60.0), ("VT_mV", -50.0, -50.0),
            ("Delta_mV", 0.0, 2.0), ("Vreset_mV", -60.0, -60.0),
            ("tauw_ms", 500.0, 500.0), ("a_nS", 0.0, 0.0),
            ("b_pA", 0.0, 0.0), ("Tref_ms", 5.0, 5.0),
        ]
    }
    params = AdExParams(
        **{k: jnp.asarray([f(x) for x in v]) for k, v in two.items()},
        Vpeak_mV=jnp.asarray([f(float("nan")), f(float("nan"))]),
        labels=("E", "E"),
    )
    state = (
        jnp.asarray([f(-50.0), f(-65.0)]),
        jnp.asarray([f(0.0), f(0.0)]),
        jnp.asarray([0, 0], dtype=jnp.int32),
    )
    I = jnp.asarray([f(0.0), f(200.0)])
    (V_new, _, _), spk = adex_step(state, I, params, DT_MS)
    got = np.asarray(V_new)
    assert bool(np.isfinite(got).all()), "mixed population must stay finite"
    assert [bool(s) for s in np.asarray(spk)] == [False, False]
    arg = (f(-65.0) - f(-50.0)) / f(2.0)
    exp_current = f(10.0) * f(2.0) * np.exp(arg, dtype=np.float32)
    dV = (f(10.0) * (f(-60.0) - f(-65.0)) + exp_current + f(200.0)) / f(200.0) * f(DT_MS)
    np.testing.assert_allclose(
        got, [_euler_lif(-50.0, 0.0), f(-65.0) + dV], rtol=1e-5,
    )


def test_zero_delta_passes_param_validation():
    """Delta_mV=0.0 is accepted by the AdEx parameter constructors."""
    params = make_adex_params(("E",), {"E": {"Delta_mV": 0.0}})
    assert float(np.asarray(params.Delta_mV)[0]) == 0.0
