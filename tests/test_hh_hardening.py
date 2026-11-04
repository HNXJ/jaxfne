"""Hardening tests for the HH emitter route (packet cl-hh-harden).

Fail-first: written before the fixes; a fail-first test fails on the
baseline tree. A test that passes on the baseline implementation is a pin,
not a fail-first test, and its docstring says ``PIN`` with the reason.
Reference: ``artifacts/etudes/papers/gsdr2026/PORT_SPEC.md``. CPU-only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne import emitters_hh
from jaxfne.emitters_hh import (
    graded_syn_step,
    hh_step,
    make_hh_params,
    ou_step,
    simulate_hh,
)

DT_MS = 0.1


def _two_cell_kwargs(**over):
    params = make_hh_params(["E", "Ig"])
    kw = dict(
        params=params,
        n_steps=500,
        dt_ms=DT_MS,
        edge_pre=[0],
        edge_post=[1],
        edge_weight=[4.0],
        edge_receptor=[0],
        receptor_e_rev=(0.0, -80.0),
        receptor_tau_r=(0.2, 0.5),
        receptor_tau_d=(2.0, 5.0),
        receptor_slope=(5.0, 5.0),
        receptor_vth=(-20.0, -20.0),
        ou_tau=[20.0, 20.0],
        ou_sigma=[0.0, 0.0],
        ou_key=jax.random.PRNGKey(0),
        I_ext=None,
    )
    kw.update(over)
    return kw


def _driven_two_cell_i_ext(n_steps=500, amp=10.0):
    i_ext = np.zeros((n_steps, 2), dtype=np.float64)
    i_ext[:, 0] = amp
    return i_ext


# --- Finding 1: hold-and-surface for non-finite updates ---------------------

def test_simulate_hh_holds_and_counts_nonfinite_synapse():
    # Denormal-but-positive tauR passes entry validation yet overflows the
    # gating update once the driven pre cell spikes: the kernel must hold
    # the old gating and report a nonzero held count (not raise: the GSDR
    # fit maps a NaN loss to inf and reverts the generation).
    kw = _two_cell_kwargs(receptor_tau_r=(1e-320, 0.5),
                          I_ext=_driven_two_cell_i_ext())
    out = simulate_hh(**kw)
    assert int(out["held_steps"]["s"]) > 0
    assert int(out["held_steps"]["ou"]) == 0
    assert bool(jnp.all(jnp.isfinite(out["s"])))
    again = simulate_hh(**kw)
    assert bool(jnp.all(again["V_mV"] == out["V_mV"]))


def test_simulate_hh_holds_and_counts_nonfinite_ou():
    # Finite-but-huge sigma/tau pass entry validation yet overflow the OU
    # update on the first step (float32 arithmetic): the kernel must hold
    # the old noise and report a nonzero held count. Edgeless, so no
    # synapse hold can fire first.
    kw = _two_cell_kwargs(ou_sigma=[1e30, 0.0], ou_tau=[1e-30, 20.0],
                          ou_key=jax.random.PRNGKey(0),
                          edge_pre=None, edge_post=None,
                          edge_weight=None, edge_receptor=None)
    out = simulate_hh(**kw)
    assert int(out["held_steps"]["ou"]) > 0
    assert int(out["held_steps"]["s"]) == 0
    assert bool(jnp.all(jnp.isfinite(out["ou"])))


def test_simulate_hh_held_steps_zero_when_clean():
    out = simulate_hh(**_two_cell_kwargs(I_ext=_driven_two_cell_i_ext()))
    assert int(out["held_steps"]["s"]) == 0
    assert int(out["held_steps"]["ou"]) == 0


def test_graded_syn_step_holds_old_s_on_nonfinite():
    """PIN: passes on the base implementation; guards the hold barrier."""
    s = jnp.asarray([np.nan])
    out = graded_syn_step(s, jnp.asarray([20.0]), tau_d=2.0, tau_r=0.2,
                          slope=5.0, v_th=-20.0, dt=DT_MS)
    assert bool(jnp.array_equal(out, s, equal_nan=True))


def test_ou_step_holds_old_n_on_nonfinite():
    """PIN: passes on the base implementation; guards the hold barrier."""
    key = jax.random.PRNGKey(0)
    n = jnp.asarray([np.inf])
    out = ou_step(n, key, mean=0.0, sigma=0.05, tau=20.0, dt=DT_MS)
    assert bool(jnp.all(out == n))


# --- Finding 2: steppers stay pure jnp; dt validated at entry ---------------

def test_hh_step_jits():
    """PIN: passes on the base implementation; guards stepper jittability.

    ``params`` is closed over (a Python object, not a JAX type); the point
    is the stepper body traces without ``float()``/``bool()``/``int()`` on
    traced values.
    """
    params = make_hh_params(["E"])
    state = (jnp.asarray([-70.0]), jnp.asarray([0.05]),
             jnp.asarray([0.6]), jnp.asarray([0.32]))
    f = jax.jit(lambda st, i: hh_step(st, i, params, DT_MS))
    v, m, h, n = f(state, jnp.asarray([0.0]))
    assert bool(jnp.all(jnp.isfinite(v)))


def test_graded_syn_step_grad_flows():
    """PIN: passes on the base implementation; guards differentiability."""
    s0 = jnp.asarray([0.1], dtype=jnp.float32)
    v_pre = jnp.asarray([20.0], dtype=jnp.float32)
    g = jax.grad(lambda s: graded_syn_step(
        s, v_pre, tau_d=2.0, tau_r=0.2,
        slope=5.0, v_th=-20.0, dt=DT_MS).sum())(s0)
    assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_simulate_hh_refuses_nonpositive_dt(bad):
    """PIN: the base entry check refused these too; guards entry validation."""
    with pytest.raises(ValueError, match="dt_ms"):
        simulate_hh(**_two_cell_kwargs(dt_ms=bad))


def test_simulate_hh_refuses_nonfinite_dt():
    with pytest.raises(ValueError, match="dt_ms"):
        simulate_hh(**_two_cell_kwargs(dt_ms=float("inf")))


# --- Finding 3: declared defaults, required OU tau --------------------------

def test_receptor_default_constants_match_port_spec():
    assert emitters_hh.HH_GRADED_SLOPE_DEFAULT == 5.0
    assert emitters_hh.HH_GRADED_VTH_DEFAULT == -20.0
    assert emitters_hh.HH_GRADED_S0_DEFAULT == 0.1
    assert emitters_hh.HH_OU_SIGMA_DEFAULT == 0.05


def test_receptor_fallback_values_unchanged():
    """PIN: passes on the base implementation; guards the fallback values."""
    from jaxfne._construct_hh import _HH_RECEPTOR_DEFAULTS

    assert _HH_RECEPTOR_DEFAULTS == {
        "tau_r_ms": 0.5, "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1,
    }


def test_simulate_hh_requires_ou_tau_when_noise_live():
    kw = _two_cell_kwargs(ou_sigma=[0.03, 0.03])
    del kw["ou_tau"]
    with pytest.raises(ValueError, match="ou_tau"):
        simulate_hh(**kw)


def test_simulate_hh_tau_not_required_when_gain_silences_noise():
    """PIN: passes on the base implementation (it never requires ou_tau).

    Noise is live only when a key is given and the effective sigma (sigma
    times gain) is nonzero: with gain 0 the fallback tau covers the run.
    """
    kw = _two_cell_kwargs(ou_sigma=[0.03, 0.03], noise_gain=0.0)
    del kw["ou_tau"]
    out = simulate_hh(**kw)
    assert bool(jnp.all(jnp.isfinite(out["V_mV"])))


def test_simulate_hh_refuses_nonfinite_ou_tau():
    with pytest.raises(ValueError, match="ou_tau"):
        simulate_hh(**_two_cell_kwargs(ou_sigma=[0.03, 0.03],
                                       ou_tau=[float("nan"), 20.0]))


# --- Finding 4: mixed s0 refused at construction ----------------------------

def test_construct_hh_refuses_mixed_s0():
    import jaxfne as jtfne
    from jaxfne._construct_hh import construct_hh_model

    cfg = (
        jtfne.Configuration()
        .network(name="mini", n=3, cell_types={"E": 2.0 / 3.0, "Ig": 1.0 / 3.0})
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0, "s0": 0.1})
        .mechanisms(name="gaba", kind="receptor",
                    params={"tau_ms": 5.0, "reversal_mV": -80.0, "s0": 0.2})
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=None, weight=4.0, mechanism="ampa")
    ).connectivity(edge_seed=0)
    with pytest.raises(ValueError, match=r"mixed s0.*kernel takes one scalar"):
        construct_hh_model(cfg)


# --- Finding 5: ou_key validation still refuses garbage ---------------------

@pytest.mark.parametrize("bad", ["not-a-key", 3, np.zeros((3,), np.uint32)])
def test_simulate_hh_refuses_bad_ou_key(bad):
    """PIN: passes on the base implementation (its broad catch refused
    these too); guards the narrowed ``(TypeError, ValueError)`` catch."""
    with pytest.raises(ValueError, match="ou_key"):
        simulate_hh(**_two_cell_kwargs(ou_sigma=[0.03, 0.03], ou_key=bad))


# --- Finding 6: OU stationary variance is dt-independent --------------------

@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
def test_ou_stationary_variance_dt_rescaling(dt):
    """PIN: passes on the base implementation; guards the OU discretization."""
    tau, sigma = 5.0, 0.05
    n_chains, burn, meas = 2048, 500, 2000
    keys = jax.random.split(jax.random.PRNGKey(7), burn + meas)
    n = jnp.zeros((n_chains,), jnp.float32)
    for k in keys[:burn]:
        n = ou_step(n, k, mean=0.0, sigma=np.float32(sigma),
                    tau=np.float32(tau), dt=np.float32(dt))
    acc = np.zeros((meas, n_chains), dtype=np.float32)
    for i, k in enumerate(keys[burn:]):
        n = ou_step(n, k, mean=0.0, sigma=np.float32(sigma),
                    tau=np.float32(tau), dt=np.float32(dt))
        acc[i] = np.asarray(n)
    vals = np.asarray(acc, dtype=np.float64)
    n_eff = n_chains * (meas * dt) / (2.0 * tau)
    assert float(np.sqrt(2.0 / n_eff)) < 0.02, \
        "run length must give a standard error below 2%%, got %r" % float(np.sqrt(2.0 / n_eff))
    var = float(vals.var())
    assert abs(var - sigma ** 2) / sigma ** 2 < 0.05, \
        "OU variance must stay within 5%% of sigma^2 at dt=%r, got %r" % (dt, var)


# --- Finding 7 (T-38d): held_steps unit; dt must be a number ----------------

def test_held_steps_docstring_states_element_count():
    """Fail-first on the base docstring: held_steps counts held element updates."""
    doc = " ".join(simulate_hh.__doc__.split())
    assert "one per edge per held step" in doc


def test_held_steps_counts_one_per_edge_per_held_step():
    """PIN: passes on the base implementation; pins the unit the docstring states.

    Two identical edges (same pre, post, receptor) hold on the same steps, so
    the synapse count doubles with the second edge: one count per held
    (step, edge) pair.
    """
    tau_r = (1e-320, 0.5)
    one = simulate_hh(**_two_cell_kwargs(receptor_tau_r=tau_r,
                                         I_ext=_driven_two_cell_i_ext()))
    two = simulate_hh(**_two_cell_kwargs(receptor_tau_r=tau_r,
                                         I_ext=_driven_two_cell_i_ext(),
                                         edge_pre=[0, 0], edge_post=[1, 1],
                                         edge_weight=[4.0, 4.0], edge_receptor=[0, 0]))
    n_one = int(one["held_steps"]["s"])
    assert n_one > 0, "fixture must hold synapse updates"
    assert int(two["held_steps"]["s"]) == 2 * n_one


@pytest.mark.parametrize("bad", ["0.1", b"0.1"])
def test_simulate_hh_refuses_string_dt(bad):
    """Fail-first: a string dt is refused with TypeError, not coerced by float()."""
    with pytest.raises(TypeError, match="dt_ms"):
        simulate_hh(**_two_cell_kwargs(dt_ms=bad))


def test_simulate_hh_batch_refuses_string_dt():
    """Fail-first: the batch entry checks dt_ms before its batch mapping."""
    params = make_hh_params(["E", "Ig"])
    with pytest.raises(TypeError, match="dt_ms"):
        emitters_hh.simulate_hh_batch(params, n_steps=500, dt_ms="0.1", batch=None)
