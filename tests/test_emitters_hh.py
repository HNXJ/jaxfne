"""Tests for the HH conductance kernel in ``jaxfne/emitters_hh.py``.

Fail-first: written before the module exists; on the baseline tree the import
fails. Reference values: PORT_SPEC.md sections 2-4 (classic HH kinetics,
tanh graded synapses, OU channel of gsdr/models.py Inoise, Fig. 6 run values
in misc/main.py, read as data). CPU-only, float32, small grids.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters_hh import (
    HH_CELL_TYPE_DEFAULTS,
    graded_syn_current,
    graded_syn_step,
    hh_step,
    make_hh_params,
    ou_step,
    simulate_hh,
)

DT_MS = 0.1


def _one_cell_params():
    return make_hh_params(["E"])


def _drive_rates(i_inj, n_steps=6000):
    """Spike count of one cell under constant depolarizing current."""
    params = _one_cell_params()
    v = jnp.asarray([-65.0], dtype=jnp.float32)
    m = jnp.asarray([0.05], dtype=jnp.float32)
    h = jnp.asarray([0.6], dtype=jnp.float32)
    n = jnp.asarray([0.32], dtype=jnp.float32)
    inj = jnp.asarray([float(i_inj)], dtype=jnp.float32)
    count = 0
    prev = float(v[0])
    for _ in range(n_steps):
        (v, m, h, n) = hh_step((v, m, h, n), inj, params, DT_MS)
        cur = float(v[0])
        if prev < -20.0 <= cur:
            count += 1
        prev = cur
    return count, cur


def test_hh_single_cell_spikes_under_drive():
    count, peak_holder = _drive_rates(10.0)
    assert count > 0, "driven HH cell must spike (fixture must spike)"


def test_hh_spike_peak_overshoots_zero():
    params = _one_cell_params()
    v = jnp.asarray([-65.0], dtype=jnp.float32)
    m = jnp.asarray([0.05], dtype=jnp.float32)
    h = jnp.asarray([0.6], dtype=jnp.float32)
    n = jnp.asarray([0.32], dtype=jnp.float32)
    inj = jnp.asarray([10.0], dtype=jnp.float32)
    peak = -100.0
    for _ in range(6000):
        (v, m, h, n) = hh_step((v, m, h, n), inj, params, DT_MS)
        peak = max(peak, float(v[0]))
    assert peak > 0.0, "HH spike peak must overshoot 0 mV, got %r" % peak


def test_hh_single_cell_quiet_at_rest():
    count, _ = _drive_rates(0.0)
    assert count == 0, "undriven HH cell must not spike, got %r spikes" % count


def test_hh_rate_current_curve_increases():
    low, _ = _drive_rates(6.0)
    high, _ = _drive_rates(12.0)
    assert high > 0, "strong drive must spike (fixture must spike)"
    assert high >= low, "rate must not fall with stronger drive: %r vs %r" % (high, low)


def test_hh_defaults_are_classic_values():
    for ctype in ("E", "Ig", "Il"):
        d = HH_CELL_TYPE_DEFAULTS[ctype]
        assert d["gNa_mS_cm2"] == 120.0
        assert d["gK_mS_cm2"] == 36.0
        assert d["gL_mS_cm2"] == 0.3
        assert d["ENa_mV"] == 50.0
        assert d["EK_mV"] == -77.0
        assert d["EL_mV"] == -54.3
        assert d["Cm_uF_cm2"] == 1.0


def test_make_hh_params_refuses_unknown_cell_type():
    with pytest.raises(ValueError, match="unknown HH cell type"):
        make_hh_params(["E", "PV"])


def test_make_hh_params_refuses_unknown_field():
    with pytest.raises(ValueError, match="unknown HH parameter field"):
        make_hh_params(["E"], overrides={"E": {"gNa_zzz": 1.0}})


def test_make_hh_params_refuses_absent_type_override():
    with pytest.raises(ValueError, match="not in labels"):
        make_hh_params(["E"], overrides={"Ig": {"gNa_mS_cm2": 100.0}})


def test_graded_synapse_steady_state():
    # Clamped depolarized presynaptic voltage: s -> tauD/(tauD+tauR).
    s = jnp.asarray([0.1], dtype=jnp.float32)
    for _ in range(2000):
        s = graded_syn_step(s, jnp.asarray([20.0]), tau_d=2.0, tau_r=0.2,
                            slope=5.0, v_th=-20.0, dt=DT_MS)
    assert abs(float(s[0]) - 2.0 / 2.2) < 0.01, \
        "s_inf must equal tauD/(tauD+tauR), got %r" % float(s[0])


def test_graded_synapse_decays_when_silent():
    s = jnp.asarray([0.5], dtype=jnp.float32)
    for _ in range(2000):
        s = graded_syn_step(s, jnp.asarray([-70.0]), tau_d=2.0, tau_r=0.2,
                            slope=5.0, v_th=-20.0, dt=DT_MS)
    assert float(s[0]) < 0.05, "silent synapse must decay below 0.05, got %r" % float(s[0])


def test_graded_syn_current_sign_convention():
    # Outward-positive: excitatory (E_rev=0) at rest drives inward (negative).
    i_ampa = float(graded_syn_current(2.5, 0.5, -65.0, 0.0))
    assert i_ampa < 0.0, "AMPA current at rest must be inward, got %r" % i_ampa
    i_gaba = float(graded_syn_current(5.0, 0.5, -65.0, -80.0))
    assert i_gaba > 0.0, "GABAa current at rest must be outward, got %r" % i_gaba


def test_ou_stationary_moments():
    key = jax.random.PRNGKey(0)
    n = jnp.asarray(0.0)
    vals = []
    for t in range(60000):
        key, sub = jax.random.split(key)
        n = ou_step(n, sub, mean=0.0, sigma=0.05, tau=20.0, dt=DT_MS)
        if t >= 5000:
            vals.append(float(n))
    arr = np.asarray(vals)
    assert abs(arr.mean()) < 0.015, "OU mean must be ~0, got %r" % arr.mean()
    assert abs(arr.var() - 0.05 ** 2) / 0.05 ** 2 < 0.15, \
        "OU variance must be ~sigma^2, got %r" % arr.var()


def test_ou_zero_sigma_relaxes_to_mean():
    key = jax.random.PRNGKey(3)
    n = jnp.asarray(1.0)
    for _ in range(2000):
        key, sub = jax.random.split(key)
        n = ou_step(n, sub, mean=0.0, sigma=0.0, tau=20.0, dt=DT_MS)
    assert abs(float(n)) < 0.01, "sigma=0 OU must relax to mean, got %r" % float(n)


def _two_cell_kwargs(**over):
    params = make_hh_params(["E", "Ig"])
    kw = dict(
        params=params,
        n_steps=3000,
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


def test_simulate_hh_driven_cell_spikes_and_shapes():
    i_ext = np.zeros((3000, 2), dtype=np.float32)
    i_ext[:, 0] = 10.0
    out = simulate_hh(**_two_cell_kwargs(I_ext=i_ext))
    assert out["spikes"].shape == (3000, 2)
    assert out["V_mV"].shape == (3000, 2)
    assert int(np.asarray(out["spikes"])[:, 0].sum()) > 0, \
        "driven cell must spike (fixture must spike)"


def test_simulate_hh_excitation_propagates():
    i_ext = np.zeros((6000, 2), dtype=np.float32)
    i_ext[:, 0] = 10.0
    out = simulate_hh(**_two_cell_kwargs(n_steps=6000, I_ext=i_ext,
                                         edge_weight=[8.0]))
    assert int(np.asarray(out["spikes"])[:, 1].sum()) > 0, \
        "strong AMPA drive must make the follower spike"


def test_simulate_hh_is_deterministic_given_key():
    out1 = simulate_hh(**_two_cell_kwargs(ou_sigma=[0.03, 0.03]))
    out2 = simulate_hh(**_two_cell_kwargs(ou_sigma=[0.03, 0.03]))
    np.testing.assert_array_equal(np.asarray(out1["V_mV"]), np.asarray(out2["V_mV"]))


def test_simulate_hh_refuses_partial_edges():
    kw = _two_cell_kwargs()
    del kw["edge_post"]
    with pytest.raises(ValueError, match="all given or all None"):
        simulate_hh(**kw)


def test_simulate_hh_refuses_receptor_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        simulate_hh(**_two_cell_kwargs(receptor_tau_d=(2.0,)))


def test_simulate_hh_refuses_nonpositive_tau():
    with pytest.raises(ValueError, match="tau"):
        simulate_hh(**_two_cell_kwargs(receptor_tau_d=(0.0, 5.0)))


def test_simulate_hh_refuses_negative_weight():
    with pytest.raises(ValueError, match="edge_weight"):
        simulate_hh(**_two_cell_kwargs(edge_weight=[-1.0]))


def test_simulate_hh_refuses_bad_drive_shape():
    with pytest.raises(ValueError, match="I_ext_pA|I_ext"):
        simulate_hh(**_two_cell_kwargs(I_ext=np.zeros((10, 5), dtype=np.float32)))


def test_simulate_hh_refuses_unconsumed_delay_kwarg():
    with pytest.raises(TypeError):
        simulate_hh(**_two_cell_kwargs(edge_delay_steps=[1]))


# --- Fidelity F1-F4 (PORT_SPEC.md fidelity order: integration, then units) ---

def test_hh_step_stays_finite_under_drive_dt01():
    """Explicit Euler goes NaN at step 21 under I=10/dt=0.1 (diag_nan.py);
    exponential-Euler gates + backward-Euler voltage must stay finite."""
    params = _one_cell_params()
    v = jnp.asarray([-70.0], dtype=jnp.float32)
    from jaxfne.emitters_hh import hh_gate_inf

    m0, h0, n0 = hh_gate_inf(jnp.asarray([-70.0], dtype=jnp.float32))
    m = jnp.asarray([float(m0[0])], dtype=jnp.float32)
    h = jnp.asarray([float(h0[0])], dtype=jnp.float32)
    n = jnp.asarray([float(n0[0])], dtype=jnp.float32)
    inj = jnp.asarray([10.0], dtype=jnp.float32)
    count = 0
    prev = -70.0
    for _ in range(3000):
        (v, m, h, n) = hh_step((v, m, h, n), inj, params, DT_MS)
        cur = float(v[0])
        assert np.all(np.isfinite(np.asarray(v))), "V went non-finite under drive"
        assert np.all(np.isfinite(np.asarray(m))), "m went non-finite under drive"
        if prev < -20.0 <= cur:
            count += 1
        prev = cur
    assert count > 0, "driven HH cell must spike (fixture must spike)"


def test_hh_gate_update_is_exponential_euler():
    """One gate update must equal the exact exponential-Euler step at frozen
    rates (jaxley solver_gate.solve_gate_exponential, read as data)."""
    from jaxfne.emitters_hh import hh_gate_inf

    params = _one_cell_params()
    v = jnp.asarray([-70.0], dtype=jnp.float32)
    m0, h0, n0 = hh_gate_inf(v)
    state = (v, jnp.asarray([float(m0[0])], dtype=jnp.float32),
             jnp.asarray([float(h0[0])], dtype=jnp.float32),
             jnp.asarray([float(n0[0])], dtype=jnp.float32))
    inj = jnp.asarray([0.0], dtype=jnp.float32)
    (v1, m1, h1, n1) = hh_step(state, inj, params, DT_MS)
    vv = -70.0 + 65.0
    am = 0.1 * (25.0 - vv) / (np.exp((25.0 - vv) / 10.0) - 1.0)
    bm = 4.0 * np.exp(-vv / 18.0)
    tau = 1.0 / (am + bm)
    expected = float(m0[0]) * np.exp(-DT_MS / tau) + (am * tau) * (1.0 - np.exp(-DT_MS / tau))
    assert abs(float(m1[0]) - expected) < 1e-4, \
        "m update must be exponential Euler, got %r vs %r" % (float(m1[0]), expected)


def test_hh_defaults_e_leak_minus_54_3():
    """eLeak is -54.3 in the reference HH channel (jaxley channels/hh.py)."""
    for ctype in ("E", "Ig", "Il"):
        assert HH_CELL_TYPE_DEFAULTS[ctype]["EL_mV"] == -54.3, \
            "EL must be the reference -54.3 mV, got %r" % HH_CELL_TYPE_DEFAULTS[ctype]["EL_mV"]


def test_hh_rest_voltage_minus_70():
    """Reference compartments init at v=-70 mV (jaxley compartment.py)."""
    from jaxfne.emitters_hh import HH_V_REST_MV

    assert HH_V_REST_MV == -70.0, "rest/V0 must be -70 mV, got %r" % HH_V_REST_MV


def test_hh_gate_inf_is_fixed_point():
    """Gate init is the rate fixed point at V0 (jaxley HH.init_state)."""
    from jaxfne.emitters_hh import HH_V_REST_MV, hh_gate_inf

    v = jnp.asarray([HH_V_REST_MV], dtype=jnp.float32)
    m0, h0, n0 = hh_gate_inf(v)
    params = _one_cell_params()
    inj = jnp.asarray([0.0], dtype=jnp.float32)
    (v1, m1, h1, n1) = hh_step(
        (v, jnp.asarray([float(m0[0])], dtype=jnp.float32),
         jnp.asarray([float(h0[0])], dtype=jnp.float32),
         jnp.asarray([float(n0[0])], dtype=jnp.float32)),
        inj, params, DT_MS)
    np.testing.assert_allclose(float(m1[0]), float(m0[0]), rtol=1e-6, atol=0.0,
                               err_msg="m must sit at its fixed point without drive")
    np.testing.assert_allclose(float(h1[0]), float(h0[0]), rtol=1e-6, atol=0.0,
                               err_msg="h must sit at its fixed point without drive")
    np.testing.assert_allclose(float(n1[0]), float(n0[0]), rtol=1e-6, atol=0.0,
                               err_msg="n must sit at its fixed point without drive")


def test_simulate_hh_no_nan_under_drive_dt01():
    """Single driven cell over 600 ms at dt=0.1: finite trace, real spikes."""
    params = _one_cell_params()
    n_steps = 6000
    i_ext = np.full((n_steps, 1), 10.0, dtype=np.float32)
    out = simulate_hh(params, n_steps=n_steps, dt_ms=DT_MS,
                      receptor_e_rev=(0.0,), receptor_tau_r=(0.2,),
                      receptor_tau_d=(2.0,), receptor_slope=(5.0,),
                      receptor_vth=(-20.0,), I_ext=i_ext,
                      ou_key=jax.random.PRNGKey(0))
    v = np.asarray(out["V_mV"])[:, 0]
    assert np.all(np.isfinite(v)), "V trace must stay finite under drive"
    spikes = int(np.asarray(out["spikes"])[:, 0].sum())
    assert spikes > 0, "driven cell must spike (fixture must spike)"


def test_simulate_hh_noise_gain_one_reproduces_default():
    """Lane-4e residual X: ``noise_gain=1.0`` is bit-exactly the default."""
    kw0 = _two_cell_kwargs(ou_sigma=[0.03, 0.03])
    out0 = simulate_hh(**kw0)
    out1 = simulate_hh(**_two_cell_kwargs(ou_sigma=[0.03, 0.03],
                                          noise_gain=1.0))
    np.testing.assert_array_equal(np.asarray(out0["V_mV"]),
                                  np.asarray(out1["V_mV"]))


def test_simulate_hh_noise_gain_zero_silences_noise():
    """``noise_gain=0.0`` equals an all-zero sigma run (same key stream)."""
    out0 = simulate_hh(**_two_cell_kwargs(ou_sigma=[0.0, 0.0]))
    out1 = simulate_hh(**_two_cell_kwargs(ou_sigma=[0.03, 0.03],
                                          noise_gain=0.0))
    np.testing.assert_array_equal(np.asarray(out0["V_mV"]),
                                  np.asarray(out1["V_mV"]))
    assert float(np.asarray(out1["ou"]).max()) == 0.0


def test_simulate_hh_noise_gain_moves_realized_noise():
    """Doubling the gain doubles the realized OU state (same key)."""
    out1 = simulate_hh(**_two_cell_kwargs(ou_sigma=[0.03, 0.03],
                                          noise_gain=1.0))
    out2 = simulate_hh(**_two_cell_kwargs(ou_sigma=[0.03, 0.03],
                                          noise_gain=2.0))
    ou1 = np.asarray(out1["ou"])
    ou2 = np.asarray(out2["ou"])
    assert np.all(np.isfinite(ou2)), "gained OU state must stay finite"
    np.testing.assert_allclose(ou2, 2.0 * ou1, rtol=1e-5)


def test_simulate_hh_refuses_bad_noise_gain():
    for bad in (-1.0, float("nan"), float("inf"), "loud"):
        with pytest.raises(ValueError, match="noise_gain"):
            simulate_hh(**_two_cell_kwargs(ou_sigma=[0.03, 0.03],
                                           noise_gain=bad))
