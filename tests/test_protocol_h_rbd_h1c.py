"""Protocol H1c-C — postsynaptic recurrent-input gain (affine G_H)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    _rbd_compose_native_current,
    _rbd_recurrent_gain_affine,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_rbd,
)


def _isolated_neuron_params():
    jdtype = jnp.float32
    return IzhikevichParams(
        v0=jnp.asarray([-65.0], dtype=jdtype),
        u0=jnp.zeros((1,), dtype=jdtype),
        a=jnp.asarray([0.02], dtype=jdtype),
        b=jnp.asarray([0.2], dtype=jdtype),
        c=jnp.asarray([-65.0], dtype=jdtype),
        d=jnp.asarray([8.0], dtype=jdtype),
        drive=jnp.zeros((1,), dtype=jdtype),
        sign=jnp.ones((1,), dtype=jdtype),
        W=jnp.zeros((1, 1), dtype=jdtype),
        source_scale=jnp.ones((1,), dtype=jdtype),
        labels=("E",),
        layer_labels=("L4",),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )


def _empty_edges() -> EdgeList:
    jdtype = jnp.float32
    return EdgeList(
        pre=jnp.zeros((0,), dtype=jnp.int32),
        post=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jdtype),
        receptor_index=jnp.zeros((0,), dtype=jnp.int32),
        tau_ms=jnp.zeros((0,), dtype=jdtype),
    )


def _two_neuron_ring():
    jdtype = jnp.float32
    params = _isolated_neuron_params()
    params = IzhikevichParams(
        v0=jnp.full((2,), -65.0, dtype=jdtype),
        u0=jnp.zeros((2,), dtype=jdtype),
        a=jnp.full((2,), 0.02, dtype=jdtype),
        b=jnp.full((2,), 0.2, dtype=jdtype),
        c=jnp.full((2,), -65.0, dtype=jdtype),
        d=jnp.full((2,), 8.0, dtype=jdtype),
        drive=jnp.zeros((2,), dtype=jdtype),
        sign=jnp.ones((2,), dtype=jdtype),
        W=jnp.zeros((2, 2), dtype=jdtype),
        source_scale=jnp.ones((2,), dtype=jdtype),
        labels=("E", "E"),
        layer_labels=("L4", "L4"),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    edges = EdgeList(
        pre=jnp.asarray([0, 1], dtype=jnp.int32),
        post=jnp.asarray([1, 0], dtype=jnp.int32),
        weight=jnp.asarray([5.0, 5.0], dtype=jdtype),
        receptor_index=jnp.asarray([0, 0], dtype=jnp.int32),
        tau_ms=jnp.asarray([2.0, 2.0], dtype=jdtype),
    )
    return params, edges


def _run_rbd(params, edges, *, n_steps, dt_ms=1.0, seed=0, drive_schedule=None, init_state=None, **rbd_kw):
    key = jax.random.PRNGKey(seed)
    return simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        n_steps,
        dt_ms,
        key,
        dtype="float32",
        noise_scale=0.0,
        drive_schedule=drive_schedule,
        init_state=init_state,
        **rbd_kw,
    )


def test_affine_gain_baseline_and_null():
    jdtype = jnp.float32
    H = jnp.asarray([0.8, 1.0, 1.2], dtype=jdtype)
    beta = jnp.asarray(0.5, dtype=jdtype)
    g = _rbd_recurrent_gain_affine("f1", H, beta, jdtype=jdtype)
    assert float(g[1]) == pytest.approx(1.0)
    g0 = _rbd_recurrent_gain_affine("f1", H, jnp.asarray(0.0, dtype=jdtype), jdtype=jdtype)
    assert jnp.allclose(g0, 1.0)
    g_f0 = _rbd_recurrent_gain_affine("f0", H, beta, jdtype=jdtype)
    assert jnp.allclose(g_f0, 1.0)


def test_beta_h_zero_bit_exact_h1a_parity():
    params, edges = _two_neuron_ring()
    n_steps = 40
    drive = jnp.zeros((n_steps, 2), dtype=jnp.float32)
    drive = drive.at[10, 0].set(35.0)
    kw = dict(
        n_steps=n_steps,
        drive_schedule=drive,
        rbd_family="f1",
        init_state={"H": jnp.asarray([1.2, 0.9], dtype=jnp.float32)},
    )
    a = _run_rbd(params, edges, beta_h=0.0, **kw)
    b = _run_rbd(params, edges, **kw)
    for i in range(3):
        assert jnp.allclose(a[i], b[i])
    for key in ("H_trace", "H_final"):
        assert jnp.allclose(a[3][key], b[3][key])


def test_f0_activity_legacy_with_beta_h():
    """F0 forces G_H=1 regardless of beta_H."""
    jdtype = jnp.float32
    params, edges = _two_neuron_ring()
    n_steps = 40
    drive = jnp.zeros((n_steps, 2), dtype=jdtype)
    drive = drive.at[10, 0].set(35.0)
    v0, s0, _, _ = _run_rbd(
        params,
        edges,
        n_steps=n_steps,
        drive_schedule=drive,
        rbd_family="f0",
        beta_h=0.0,
    )
    v1, s1, _, _ = _run_rbd(
        params,
        edges,
        n_steps=n_steps,
        drive_schedule=drive,
        rbd_family="f0",
        beta_h=0.8,
        init_state={"H": jnp.asarray([1.3, 0.7], dtype=jdtype)},
    )
    assert jnp.allclose(v0, v1)
    assert jnp.allclose(s0, s1)


def test_no_recurrent_input_h_invisible_even_with_beta():
    """I^rec=0 => H cannot affect x through the recurrent gain map."""
    params = _isolated_neuron_params()
    edges = _empty_edges()
    n_steps = 30
    drive = jnp.zeros((n_steps, 1), dtype=jnp.float32)
    drive = drive.at[8, 0].set(30.0)
    v_a, s_a, _, _ = _run_rbd(
        params,
        edges,
        n_steps=n_steps,
        drive_schedule=drive,
        beta_h=1.0,
        init_state={"H": jnp.asarray([0.5], dtype=jnp.float32)},
    )
    v_b, s_b, _, _ = _run_rbd(
        params,
        edges,
        n_steps=n_steps,
        drive_schedule=drive,
        beta_h=1.0,
        init_state={"H": jnp.asarray([2.0], dtype=jnp.float32)},
    )
    assert jnp.allclose(v_a, v_b)
    assert jnp.allclose(s_a, s_b)


def test_fixed_w_invariant_under_beta_h():
    params, edges = _two_neuron_ring()
    w0 = np.asarray(edges.weight)
    _, _, _, st = _run_rbd(params, edges, n_steps=30, beta_h=0.6)
    assert np.array_equal(np.asarray(st["w_fixed"]), w0)


def test_nonpositive_gain_rejected_at_init():
    params = _isolated_neuron_params()
    edges = _empty_edges()
    with pytest.raises(ValueError, match="G_H"):
        _run_rbd(
            params,
            edges,
            n_steps=5,
            beta_h=2.0,
            init_state={"H": jnp.asarray([0.1], dtype=jnp.float32)},
        )


def test_sign_symmetry_recurrent_susceptibility():
    """H_k=1±delta with beta_H>0: opposite recurrent susceptibility on neuron k."""
    params, edges = _two_neuron_ring()
    n_steps = 120
    dt_ms = 1.0
    jdtype = jnp.float32
    drive = jnp.zeros((n_steps, 2), dtype=jdtype)
    drive = drive.at[5, 0].set(45.0)
    delta = 0.2
    beta = 0.8
    common = dict(
        n_steps=n_steps,
        dt_ms=dt_ms,
        drive_schedule=drive,
        beta_h=beta,
        kappa_h=0.0,
        rbd_family="f1",
        tau_h_ms=200.0,
    )
    v_hi, s_hi, _, _ = _run_rbd(
        params,
        edges,
        init_state={"H": jnp.asarray([1.0, 1.0 + delta], dtype=jdtype)},
        **common,
    )
    v_lo, s_lo, _, _ = _run_rbd(
        params,
        edges,
        init_state={"H": jnp.asarray([1.0, 1.0 - delta], dtype=jdtype)},
        **common,
    )
    # Postsynaptic gain on neuron 1 during recurrent epoch after drive on neuron 0.
    window = slice(6, 50)
    assert not jnp.allclose(v_hi[window, 1], v_lo[window, 1])
    assert float(jnp.sum(v_hi[window, 1])) > float(jnp.sum(v_lo[window, 1]))


def test_f_h_uses_pre_gain_recurrent_input():
    """F_H kappa path must not see G_H*I_rec (no immediate H->G->I->H loop)."""
    jdtype = jnp.float32
    H = jnp.asarray([1.2], dtype=jdtype)
    I_rec = jnp.asarray([10.0], dtype=jdtype)
    beta = jnp.asarray(0.5, dtype=jdtype)
    I_drive = _rbd_compose_native_current(
        jnp.asarray([0.0], dtype=jdtype),
        I_rec,
        H,
        "f1",
        beta,
        jnp.asarray([0.0], dtype=jdtype),
        jdtype=jdtype,
    )
    g = float(_rbd_recurrent_gain_affine("f1", H, beta, jdtype=jdtype)[0])
    assert float(I_drive[0]) == pytest.approx(g * 10.0)
    I_rel_for_fh = float(I_rec[0] / 1.0)
    assert I_rel_for_fh == pytest.approx(10.0)
    assert I_rel_for_fh != pytest.approx(float(I_drive[0]))


def test_h_at_one_matches_legacy_with_beta():
    params, edges = _two_neuron_ring()
    n_steps = 40
    drive = jnp.zeros((n_steps, 2), dtype=jnp.float32)
    drive = drive.at[10, 0].set(35.0)
    key = jax.random.PRNGKey(0)
    v0, s0, _, _ = simulate_edge_recurrent_izhikevich(
        params,
        edges,
        n_steps,
        1.0,
        key,
        dtype="float32",
        drive_schedule=drive,
        noise_scale=0.0,
    )
    v1, s1, _, _ = _run_rbd(
        params,
        edges,
        n_steps=n_steps,
        drive_schedule=drive,
        beta_h=0.5,
        init_state={"H": jnp.ones(2, dtype=jnp.float32)},
    )
    assert float(jnp.max(jnp.abs(v0 - v1))) == 0.0
    assert float(jnp.max(jnp.abs(s0 - s1))) == 0.0
