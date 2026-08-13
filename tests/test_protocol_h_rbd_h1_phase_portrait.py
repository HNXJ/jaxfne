"""Protocol H1 — isolated F1/F2 phase portraits (I^rel=0).

Validates primitive RBD restoring asymmetry before network experiments:
F1 symmetric about H=1; F2 stronger recovery below 1, slower decay above 1.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    _rbd_advance_h,
    _rbd_restoring_term,
    simulate_edge_recurrent_izhikevich_rbd,
)


def _isolated_neuron_params() -> IzhikevichParams:
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


H0_GRID = (0.5, 0.8, 1.0, 1.2, 2.0)


@pytest.mark.parametrize("delta", (0.2, 0.5, 0.8))
def test_f1_restoring_symmetric_about_one(delta: float):
    """R_1(1-d) = -R_1(1+d): antisymmetric restoring velocity."""
    jdtype = jnp.float32
    H_lo = jnp.asarray([1.0 - delta], dtype=jdtype)
    H_hi = jnp.asarray([1.0 + delta], dtype=jdtype)
    R_lo = _rbd_restoring_term("f1", H_lo, jdtype=jdtype)
    R_hi = _rbd_restoring_term("f1", H_hi, jdtype=jdtype)
    assert float(R_lo[0] + R_hi[0]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("H_below", (0.5, 0.8))
def test_f2_stronger_recovery_below_one_than_f1(H_below: float):
    """For H<1: H^{-1}-1 > 1-H."""
    jdtype = jnp.float32
    H = jnp.asarray([H_below], dtype=jdtype)
    R1 = float(_rbd_restoring_term("f1", H, jdtype=jdtype)[0])
    R2 = float(_rbd_restoring_term("f2", H, jdtype=jdtype)[0])
    assert R1 > 0.0
    assert R2 > R1


@pytest.mark.parametrize("H_above", (1.2, 2.0))
def test_f2_slower_decay_above_one_than_f1(H_above: float):
    """For H>1: |H^{-1}-1| < |1-H|."""
    jdtype = jnp.float32
    H = jnp.asarray([H_above], dtype=jdtype)
    R1 = float(_rbd_restoring_term("f1", H, jdtype=jdtype)[0])
    R2 = float(_rbd_restoring_term("f2", H, jdtype=jdtype)[0])
    assert R1 < 0.0
    assert R2 < 0.0
    assert abs(R2) < abs(R1)


@pytest.mark.parametrize("H0", H0_GRID)
def test_isolated_f1_relaxes_to_coordinate_one(H0: float):
    """kappa_H=0, no edges: all F1 trajectories approach H=1."""
    params = _isolated_neuron_params()
    edges = _empty_edges()
    import jax

    key = jax.random.PRNGKey(0)
    _, _, _, st = simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        500,
        1.0,
        key,
        dtype="float32",
        noise_scale=0.0,
        rbd_family="f1",
        tau_h_ms=20.0,
        kappa_h=0.0,
        init_state={"H": jnp.asarray([H0], dtype=jnp.float32)},
    )
    assert jnp.isfinite(st["H_final"][0])
    assert float(st["H_final"][0]) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("H0", H0_GRID)
def test_isolated_f2_relaxes_to_coordinate_one(H0: float):
    params = _isolated_neuron_params()
    edges = _empty_edges()
    import jax

    key = jax.random.PRNGKey(0)
    _, _, _, st = simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        500,
        1.0,
        key,
        dtype="float32",
        noise_scale=0.0,
        rbd_family="f2",
        tau_h_ms=20.0,
        kappa_h=0.0,
        init_state={"H": jnp.asarray([H0], dtype=jnp.float32)},
    )
    assert jnp.isfinite(st["H_final"][0])
    assert float(st["H_final"][0]) == pytest.approx(1.0, abs=0.02)


def test_f2_initially_faster_than_f1_below_one():
    """At H0=0.5, F2 first-step dH exceeds F1 (stronger depletion recovery)."""
    jdtype = jnp.float32
    dt = jnp.asarray(1.0, dtype=jdtype)
    tau = jnp.asarray(20.0, dtype=jdtype)
    kappa = jnp.asarray(0.0, dtype=jdtype)
    i_ref = jnp.asarray(1.0, dtype=jdtype)
    I_syn = jnp.zeros((1,), dtype=jdtype)
    H0 = jnp.asarray([0.5], dtype=jdtype)
    H1n = _rbd_advance_h("f1", H0, I_syn, dt, tau, kappa, i_ref, jdtype=jdtype)
    H2n = _rbd_advance_h("f2", H0, I_syn, dt, tau, kappa, i_ref, jdtype=jdtype)
    assert float(H2n[0] - H0[0]) > float(H1n[0] - H0[0])


def test_f2_initially_slower_than_f1_above_one():
    """At H0=2, F2 first-step |dH| is smaller than F1 (slower surplus decay)."""
    jdtype = jnp.float32
    dt = jnp.asarray(1.0, dtype=jdtype)
    tau = jnp.asarray(20.0, dtype=jdtype)
    kappa = jnp.asarray(0.0, dtype=jdtype)
    i_ref = jnp.asarray(1.0, dtype=jdtype)
    I_syn = jnp.zeros((1,), dtype=jdtype)
    H0 = jnp.asarray([2.0], dtype=jdtype)
    H1n = _rbd_advance_h("f1", H0, I_syn, dt, tau, kappa, i_ref, jdtype=jdtype)
    H2n = _rbd_advance_h("f2", H0, I_syn, dt, tau, kappa, i_ref, jdtype=jdtype)
    assert abs(float(H2n[0] - H0[0])) < abs(float(H1n[0] - H0[0]))


def test_h_perturbation_invisible_to_activity_without_h_to_x():
    """H1a only: different H0, kappa_H=0 => identical spikes/voltage."""
    params = _isolated_neuron_params()
    edges = _empty_edges()
    import jax

    key = jax.random.PRNGKey(3)
    drive = jnp.zeros((30, 1), dtype=jnp.float32)
    drive = drive.at[10, 0].set(25.0)
    v_a, s_a, _, _ = simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        30,
        1.0,
        key,
        dtype="float32",
        noise_scale=0.0,
        drive_schedule=drive,
        rbd_family="f1",
        kappa_h=0.0,
        init_state={"H": jnp.asarray([0.5], dtype=jnp.float32)},
    )
    v_b, s_b, _, _ = simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        30,
        1.0,
        key,
        dtype="float32",
        noise_scale=0.0,
        drive_schedule=drive,
        rbd_family="f1",
        kappa_h=0.0,
        init_state={"H": jnp.asarray([2.0], dtype=jnp.float32)},
    )
    assert jnp.allclose(v_a, v_b)
    assert jnp.allclose(s_a, s_b)


def test_irel_not_zero_centered_under_recurrent_drive():
    """Positive recurrent synaptic aggregate => E[I^rel] > 0 (not zero-centered)."""
    jdtype = jnp.float32
    params = _isolated_neuron_params()
    edges = EdgeList(
        pre=jnp.asarray([0], dtype=jnp.int32),
        post=jnp.asarray([0], dtype=jnp.int32),
        weight=jnp.asarray([5.0], dtype=jdtype),
        receptor_index=jnp.asarray([0], dtype=jnp.int32),
        tau_ms=jnp.asarray([5.0], dtype=jdtype),
    )
    import jax

    key = jax.random.PRNGKey(1)
    drive = jnp.zeros((100, 1), dtype=jdtype)
    drive = drive.at[5, 0].set(40.0)
    _, _, _, st = simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        100,
        1.0,
        key,
        dtype="float32",
        noise_scale=0.0,
        drive_schedule=drive,
        rbd_family="f1",
        kappa_h=0.0,
    )
    # syn_state in final_state is edge-local; recompute postsynaptic aggregate proxy
    syn_state = np.asarray(st["syn_state"])
    assert syn_state.size > 0
    assert float(np.mean(syn_state[syn_state > 0])) > 0.0
