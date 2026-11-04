"""D34: H reaches the recurrent drive only through the typed coupling map G_H."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne import emitters
from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    simulate_edge_recurrent_izhikevich_rbd,
)

pytestmark = pytest.mark.x64


def _two_neuron_ring():
    jdtype = jnp.float64
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


def _run_rbd(params, edges, *, n_steps, drive_schedule=None, init_state=None, **rbd_kw):
    key = jax.random.PRNGKey(0)
    return simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        n_steps,
        1.0,
        key,
        dtype="float64",
        noise_scale=0.0,
        drive_schedule=drive_schedule,
        init_state=init_state,
        **rbd_kw,
    )


def _check_neutral_reduction():
    """beta_h=0 with moving H reduces to the H-free (f0) run (public sim path)."""
    params, edges = _two_neuron_ring()
    n_steps = 40
    drive = jnp.zeros((n_steps, 2), dtype=jnp.float64)
    for t in range(10, 15):
        drive = drive.at[t, 0].set(35.0)
    h0 = jnp.asarray([1.2, 0.9], dtype=jnp.float64)
    kw = dict(
        n_steps=n_steps,
        drive_schedule=drive,
        init_state={"H": h0},
        tau_h_ms=100.0,
        kappa_h=0.5,
        i_ref=1.0,
    )
    v_f1, s_f1, _, st_f1 = _run_rbd(params, edges, rbd_family="f1", beta_h=0.0, **kw)
    v_f0, s_f0, _, _ = _run_rbd(params, edges, rbd_family="f0", beta_h=0.0, **kw)
    h_trace = np.asarray(st_f1["H_trace"])
    assert float(np.max(np.abs(h_trace - h_trace[0]))) > 0.0, "fixture must move H"
    np.testing.assert_allclose(np.asarray(v_f1), np.asarray(v_f0), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(s_f1), np.asarray(s_f0), rtol=1e-12, atol=0.0)


def _check_map_at_composition_site():
    """With beta_h != 0 the drive equals I_ext + G_H(H) * I_rec + noise."""
    jdtype = jnp.float64
    H = jnp.asarray([0.7, 1.4], dtype=jdtype)
    beta = jnp.asarray(0.5, dtype=jdtype)
    I_ext = jnp.asarray([2.0, -1.0], dtype=jdtype)
    I_rec = jnp.asarray([3.0, 4.0], dtype=jdtype)
    noise = jnp.asarray([0.1, -0.2], dtype=jdtype)
    got = emitters._rbd_compose_native_current(I_ext, I_rec, H, "f1", beta, noise, jdtype=jdtype)
    expected = (
        I_ext + emitters.rbd_recurrent_gain("f1", H, beta, jdtype=jdtype) * I_rec + noise
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-12, atol=0.0)


def test_neutral_reduction_public_path():
    _check_neutral_reduction()


def test_coupling_through_map():
    _check_map_at_composition_site()


def _bypass_compose(I_ext, I_rec, H, family, beta_h, noise, *, jdtype):
    return I_ext + H * I_rec + noise


def _drop_compose(I_ext, I_rec, H, family, beta_h, noise, *, jdtype):
    return I_ext + I_rec + noise


def test_mutant_bypass_breaks_typed_coupling(monkeypatch):
    monkeypatch.setattr(emitters, "_rbd_compose_native_current", _bypass_compose)
    with pytest.raises(AssertionError):
        _check_neutral_reduction()


def test_mutant_drop_breaks_typed_coupling(monkeypatch):
    monkeypatch.setattr(emitters, "_rbd_compose_native_current", _drop_compose)
    with pytest.raises(AssertionError):
        _check_map_at_composition_site()
