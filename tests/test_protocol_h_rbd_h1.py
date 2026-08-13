"""Protocol H1 — RBD kernel with fixed weights (d_H=1, F0/F1/F2, Protocol D delays).

Tests: analytic F1 relaxation, equilibrium, matched Jacobian at H=1, fixed-W
invariance, deterministic repeatability, finiteness, D0 zero-delay parity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    _rbd_advance_h,
    _rbd_restoring_term,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_rbd,
)


def _isolated_neuron_params(
    *,
    n: int = 1,
    v0: float = -65.0,
) -> IzhikevichParams:
    jdtype = jnp.float32
    return IzhikevichParams(
        v0=jnp.full((n,), v0, dtype=jdtype),
        u0=jnp.zeros((n,), dtype=jdtype),
        a=jnp.full((n,), 0.02, dtype=jdtype),
        b=jnp.full((n,), 0.2, dtype=jdtype),
        c=jnp.full((n,), -65.0, dtype=jdtype),
        d=jnp.full((n,), 8.0, dtype=jdtype),
        drive=jnp.zeros((n,), dtype=jdtype),
        sign=jnp.ones((n,), dtype=jdtype),
        W=jnp.zeros((n, n), dtype=jdtype),
        source_scale=jnp.ones((n,), dtype=jdtype),
        labels=tuple("E" for _ in range(n)),
        layer_labels=tuple("L4" for _ in range(n)),
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


def _two_neuron_ring(*, delay_steps: int = 0) -> tuple[IzhikevichParams, EdgeList]:
    jdtype = jnp.float32
    n = 2
    params = _isolated_neuron_params(n=n)
    edges = EdgeList(
        pre=jnp.asarray([0, 1], dtype=jnp.int32),
        post=jnp.asarray([1, 0], dtype=jnp.int32),
        weight=jnp.asarray([5.0, 5.0], dtype=jdtype),
        receptor_index=jnp.asarray([0, 0], dtype=jnp.int32),
        tau_ms=jnp.asarray([2.0, 2.0], dtype=jdtype),
        delay_steps=jnp.asarray([delay_steps, delay_steps], dtype=jnp.int32),
    )
    return params, edges


def _run_rbd(
    params: IzhikevichParams,
    edges: EdgeList,
    *,
    n_steps: int,
    dt_ms: float = 1.0,
    seed: int = 0,
    drive_schedule: jax.Array | None = None,
    init_state: dict | None = None,
    **rbd_kw,
):
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


def test_f1_analytic_relaxation_without_input():
    """H_i(t)-1 = (H_i(0)-1) exp(-t/tau_H) for F1 with kappa_H=0."""
    params = _isolated_neuron_params()
    edges = _empty_edges()
    dt_ms = 1.0
    tau_h = 40.0
    n_steps = 200
    delta = 0.25
    H0 = jnp.asarray([1.0 + delta], dtype=jnp.float32)
    _, _, _, st = _run_rbd(
        params,
        edges,
        n_steps=n_steps,
        dt_ms=dt_ms,
        rbd_family="f1",
        tau_h_ms=tau_h,
        kappa_h=0.0,
        init_state={"H": H0},
    )
    steps = jnp.arange(n_steps, dtype=jnp.float32) + 1.0
    decay = 1.0 - dt_ms / tau_h
    expected = 1.0 + delta * jnp.power(decay, steps)
    got = st["H_trace"][:, 0]
    assert jnp.allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_f1_f2_equilibrium_at_h_one():
    """R_1(1)=R_2(1)=0 with zero relative input."""
    jdtype = jnp.float32
    H = jnp.asarray([1.0], dtype=jdtype)
    for fam in ("f1", "f2"):
        R = _rbd_restoring_term(fam, H, jdtype=jdtype)
        assert float(R[0]) == 0.0


def test_f1_f2_matched_jacobian_at_equilibrium():
    """R_1'(1)=R_2'(1)=-1 => same first-order relaxation rate 1/tau_H."""
    eps = 1e-4
    H = jnp.asarray([1.0], dtype=jnp.float32)
    dt = jnp.asarray(1.0, dtype=jnp.float32)
    tau = jnp.asarray(100.0, dtype=jnp.float32)
    kappa = jnp.asarray(0.0, dtype=jnp.float32)
    i_ref = jnp.asarray(1.0, dtype=jnp.float32)
    I_syn = jnp.zeros((1,), dtype=jnp.float32)

    def dh_dt(family: str, h_val: float) -> float:
        H_loc = jnp.asarray([h_val], dtype=jnp.float32)
        Hn = _rbd_advance_h(
            family, H_loc, I_syn, dt, tau, kappa, i_ref, jdtype=jnp.float32
        )
        return float((Hn[0] - H_loc[0]) / dt)

    for fam in ("f1", "f2"):
        d1 = (dh_dt(fam, 1.0 + eps) - dh_dt(fam, 1.0 - eps)) / (2 * eps)
        assert abs(d1 - (-1.0 / 100.0)) < 1e-3


def test_f0_rbs_disabled_holds_at_one():
    params = _isolated_neuron_params()
    edges = _empty_edges()
    H0 = jnp.asarray([1.4], dtype=jnp.float32)
    _, _, _, st = _run_rbd(
        params,
        edges,
        n_steps=50,
        rbd_family="f0",
        init_state={"H": H0},
    )
    assert jnp.allclose(st["H_trace"], 1.0)
    assert jnp.allclose(st["H_final"], 1.0)


def test_f2_rejects_nonpositive_initial_h():
    params = _isolated_neuron_params()
    edges = _empty_edges()
    with pytest.raises(ValueError, match="F2 requires H>0"):
        _run_rbd(
            params,
            edges,
            n_steps=5,
            rbd_family="f2",
            init_state={"H": jnp.asarray([-0.1], dtype=jnp.float32)},
        )


def test_f2_invalidates_on_h_crossing_zero():
    """Euler overshoot below H=0 propagates nan (no clip)."""
    params = _isolated_neuron_params()
    edges = _empty_edges()
    _, _, _, st = _run_rbd(
        params,
        edges,
        n_steps=5,
        dt_ms=10.0,
        rbd_family="f2",
        tau_h_ms=1.0,
        kappa_h=0.0,
        init_state={"H": jnp.asarray([3.0], dtype=jnp.float32)},
    )
    assert not jnp.all(jnp.isfinite(st["H_trace"]))


def test_fixed_w_invariance():
    params, edges = _two_neuron_ring()
    w_before = np.asarray(edges.weight)
    _, _, _, st_a = _run_rbd(
        params,
        edges,
        n_steps=30,
        rbd_family="f1",
        init_state={"H": jnp.asarray([1.0, 1.2], dtype=jnp.float32)},
    )
    _, _, _, st_b = _run_rbd(
        params,
        edges,
        n_steps=30,
        rbd_family="f2",
        init_state={"H": jnp.asarray([1.0, 0.9], dtype=jnp.float32)},
    )
    assert np.array_equal(np.asarray(st_a["w_fixed"]), w_before)
    assert np.array_equal(np.asarray(st_b["w_fixed"]), w_before)
    assert np.array_equal(np.asarray(st_a["w_fixed"]), np.asarray(st_b["w_fixed"]))


def test_deterministic_repeatability():
    params, edges = _two_neuron_ring()
    out_a = _run_rbd(params, edges, n_steps=40, seed=7, rbd_family="f1")
    out_b = _run_rbd(params, edges, n_steps=40, seed=7, rbd_family="f1")
    for a, b in zip(out_a, out_b[:3]):
        assert jnp.allclose(a, b)
    for key in ("H_trace", "H_final", "syn_state"):
        assert jnp.allclose(out_a[3][key], out_b[3][key])


def test_finiteness_f0_f1_f2_zero_delay():
    params, edges = _two_neuron_ring(delay_steps=0)
    for fam in ("f0", "f1", "f2"):
        v, s, q, st = _run_rbd(
            params,
            edges,
            n_steps=60,
            rbd_family=fam,
            init_state={"H": jnp.asarray([1.0, 1.05], dtype=jnp.float32)},
        )
        assert jnp.all(jnp.isfinite(v))
        assert jnp.all(jnp.isfinite(s))
        assert jnp.all(jnp.isfinite(q))
        if fam != "f2":
            assert jnp.all(jnp.isfinite(st["H_trace"]))


def test_d0_f0_parity_with_legacy_kernel():
    """F0 + zero delay: activity matches simulate_edge_recurrent_izhikevich."""
    jdtype = jnp.float32
    params, edges = _two_neuron_ring(delay_steps=0)
    n_steps = 40
    drive = jnp.zeros((n_steps, 2), dtype=jdtype)
    drive = drive.at[10, 0].set(35.0)
    key = jax.random.PRNGKey(0)
    v0, s0, q0, _ = simulate_edge_recurrent_izhikevich(
        params,
        edges,
        n_steps,
        1.0,
        key,
        dtype="float32",
        drive_schedule=drive,
        noise_scale=0.0,
    )
    v1, s1, q1, st1 = _run_rbd(
        params,
        edges,
        n_steps=n_steps,
        drive_schedule=drive,
        rbd_family="f0",
    )
    assert float(jnp.max(jnp.abs(v0 - v1))) == 0.0
    assert float(jnp.max(jnp.abs(s0 - s1))) == 0.0
    assert float(jnp.max(jnp.abs(q0 - q1))) == 0.0
    assert jnp.allclose(st1["H_trace"], 1.0)


def test_activity_unchanged_across_families_when_kappa_zero():
    """H1 does not feed H back into F_x; v/s/q identical across F0/F1/F2."""
    params, edges = _two_neuron_ring()
    n_steps = 50
    drive = jnp.zeros((n_steps, 2), dtype=jnp.float32)
    drive = drive.at[8, 0].set(30.0)
    outs = []
    for fam in ("f0", "f1", "f2"):
        outs.append(
            _run_rbd(
                params,
                edges,
                n_steps=n_steps,
                drive_schedule=drive,
                rbd_family=fam,
                init_state={"H": jnp.asarray([1.1, 0.95], dtype=jnp.float32)},
            )
        )
    for i in range(1, 3):
        assert jnp.allclose(outs[0][0], outs[i][0])
        assert jnp.allclose(outs[0][1], outs[i][1])
        assert jnp.allclose(outs[0][2], outs[i][2])


def test_delayed_rbd_finite():
    params, edges = _two_neuron_ring(delay_steps=3)
    v, s, q, st = _run_rbd(params, edges, n_steps=80, rbd_family="f1")
    assert jnp.all(jnp.isfinite(v))
    assert jnp.all(jnp.isfinite(s))
    assert jnp.all(jnp.isfinite(q))
    assert jnp.all(jnp.isfinite(st["H_trace"]))
    assert "spike_history" in st


def test_delayed_rbd_rejects_init_state():
    params, edges = _two_neuron_ring(delay_steps=2)
    with pytest.raises(ValueError, match="delay history"):
        _run_rbd(
            params,
            edges,
            n_steps=10,
            rbd_family="f1",
            init_state={"v": params.v0, "u": params.u0, "prev_spikes": jnp.zeros(2), "syn_state": jnp.zeros(2)},
        )


def test_jit_compatible():
    params, edges = _two_neuron_ring()

    @jax.jit
    def run(key):
        v, s, q, st = simulate_edge_recurrent_izhikevich_rbd(
            params,
            edges,
            20,
            1.0,
            key,
            dtype="float32",
            noise_scale=0.0,
            rbd_family="f1",
        )
        return v, s, q, st["H_trace"]

    key = jax.random.PRNGKey(1)
    v, s, q, H_trace = run(key)
    assert v.shape == (20, 2)
    assert H_trace.shape == (20, 2)
