"""Protocol H2 — complete RBD + delay_state continuation.

Pre-registered tolerance: bit-exact (float32, max abs diff == 0) with
noise_scale=0 and deterministic drive schedules.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters import EdgeList, IzhikevichParams, simulate_edge_recurrent_izhikevich_rbd

BETA_TEST = 0.5
FAMILIES = ("f0", "f1", "f2")
BETA_VALUES = (0.0, BETA_TEST)


def _params_edges_ring(*, delay_steps: int | list[int] = 0) -> tuple[IzhikevichParams, EdgeList]:
    jdtype = jnp.float32
    n = 3
    params = IzhikevichParams(
        v0=jnp.full((n,), -65.0, dtype=jdtype),
        u0=jnp.zeros((n,), dtype=jdtype),
        a=jnp.full((n,), 0.02, dtype=jdtype),
        b=jnp.full((n,), 0.2, dtype=jdtype),
        c=jnp.full((n,), -65.0, dtype=jdtype),
        d=jnp.full((n,), 8.0, dtype=jdtype),
        drive=jnp.zeros((n,), dtype=jdtype),
        sign=jnp.ones((n,), dtype=jdtype),
        W=jnp.zeros((n, n), dtype=jdtype),
        source_scale=jnp.ones((n,), dtype=jdtype),
        labels=("E", "E", "E"),
        layer_labels=("L4", "L4", "L4"),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    if isinstance(delay_steps, int):
        ds = jnp.asarray([delay_steps, delay_steps, delay_steps], dtype=jnp.int32)
    else:
        ds = jnp.asarray(delay_steps, dtype=jnp.int32)
    edges = EdgeList(
        pre=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        post=jnp.asarray([1, 2, 0], dtype=jnp.int32),
        weight=jnp.asarray([6.0, 6.0, 6.0], dtype=jdtype),
        receptor_index=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        tau_ms=jnp.asarray([3.0, 3.0, 3.0], dtype=jdtype),
        delay_steps=ds,
    )
    return params, edges


def _drive_schedule(n_steps: int, n_neurons: int) -> jax.Array:
    drive = jnp.zeros((n_steps, n_neurons), dtype=jnp.float32)
    drive = drive.at[7, 0].set(42.0)
    drive = drive.at[22, 1].set(38.0)
    return drive


def _run(
    params: IzhikevichParams,
    edges: EdgeList,
    *,
    n_steps: int,
    drive: jax.Array,
    seed: int = 0,
    init_state: dict | None = None,
    **rbd_kw,
):
    key = jax.random.PRNGKey(seed)
    return simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        n_steps,
        1.0,
        key,
        dtype="float32",
        noise_scale=0.0,
        drive_schedule=drive,
        init_state=init_state,
        **rbd_kw,
    )


def _concat_outputs(a, b):
    v0, s0, q0, st0 = a
    v1, s1, q1, st1 = b
    return (
        jnp.concatenate([v0, v1], axis=0),
        jnp.concatenate([s0, s1], axis=0),
        jnp.concatenate([q0, q1], axis=0),
        st1,
    )


def _assert_bit_exact_states(st_full: dict, st_split: dict, *, delayed: bool):
    for key in ("v", "u", "prev_spikes", "syn_state", "H_final", "w_fixed"):
        assert float(jnp.max(jnp.abs(st_full[key] - st_split[key]))) == 0.0
    if delayed:
        assert float(jnp.max(jnp.abs(st_full["delay_state"] - st_split["delay_state"]))) == 0.0
        assert "delay_state" in st_full
        assert int(st_full["continuation_step_offset"]) == int(st_split["continuation_step_offset"])


@pytest.mark.parametrize("rbd_family", FAMILIES)
@pytest.mark.parametrize("beta_h", BETA_VALUES)
def test_h2_zero_delay_segmented_bit_exact(rbd_family: str, beta_h: float):
    params, edges = _params_edges_ring(delay_steps=0)
    t1, t2 = 30, 25
    drive = _drive_schedule(t1 + t2, 3)
    h0 = jnp.asarray([1.0, 1.15, 0.92], dtype=jnp.float32)
    base_kw = dict(rbd_family=rbd_family, beta_h=beta_h)
    v_a, s_a, q_a, st_a = _run(
        params, edges, n_steps=t1 + t2, drive=drive, init_state={"H": h0}, **base_kw
    )
    v1, s1, q1, st1 = _run(
        params, edges, n_steps=t1, drive=drive[:t1], init_state={"H": h0}, **base_kw
    )
    cont = dict(st1)
    cont["H"] = st1["H_final"]
    v2, s2, q2, st2 = _run(
        params, edges, n_steps=t2, drive=drive[t1:], init_state=cont, **base_kw
    )
    v_b, s_b, q_b, st_b = _concat_outputs((v1, s1, q1, st1), (v2, s2, q2, st2))
    assert float(jnp.max(jnp.abs(v_a - v_b))) == 0.0
    assert float(jnp.max(jnp.abs(s_a - s_b))) == 0.0
    assert float(jnp.max(jnp.abs(q_a - q_b))) == 0.0
    _assert_bit_exact_states(st_a, st_b, delayed=False)


@pytest.mark.parametrize("rbd_family", FAMILIES)
@pytest.mark.parametrize("beta_h", BETA_VALUES)
def test_h2_uniform_delay_segmented_bit_exact(rbd_family: str, beta_h: float):
    params, edges = _params_edges_ring(delay_steps=4)
    t1, t2 = 35, 30
    drive = _drive_schedule(t1 + t2, 3)
    h0 = jnp.asarray([1.0, 1.1, 0.95], dtype=jnp.float32)
    base_kw = dict(rbd_family=rbd_family, beta_h=beta_h)
    v_a, s_a, q_a, st_a = _run(
        params, edges, n_steps=t1 + t2, drive=drive, init_state={"H": h0}, **base_kw
    )
    v1, s1, q1, st1 = _run(
        params, edges, n_steps=t1, drive=drive[:t1], init_state={"H": h0}, **base_kw
    )
    cont = dict(st1)
    cont["H"] = st1["H_final"]
    v2, s2, q2, st2 = _run(
        params, edges, n_steps=t2, drive=drive[t1:], init_state=cont, **base_kw
    )
    v_b, s_b, q_b, st_b = _concat_outputs((v1, s1, q1, st1), (v2, s2, q2, st2))
    assert float(jnp.max(jnp.abs(v_a - v_b))) == 0.0
    assert float(jnp.max(jnp.abs(s_a - s_b))) == 0.0
    assert float(jnp.max(jnp.abs(q_a - q_b))) == 0.0
    _assert_bit_exact_states(st_a, st_b, delayed=True)


@pytest.mark.parametrize("rbd_family", FAMILIES)
@pytest.mark.parametrize("beta_h", BETA_VALUES)
def test_h2_heterogeneous_delay_segmented_bit_exact(rbd_family: str, beta_h: float):
    params, edges = _params_edges_ring(delay_steps=[0, 3, 6])
    t1, t2 = 40, 35
    drive = _drive_schedule(t1 + t2, 3)
    h0 = jnp.asarray([1.05, 1.0, 1.08], dtype=jnp.float32)
    base_kw = dict(rbd_family=rbd_family, beta_h=beta_h)
    v_a, s_a, q_a, st_a = _run(
        params, edges, n_steps=t1 + t2, drive=drive, init_state={"H": h0}, **base_kw
    )
    v1, s1, q1, st1 = _run(
        params, edges, n_steps=t1, drive=drive[:t1], init_state={"H": h0}, **base_kw
    )
    cont = dict(st1)
    cont["H"] = st1["H_final"]
    v2, s2, q2, st2 = _run(
        params, edges, n_steps=t2, drive=drive[t1:], init_state=cont, **base_kw
    )
    v_b, s_b, q_b, st_b = _concat_outputs((v1, s1, q1, st1), (v2, s2, q2, st2))
    assert float(jnp.max(jnp.abs(v_a - v_b))) == 0.0
    assert float(jnp.max(jnp.abs(s_a - s_b))) == 0.0
    assert float(jnp.max(jnp.abs(q_a - q_b))) == 0.0
    _assert_bit_exact_states(st_a, st_b, delayed=True)


def test_h2_memory_preservation_h_perturbation_with_delay():
    """Localized H_k=1+delta survives split continuation (X, H, B, W)."""
    params, edges = _params_edges_ring(delay_steps=[0, 4, 8])
    t1, t2 = 45, 40
    total = t1 + t2
    drive = _drive_schedule(total, 3)
    delta = 0.18
    h0 = jnp.asarray([1.0, 1.0 + delta, 1.0], dtype=jnp.float32)
    base_kw = dict(rbd_family="f1", beta_h=BETA_TEST)
    v_a, s_a, _, st_a = _run(
        params, edges, n_steps=total, drive=drive, init_state={"H": h0}, **base_kw
    )
    v1, s1, _, st1 = _run(
        params, edges, n_steps=t1, drive=drive[:t1], init_state={"H": h0}, **base_kw
    )
    cont = dict(st1)
    cont["H"] = st1["H_final"]
    v2, s2, _, st2 = _run(
        params, edges, n_steps=t2, drive=drive[t1:], init_state=cont, **base_kw
    )
    v_b = jnp.concatenate([v1, v2], axis=0)
    s_b = jnp.concatenate([s1, s2], axis=0)
    assert float(jnp.max(jnp.abs(v_a - v_b))) == 0.0
    assert float(jnp.max(jnp.abs(s_a - s_b))) == 0.0
    _assert_bit_exact_states(st_a, st2, delayed=True)
    w0 = np.asarray(edges.weight)
    assert np.array_equal(np.asarray(st_a["w_fixed"]), w0)
    assert np.array_equal(np.asarray(st2["w_fixed"]), w0)
    assert not jnp.allclose(st_a["H_final"], jnp.ones(3))


def test_h2_rejects_delayed_continuation_without_delay_state():
    params, edges = _params_edges_ring(delay_steps=3)
    drive = _drive_schedule(10, 3)
    v1, _, _, st1 = _run(params, edges, n_steps=5, drive=drive[:5])
    bad = {
        "v": st1["v"],
        "u": st1["u"],
        "prev_spikes": st1["prev_spikes"],
        "syn_state": st1["syn_state"],
        "H": st1["H_final"],
    }
    with pytest.raises(ValueError, match="delay_state"):
        _run(params, edges, n_steps=5, drive=drive[5:10], init_state=bad)
