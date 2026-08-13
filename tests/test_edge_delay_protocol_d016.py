"""Protocol D0/D1 — finite edge-delay dynamics (jaxfne 0.4.16).

D0: tau_ij=0 embeds the legacy instantaneous recurrent kernel.
D1: known grid-aligned delays recover at the presynaptic synaptic-input level.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    edge_delay_steps_from_ms,
    edge_list_with_delay_ms,
    simulate_edge_recurrent_izhikevich,
)


def _two_neuron_pulse_circuit(
  *,
  delay_steps: int | np.ndarray = 0,
  dt_ms: float = 1.0,
  pulse_step: int = 5,
  n_steps: int = 40,
) -> tuple[IzhikevichParams, EdgeList, jax.Array, float]:
    """Minimal causal chain: neuron 0 -> neuron 1, pulse drive on 0 only."""
    jdtype = jnp.float32
    n = 2
    params = IzhikevichParams(
        v0=jnp.asarray([-65.0, -65.0], dtype=jdtype),
        u0=jnp.zeros((n,), dtype=jdtype),
        a=jnp.full((n,), 0.02, dtype=jdtype),
        b=jnp.full((n,), 0.2, dtype=jdtype),
        c=jnp.full((n,), -65.0, dtype=jdtype),
        d=jnp.full((n,), 8.0, dtype=jdtype),
        drive=jnp.zeros((n,), dtype=jdtype),
        sign=jnp.ones((n,), dtype=jdtype),
        W=jnp.zeros((n, n), dtype=jdtype),
        source_scale=jnp.ones((n,), dtype=jdtype),
        labels=("E", "E"),
        layer_labels=("L4", "L4"),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    edges = EdgeList(
        pre=jnp.asarray([0], dtype=jnp.int32),
        post=jnp.asarray([1], dtype=jnp.int32),
        weight=jnp.asarray([8.0], dtype=jdtype),
        receptor_index=jnp.asarray([0], dtype=jnp.int32),
        tau_ms=jnp.asarray([2.0], dtype=jdtype),
    )
    if isinstance(delay_steps, int):
        delay_steps_arr = jnp.asarray([delay_steps], dtype=jnp.int32)
    else:
        delay_steps_arr = jnp.asarray(delay_steps, dtype=jnp.int32)
    edges = EdgeList(
        pre=edges.pre,
        post=edges.post,
        weight=edges.weight,
        receptor_index=edges.receptor_index,
        tau_ms=edges.tau_ms,
        delay_steps=delay_steps_arr,
    )
    drive = jnp.zeros((n_steps, n), dtype=jdtype)
    drive = drive.at[pulse_step, 0].set(40.0)
    return params, edges, drive, dt_ms


def _run(
    params: IzhikevichParams,
    edges: EdgeList,
    *,
    n_steps: int,
    dt_ms: float,
    drive_schedule: jax.Array,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, dict]:
    key = jax.random.PRNGKey(seed)
    return simulate_edge_recurrent_izhikevich(
        params,
        edges,
        n_steps,
        dt_ms,
        key,
        dtype="float32",
        drive_schedule=drive_schedule,
        noise_scale=0.0,
    )


def test_d0_no_delay_spec_matches_legacy():
    """No delay_steps field (defaults to zero) matches legacy kernel."""
    params, edges_default, drive, dt_ms = _two_neuron_pulse_circuit()
    edges_explicit = EdgeList(
        pre=edges_default.pre,
        post=edges_default.post,
        weight=edges_default.weight,
        receptor_index=edges_default.receptor_index,
        tau_ms=edges_default.tau_ms,
        delay_steps=jnp.asarray([0], dtype=jnp.int32),
    )
    n_steps = drive.shape[0]
    v0, s0, q0, st0 = _run(params, edges_default, n_steps=n_steps, dt_ms=dt_ms, drive_schedule=drive)
    v1, s1, q1, st1 = _run(params, edges_explicit, n_steps=n_steps, dt_ms=dt_ms, drive_schedule=drive)
    assert float(jnp.max(jnp.abs(v0 - v1))) == 0.0
    assert float(jnp.max(jnp.abs(s0 - s1))) == 0.0
    assert float(jnp.max(jnp.abs(q0 - q1))) == 0.0
    assert float(jnp.max(jnp.abs(st0["syn_state"] - st1["syn_state"]))) == 0.0


def test_d0_explicit_zero_delay_ms_embedding():
    """edge_list_with_delay_ms(..., 0) matches legacy."""
    params, edges, drive, dt_ms = _two_neuron_pulse_circuit()
    n_steps = drive.shape[0]
    edges_zero_ms = edge_list_with_delay_ms(edges, 0.0, dt_ms)
    v0, s0, q0, st0 = _run(params, edges, n_steps=n_steps, dt_ms=dt_ms, drive_schedule=drive)
    v1, s1, q1, st1 = _run(params, edges_zero_ms, n_steps=n_steps, dt_ms=dt_ms, drive_schedule=drive)
    assert float(jnp.max(jnp.abs(v0 - v1))) == 0.0
    assert float(jnp.max(jnp.abs(s0 - s1))) == 0.0
    assert float(jnp.max(jnp.abs(q0 - q1))) == 0.0


def test_d1_known_delay_recovery_at_presynaptic_input():
    """Presynaptic synaptic event arrives n steps after presynaptic spike."""
    dt_ms = 1.0
    pulse_step = 5
    for delay_steps in (1, 3, 7):
        params, edges, drive, dt_ms = _two_neuron_pulse_circuit(
            delay_steps=delay_steps, dt_ms=dt_ms, pulse_step=pulse_step
        )
        n_steps = drive.shape[0]
        _, spikes, _, final = _run(
            params, edges, n_steps=n_steps, dt_ms=dt_ms, drive_schedule=drive
        )
        presyn = np.asarray(final["presynaptic_drive_trace"][:, 0])
        spike_train = np.asarray(spikes[:, 0])
        spike_idx = int(np.argmax(spike_train > 0.5))
        assert spike_train[spike_idx] > 0.5
        arrival = int(np.argmax(presyn))
        assert arrival == spike_idx + delay_steps, (
            f"delay_steps={delay_steps}: presyn arrival {arrival}, "
            f"expected {spike_idx + delay_steps} (spike at {spike_idx})"
        )
        assert presyn[arrival] == pytest.approx(1.0)
        assert float(jnp.sum(presyn)) == pytest.approx(1.0)


def test_d1_heterogeneous_edge_delays_route_correctly():
    """Two edges from the same presynaptic neuron with distinct delays."""
    dt_ms = 1.0
    pulse_step = 4
    n_steps = 30
    dtype = jnp.float32
    params = IzhikevichParams(
        v0=jnp.asarray([-65.0, -65.0, -65.0], dtype=dtype),
        u0=jnp.zeros((3,), dtype=dtype),
        a=jnp.full((3,), 0.02, dtype=dtype),
        b=jnp.full((3,), 0.2, dtype=dtype),
        c=jnp.full((3,), -65.0, dtype=dtype),
        d=jnp.full((3,), 8.0, dtype=dtype),
        drive=jnp.zeros((3,), dtype=dtype),
        sign=jnp.ones((3,), dtype=dtype),
        W=jnp.zeros((3, 3), dtype=dtype),
        source_scale=jnp.ones((3,), dtype=dtype),
        labels=("E", "E", "E"),
        layer_labels=("L4", "L4", "L4"),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    edges = EdgeList(
        pre=jnp.asarray([0, 0], dtype=jnp.int32),
        post=jnp.asarray([1, 2], dtype=jnp.int32),
        weight=jnp.asarray([5.0, 5.0], dtype=dtype),
        receptor_index=jnp.asarray([0, 0], dtype=jnp.int32),
        tau_ms=jnp.asarray([2.0, 2.0], dtype=dtype),
        delay_steps=jnp.asarray([2, 6], dtype=jnp.int32),
    )
    drive = jnp.zeros((n_steps, 3), dtype=dtype)
    drive = drive.at[pulse_step, 0].set(40.0)
    _, spikes, _, final = _run(params, edges, n_steps=n_steps, dt_ms=dt_ms, drive_schedule=drive)
    spike_train = np.asarray(spikes[:, 0])
    spike_idx = int(np.argmax(spike_train > 0.5))
    assert spike_train[spike_idx] > 0.5
    presyn = np.asarray(final["presynaptic_drive_trace"])
    assert int(np.argmax(presyn[:, 0])) == spike_idx + 2
    assert int(np.argmax(presyn[:, 1])) == spike_idx + 6


def test_delay_validation_rejects_negative_and_fractional():
    with pytest.raises(ValueError, match=">= 0"):
        edge_delay_steps_from_ms(-0.5, 1.0)
    with pytest.raises(ValueError, match="grid-aligned"):
        edge_delay_steps_from_ms(1.5, 1.0)
    params, edges, drive, dt_ms = _two_neuron_pulse_circuit()
    with pytest.raises(ValueError, match=">= 0"):
        bad = EdgeList(
            pre=edges.pre,
            post=edges.post,
            weight=edges.weight,
            receptor_index=edges.receptor_index,
            tau_ms=edges.tau_ms,
            delay_steps=jnp.asarray([-1], dtype=jnp.int32),
        )
        _run(params, bad, n_steps=drive.shape[0], dt_ms=dt_ms, drive_schedule=drive)


def test_nonzero_delay_rejects_init_state_continuation():
    params, edges, drive, dt_ms = _two_neuron_pulse_circuit(delay_steps=2)
    with pytest.raises(ValueError, match="continuation is not supported"):
        simulate_edge_recurrent_izhikevich(
            params,
            edges,
            drive.shape[0],
            dt_ms,
            jax.random.PRNGKey(0),
            drive_schedule=drive,
            noise_scale=0.0,
            init_state={"v": params.v0, "u": params.u0, "prev_spikes": jnp.zeros(2), "syn_state": jnp.zeros(1)},
        )


def test_finite_trajectories_under_delay():
    params, edges, drive, dt_ms = _two_neuron_pulse_circuit(delay_steps=4)
    v, spikes, sources, final = _run(
        params, edges, n_steps=drive.shape[0], dt_ms=dt_ms, drive_schedule=drive
    )
    for arr in (v, spikes, sources, final["syn_state"], final["spike_history"]):
        assert jnp.all(jnp.isfinite(arr))


def test_delay_does_not_change_source_field_contract_except_via_x():
    """Sources follow X; field projection remains downstream observation."""
    dt_ms = 0.5
    n_steps = 20
    cfg = (
        jtfne.configuration()
        .network(n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy")
        .probe(name="lp", n_contacts=8)
    )
    model = jtfne.construct(cfg)
    edges = model.params["edge_list"]
    delay_ms = jnp.zeros((edges.n_edges,), dtype=jnp.float32)
    if edges.n_edges > 0:
        delay_ms = delay_ms.at[0].set(2.0 * dt_ms)
        delayed_edges = edge_list_with_delay_ms(edges, delay_ms, dt_ms)
        assert int(np.max(np.asarray(delayed_edges.delay_steps))) == 2
    sim = jtfne.simulation(duration_ms=n_steps * dt_ms, dt_ms=dt_ms, seed=0)
    sig_legacy = model.simulate(sim)
    assert sig_legacy.field is not None
    assert jnp.all(jnp.isfinite(sig_legacy.field.lfp_proxy))


def test_delayed_kernel_jit_compiles():
    params, edges, drive, dt_ms = _two_neuron_pulse_circuit(delay_steps=3)
    jit_fn = jax.jit(
        lambda key: simulate_edge_recurrent_izhikevich(
            params,
            edges,
            drive.shape[0],
            dt_ms,
            key,
            drive_schedule=drive,
            noise_scale=0.0,
        )
    )
    v, spikes, sources, _ = jit_fn(jax.random.PRNGKey(1))
    assert jnp.all(jnp.isfinite(v))
    assert jnp.all(jnp.isfinite(spikes))
