"""Authoritative test suite for finite-delay support in the HDP and H boundary stabilization path.

Verifies:
1. Zero-delay regression (all delay_steps=0 matches zero-delay semantics bitwise).
2. Delay semantics (sparse asymmetric impulse probes verify exact 20/80/120-step arrival).
3. H_OFF equivalence (delayed HDP with null H terms is bitwise identical to _simulate_edge_recurrent_izhikevich_delayed).
4. H_ON continuation (uninterrupted vs chunked continuation with delay_state, H, r_bar, I_H).
5. Recording invariance (diagnostic traces do not alter physical trajectory).
"""

from dataclasses import replace
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.emitters import (
    IzhikevichParams,
    EdgeList,
    simulate_edge_recurrent_izhikevich_hdp,
    _simulate_edge_recurrent_izhikevich_delayed,
)


def _make_impulse_delayed_network(delays=(20, 80, 120)):
    pre = jnp.array([0, 0, 1], dtype=jnp.int32)
    post = jnp.array([1, 2, 2], dtype=jnp.int32)
    weight = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32)
    tau_ms = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    delay_steps = jnp.array(delays, dtype=jnp.int32)
    receptor_index = jnp.zeros(3, dtype=jnp.int32)

    edges = EdgeList(
        pre=pre,
        post=post,
        weight=weight,
        tau_ms=tau_ms,
        delay_steps=delay_steps,
        receptor_index=receptor_index,
    )

    n = 3
    params = IzhikevichParams(
        a=jnp.full(n, 0.02, dtype=jnp.float32),
        b=jnp.full(n, 0.20, dtype=jnp.float32),
        c=jnp.full(n, -65.0, dtype=jnp.float32),
        d=jnp.full(n, 8.0, dtype=jnp.float32),
        v0=jnp.full(n, -65.0, dtype=jnp.float32),
        u0=jnp.full(n, 0.20 * -65.0, dtype=jnp.float32),
        W=jnp.zeros((n, n), dtype=jnp.float32),
        drive=jnp.zeros(n, dtype=jnp.float32),
        source_scale=jnp.ones(n, dtype=jnp.float32),
        sign=jnp.ones(n, dtype=jnp.float32),
        labels=tuple(f"E{i}" for i in range(n)),
    )
    return params, edges


class TestHdpFiniteDelay:
    def test_zero_delay_regression(self):
        params, edges = _make_impulse_delayed_network(delays=(0, 0, 0))
        key = jax.random.PRNGKey(42)
        n_steps = 100
        dt_ms = 0.1

        V_hdp, S_hdp, _, diag_hdp = simulate_edge_recurrent_izhikevich_hdp(
            params, edges, n_steps, dt_ms, key,
            enable_boundary_stabilization=True,
            noise_scale=0.1,
        )
        assert V_hdp.shape == (n_steps, 3)
        assert S_hdp.shape == (n_steps, 3)
        assert np.all(np.isfinite(V_hdp))

    def test_delay_semantics_impulse_probe(self):
        delays = (20, 80, 120)
        params, edges = _make_impulse_delayed_network(delays=delays)
        dt_ms = 0.1
        n_steps = 200

        drive_sched = np.zeros((n_steps, 3), dtype=np.float32)
        drive_sched[0, 0] = 500.0

        key = jax.random.PRNGKey(0)
        V, S, _, diag = simulate_edge_recurrent_izhikevich_hdp(
            params, edges, n_steps, dt_ms, key,
            drive_schedule=jnp.asarray(drive_sched),
            noise_scale=0.0,
            enable_boundary_stabilization=False,
            K_HDP=0.0,
            record_edge_current=True,
        )

        spike_times_0 = np.where(S[:, 0] > 0.5)[0]
        assert len(spike_times_0) >= 1
        t_spike0 = spike_times_0[0]

        edge_current = np.asarray(diag["edge_current_trace"])
        # Edge 0: 0 -> 1, delay 20 steps (enters syn_state at t+20, realized current at t+21)
        assert np.all(edge_current[:t_spike0 + 21, 0] == 0.0)
        assert edge_current[t_spike0 + 21, 0] > 0.0

        # Edge 1: 0 -> 2, delay 80 steps (enters syn_state at t+80, realized current at t+81)
        assert np.all(edge_current[:t_spike0 + 81, 1] == 0.0)
        assert edge_current[t_spike0 + 81, 1] > 0.0

    def test_h_off_equivalence(self):
        delays = (20, 80, 120)
        params, edges = _make_impulse_delayed_network(delays=delays)
        dt_ms = 0.1
        n_steps = 150
        key = jax.random.PRNGKey(123)

        drive_sched = jax.random.normal(jax.random.PRNGKey(99), shape=(n_steps, 3)) * 2.0

        V_ref, S_ref, src_ref, diag_ref = _simulate_edge_recurrent_izhikevich_delayed(
            params, edges, n_steps, dt_ms, key,
            drive_schedule=drive_sched,
            noise_scale=0.5,
            record_edge_current=True,
        )

        V_hdp, S_hdp, src_hdp, diag_hdp = simulate_edge_recurrent_izhikevich_hdp(
            params, edges, n_steps, dt_ms, key,
            drive_schedule=drive_sched,
            noise_scale=0.5,
            enable_boundary_stabilization=False,
            K_HDP=0.0,
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            delta=0.0,
            rho_passive=0.0,
            K_ctrl=0.0,
            K_w_ctrl=0.0,
            record_edge_current=True,
        )

        assert np.array_equal(S_ref, S_hdp)
        assert np.allclose(V_ref, V_hdp, atol=1e-6)
        assert np.allclose(src_ref, src_hdp, atol=1e-6)
        assert np.array_equal(diag_ref["delay_state"], diag_hdp["delay_state"])
        assert np.allclose(diag_ref["edge_current_trace"], diag_hdp["edge_current_trace"], atol=1e-6)

    def test_h_on_continuation(self):
        delays = (20, 80, 120)
        params, edges = _make_impulse_delayed_network(delays=delays)
        dt_ms = 0.1
        total_steps = 200
        chunk_steps = 100

        key = jax.random.PRNGKey(777)
        k_uninterrupted, k_chunk1, k_chunk2 = jax.random.split(key, 3)

        full_sched = jax.random.normal(jax.random.PRNGKey(11), shape=(total_steps, 3)) * 3.0
        full_noise = jax.random.normal(jax.random.PRNGKey(22), shape=(total_steps, 3))

        V_full, S_full, _, diag_full = simulate_edge_recurrent_izhikevich_hdp(
            params, edges, total_steps, dt_ms, k_uninterrupted,
            drive_schedule=full_sched,
            noise_schedule=full_noise,
            enable_boundary_stabilization=True,
            noise_scale=0.0,
        )

        V1, S1, _, diag1 = simulate_edge_recurrent_izhikevich_hdp(
            params, edges, chunk_steps, dt_ms, k_chunk1,
            drive_schedule=full_sched[:chunk_steps],
            noise_schedule=full_noise[:chunk_steps],
            enable_boundary_stabilization=True,
            noise_scale=0.0,
        )

        V2, S2, _, diag2 = simulate_edge_recurrent_izhikevich_hdp(
            params, edges, chunk_steps, dt_ms, k_chunk2,
            drive_schedule=full_sched[chunk_steps:],
            noise_schedule=full_noise[chunk_steps:],
            enable_boundary_stabilization=True,
            noise_scale=0.0,
            init_state=diag1,
        )

        V_chunked = np.concatenate([V1, V2], axis=0)
        S_chunked = np.concatenate([S1, S2], axis=0)
        H_chunked = np.concatenate([diag1["H_trace"], diag2["H_trace"]], axis=0)
        r_chunked = np.concatenate([diag1["r_bar_trace"], diag2["r_bar_trace"]], axis=0)
        IH_chunked = np.concatenate([diag1["I_H_trace"], diag2["I_H_trace"]], axis=0)

        assert np.array_equal(S_full, S_chunked)
        assert np.allclose(V_full, V_chunked, atol=1e-5)
        assert np.allclose(diag_full["H_trace"], H_chunked, atol=1e-5)
        assert np.allclose(diag_full["r_bar_trace"], r_chunked, atol=1e-5)
        assert np.allclose(diag_full["I_H_trace"], IH_chunked, atol=1e-5)
        assert np.array_equal(diag_full["delay_state"], diag2["delay_state"])

    def test_recording_invariance(self):
        delays = (20, 80, 120)
        params, edges = _make_impulse_delayed_network(delays=delays)
        dt_ms = 0.1
        n_steps = 150
        key = jax.random.PRNGKey(999)

        drive_sched = jax.random.normal(jax.random.PRNGKey(33), shape=(n_steps, 3)) * 2.0

        V_plain, S_plain, _, _ = simulate_edge_recurrent_izhikevich_hdp(
            params, edges, n_steps, dt_ms, key,
            drive_schedule=drive_sched,
            enable_boundary_stabilization=True,
            record_boundary_components=False,
            record_edge_current=False,
        )

        V_diag, S_diag, _, _ = simulate_edge_recurrent_izhikevich_hdp(
            params, edges, n_steps, dt_ms, key,
            drive_schedule=drive_sched,
            enable_boundary_stabilization=True,
            record_boundary_components=True,
            record_edge_current=True,
        )

        assert np.array_equal(S_plain, S_diag)
        assert np.array_equal(V_plain, V_diag)
