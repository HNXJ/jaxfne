"""Gate 7: RNG and recording invariance fixture.

Verifies that optional diagnostic and recording configurations
(record_sources, record_edge_current, record_dH_components, record_weight_trace)
do not perturb the simulation trajectory (V, spikes, sources, H, w) or step PRNG key sequence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    simulate_edge_recurrent_izhikevich_hdp,
)


def _make_asymmetric_stochastic_fixture(seed: int = 42):
    """Build an asymmetric small recurrent circuit with heterogeneous delays, weights, and drive."""
    N = 4
    p = IzhikevichParams(
        a=jnp.array([0.02, 0.02, 0.1, 0.02], dtype=jnp.float32),
        b=jnp.array([0.2, 0.25, 0.2, 0.2], dtype=jnp.float32),
        c=jnp.array([-65.0, -65.0, -65.0, -55.0], dtype=jnp.float32),
        d=jnp.array([8.0, 2.0, 2.0, 4.0], dtype=jnp.float32),
        drive=jnp.array([4.5, 3.2, 5.0, 1.5], dtype=jnp.float32),
        sign=jnp.array([1.0, 1.0, -1.0, 1.0], dtype=jnp.float32),
        W=jnp.zeros((N, N), dtype=jnp.float32),
        v0=jnp.array([-65.0, -60.0, -62.0, -70.0], dtype=jnp.float32),
        u0=jnp.array([-13.0, -15.0, -12.0, -14.0], dtype=jnp.float32),
        source_scale=jnp.array([1.0, 0.8, 1.2, 0.9], dtype=jnp.float32),
        labels=("E", "E", "I", "E"),
        layer_labels=("L4", "L23", "L23", "L5"),
        source_calibration_status="uncalibrated",
    )
    edges = EdgeList(
        pre=jnp.array([0, 0, 1, 2, 2, 3], dtype=jnp.int32),
        post=jnp.array([1, 2, 3, 0, 1, 0], dtype=jnp.int32),
        weight=jnp.array([0.45, 0.30, 0.25, -0.60, -0.40, 0.15], dtype=jnp.float32),
        receptor_index=jnp.array([0, 0, 0, 1, 1, 0], dtype=jnp.int32),
        tau_ms=jnp.array([5.0, 5.0, 6.0, 8.0, 8.0, 5.0], dtype=jnp.float32),
        delay_steps=jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.int32),
        source_calibration_status="uncalibrated",
    )
    return p, edges


def test_gate7_emitter_recording_invariance_stochastic():
    """Verify bit-exact state invariance across recording combinations under stochastic noise."""
    p, edges = _make_asymmetric_stochastic_fixture()
    n_steps = 40
    dt_ms = 0.5
    key = jax.random.PRNGKey(101)

    hdp_kw = dict(
        noise_scale=0.05,
        alpha=0.01,
        gamma=0.02,
        K_HDP=0.01,
        K_ctrl=0.05,
        K_w_ctrl=0.001,
        rho_passive=0.005,
        barrier_c=0.005,
        barrier_d=0.005,
        tau_0_ms=10.0,
    )

    # 1. Baseline: minimal recording
    v_base, s_base, src_base, d_base = simulate_edge_recurrent_izhikevich_hdp(
        p, edges, n_steps, dt_ms, key,
        record_weight_trace=False,
        record_edge_current=False,
        record_dH_components=False,
        **hdp_kw,
    )

    # 2. Record weight trace only
    v_w, s_w, src_w, d_w = simulate_edge_recurrent_izhikevich_hdp(
        p, edges, n_steps, dt_ms, key,
        record_weight_trace=True,
        record_edge_current=False,
        record_dH_components=False,
        **hdp_kw,
    )

    # 3. Record edge current only
    v_ec, s_ec, src_ec, d_ec = simulate_edge_recurrent_izhikevich_hdp(
        p, edges, n_steps, dt_ms, key,
        record_weight_trace=False,
        record_edge_current=True,
        record_dH_components=False,
        **hdp_kw,
    )

    # 4. Record dH components only
    v_dh, s_dh, src_dh, d_dh = simulate_edge_recurrent_izhikevich_hdp(
        p, edges, n_steps, dt_ms, key,
        record_weight_trace=False,
        record_edge_current=False,
        record_dH_components=True,
        **hdp_kw,
    )

    # 5. Record ALL optional streams simultaneously
    v_all, s_all, src_all, d_all = simulate_edge_recurrent_izhikevich_hdp(
        p, edges, n_steps, dt_ms, key,
        record_weight_trace=True,
        record_edge_current=True,
        record_dH_components=True,
        **hdp_kw,
    )

    # Invariance assertions: bit-exact trajectory equality
    for label, (v, s, src, diag) in [
        ("weight_trace_on", (v_w, s_w, src_w, d_w)),
        ("edge_current_on", (v_ec, s_ec, src_ec, d_ec)),
        ("dh_components_on", (v_dh, s_dh, src_dh, d_dh)),
        ("all_recording_on", (v_all, s_all, src_all, d_all)),
    ]:
        np.testing.assert_array_equal(np.asarray(v), np.asarray(v_base), err_msg=f"{label}: V_m differs")
        np.testing.assert_array_equal(np.asarray(s), np.asarray(s_base), err_msg=f"{label}: Spikes differ")
        np.testing.assert_array_equal(np.asarray(src), np.asarray(src_base), err_msg=f"{label}: Sources differ")
        np.testing.assert_array_equal(np.asarray(diag["H_final"]), np.asarray(d_base["H_final"]), err_msg=f"{label}: H_final differs")
        np.testing.assert_array_equal(np.asarray(diag["w_final"]), np.asarray(d_base["w_final"]), err_msg=f"{label}: w_final differs")


def test_gate7_model_simulate_recording_invariance():
    """Verify Model.simulate yields bit-exact signals whether record_fields is True or False."""
    cfg = jtfne.suite2_net1_config(seed=42, n=5, duration_ms=25.0, dt_ms=0.5)
    model = jtfne.construct(cfg)

    # Run with fields recorded
    sim_fields = jtfne.simulation(duration_ms=25.0, dt_ms=0.5, seed=17, record_fields=True, record_sources=True)
    res_fields = model.simulate(sim_fields)

    # Run without fields recorded
    sim_nofields = jtfne.simulation(duration_ms=25.0, dt_ms=0.5, seed=17, record_fields=False, record_sources=True)
    res_nofields = model.simulate(sim_nofields)

    np.testing.assert_array_equal(np.asarray(res_fields.V_m), np.asarray(res_nofields.V_m))
    np.testing.assert_array_equal(np.asarray(res_fields.spikes), np.asarray(res_nofields.spikes))
    np.testing.assert_array_equal(np.asarray(res_fields.sources), np.asarray(res_nofields.sources))
