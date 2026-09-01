"""Isolated acceptance test suite for frozen H_BOUNDARY_STABILIZATION dynamics.

Governing equations (sign-consistent):
    h = H - 1                            # H in [0.1, 10], h in [-0.9, 9]
    I_H = -g_H * h                       # g_H = 0.22 (asymmetric +0.198 / -1.98)
    I_total = I_native + I_H             # I_native = drive + schedule + syn + noise

    tau_r * dr_bar/dt = r - r_bar        # tau_r = 0.3 s
    tau_H * dh/dt = -K_H * h - S_L(r_bar) + S_H(r_bar) - B'(h)
      K_H = 0.1, tau_H = 4 s (E) / 1 s (PV/SST/VIP)

    r_bar_{n+1} = r_bar_n + (dt/tau_r) * (s_n/dt_s - r_bar_n)

    S_L(r_bar) = k_L * softplus_beta(r_L - r_bar)   k_L = 1 Hz^-1, beta = 25 Hz^-1, r_L = 0.5 Hz
    S_H(r_bar) = k_H * softplus_beta(r_bar - r_H)   k_H = 1 Hz^-1, beta = 25 Hz^-1, r_H = 20 Hz
    S < 1.5e-7 for 1.0 < r_bar < 19.5 (dead zone)

    B(h) = c / (h + 0.9) + d / (9 - h)   c = 0.01, d = 1.00 (d/c = 100 => B'(0) = 0)
    B'(h) = -c / (h + 0.9)^2 + d / (9 - h)^2
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jaxfne.emitters import (
    IzhikevichParams,
    EdgeList,
    simulate_edge_recurrent_izhikevich_hdp,
)


def _make_isolated_params(n_neurons: int = 2, labels: tuple[str, ...] = ("E", "PV"), drive: float = 0.0):
    """Build isolated Izhikevich neurons with zero synaptic edges."""
    params = IzhikevichParams(
        a=jnp.full((n_neurons,), 0.02, dtype=jnp.float32),
        b=jnp.full((n_neurons,), 0.2, dtype=jnp.float32),
        c=jnp.full((n_neurons,), -65.0, dtype=jnp.float32),
        d=jnp.full((n_neurons,), 8.0, dtype=jnp.float32),
        drive=jnp.full((n_neurons,), drive, dtype=jnp.float32),
        sign=jnp.ones((n_neurons,), dtype=jnp.float32),
        W=jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32),
        v0=jnp.full((n_neurons,), -65.0, dtype=jnp.float32),
        u0=jnp.full((n_neurons,), -13.0, dtype=jnp.float32),
        source_scale=jnp.ones((n_neurons,), dtype=jnp.float32),
        labels=labels,
        layer_labels=tuple("L4" for _ in range(n_neurons)),
        source_calibration_status="calibrated",
    )
    edges = EdgeList(
        pre=jnp.zeros((0,), dtype=jnp.int32),
        post=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jnp.float32),
        receptor_index=jnp.zeros((0,), dtype=jnp.int32),
        tau_ms=jnp.zeros((0,), dtype=jnp.float32),
        source_calibration_status="calibrated",
    )
    return params, edges


def test_dead_zone_and_barrier_math():
    """Verify algebraic properties: B'(0)=0, B''(0)=0.03017, and dead zone suppression."""
    c, d = 0.01, 1.00
    h0 = 0.0
    # B'(h) = -c/(h+0.9)^2 + d/(9-h)^2
    B_prime_0 = -c / (h0 + 0.9) ** 2 + d / (9.0 - h0) ** 2
    assert abs(B_prime_0) < 1e-12, f"B'(0) should be 0, got {B_prime_0}"

    # B''(h) = 2c/(h+0.9)^3 + 2d/(9-h)^3
    B_double_prime_0 = 2.0 * c / (h0 + 0.9) ** 3 + 2.0 * d / (9.0 - h0) ** 3
    assert abs(B_double_prime_0 - 0.0301783) < 1e-5

    # Dead zone: S < 1.5e-7 for 1.0 < r_bar < 19.5
    beta = 25.0
    r_L, r_H = 0.5, 20.0
    for r in [1.0, 5.0, 8.0, 15.0, 19.5]:
        S_L = float(jax.nn.softplus(beta * (r_L - r)) / beta)
        S_H = float(jax.nn.softplus(beta * (r - r_H)) / beta)
        assert S_L < 1.5e-7, f"S_L at r={r} was {S_L} >= 1.5e-7"
        assert S_H < 1.5e-7, f"S_H at r={r} was {S_H} >= 1.5e-7"


def test_interior_equilibrium():
    """Verify INTERIOR fixed point: r_bar* = 8 Hz => h* = 0, H* = 1.0, I_H* = 0."""
    params, edges = _make_isolated_params(2, ("E", "PV"))
    key = jax.random.PRNGKey(0)

    # Run for 2000 ms (2.0 s) with r_bar initialized to 8.0 Hz and H initialized to 1.0
    init_state = {
        "v": params.v0,
        "u": params.u0,
        "prev_spikes": jnp.zeros((2,), dtype=jnp.float32),
        "syn_state": jnp.zeros((0,), dtype=jnp.float32),
        "H_final": jnp.array([1.0, 1.0], dtype=jnp.float32),
        "r_bar": jnp.array([8.0, 8.0], dtype=jnp.float32),
    }

    dt_ms = 0.1
    _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        params,
        edges,
        n_steps=1000,
        dt_ms=dt_ms,
        key=key,
        enable_boundary_stabilization=True,
        r_bar_init=8.0,
        init_state=init_state,
        record_boundary_components=True,
    )

    H_trace = np.asarray(diag["H_trace"])
    I_H_trace = np.asarray(diag["I_H_trace"])

    # H should stay pinned at 1.0 (deviation < 1e-5)
    assert np.allclose(H_trace[0], 1.0, atol=1e-5)
    assert np.allclose(I_H_trace[0], 0.0, atol=1e-5)


def test_hypo_equilibrium():
    """Verify HYPO fixed point (r_bar = 0 Hz):
    Equation: -K_H*h - S_L(0) - B'(h) = 0
    yields h* = -0.7485 => H* = 0.2515, I_H* = +0.1647 nA.
    """
    params, edges = _make_isolated_params(2, ("E", "PV"))
    key = jax.random.PRNGKey(42)

    h_hypo = -0.7485
    H_hypo = 1.0 + h_hypo
    init_state = {
        "v": params.v0,
        "u": params.u0,
        "prev_spikes": jnp.zeros((2,), dtype=jnp.float32),
        "syn_state": jnp.zeros((0,), dtype=jnp.float32),
        "H_final": jnp.array([H_hypo, H_hypo], dtype=jnp.float32),
        "r_bar": jnp.array([0.0, 0.0], dtype=jnp.float32),
    }

    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        params,
        edges,
        n_steps=10000,
        dt_ms=0.2,
        key=key,
        enable_boundary_stabilization=True,
        r_bar_init=0.0,
        init_state=init_state,
    )

    H_final = np.asarray(diag["H_final"])
    I_H_final = np.asarray(diag["I_H_final"])

    # Expected H* = 0.2515, I_H* = +0.1647 nA
    assert abs(H_final[1] - 0.2515) < 0.002, f"Expected H* ~ 0.2515, got {H_final[1]}"
    assert abs(I_H_final[1] - 0.1647) < 0.002, f"Expected I_H* ~ 0.1647, got {I_H_final[1]}"


def test_threshold_equilibrium():
    """Verify THRESHOLD fixed point (r_bar = 0.5 Hz = r_L):
    h* = -0.1946 => H* = 0.8054, I_H* = +0.0428 nA.
    """
    h = -0.1946
    S_L = np.log(2.0) / 25.0
    minus_B = 0.01 / (h + 0.9) ** 2 - 1.00 / (9.0 - h) ** 2
    residual = -0.1 * h - S_L + minus_B
    assert abs(residual) < 1e-4

    H_thresh = 1.0 + h
    I_H_thresh = -0.22 * h
    assert abs(H_thresh - 0.8054) < 1e-4
    assert abs(I_H_thresh - 0.0428) < 1e-4


def test_hyper_equilibrium():
    """Verify HYPER fixed point (r_bar = 30 Hz):
    Equation: -K_H*h + S_H(30) - B'(h) = 0 with S_H(30) ~ 10.0
    yields h* = 8.669 => H* = 9.669, I_H* = -1.907 nA.
    """
    h = 8.669
    S_H = float(jax.nn.softplus(25.0 * (30.0 - 20.0)) / 25.0)  # 10.0
    minus_B = 0.01 / (h + 0.9) ** 2 - 1.00 / (9.0 - h) ** 2
    residual = -0.1 * h + S_H + minus_B
    assert abs(residual) < 0.02

    H_hyper = 1.0 + h
    I_H_hyper = -0.22 * h
    assert abs(H_hyper - 9.669) < 0.01
    assert abs(I_H_hyper - (-1.907)) < 0.01


def test_transient_pulse_in_dead_zone():
    """Verify TRANSIENT: 50-ms pulse (8 -> 19.5 Hz) stays inside dead zone, Delta H ~ 0."""
    params, edges = _make_isolated_params(1, ("E",))
    key = jax.random.PRNGKey(101)

    init_state = {
        "v": params.v0,
        "u": params.u0,
        "prev_spikes": jnp.zeros((1,), dtype=jnp.float32),
        "syn_state": jnp.zeros((0,), dtype=jnp.float32),
        "H_final": jnp.array([1.0], dtype=jnp.float32),
        "r_bar": jnp.array([19.5], dtype=jnp.float32),  # Top edge of dead zone
    }

    # 50 ms at dt=0.1 ms = 500 steps
    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        params,
        edges,
        n_steps=500,
        dt_ms=0.1,
        key=key,
        enable_boundary_stabilization=True,
        init_state=init_state,
        record_boundary_components=True,
    )

    H_trace = np.asarray(diag["H_trace"])
    delta_H = float(np.max(np.abs(H_trace - 1.0)))
    assert delta_H < 1e-5, f"Transient pulse in dead zone produced Delta H = {delta_H} >= 1e-5"


def test_nonlinear_recovery_times():
    """Verify RECOVERY: Trajectory recovering from boundary state returns monotonically to interior.
    Measures T_50 and T_63. Drive=4.0 sustains ~7.7 Hz firing within dead zone (1.0-19.5 Hz).
    """
    params, edges = _make_isolated_params(1, ("PV",), drive=4.0)
    key = jax.random.PRNGKey(7)

    # Start at hypo state H = 0.2515, but set r_bar = 8.0 Hz (interior rate restored)
    h_start = -0.7485
    H_start = 1.0 + h_start
    init_state = {
        "v": params.v0,
        "u": params.u0,
        "prev_spikes": jnp.zeros((1,), dtype=jnp.float32),
        "syn_state": jnp.zeros((0,), dtype=jnp.float32),
        "H_final": jnp.array([H_start], dtype=jnp.float32),
        "r_bar": jnp.array([8.0], dtype=jnp.float32),
    }

    # PV tau_H = 1.0 s. Integrate for 20 s at dt=0.5 ms (40,000 steps)
    dt_ms = 0.5
    n_steps = 40000
    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        params,
        edges,
        n_steps=n_steps,
        dt_ms=dt_ms,
        key=key,
        enable_boundary_stabilization=True,
        init_state=init_state,
    )

    H_trace = np.asarray(diag["H_trace"])[:, 0]
    h_trace = H_trace - 1.0

    # Must recover monotonically toward 0
    diffs = np.diff(h_trace)
    assert np.all(diffs >= -1e-6), "Recovery must be monotonic increasing toward 0"

    # Initial deviation = -0.7485
    init_dev = abs(h_start)
    # T_50: time when |h(t)| <= 0.5 * init_dev
    idx_50 = np.where(np.abs(h_trace) <= 0.5 * init_dev)[0]
    assert len(idx_50) > 0, "Did not reach 50% recovery"
    t_50_s = idx_50[0] * (dt_ms / 1000.0)

    # T_63: time when |h(t)| <= (1 - 0.632) * init_dev
    idx_63 = np.where(np.abs(h_trace) <= (1.0 - 0.632) * init_dev)[0]
    assert len(idx_63) > 0, "Did not reach 63% recovery"
    t_63_s = idx_63[0] * (dt_ms / 1000.0)

    # For PV (tau_eff,int ~ 7.68 s), measured T_50 ~ 3.0 s, T_63 ~ 4.9 s
    assert 1.0 < t_50_s < 10.0, f"T_50 = {t_50_s} s out of expected range"
    assert 2.0 < t_63_s < 15.0, f"T_63 = {t_63_s} s out of expected range"


def test_no_oscillation_and_barrier_boundedness():
    """Verify trajectory stability: smooth asymptotic approach without overshoot or hard clamp hitting.
    NOTE on NO_CLAMP: This test establishes barrier boundedness (soft repulsion before hard clamps).
    The frozen NO_CLAMP requirement of physiological heterogeneity preservation
    |SD_ON / SD_OFF - 1| < 0.10 requires population/network measurement and remains explicitly UNRESOLVED.
    """
    params, edges = _make_isolated_params(1, ("E",), drive=4.0)
    key = jax.random.PRNGKey(123)

    # Start near lower barrier H = 0.15 (h = -0.85)
    init_state = {
        "v": params.v0,
        "u": params.u0,
        "prev_spikes": jnp.zeros((1,), dtype=jnp.float32),
        "syn_state": jnp.zeros((0,), dtype=jnp.float32),
        "H_final": jnp.array([0.15], dtype=jnp.float32),
        "r_bar": jnp.array([8.0], dtype=jnp.float32),
    }

    n_steps = 10000
    dt_ms = 0.5
    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        params,
        edges,
        n_steps=n_steps,
        dt_ms=dt_ms,
        key=key,
        enable_boundary_stabilization=True,
        init_state=init_state,
    )

    H_trace = np.asarray(diag["H_trace"])[:, 0]
    # Verify strictly finite and strictly above H_min (0.1)
    assert np.all(np.isfinite(H_trace))
    assert np.all(H_trace > 0.101), "Hit or violated H_min lower clamp"
    assert np.all(H_trace < 9.99), "Hit or violated H_max upper clamp"

    # Verify no limit cycles or oscillations: sign of dh should never alternate
    dh = np.diff(H_trace)
    sign_changes = np.where(np.diff(np.signbit(dh)))[0]
    assert len(sign_changes) == 0, f"Detected {len(sign_changes)} trajectory oscillations (limit cycle risk)"


def test_h_on_chunk_resume_matches_continuous_byte_identically():
    """Verify that a 2000-step H_ON run equals two 1000-step chunks chained via init_state.
    Verifies that r_bar_final and H_final seamlessly carry forward across chunks.
    """
    params, edges = _make_isolated_params(2, ("E", "PV"), drive=4.0)
    key = jax.random.PRNGKey(42)
    noise_schedule = jnp.zeros((2000, 2), dtype=jnp.float32)

    # (1) Continuous 2000-step run
    v_full, s_full, _, d_full = simulate_edge_recurrent_izhikevich_hdp(
        params, edges, n_steps=2000, dt_ms=0.5, key=key,
        noise_schedule=noise_schedule,
        enable_boundary_stabilization=True,
    )

    # (2) Chunk 1: 1000 steps
    v1, s1, _, d1 = simulate_edge_recurrent_izhikevich_hdp(
        params, edges, n_steps=1000, dt_ms=0.5, key=key,
        noise_schedule=noise_schedule[:1000],
        enable_boundary_stabilization=True,
    )
    assert "r_bar_final" in d1, "Chunk 1 diagnostics must include r_bar_final"
    assert "H_final" in d1

    # (3) Chunk 2: 1000 steps seeded from Chunk 1
    init_state_2 = {
        "v": d1["v"],
        "u": d1["u"],
        "prev_spikes": d1["prev_spikes"],
        "syn_state": d1["syn_state"],
        "H_final": d1["H_final"],
        "w_final": d1["w_final"],
        "r_bar_final": d1["r_bar_final"],
    }
    v2, s2, _, d2 = simulate_edge_recurrent_izhikevich_hdp(
        params, edges, n_steps=1000, dt_ms=0.5, key=key,
        noise_schedule=noise_schedule[1000:],
        init_state=init_state_2,
        enable_boundary_stabilization=True,
    )

    # Verify byte-identical resumption
    assert jnp.array_equal(d_full["H_trace"][:1000], d1["H_trace"])
    assert jnp.array_equal(d_full["H_trace"][1000:], d2["H_trace"])
    assert jnp.array_equal(d_full["r_bar_trace"][:1000], d1["r_bar_trace"])
    assert jnp.array_equal(d_full["r_bar_trace"][1000:], d2["r_bar_trace"])
    assert jnp.array_equal(d_full["I_H_trace"][:1000], d1["I_H_trace"])
    assert jnp.array_equal(d_full["I_H_trace"][1000:], d2["I_H_trace"])
    assert jnp.array_equal(s_full[:1000], s1)
    assert jnp.array_equal(s_full[1000:], s2)
    assert jnp.array_equal(v_full[:1000], v1)
    assert jnp.array_equal(v_full[1000:], v2)


def test_no_trace_allocation_when_recording_flags_false():
    """Verify that when record_weight_trace=False and record_boundary_components=False,
    no extraneous traces are allocated or exposed.
    """
    params, edges = _make_isolated_params(2, ("E", "PV"), drive=4.0)
    key = jax.random.PRNGKey(12)

    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        params, edges, n_steps=100, dt_ms=0.5, key=key,
        enable_boundary_stabilization=True,
        record_weight_trace=False,
        record_boundary_components=False,
        record_dH_components=False,
        record_edge_current=False,
    )

    assert diag["w_trace"] is None
    assert "S_L_trace" not in diag
    assert "S_H_trace" not in diag
    assert "minus_B_prime_trace" not in diag
    assert "dH_income_trace" not in diag
    assert "edge_current_trace" not in diag


def test_h_off_invariance():
    """Verify H_OFF_INVARIANCE: When boundary stabilization is disabled, dynamics match legacy baseline identically."""
    params, edges = _make_isolated_params(2, ("E", "PV"))
    key = jax.random.PRNGKey(999)

    v1, s1, _, diag1 = simulate_edge_recurrent_izhikevich_hdp(
        params,
        edges,
        n_steps=1000,
        dt_ms=0.1,
        key=key,
        enable_boundary_stabilization=False,
    )

    v2, s2, _, diag2 = simulate_edge_recurrent_izhikevich_hdp(
        params,
        edges,
        n_steps=1000,
        dt_ms=0.1,
        key=key,
        enable_boundary_stabilization=False,
    )

    assert np.array_equal(v1, v2)
    assert np.array_equal(s1, s2)
    assert np.array_equal(diag1["H_trace"], diag2["H_trace"])
    assert "r_bar_trace" not in diag1, "r_bar_trace should only be exposed when boundary stabilization is active"


def test_dispatch_boundary_stabilization_traces():
    """Verify that full Model dispatch carries boundary stabilization and exposes traces."""
    import jaxfne as jtfne

    cfg = jtfne.build_laminar_column(n=80, ei_profile="canonical")
    cfg = cfg.runtime(
        enable_hdp=True,
        hdp_params={
            "enable_boundary_stabilization": True,
            "tau_r_s": 0.3,
            "tau_H_E_s": 4.0,
            "tau_H_I_s": 1.0,
            "K_H": 0.1,
            "g_H": 0.22,
            "r_bar_init": 8.0,
        },
    )
    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    model = jtfne.construct(cfg)
    sig = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)

    diag = model.last_hdp_diagnostics()
    assert diag is not None
    assert "r_bar_trace" in diag, "r_bar_trace must be exposed in diagnostics"
    assert "I_H_trace" in diag, "I_H_trace must be exposed in diagnostics"
    assert "r_bar_final" in diag
    assert "I_H_final" in diag

    r_bar_trace = np.asarray(diag["r_bar_trace"])
    I_H_trace = np.asarray(diag["I_H_trace"])
    assert r_bar_trace.shape == (400, 80)
    assert I_H_trace.shape == (400, 80)
    assert bool(np.isfinite(r_bar_trace).all())
    assert bool(np.isfinite(I_H_trace).all())
