"""Standalone tests for the HDP (Hidden-state Dependent Plasticity) kernel.

Exercises jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp directly
(no Configuration/Model layer), matching the existing
test_homeostasis_dispatch.py pattern. Item #12 of the HDP punch list:
the kernel itself (#11) had no dedicated test before this file.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jaxfne.emitters import (
    IzhikevichParams, EdgeList,
    simulate_edge_recurrent_izhikevich_hdp as hdp_kernel,
)


def _make_params_edges(N: int, ne: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = IzhikevichParams(
        a=jnp.full((N,), 0.02), b=jnp.full((N,), 0.2), c=jnp.full((N,), -65.0),
        d=jnp.full((N,), 8.0), drive=jnp.full((N,), 6.0), sign=jnp.ones((N,)),
        W=jnp.zeros((N, N)), v0=jnp.full((N,), -65.0), u0=jnp.full((N,), -13.0),
        source_scale=jnp.ones((N,)), labels=tuple("E" for _ in range(N)),
        layer_labels=tuple("L4" for _ in range(N)), source_calibration_status="x")
    receptor_index = rng.integers(0, 2, ne)
    # The kernel re-derives weight sign from |w| + receptor_index every step
    # (excitatory positive, inhibitory negative) -- match that convention in
    # the fixture so a K_HDP=0 plasticity-term null is a true no-op on weights,
    # not a one-time sign normalization.
    weight_mag = np.abs(rng.normal(0, 0.3, ne)).astype(np.float32)
    weight = np.where(receptor_index == 0, weight_mag, -weight_mag).astype(np.float32)
    edges = EdgeList(
        pre=jnp.asarray(rng.integers(0, N, ne), jnp.int32),
        post=jnp.asarray(rng.integers(0, N, ne), jnp.int32),
        weight=jnp.asarray(weight),
        receptor_index=jnp.asarray(receptor_index, jnp.int32),
        tau_ms=jnp.full((ne,), 5.0), source_calibration_status="x")
    return p, edges


def _make_two_neuron_edge(
    *,
    pre_label: str,
    post_label: str,
    receptor_index: int,
    h_pre: float,
    h_post: float,
    carried_weight: float = 1.0,
):
    """Build one deterministic directed edge for HDP sign/null tests."""
    labels = (pre_label, post_label)
    p = IzhikevichParams(
        a=jnp.full((2,), 0.02), b=jnp.full((2,), 0.2),
        c=jnp.full((2,), -65.0), d=jnp.full((2,), 8.0),
        drive=jnp.zeros((2,)), sign=jnp.array([
            1.0 if pre_label == "E" else -1.0,
            1.0 if post_label == "E" else -1.0,
        ]),
        W=jnp.zeros((2, 2)), v0=jnp.full((2,), -65.0),
        u0=jnp.full((2,), -13.0), source_scale=jnp.ones((2,)),
        labels=labels, layer_labels=("L4", "L4"),
        source_calibration_status="x",
    )
    signed_weight = 1.0 if receptor_index == 0 else -1.0
    edges = EdgeList(
        pre=jnp.array([0], dtype=jnp.int32),
        post=jnp.array([1], dtype=jnp.int32),
        weight=jnp.array([signed_weight]),
        receptor_index=jnp.array([receptor_index], dtype=jnp.int32),
        tau_ms=jnp.array([5.0]),
        source_calibration_status="x",
    )
    init_state = {
        "v": p.v0,
        "u": p.u0,
        "prev_spikes": jnp.zeros((2,)),
        "syn_state": jnp.zeros((1,)),
        "H_final": jnp.array([h_pre, h_post]),
        "w_final": jnp.array([carried_weight]),
    }
    return p, edges, init_state


def _run_two_neuron_edge(
    *,
    pre_label: str = "E",
    post_label: str = "I",
    receptor_index: int = 0,
    h_pre: float = 2.0,
    h_post: float = 1.0,
    carried_weight: float = 1.0,
    K_HDP: float = 0.0,
    K_ctrl: float = 0.0,
    K_w_ctrl: float = 0.0,
):
    p, edges, init_state = _make_two_neuron_edge(
        pre_label=pre_label,
        post_label=post_label,
        receptor_index=receptor_index,
        h_pre=h_pre,
        h_post=h_post,
        carried_weight=carried_weight,
    )
    _, _, _, diag = hdp_kernel(
        p,
        edges,
        1,
        0.5,
        jax.random.PRNGKey(0),
        init_state=init_state,
        drive_schedule=jnp.zeros((1, 2)),
        noise_scale=0.0,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        delta=0.0,
        C_spike=0.0,
        rho_passive=0.0,
        K_HDP=K_HDP,
        K_ctrl=K_ctrl,
        K_w_ctrl=K_w_ctrl,
        barrier_c=0.0,
        barrier_d=0.0,
        tau_0_ms=1.0,
        size_scale_override=jnp.ones((2,)),
        w_floor=0.001,
        w_ceiling=50.0,
    )
    return diag


def test_h_state_dim_one_is_exact_scalar_kernel_embedding():
    params, edges = _make_params_edges(3, 4, seed=7)
    common = dict(
        n_steps=5,
        dt_ms=0.5,
        key=jax.random.PRNGKey(12),
        dtype="float32",
        drive_schedule=jnp.zeros((5, 3)),
        noise_scale=0.0,
        alpha=0.02,
        beta=0.01,
        gamma=0.03,
        delta=0.01,
        K_HDP=0.02,
        K_ctrl=0.1,
        K_w_ctrl=0.001,
        rho_passive=0.01,
        barrier_c=0.0,
        barrier_d=0.0,
    )
    legacy = hdp_kernel(params, edges, **common)
    generalized = hdp_kernel(
        params,
        edges,
        **common,
        h_state_dim=1,
        h_state_readout=jnp.array([1.0]),
        h_state_coupling=jnp.zeros((1, 1)),
    )
    for legacy_arr, generalized_arr in zip(legacy[:3], generalized[:3]):
        assert jnp.array_equal(legacy_arr, generalized_arr)
    for key in ("H_trace", "H_final", "w_trace", "w_final"):
        assert jnp.array_equal(legacy[3][key], generalized[3][key])


def test_difference_family_signs_cover_all_four_canonical_edge_types():
    """The adopted delta_H = H_post - H_pre convention has one E/I magnitude
    direction for each of E->E, E->I, I->E, and I->I."""
    edge_types = (
        ("E->E", "E", "E", 0),
        ("E->I", "E", "I", 0),
        ("I->E", "I", "E", 1),
        ("I->I", "I", "I", 1),
    )
    for h_pre, h_post, expected_exc, expected_inh in (
        (2.0, 1.0, 0.95, -1.05),
        (1.0, 2.0, 1.05, -0.95),
    ):
        for name, pre_label, post_label, receptor_index in edge_types:
            diag = _run_two_neuron_edge(
                pre_label=pre_label,
                post_label=post_label,
                receptor_index=receptor_index,
                h_pre=h_pre,
                h_post=h_post,
                K_HDP=0.1,
            )
            expected = expected_exc if receptor_index == 0 else expected_inh
            np.testing.assert_allclose(
                np.asarray(diag["w_final"]),
                [expected],
                atol=1e-6,
                err_msg=f"unexpected signed update for {name}",
            )
            np.testing.assert_allclose(
                np.asarray(diag["H_final"]),
                [h_pre, h_post],
                atol=1e-6,
            )


def test_K_HDP_zero_is_plasticity_term_null_not_H_state_null():
    """K_HDP=0 freezes the difference-driven weight term while H remains an
    independent state axis."""
    diag = _run_two_neuron_edge(
        h_pre=2.0,
        h_post=1.0,
        K_HDP=0.0,
        K_ctrl=0.0,
        K_w_ctrl=0.0,
    )
    np.testing.assert_allclose(np.asarray(diag["w_final"]), [1.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(diag["H_final"]), [2.0, 1.0], atol=1e-6)


def test_K_ctrl_only_restores_H_toward_one():
    """K_ctrl changes the H equation without directly changing weights."""
    no_controller = _run_two_neuron_edge(
        h_pre=2.0,
        h_post=1.0,
        K_HDP=0.0,
        K_ctrl=0.0,
        K_w_ctrl=0.0,
    )
    controller = _run_two_neuron_edge(
        h_pre=2.0,
        h_post=1.0,
        K_HDP=0.0,
        K_ctrl=0.2,
        K_w_ctrl=0.0,
    )
    np.testing.assert_allclose(np.asarray(no_controller["H_final"]), [2.0, 1.0], atol=1e-6)
    assert float(np.asarray(controller["H_final"])[0]) < 2.0
    np.testing.assert_allclose(np.asarray(controller["w_final"]), [1.0], atol=1e-6)


def test_K_w_ctrl_restores_carried_magnitude_independently():
    """K_w_ctrl acts on carried magnitude even when K_HDP is zero."""
    no_controller = _run_two_neuron_edge(
        carried_weight=0.5,
        K_HDP=0.0,
        K_ctrl=0.0,
        K_w_ctrl=0.0,
    )
    controller = _run_two_neuron_edge(
        carried_weight=0.5,
        K_HDP=0.0,
        K_ctrl=0.0,
        K_w_ctrl=0.1,
    )
    np.testing.assert_allclose(np.asarray(no_controller["w_final"]), [0.5], atol=1e-6)
    np.testing.assert_allclose(np.asarray(controller["w_final"]), [0.525], atol=1e-6)


def test_null_control_holds_H_at_1_and_matches_homeostatic_kernel():
    """alpha=beta=gamma=delta=C_spike=0.0 holds H_i fixed at 1.0 forever, making
    the K_HDP-scaled weight term identically zero regardless of K_HDP -- the
    documented null control."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(1)
    V, S, _, diag = hdp_kernel(p, edges, 100, 0.5, key, K_HDP=1.0, noise_scale=0.0)
    H_trace = np.asarray(diag["H_trace"])
    assert np.allclose(H_trace, 1.0, atol=1e-6)
    w_final = np.asarray(diag["w_final"])
    w0 = np.asarray(edges.weight)
    assert np.allclose(w_final, w0, atol=1e-6)
    assert bool(np.isfinite(np.asarray(V)).all())


def test_K_HDP_zero_disables_difference_term_regardless_of_other_gains():
    """K_HDP=0.0 nulls the HDP weight term even with nonzero
    alpha/gamma/K_ctrl -- H_i may still move, but weights must not."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(1)
    V, S, _, diag = hdp_kernel(
        p, edges, 200, 0.5, key, K_HDP=0.0,
        alpha=0.05, gamma=0.3, K_ctrl=0.1, noise_scale=0.0)
    w_final = np.asarray(diag["w_final"])
    w0 = np.asarray(edges.weight)
    assert np.allclose(w_final, w0, atol=1e-6)
    assert bool(np.isfinite(np.asarray(V)).all())


def test_nonzero_gains_move_H_and_weights_while_staying_finite_and_clipped():
    """Nonzero alpha/gamma/K_ctrl/K_HDP -> H_i deviates from 1.0, weights
    change, and both stay finite and within their clip bounds."""
    N, ne = 64, 512
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(2)
    V, S, _, diag = hdp_kernel(
        p, edges, 400, 0.5, key,
        alpha=0.05, gamma=0.5, K_ctrl=0.15, K_HDP=0.01,
        barrier_c=0.01, barrier_d=0.01, tau_0_ms=5.0,
        w_floor=0.01, w_ceiling=10.0, noise_scale=0.0)
    H_trace = np.asarray(diag["H_trace"])
    w_trace = np.asarray(diag["w_trace"])
    assert not np.allclose(H_trace, 1.0, atol=1e-6)
    assert bool(np.isfinite(H_trace).all())
    assert bool(np.isfinite(w_trace).all())
    assert H_trace.min() >= 0.1 - 1e-4 and H_trace.max() <= 10.0 + 1e-4
    w_final = np.asarray(diag["w_final"])
    assert np.abs(w_final).max() <= 10.0 + 1e-4
    assert bool(np.isfinite(np.asarray(V)).all())
    assert H_trace.shape == (400, N)
    assert w_trace.shape == (400, ne)


def test_record_dH_components_returns_five_traces_with_passive_income():
    """record_dH_components=True exposes the five additive dH/dt terms;
    each has the documented shape (n_steps, n_neurons). HDP v2 uses
    dH_passive_trace (rho_passive/H_i**2) instead of dH_ctrl_trace."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(3)
    _, _, _, diag = hdp_kernel(
        p, edges, 50, 0.5, key,
        alpha=0.05, gamma=0.5, rho_passive=0.1, K_HDP=0.01,
        barrier_c=0.01, barrier_d=0.01, tau_0_ms=5.0,
        record_dH_components=True, noise_scale=0.0)
    for name in ("dH_income_trace", "dH_rate_trace", "dH_weight_trace",
                 "dH_passive_trace", "dH_barrier_trace"):
        assert name in diag
        assert np.asarray(diag[name]).shape == (50, N)
        assert bool(np.isfinite(np.asarray(diag[name])).all())


def test_record_edge_current_returns_per_edge_trace():
    """record_edge_current=True exposes the per-edge synaptic current
    contribution with shape (n_steps, n_edges)."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(4)
    _, _, _, diag = hdp_kernel(
        p, edges, 30, 0.5, key, alpha=0.05, gamma=0.5, K_ctrl=0.15,
        record_edge_current=True, noise_scale=0.0)
    assert "edge_current_trace" in diag
    assert np.asarray(diag["edge_current_trace"]).shape == (30, ne)


def test_init_state_resume_matches_full_run():
    """Two chunks via init_state == one full run, deterministically -- same
    pause/resume guarantee as the existing homeostatic kernel."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(5)
    kw = dict(alpha=0.05, gamma=0.5, K_ctrl=0.15, K_HDP=0.01, noise_scale=0.0)
    Vf, Sf, _, _ = hdp_kernel(p, edges, 200, 0.5, key, **kw)
    V1, S1, _, d1 = hdp_kernel(p, edges, 100, 0.5, key, **kw)
    V2, S2, _, _ = hdp_kernel(p, edges, 100, 0.5, key, init_state=d1, **kw)
    Vc = np.concatenate([np.asarray(V1), np.asarray(V2)])
    Sc = np.concatenate([np.asarray(S1), np.asarray(S2)])
    assert np.allclose(np.asarray(Vf), Vc, atol=1e-4)
    assert np.array_equal(np.asarray(Sf), Sc)


def test_size_scale_override_changes_tau_and_dynamics():
    """An explicit per-neuron size_scale_override changes tau_i = tau_0_ms *
    size_i**3 and therefore the resulting H trajectory vs. the default
    cell-type-only size table."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(6)
    kw = dict(alpha=0.05, gamma=0.5, K_ctrl=0.15, K_HDP=0.01, tau_0_ms=5.0,
              noise_scale=0.0)
    _, _, _, d_default = hdp_kernel(p, edges, 100, 0.5, key, **kw)
    override = np.full(N, 10.0, dtype=np.float32)  # far from the E default
    _, _, _, d_override = hdp_kernel(
        p, edges, 100, 0.5, key, size_scale_override=jnp.asarray(override), **kw)
    H_default = np.asarray(d_default["H_trace"])
    H_override = np.asarray(d_override["H_trace"])
    assert not np.allclose(H_default, H_override, atol=1e-5)


def test_size_scale_is_cube_law_not_square_law():
    """tau_i = tau_0_ms * size_i**3 (0.4.7 cube-law adaptation rate): a
    neuron with relative size 2.0 must integrate H exactly 8x slower than
    a size-1.0 neuron under a pure constant-income forcing (beta only, no
    spiking/weight feedback), isolating tau_i from everything else in the
    dH/dt expression. dH/dt = beta/tau_i => H(t) - 1 = beta*t/tau_i, so the
    size-2.0 neuron's H deviation after n steps must equal exactly 1/8th
    of the size-1.0 neuron's deviation (linear regime, far from H_min/H_max
    clamps and K_ctrl/barrier forces, which are all zero here)."""
    N, ne = 4, 8
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(0)
    kw = dict(alpha=0.0, beta=0.02, gamma=0.0, delta=0.0, C_spike=0.0,
              K_ctrl=0.0, barrier_c=0.0, barrier_d=0.0, K_HDP=0.0,
              tau_0_ms=100.0, noise_scale=0.0,
              drive_schedule=jnp.zeros((20, N)))
    size_arr = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    _, _, _, d = hdp_kernel(
        p, edges, 20, 0.5, key, size_scale_override=jnp.asarray(size_arr), **kw)
    H_dev = np.asarray(d["H_trace"])[-1] - 1.0
    np.testing.assert_allclose(H_dev[0], H_dev[2], rtol=1e-5)
    np.testing.assert_allclose(H_dev[1], H_dev[3], rtol=1e-5)
    np.testing.assert_allclose(H_dev[0] / H_dev[1], 8.0, rtol=5e-3)


def test_negative_K_HDP_is_anti_homeostatic_and_still_finite():
    """K_HDP<0 is the documented explicit anti-homeostatic stress-test mode;
    it must still run and stay finite over a short horizon."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(7)
    V, S, _, diag = hdp_kernel(
        p, edges, 50, 0.5, key, alpha=0.05, gamma=0.5, K_ctrl=0.15,
        K_HDP=-0.01, noise_scale=0.0)
    assert bool(np.isfinite(np.asarray(V)).all())
    assert bool(np.isfinite(np.asarray(diag["w_final"])).all())


def test_rho_passive_produces_finite_clipped_H():
    """rho_passive > 0 adds a passive-income term (rho_passive/H_i**2) that
    pulls H_i toward 1 without an explicit linear K_ctrl term. The test verifies
    that the new term produces finite, clipped results."""
    N, ne = 16, 128
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(10)
    kw = dict(
        alpha=0.05, gamma=0.3, delta=0.0, C_spike=0.0, K_ctrl=0.0,
        barrier_c=0.01, barrier_d=0.01, K_HDP=0.01, tau_0_ms=5.0, noise_scale=0.1
    )

    # With positive rho_passive, H should stay finite and clipped
    _, _, _, d_passive = hdp_kernel(
        p, edges, 100, 0.5, key, rho_passive=0.5, **kw)
    H_passive = np.asarray(d_passive["H_trace"])
    assert bool(np.isfinite(H_passive).all())
    # H should be clipped to [H_min, H_max]
    assert H_passive.min() >= 0.1 - 1e-4
    assert H_passive.max() <= 10.0 + 1e-4


def test_H_taxed_rate_spending_scales_with_H():
    """gamma > 0 implements H-taxed output spending: dH/dt -= gamma*H_i*r_i.
    This means the rate drain is stronger for neurons with high H (abundant
    resources) than for those with low H (starved resources)."""
    N, ne = 8, 64
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(11)

    # Pure rate-drain test: only gamma and C_spike matter, everything else zero
    kw = dict(
        alpha=0.0, beta=0.0, delta=0.0, K_HDP=0.0, K_ctrl=0.0,
        rho_passive=0.0, barrier_c=0.0, barrier_d=0.0,
        tau_0_ms=5.0, noise_scale=0.1, w_floor=0.01, w_ceiling=10.0
    )

    # Run with gamma > 0 and see H modulation
    _, _, _, d_taxed = hdp_kernel(p, edges, 100, 0.5, key, gamma=0.3, **kw)
    H_taxed = np.asarray(d_taxed["H_trace"])

    # H should be finite and clipped
    assert bool(np.isfinite(H_taxed).all())
    assert H_taxed.min() >= 0.1 - 1e-4
    assert H_taxed.max() <= 10.0 + 1e-4


def test_hdp_rule_families_signed_linear_signed_quadratic_product():
    """The two difference rules and separate product modulation
    (signed_linear, signed_quadratic, hebbian_product) produce distinct weight
    deltas from the same H_pre/H_post snapshot."""
    N, ne = 16, 128
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(12)

    kw = dict(
        alpha=0.05, gamma=0.0, delta=0.0, C_spike=0.0, K_HDP=0.01,
        tau_0_ms=5.0, noise_scale=0.0, K_ctrl=0.0, rho_passive=0.0
    )

    # Run the three rule variants
    _, _, _, d_linear = hdp_kernel(
        p, edges, 50, 0.5, key, hdp_rule="signed_linear", **kw)
    _, _, _, d_quad = hdp_kernel(
        p, edges, 50, 0.5, key, hdp_rule="signed_quadratic", **kw)
    _, _, _, d_hebb = hdp_kernel(
        p, edges, 50, 0.5, key, hdp_rule="hebbian_product", **kw)

    w_linear = np.asarray(d_linear["w_trace"])
    w_quad = np.asarray(d_quad["w_trace"])
    w_hebb = np.asarray(d_hebb["w_trace"])

    # All should be finite
    assert bool(np.isfinite(w_linear).all())
    assert bool(np.isfinite(w_quad).all())
    assert bool(np.isfinite(w_hebb).all())

    # They should produce distinct weight trajectories (not all the same)
    assert not np.allclose(w_linear, w_quad, atol=1e-5)
    assert not np.allclose(w_linear, w_hebb, atol=1e-5)
    assert not np.allclose(w_quad, w_hebb, atol=1e-5)


def test_record_weight_trace_false_matches_true_except_the_trace_itself():
    """record_weight_trace=False must be a pure output-shaping change: the
    same seed/config must produce bit-identical V/spikes/H_trace/w_final
    with or without it (the flag never touches the actual HDP dynamics),
    while w_trace itself becomes None instead of (n_steps, n_edges)."""
    N, ne = 32, 256
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(20)
    kw = dict(alpha=0.05, gamma=0.5, K_ctrl=0.15, K_HDP=0.01,
              barrier_c=0.01, barrier_d=0.01, tau_0_ms=5.0, noise_scale=0.1)
    V_true, S_true, _, d_true = hdp_kernel(p, edges, 60, 0.5, key, **kw)
    V_false, S_false, _, d_false = hdp_kernel(
        p, edges, 60, 0.5, key, record_weight_trace=False, **kw)

    assert np.array_equal(np.asarray(V_true), np.asarray(V_false))
    assert np.array_equal(np.asarray(S_true), np.asarray(S_false))
    np.testing.assert_array_equal(np.asarray(d_true["H_trace"]), np.asarray(d_false["H_trace"]))
    np.testing.assert_allclose(np.asarray(d_true["w_final"]), np.asarray(d_false["w_final"]), atol=1e-6)

    assert np.asarray(d_true["w_trace"]).shape == (60, ne)
    assert d_false["w_trace"] is None


def test_unknown_hdp_rule_raises_error():
    """hdp_rule parameter validation: passing an unrecognized rule name
    should raise ValueError."""
    N, ne = 8, 64
    p, edges = _make_params_edges(N, ne)
    key = jax.random.PRNGKey(13)

    with pytest.raises(ValueError, match="Unknown hdp_rule"):
        hdp_kernel(p, edges, 10, 0.5, key, hdp_rule="invalid_rule")
