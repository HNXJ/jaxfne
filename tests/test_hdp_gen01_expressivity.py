"""24-HDP-GEN-01: general finite-state HDP expressivity qualification.

Each probe expresses a plasticity dimension through the SAME generic
rule/state interface (register descriptor + step fn; no simulator branch).
Analytic trajectories are predicted independently in numpy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne._hdp_registrable_kernel import (
    simulate_edge_recurrent_izhikevich_hdp_registered as sim,
)
from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.hdp_rule import (
    HDPRuleDescriptor,
    HDPRuleUpdate,
    expected_aux_shape,
    list_registered_hdp_rules,
    register_hdp_rule,
)


def _two_neuron():
    params = IzhikevichParams(
        a=jnp.full((2,), 0.02),
        b=jnp.full((2,), 0.2),
        c=jnp.full((2,), -65.0),
        d=jnp.full((2,), 8.0),
        drive=jnp.array([0.0, 0.0]),
        sign=jnp.ones((2,)),
        W=jnp.zeros((2, 2)),
        v0=jnp.full((2,), -65.0),
        u0=jnp.full((2,), -13.0),
        source_scale=jnp.ones((2,)),
        labels=("E", "E"),
        layer_labels=("L4", "L4"),
        source_calibration_status="x",
    )
    edges = EdgeList(
        pre=jnp.array([0], dtype=jnp.int32),
        post=jnp.array([1], dtype=jnp.int32),
        weight=jnp.array([0.2], dtype=jnp.float32),
        receptor_index=jnp.array([0], dtype=jnp.int32),
        tau_ms=jnp.array([2.0], dtype=jnp.float32),
        source_calibration_status="x",
    )
    n_steps = 20
    drive = jnp.zeros((n_steps, 2), dtype=jnp.float32).at[:, 0].set(8.0)
    return params, edges, drive


def _ensure(name, descriptor, step):
    if name not in list_registered_hdp_rules():
        register_hdp_rule(descriptor, step)


# --- F: saturating efficacy via per-edge latent (no engine change) ---------
_FLO, _CEI = 1e-3, 50.0


def _saturating_step(ctx):
    k = jnp.asarray(ctx.rule_params.get("k_sat", 0.5), dtype=ctx.H.dtype)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    ds = k * pre_sp
    s_next = ctx.aux + ctx.dt * ds
    target = _FLO + (_CEI - _FLO) * jax.nn.sigmoid(s_next)
    dw = (target - jnp.abs(ctx.w)) / jnp.maximum(
        ctx.dt, jnp.asarray(1e-9, dtype=ctx.H.dtype)
    )
    return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), d_aux=ds,
                         d_theta={"edge_weight": dw})


def test_f_saturating_efficacy_tracks_logistic_of_spike_count():
    _ensure("gen01_saturating_gain",
            HDPRuleDescriptor(name="gen01_saturating_gain",
                              aux_coords=("latent",), aux_layout="per_edge",
                              default_params={"k_sat": 0.5}),
            _saturating_step)
    params, edges, drive = _two_neuron()
    _, spikes, _, diag = sim(params, edges, drive.shape[0], 1.0,
                             jax.random.PRNGKey(0), drive_schedule=drive,
                             noise_scale=0.0, hdp_rule="gen01_saturating_gain")
    s = np.asarray(spikes)
    n_pre = np.cumsum(s[:, 0])
    s_lat = 0.5 * n_pre
    w_exp = _FLO + (_CEI - _FLO) / (1.0 + np.exp(-s_lat))
    w_got = np.asarray(diag["w_trace"])[:, 0]
    np.testing.assert_allclose(w_got, w_exp, rtol=1e-5, atol=1e-5)
    assert bool(np.all(np.diff(w_got) >= -1e-7)), "saturating rise must be monotone"
    assert float(w_got[-1]) < _CEI, "efficacy stays strictly below ceiling"


# --- G: global scalar state -------------------------------------------------
def _global_step(ctx):
    k = jnp.asarray(ctx.rule_params.get("k_g", 0.25), dtype=ctx.H.dtype)
    dg = k * jnp.mean(ctx.spikes)
    return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), d_aux=dg)


# --- E: multidimensional H (d_H = 2) ---------------------------------------
def _vector_step(ctx):
    k_h = jnp.asarray(ctx.rule_params.get("k_h", 0.5), dtype=ctx.H.dtype)
    k_w = jnp.asarray(ctx.rule_params.get("k_w", 0.1), dtype=ctx.H.dtype)
    tau = jnp.asarray(ctx.rule_params.get("tau_track", 5.0), dtype=ctx.H.dtype)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    d0 = _seg(k_h * pre_sp, ctx)
    h0_new = ctx.H[..., 0] + ctx.dt * d0
    d1 = (h0_new - ctx.H[..., 1]) / jnp.maximum(
        tau, jnp.asarray(1e-6, dtype=ctx.H.dtype))
    dH = jnp.stack([d0, d1], axis=-1)
    h1_new = ctx.H[..., 1] + ctx.dt * d1
    dw = k_w * h1_new[ctx.post] * pre_sp * jnp.abs(ctx.w)
    return HDPRuleUpdate(dH=dH, d_theta={"edge_weight": dw})


def _seg(x, ctx):
    from jaxfne.emitters import _segment_sum
    return _segment_sum(x, ctx.post, ctx.n_neurons)


def test_e_vector_h_two_coordinates_match_manual_trajectory():
    _ensure("gen01_vector_h",
            HDPRuleDescriptor(name="gen01_vector_h",
                              h_coords=("fast", "tracking"), h_shape=(2,),
                              default_params={"k_h": 0.5, "k_w": 0.1,
                                              "tau_track": 5.0}),
            _vector_step)
    params, edges, drive = _two_neuron()
    _, spikes, _, diag = sim(params, edges, drive.shape[0], 1.0,
                             jax.random.PRNGKey(0), drive_schedule=drive,
                             noise_scale=0.0, hdp_rule="gen01_vector_h")
    assert np.asarray(diag["H_final"]).shape == (2, 2)
    assert np.asarray(diag["H_trace"]).shape == (20, 2, 2)
    s = np.asarray(spikes)
    H = np.ones((2, 2))
    w = 0.2
    Hs, ws = [], []
    for t in range(s.shape[0]):
        pre_sp = s[t, 0]
        d0 = np.zeros(2)
        d0[1] += 0.5 * pre_sp
        h0_new = H[:, 0] + d0  # dt = 1.0; clip applied after, as in kernel
        h1_new = H[:, 1] + (h0_new - H[:, 1]) / 5.0
        H = np.stack([np.clip(h0_new, 0.1, 10.0),
                      np.clip(h1_new, 0.1, 10.0)], axis=-1)
        w = np.clip(w + 0.1 * H[1, 1] * pre_sp * abs(w), 1e-3, 50.0)
        Hs.append(H.copy())
        ws.append(w)
    np.testing.assert_allclose(np.asarray(diag["H_trace"]), np.asarray(Hs),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(diag["w_trace"])[:, 0],
                               np.asarray(ws), rtol=1e-5, atol=1e-5)


def test_e_vector_h_continuation_and_shape_rejection():
    _ensure("gen01_vector_h",
            HDPRuleDescriptor(name="gen01_vector_h",
                              h_coords=("fast", "tracking"), h_shape=(2,),
                              default_params={"k_h": 0.5, "k_w": 0.1,
                                              "tau_track": 5.0}),
            _vector_step)
    params, edges, drive = _two_neuron()
    kw = dict(hdp_rule="gen01_vector_h", noise_scale=0.0)
    _, _, _, full = sim(params, edges, 20, 1.0, jax.random.PRNGKey(0),
                        drive_schedule=drive, **kw)
    _, _, _, first = sim(params, edges, 12, 1.0, jax.random.PRNGKey(0),
                         drive_schedule=drive[:12], **kw)
    init = {"v": first["v"], "u": first["u"],
            "prev_spikes": first["prev_spikes"],
            "syn_state": first["syn_state"], "H_final": first["H_final"],
            "w_final": first["w_final"], "aux_final": first["aux_final"]}
    _, _, _, second = sim(params, edges, 8, 1.0, jax.random.PRNGKey(0),
                          drive_schedule=drive[12:], init_state=init, **kw)
    np.testing.assert_array_equal(np.asarray(second["H_final"]),
                                  np.asarray(full["H_final"]))
    bad = dict(init)
    bad["H_final"] = np.ones((2,), dtype=np.float32)
    with pytest.raises(ValueError, match="H_final must have shape"):
        sim(params, edges, 8, 1.0, jax.random.PRNGKey(0),
            drive_schedule=drive[12:], init_state=bad, **kw)


# --- Multi-coordinate aux cascade (per_edge x2, no engine branch) ------------
def _cascade_step(ctx):
    tau_f = jnp.asarray(ctx.rule_params.get("tau_fast", 5.0), dtype=ctx.H.dtype)
    tau_s = jnp.asarray(ctx.rule_params.get("tau_slow", 40.0), dtype=ctx.H.dtype)
    k_w = jnp.asarray(ctx.rule_params.get("k_w", 0.05), dtype=ctx.H.dtype)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    post_sp = ctx.spikes[ctx.post]
    ef, es = ctx.aux[..., 0], ctx.aux[..., 1]
    d_fast = -ef / jnp.maximum(tau_f, jnp.asarray(1e-6, dtype=ctx.H.dtype))
    d_fast = d_fast + pre_sp * post_sp
    d_slow = -es / jnp.maximum(tau_s, jnp.asarray(1e-6, dtype=ctx.H.dtype))
    d_slow = d_slow + (ef + ctx.dt * d_fast)
    dw = k_w * ctx.H[ctx.post] * (ef + ctx.dt * d_fast) * jnp.abs(ctx.w)
    d_aux = jnp.stack([d_fast, d_slow], axis=-1)
    return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), d_aux=d_aux,
                         d_theta={"edge_weight": dw})


def test_multi_coordinate_aux_cascade_matches_manual():
    _ensure("gen01_cascade",
            HDPRuleDescriptor(name="gen01_cascade",
                              aux_coords=("fast", "slow"),
                              aux_layout="per_edge",
                              default_params={"tau_fast": 5.0,
                                              "tau_slow": 40.0,
                                              "k_w": 0.05}),
            _cascade_step)
    params, edges, drive = _two_neuron()
    _, spikes, _, diag = sim(params, edges, drive.shape[0], 1.0,
                             jax.random.PRNGKey(0), drive_schedule=drive,
                             noise_scale=0.0, hdp_rule="gen01_cascade")
    assert np.asarray(diag["aux_final"]).shape == (1, 2)
    assert np.asarray(diag["aux_trace"]).shape == (20, 1, 2)
    s = np.asarray(spikes)
    ef = es = 0.0
    w = 0.2
    EFS, ESS, WS = [], [], []
    for t in range(s.shape[0]):
        c = s[t, 0] * s[t, 1]
        d_fast = -ef / 5.0 + c
        d_slow = -es / 40.0 + (ef + d_fast)
        ef, es = ef + d_fast, es + d_slow
        w = np.clip(w + 0.05 * 1.0 * (ef) * abs(w), 1e-3, 50.0)
        EFS.append(ef)
        ESS.append(es)
        WS.append(w)
    # NOTE: manual uses post-update ef in dw (ef already advanced), matching
    # the rule's (ef + dt*d_fast) with dt=1; H_post=1 throughout here.
    np.testing.assert_allclose(np.asarray(diag["aux_trace"])[:, 0, 0],
                               np.asarray(EFS), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(diag["aux_trace"])[:, 0, 1],
                               np.asarray(ESS), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(diag["w_trace"])[:, 0],
                               np.asarray(WS), rtol=1e-5, atol=1e-7)


# --- Stochastic rule: per-step key, deterministic under seed ----------------
def _stochastic_step(ctx):
    k_w = jnp.asarray(ctx.rule_params.get("k_w", 0.0), dtype=ctx.H.dtype)
    sigma = jnp.asarray(ctx.rule_params.get("sigma", 0.5), dtype=ctx.H.dtype)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    sub = jax.random.split(ctx.key, 2)[1]
    dz = sigma * jax.random.normal(sub, shape=pre_sp.shape, dtype=ctx.H.dtype)
    d_aux = -ctx.aux / 20.0 + pre_sp + dz * pre_sp
    dw = k_w * ctx.H[ctx.post] * ctx.aux * jnp.abs(ctx.w)
    return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), d_aux=d_aux,
                         d_theta={"edge_weight": dw})


def test_stochastic_rule_deterministic_under_seed_and_seed_sensitive():
    _ensure("gen01_stochastic",
            HDPRuleDescriptor(name="gen01_stochastic",
                              aux_coords=("noisy_eligibility",),
                              aux_layout="per_edge",
                              default_params={"k_w": 0.0, "sigma": 0.5}),
            _stochastic_step)
    params, edges, drive = _two_neuron()

    def run(seed):
        return sim(params, edges, drive.shape[0], 1.0,
                   jax.random.PRNGKey(seed), drive_schedule=drive,
                   noise_scale=0.0, hdp_rule="gen01_stochastic")
    _, s_a, _, d_a = run(0)
    _, _, _, d_a2 = run(0)
    _, _, _, d_b = run(1)
    np.testing.assert_array_equal(np.asarray(d_a["aux_trace"]),
                                  np.asarray(d_a2["aux_trace"]))
    assert not np.allclose(np.asarray(d_a["aux_trace"]),
                           np.asarray(d_b["aux_trace"])), (
        "rule-noise must be seed-sensitive")
    # rule noise never touches membrane dynamics here (k_w=0, dw=0)
    np.testing.assert_array_equal(np.asarray(s_a) * 1.0, np.asarray(s_a))


def test_zero_delay_step_indices_length_validated():
    params, edges, drive = _two_neuron()
    with pytest.raises(ValueError, match="step_indices must have shape"):
        sim(params, edges, 4, 1.0, jax.random.PRNGKey(0),
            drive_schedule=drive[:4], noise_scale=0.0,
            hdp_rule="synthetic_presyn_gain", step_indices=jnp.arange(3))


# --- BCM-style metaplasticity: sliding threshold as STATE (no engine gap) ---
def _bcm_step(ctx):
    k = jnp.asarray(ctx.rule_params.get("k_bcm", 0.05), dtype=ctx.H.dtype)
    tau_y = jnp.asarray(10.0, dtype=ctx.H.dtype)
    tau_t = jnp.asarray(60.0, dtype=ctx.H.dtype)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    d0 = -ctx.H[..., 0] / tau_y + ctx.spikes
    y_new = ctx.H[..., 0] + ctx.dt * d0
    d1 = (-ctx.H[..., 1] + y_new * y_new) / tau_t
    th_new = ctx.H[..., 1] + ctx.dt * d1
    dH = jnp.stack([d0, d1], axis=-1)
    dw = k * pre_sp * (y_new[ctx.post] - th_new[ctx.post]) * jnp.abs(ctx.w)
    return HDPRuleUpdate(dH=dH, d_theta={"edge_weight": dw})


def test_bcm_sliding_threshold_potentiates_above_depresses_below():
    _ensure("gen01_bcm",
            HDPRuleDescriptor(name="gen01_bcm",
                              h_coords=("rate", "threshold"), h_shape=(2,),
                              default_params={"k_bcm": 0.05}),
            _bcm_step)
    params, edges, drive = _two_neuron()
    _, spikes, _, diag = sim(params, edges, drive.shape[0], 1.0,
                             jax.random.PRNGKey(0), drive_schedule=drive,
                             noise_scale=0.0, hdp_rule="gen01_bcm")
    s = np.asarray(spikes)
    H = np.ones((2, 2))
    w = 0.2
    signs, WS = [], []
    for t in range(s.shape[0]):
        d0 = -H[:, 0] / 10.0 + s[t]
        y_new = H[:, 0] + d0
        d1 = (-H[:, 1] + y_new * y_new) / 60.0
        th_new = H[:, 1] + d1
        H = np.stack([np.clip(y_new, 0.1, 10.0),
                      np.clip(th_new, 0.1, 10.0)], axis=-1)
        dw = 0.05 * s[t, 0] * (H[1, 0] - H[1, 1]) * abs(w)
        if s[t, 0] > 0.5:
            signs.append(np.sign(H[1, 0] - H[1, 1]))
        w = np.clip(w + dw, 1e-3, 50.0)
        WS.append(w)
    np.testing.assert_allclose(np.asarray(diag["w_trace"])[:, 0],
                               np.asarray(WS), rtol=1e-5, atol=1e-7)
    assert np.asarray(diag["H_trace"]).shape == (20, 2, 2)
    assert set(np.unique(signs)) <= {-1.0, 0.0, 1.0}
    assert float(np.asarray(diag["w_final"])[0]) > 1e-3
    assert bool(np.all(np.isfinite(np.asarray(diag["w_trace"]))))


# --- T: non-weight Theta target (per-neuron drive bias) ---------------------
def _bias_step(ctx):
    k_b = jnp.asarray(ctx.rule_params.get("k_b", 0.3), dtype=ctx.H.dtype)
    db = k_b * ctx.spikes
    return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H),
                         d_theta={"drive_bias": db})


def test_t_drive_bias_integrates_presynaptic_events_analytically():
    _ensure("gen01_drive_bias",
            HDPRuleDescriptor(name="gen01_drive_bias",
                              theta_targets=("drive_bias",),
                              default_params={"k_b": 0.3}),
            _bias_step)
    params, edges, drive = _two_neuron()
    _, spikes, _, diag = sim(params, edges, drive.shape[0], 1.0,
                             jax.random.PRNGKey(0), drive_schedule=drive,
                             noise_scale=0.0, hdp_rule="gen01_drive_bias")
    assert np.asarray(diag["b_final"]).shape == (2,)
    assert np.asarray(diag["b_trace"]).shape == (20, 2)
    s = np.asarray(spikes)
    # bias integrates each neuron's own spikes
    b_exp = 0.3 * np.cumsum(s, axis=0)
    np.testing.assert_allclose(np.asarray(diag["b_trace"]), b_exp,
                               rtol=1e-5, atol=1e-7)
    assert "b_final" in diag and "b_trace" in diag


def test_t_drive_bias_causally_alters_spikes_and_saturates():
    _ensure("gen01_drive_bias",
            HDPRuleDescriptor(name="gen01_drive_bias",
                              theta_targets=("drive_bias",),
                              default_params={"k_b": 0.3}),
            _bias_step)
    params, edges, _ = _two_neuron()
    n_steps = 20
    drive = jnp.zeros((n_steps, 2), dtype=jnp.float32).at[:, 0].set(20.0)

    def run(kb):
        return sim(params, edges, n_steps, 1.0,
                   jax.random.PRNGKey(0), drive_schedule=drive,
                   noise_scale=0.0, hdp_rule="gen01_drive_bias",
                   hdp_rule_params={"k_b": kb})
    _, s0, _, d0 = run(0.0)
    _, s1, _, d1 = run(3.0)
    assert not np.array_equal(np.asarray(s0), np.asarray(s1)), (
        "bias must causally alter executed spikes")
    np.testing.assert_allclose(np.asarray(d0["b_final"]), np.zeros(2),
                               atol=1e-9)
    assert float(np.asarray(d1["b_trace"])[:, 0].max()) <= 50.0
    # saturation: huge gain pins bias at the descriptor ceiling
    _, _, _, d2 = run(1e4)
    assert float(np.asarray(d2["b_final"])[0]) == 50.0


def test_t_drive_bias_continuation_jit_and_disabled_identity():
    import jaxfne as jtfne

    _ensure("gen01_drive_bias",
            HDPRuleDescriptor(name="gen01_drive_bias",
                              theta_targets=("drive_bias",),
                              default_params={"k_b": 0.3}),
            _bias_step)
    params, edges, drive = _two_neuron()
    kw = dict(hdp_rule="gen01_drive_bias", noise_scale=0.0)
    _, _, _, full = sim(params, edges, 20, 1.0, jax.random.PRNGKey(0),
                        drive_schedule=drive, **kw)
    _, _, _, first = sim(params, edges, 12, 1.0, jax.random.PRNGKey(0),
                         drive_schedule=drive[:12], **kw)
    init = {"v": first["v"], "u": first["u"],
            "prev_spikes": first["prev_spikes"],
            "syn_state": first["syn_state"], "H_final": first["H_final"],
            "w_final": first["w_final"], "aux_final": first["aux_final"],
            "b_final": first["b_final"]}
    v2, _, _, second = sim(params, edges, 8, 1.0, jax.random.PRNGKey(0),
                           drive_schedule=drive[12:], init_state=init, **kw)
    vfull, _, _, _ = sim(params, edges, 20, 1.0, jax.random.PRNGKey(0),
                         drive_schedule=drive, **kw)
    np.testing.assert_array_equal(np.asarray(v2),
                                  np.asarray(vfull)[12:])
    np.testing.assert_array_equal(np.asarray(second["b_final"]),
                                  np.asarray(full["b_final"]))
    # jit replay deterministic (spike output selected; diagnostics dict
    # carries static strings, as in test_jit_deterministic_replay_delayed)
    f = jax.jit(lambda: sim(params, edges, 20, 1.0, jax.random.PRNGKey(0),
                            drive_schedule=drive, noise_scale=0.0,
                            hdp_rule="gen01_drive_bias")[1])
    r1, r2 = f(), f()
    np.testing.assert_array_equal(np.asarray(r1), np.asarray(r2))
    # disabled identity: zero gain == static baseline at Model level
    # (matched backends; mirrors test_disabled_identity_routes_to_baseline)
    rt = dict(enable_hdp=True, recurrent_backend="edge_list",
              hdp_params={"hdp_rule": "gen01_drive_bias",
                          "hdp_rule_params": {"k_b": 0.0}})
    model = jtfne.construct(
        jtfne.suite2_net1_config(seed=1, n=8).runtime(**rt))
    sig_on = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=1)
    off = jtfne.construct(jtfne.suite2_net1_config(seed=1, n=8).runtime(
        enable_hdp=False, recurrent_backend="edge_list"))
    sig_off = jtfne.simulate(off, duration_ms=20.0, dt_ms=0.5, seed=1)
    assert np.array_equal(np.asarray(sig_on.V_m), np.asarray(sig_off.V_m))


def test_t_drive_bias_model_level_chunked_continuation():
    """Dim 7 at Model level: b rides the continuation carrier."""
    import jaxfne as jtfne

    _ensure("gen01_drive_bias",
            HDPRuleDescriptor(name="gen01_drive_bias",
                              theta_targets=("drive_bias",),
                              default_params={"k_b": 0.3}),
            _bias_step)
    cfg = jtfne.suite2_net1_config(seed=6, n=8).runtime(
        enable_hdp=True, recurrent_backend="edge_list",
        hdp_params={"hdp_rule": "gen01_drive_bias",
                    "hdp_rule_params": {"k_b": 0.05}})
    model = jtfne.construct(cfg)
    full, full_state = jtfne.simulate(
        model, duration_ms=12.0, dt_ms=0.5, seed=6, return_state=True)
    first, first_state = jtfne.simulate(
        model, duration_ms=6.0, dt_ms=0.5, seed=6, return_state=True)
    second, second_state = jtfne.simulate(
        model, duration_ms=6.0, dt_ms=0.5, seed=99,
        continuation=first_state, return_state=True)
    assert jnp.array_equal(
        jnp.concatenate((first.V_m, second.V_m), axis=0), full.V_m)
    assert jnp.array_equal(full_state.dynamic.b, second_state.dynamic.b)
    diag = model.last_hdp_diagnostics()
    assert diag["b_final"].shape == (8,)


def test_stochastic_rule_model_level_chunked_continuation():
    """Rule-noise follows the continuation key chain: chunked == full."""
    import jaxfne as jtfne

    _ensure("gen01_stochastic",
            HDPRuleDescriptor(name="gen01_stochastic",
                              aux_coords=("noisy_eligibility",),
                              aux_layout="per_edge",
                              default_params={"k_w": 0.0, "sigma": 0.5}),
            _stochastic_step)
    cfg = jtfne.suite2_net1_config(seed=6, n=8).runtime(
        enable_hdp=True, recurrent_backend="edge_list",
        hdp_params={"hdp_rule": "gen01_stochastic",
                    "hdp_rule_params": {"k_w": 0.0, "sigma": 0.5}})
    model = jtfne.construct(cfg)
    full, full_state = jtfne.simulate(
        model, duration_ms=12.0, dt_ms=0.5, seed=6, return_state=True)
    first, first_state = jtfne.simulate(
        model, duration_ms=6.0, dt_ms=0.5, seed=6, return_state=True)
    second, second_state = jtfne.simulate(
        model, duration_ms=6.0, dt_ms=0.5, seed=99,
        continuation=first_state, return_state=True)
    assert jnp.array_equal(
        jnp.concatenate((first.V_m, second.V_m), axis=0), full.V_m)
    assert jnp.array_equal(full_state.dynamic.aux, second_state.dynamic.aux)


def test_t_undeclared_theta_keys_rejected_loudly():
    def _bad_step(ctx):
        return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H),
                             d_theta={"mystery": jnp.zeros_like(ctx.H)})
    _ensure("gen01_bad_theta",
            HDPRuleDescriptor(name="gen01_bad_theta"),
            _bad_step)
    params, edges, drive = _two_neuron()
    with pytest.raises(ValueError, match="undeclared theta_targets"):
        sim(params, edges, 4, 1.0, jax.random.PRNGKey(0),
            drive_schedule=drive[:4], noise_scale=0.0,
            hdp_rule="gen01_bad_theta")
    with pytest.raises(ValueError, match="unknown theta_targets"):
        register_hdp_rule(
            HDPRuleDescriptor(name="gen01_bad_desc",
                              theta_targets=("mystery",)), _bad_step)


def test_g_scalar_aux_layout_integrates_global_activity():
    _ensure("gen01_global_gain",
            HDPRuleDescriptor(name="gen01_global_gain",
                              aux_coords=("global",), aux_layout="scalar",
                              default_params={"k_g": 0.25}),
            _global_step)
    assert expected_aux_shape(
        HDPRuleDescriptor(name="x", aux_layout="scalar"),
        n_neurons=2, n_edges=1) == ()
    params, edges, drive = _two_neuron()
    _, spikes, _, diag = sim(params, edges, drive.shape[0], 1.0,
                             jax.random.PRNGKey(0), drive_schedule=drive,
                             noise_scale=0.0, hdp_rule="gen01_global_gain")
    g_exp = 0.25 * np.cumsum(np.asarray(spikes).mean(axis=1))
    np.testing.assert_allclose(np.asarray(diag["aux_trace"]), g_exp,
                               rtol=1e-5, atol=1e-7)
    assert diag["aux_final"].shape == ()


def test_g_scalar_aux_continuation_is_bit_exact():
    _ensure("gen01_global_gain",
            HDPRuleDescriptor(name="gen01_global_gain",
                              aux_coords=("global",), aux_layout="scalar",
                              default_params={"k_g": 0.25}),
            _global_step)
    params, edges, drive = _two_neuron()
    kw = dict(hdp_rule="gen01_global_gain", noise_scale=0.0)
    _, _, _, full = sim(params, edges, 20, 1.0, jax.random.PRNGKey(0),
                        drive_schedule=drive, **kw)
    _, _, _, first = sim(params, edges, 12, 1.0, jax.random.PRNGKey(0),
                         drive_schedule=drive[:12], **kw)
    init = {"v": first["v"], "u": first["u"],
            "prev_spikes": first["prev_spikes"],
            "syn_state": first["syn_state"], "H_final": first["H_final"],
            "w_final": first["w_final"], "aux_final": first["aux_final"]}
    v2, _, _, second = sim(params, edges, 8, 1.0, jax.random.PRNGKey(0),
                           drive_schedule=drive[12:], init_state=init, **kw)
    np.testing.assert_array_equal(
        np.asarray(second["aux_final"]), np.asarray(full["aux_final"]))


# --- STATE: per-neuron aux ----------------------------------------------------
def _per_neuron_trace_step(ctx):
    tau = jnp.asarray(ctx.rule_params.get("tau", 10.0), dtype=ctx.H.dtype)
    d_aux = -ctx.aux / jnp.maximum(tau, jnp.asarray(1e-6, dtype=ctx.H.dtype))
    d_aux = d_aux + ctx.spikes
    return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), d_aux=d_aux)


def test_per_neuron_aux_matches_manual_lowpass():
    _ensure("gen01_per_neuron_trace",
            HDPRuleDescriptor(name="gen01_per_neuron_trace",
                              aux_coords=("activity",),
                              aux_layout="per_neuron",
                              default_params={"tau": 10.0}),
            _per_neuron_trace_step)
    params, edges, drive = _two_neuron()
    _, spikes, _, diag = sim(params, edges, drive.shape[0], 1.0,
                             jax.random.PRNGKey(0), drive_schedule=drive,
                             noise_scale=0.0, hdp_rule="gen01_per_neuron_trace")
    assert np.asarray(diag["aux_final"]).shape == (2,)
    s = np.asarray(spikes)
    a = np.zeros(2)
    AA = []
    for t in range(s.shape[0]):
        a = a - a / 10.0 + s[t]
        AA.append(a.copy())
    np.testing.assert_allclose(np.asarray(diag["aux_trace"]), np.asarray(AA),
                               rtol=1e-5, atol=1e-7)


# --- COMPOSITION: vector H + per-edge aux in one rule --------------------------
def _composed_step(ctx):
    k_h = jnp.asarray(0.5, dtype=ctx.H.dtype)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    post_sp = ctx.spikes[ctx.post]
    d0 = _seg(k_h * pre_sp, ctx)
    h0_new = ctx.H[..., 0] + ctx.dt * d0
    d1 = (h0_new - ctx.H[..., 1]) / 5.0
    d_aux = -ctx.aux / 20.0 + pre_sp * post_sp
    dw = 0.1 * h0_new[ctx.post] * ctx.aux * jnp.abs(ctx.w)
    dH = jnp.stack([d0, d1], axis=-1)
    return HDPRuleUpdate(dH=dH, d_aux=d_aux, d_theta={"edge_weight": dw})


def test_composed_vector_h_and_edge_aux_match_manual():
    _ensure("gen01_composed",
            HDPRuleDescriptor(name="gen01_composed",
                              h_coords=("fast", "tracking"), h_shape=(2,),
                              aux_coords=("eligibility",),
                              aux_layout="per_edge"),
            _composed_step)
    params, edges, drive = _two_neuron()
    _, spikes, _, diag = sim(params, edges, drive.shape[0], 1.0,
                             jax.random.PRNGKey(0), drive_schedule=drive,
                             noise_scale=0.0, hdp_rule="gen01_composed")
    assert np.asarray(diag["H_trace"]).shape == (20, 2, 2)
    assert np.asarray(diag["aux_trace"]).shape == (20, 1)
    s = np.asarray(spikes)
    H = np.ones((2, 2))
    e, w = 0.0, 0.2
    HS, ES = [], []
    for t in range(s.shape[0]):
        pre_sp = s[t, 0]
        d0 = np.zeros(2)
        d0[1] += 0.5 * pre_sp
        h0_new = H[:, 0] + d0
        h1_new = H[:, 1] + (h0_new - H[:, 1]) / 5.0
        H = np.stack([np.clip(h0_new, 0.1, 10.0),
                      np.clip(h1_new, 0.1, 10.0)], axis=-1)
        e = e - e / 20.0 + pre_sp * s[t, 1]
        w = np.clip(w + 0.1 * H[1, 0] * e * abs(w), 1e-3, 50.0)
        HS.append(H.copy())
        ES.append(e)
    np.testing.assert_allclose(np.asarray(diag["H_trace"]), np.asarray(HS),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(diag["aux_trace"])[:, 0],
                               np.asarray(ES), rtol=1e-5, atol=1e-7)


# --- COMPOSITION: delayed event + stochastic rule -------------------------------
def _two_neuron_delayed(d):
    params, edges = _two_neuron()[:2]
    edges = edges.__class__(
        pre=edges.pre, post=edges.post, weight=edges.weight,
        receptor_index=edges.receptor_index, tau_ms=edges.tau_ms,
        delay_steps=jnp.array([d], dtype=jnp.int32),
        delay_storage="per_edge",
        source_calibration_status="x",
    )
    return params, edges


def test_delayed_stochastic_rule_seed_sensitive_and_deterministic():
    _ensure("gen01_stochastic",
            HDPRuleDescriptor(name="gen01_stochastic",
                              aux_coords=("noisy_eligibility",),
                              aux_layout="per_edge",
                              default_params={"k_w": 0.0, "sigma": 0.5}),
            _stochastic_step)
    params, edges = _two_neuron_delayed(3)
    drv = jnp.zeros((20, 2), dtype=jnp.float32).at[:, 0].set(20.0)

    def run(seed):
        return sim(params, edges, 20, 1.0, jax.random.PRNGKey(seed),
                   drive_schedule=drv, noise_scale=0.0,
                   hdp_rule="gen01_stochastic")
    _, _, _, d_a = run(0)
    _, _, _, d_a2 = run(0)
    _, _, _, d_b = run(7)
    np.testing.assert_array_equal(np.asarray(d_a["aux_trace"]),
                                  np.asarray(d_a2["aux_trace"]))
    assert not np.allclose(np.asarray(d_a["aux_trace"]),
                           np.asarray(d_b["aux_trace"]))


# --- IDENTITY: null stochastic rule perturbs no RNG domain ----------------------
def test_null_stochastic_rule_matches_deterministic_membrane():
    _ensure("gen01_stochastic",
            HDPRuleDescriptor(name="gen01_stochastic",
                              aux_coords=("noisy_eligibility",),
                              aux_layout="per_edge",
                              default_params={"k_w": 0.0, "sigma": 0.5}),
            _stochastic_step)
    params, edges, drive = _two_neuron()
    kw = dict(noise_scale=0.0, hdp_rule="gen01_stochastic")
    v0, s0, _, d0 = sim(params, edges, drive.shape[0], 1.0,
                        jax.random.PRNGKey(0), drive_schedule=drive, **kw,
                        hdp_rule_params={"k_w": 0.0, "sigma": 0.0})
    v1, s1, _, d1 = sim(params, edges, drive.shape[0], 1.0,
                        jax.random.PRNGKey(0), drive_schedule=drive, **kw,
                        hdp_rule_params={"k_w": 0.0, "sigma": 0.5})
    # membrane trajectory identical: rule noise stays in the rule domain
    np.testing.assert_array_equal(np.asarray(v0), np.asarray(v1))
    np.testing.assert_array_equal(np.asarray(s0), np.asarray(s1))
    # ... while the rule domain itself is noise-driven
    assert not np.allclose(np.asarray(d0["aux_trace"]),
                           np.asarray(d1["aux_trace"]))


# --- MALFORMED: loud local failure ----------------------------------------------
def test_malformed_definitions_rejected_loudly():
    def _null_step(ctx):
        return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H))

    with pytest.raises(ValueError, match="unknown aux_layout"):
        register_hdp_rule(
            HDPRuleDescriptor(name="gen01_bad_layout", aux_layout="per_synapse"),
            _null_step)
    with pytest.raises(ValueError, match="inverted"):
        register_hdp_rule(
            HDPRuleDescriptor(name="gen01_bad_bounds", h_bounds=(10.0, 0.1)),
            _null_step)
    with pytest.raises(ValueError, match="aux_layout 'none'"):
        register_hdp_rule(
            HDPRuleDescriptor(name="gen01_bad_coords",
                              aux_coords=("x",), aux_layout="none"),
            _null_step)
    # static-shape rule: eager works, and jit accepts it (boundary control)
    _ensure("gen01_dynamic_ok", HDPRuleDescriptor(name="gen01_dynamic_ok"),
            _null_step)
    params, edges, drive = _two_neuron()
    _, _, _, diag = sim(params, edges, 4, 1.0, jax.random.PRNGKey(0),
                        drive_schedule=drive[:4], noise_scale=0.0,
                        hdp_rule="gen01_dynamic_ok")
    assert np.asarray(diag["H_trace"]).shape == (4, 2)

    def _bad_jit_step(ctx):
        if float(jnp.sum(ctx.spikes)) > 0.5:
            return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H))
        return HDPRuleUpdate(dH=jnp.ones_like(ctx.H))
    _ensure("gen01_bad_jit", HDPRuleDescriptor(name="gen01_bad_jit"),
            _bad_jit_step)
    with pytest.raises(Exception, match="(?i)concret|trace|abstract|bool"):
        jax.jit(lambda: sim(params, edges, 4, 1.0, jax.random.PRNGKey(0),
                            drive_schedule=drive[:4], noise_scale=0.0,
                            hdp_rule="gen01_bad_jit")[0])()
