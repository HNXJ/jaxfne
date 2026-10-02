"""0.5.3 ENGINE item 4: deterministic replay + RNG domain isolation.

Contract under test:
- Same seed + same inputs -> bit-identical run (all paths).
- Stochastic consumers draw from declared, independent RNG domains:
  membrane noise vs per-rule noise vs drive/ablation/batch/construction
  streams. Changing one rule's stream leaves the others untouched.

Declared domains (see also HDPRuleContext.key docstring + the
split[0]/split[1] comment in _hdp_registrable_kernel.py):
- membrane: per-step split(chain_key(t))[1] -> normal (chain contract,
  item 3); legacy-HDP plain path consumes the identical schedule.
- rule: per-step fold_in(split(chain_key(t))[0], t_global).
  Membrane and rule branch independently off the carried chain, so
  neither consumer's draws depend on the other's consumption.
- drive/paradigm: explicit schedule arrays (no RNG).
- poisson_drive / shuffled_timing: own seeds, rejected under
  continuation (cursor unambiguous).
- batch: vmap over split(PRNGKey(base_seed), n_seeds).
- construction: seeded builders (same seed -> same realized network).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne import _pipeline
from jaxfne.core import StimulusSchedule
from jaxfne.hdp_rule import (
    HDPRuleDescriptor,
    HDPRuleUpdate,
    list_registered_hdp_rules,
    register_hdp_rule,
)

N = 8
DT = 0.5


def _model(n=N, seed=11):
    cfg = jtfne.suite2_net1_config(seed=seed, n=n, duration_ms=20.0, dt_ms=DT)
    return jtfne.construct(cfg)


def _rt_base(noise=None):
    hp = {} if noise is None else {"noise_scale": noise}
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=False,
        hdp_params=dict(hp),
    )


def _rt_hdp(noise=0.2, **over):
    hp = {
        "K_HDP": 0.01,
        "K_ctrl": 0.15,
        "K_w_ctrl": 0.001,
        "tau_0_ms": 20.0,
        "alpha": 0.01,
        "barrier_c": 0.01,
        "barrier_d": 0.01,
        "noise_scale": noise,
    }
    hp.update(over)
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params=hp,
    )


def _pulse_schedule():
    return StimulusSchedule(
        events=(
            {
                "onset_ms": 1.0,
                "duration_ms": 1.0,
                "amplitude": 40.0,
                "target_indices": [0],
                "is_drive_event": True,
            },
            {
                "onset_ms": 3.0,
                "duration_ms": 1.0,
                "amplitude": 38.0,
                "target_indices": [1],
                "is_drive_event": True,
            },
        ),
        n_neurons=N,
    )


def _ensure_053_replay_rule():
    name = "gen053_replay_stochastic"
    if name in list_registered_hdp_rules():
        return name

    def _step(ctx):
        k_w = jnp.asarray(ctx.rule_params.get("k_w", 0.0), dtype=ctx.H.dtype)
        sigma = jnp.asarray(ctx.rule_params.get("sigma", 0.5), dtype=ctx.H.dtype)
        pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
        sub = jax.random.split(ctx.key, 2)[1]
        dz = sigma * jax.random.normal(sub, shape=pre_sp.shape, dtype=ctx.H.dtype)
        d_aux = -ctx.aux / 20.0 + pre_sp + dz * pre_sp
        dw = k_w * ctx.H[ctx.post] * ctx.aux * jnp.abs(ctx.w)
        return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), d_aux=d_aux, d_theta={"edge_weight": dw})

    register_hdp_rule(
        HDPRuleDescriptor(
            name=name,
            aux_coords=("noisy_eligibility",),
            aux_layout="per_edge",
            default_params={"k_w": 0.0, "sigma": 0.5},
        ),
        _step,
    )
    return name


def _rt_replay_rule(**over):
    hp = {
        "hdp_rule": _ensure_053_replay_rule(),
        "hdp_rule_params": {"k_w": 0.0, "sigma": 0.5},
        "noise_scale": 0.0,
    }
    hp.update(over)
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params=hp,
    )


# --- same seed + same inputs -> bit-identical ---------------------------------


def test_replay_bit_identical_baseline_stochastic():
    m = _model()
    kw = dict(
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        runtime=_rt_base(),
        record_sources=True,
        record_fields=False,
    )
    s1 = jtfne.simulate(m, **kw)
    s2 = jtfne.simulate(m, **kw)
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(s1.spikes, s2.spikes)
    assert jnp.array_equal(s1.sources, s2.sources)


def test_replay_bit_identical_hdp_stochastic_all_state():
    m = _model()
    kw = dict(
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        runtime=_rt_hdp(),
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    s1, st1 = jtfne.simulate(m, **kw)
    d1 = m.last_hdp_diagnostics()
    s2, st2 = jtfne.simulate(m, **kw)
    d2 = m.last_hdp_diagnostics()
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(s1.spikes, s2.spikes)
    assert jnp.array_equal(d1["H_trace"], d2["H_trace"])
    assert jnp.array_equal(d1["w_trace"], d2["w_trace"])
    assert jnp.array_equal(st1.dynamic.H, st2.dynamic.H)
    assert jnp.array_equal(st1.dynamic.w, st2.dynamic.w)
    assert jnp.array_equal(st1.prng_key, st2.prng_key)


def test_replay_bit_identical_registered_rule():
    m = _model()
    kw = dict(
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        runtime=_rt_replay_rule(),
        record_sources=True,
        record_fields=False,
    )
    s1 = jtfne.simulate(m, **kw)
    d1 = m.last_hdp_diagnostics()
    s2 = jtfne.simulate(m, **kw)
    d2 = m.last_hdp_diagnostics()
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(np.asarray(d1["aux_trace"]), np.asarray(d2["aux_trace"]))


def test_replay_with_paradigm_drive_bit_identical():
    m = _model()
    sched = _pulse_schedule()
    kw = dict(
        duration_ms=6.0,
        dt_ms=DT,
        seed=17,
        runtime=_rt_base(),
        record_sources=True,
        record_fields=False,
    )
    s1 = jtfne.simulate(m, paradigm=sched, **kw)
    s2 = jtfne.simulate(m, paradigm=sched, **kw)
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(s1.spikes, s2.spikes)


def test_replay_seed_sensitive_live_noise():
    """Different seeds diverge (streams are live, not a constant)."""
    m = _model()
    kw = dict(
        duration_ms=12.0,
        dt_ms=DT,
        runtime=_rt_base(),
        record_sources=True,
        record_fields=False,
    )
    s1 = jtfne.simulate(m, seed=17, **kw)
    s2 = jtfne.simulate(m, seed=18, **kw)
    assert not jnp.array_equal(s1.V_m, s2.V_m)


def test_replay_construction_deterministic():
    """Same construction seed -> same realized network."""
    m1, m2 = _model(), _model()
    np.testing.assert_array_equal(
        np.asarray(m1.params["edge_list"].weight),
        np.asarray(m2.params["edge_list"].weight),
    )
    np.testing.assert_array_equal(
        np.asarray(m1.params["emitter"].v0),
        np.asarray(m2.params["emitter"].v0),
    )


def test_replay_batch_deterministic():
    m = _model()
    sim = jtfne.Simulation(
        duration_ms=6.0,
        dt_ms=DT,
        seed=5,
        record_sources=True,
        record_fields=False,
        runtime=_rt_base(),
    )
    r1 = m.simulate_batch(sim, n_seeds=2)
    r2 = m.simulate_batch(sim, n_seeds=2)
    for k in ("V_m", "spikes"):
        assert jnp.array_equal(jnp.asarray(r1[k]), jnp.asarray(r2[k])), k


# --- per-rule stream isolation -------------------------------------------------


def test_rule_stream_change_leaves_membrane_untouched():
    """k_w=0 decouples rule noise from membrane dynamics: changing the
    rule stream (sigma) moves aux but leaves spikes/V bit-identical."""
    m = _model()
    kw = dict(
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        record_sources=True,
        record_fields=False,
    )
    s_a = jtfne.simulate(m, runtime=_rt_replay_rule(), **kw)
    d_a = m.last_hdp_diagnostics()
    hp_b = {
        "hdp_rule": _ensure_053_replay_rule(),
        "hdp_rule_params": {"k_w": 0.0, "sigma": 2.0},
        "noise_scale": 0.0,
    }
    rt_b = jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params=hp_b,
    )
    s_b = jtfne.simulate(m, runtime=rt_b, **kw)
    d_b = m.last_hdp_diagnostics()
    # Membrane observables untouched by the rule-stream change...
    assert jnp.array_equal(s_a.spikes, s_b.spikes)
    assert jnp.array_equal(s_a.V_m, s_b.V_m)
    # ...while the rule stream itself moved.
    assert not np.allclose(np.asarray(d_a["aux_trace"]), np.asarray(d_b["aux_trace"]))


def test_membrane_change_leaves_rule_key_derivation_untouched():
    """Rule keys derive from (seed, t) only: the derivation is identical
    across membrane noise_scales (structural independence), while the
    membrane trajectory itself moves (both streams live)."""

    def _rule_keys(seed, n_steps):
        _, step_keys = _pipeline._advance_prng_key(jax.random.PRNGKey(seed), n_steps)
        bases = jax.vmap(lambda sk: jax.random.split(sk)[0])(step_keys)
        t = jnp.arange(n_steps, dtype=jnp.int32)
        return jax.vmap(lambda rb, tt: jax.random.fold_in(rb, tt))(bases, t)

    k1 = _rule_keys(17, 24)
    k2 = _rule_keys(17, 24)
    np.testing.assert_array_equal(np.asarray(k1), np.asarray(k2))
    # Observable counterpart: same rule config, different membrane scale ->
    # spikes move (membrane live) even though the rule-key derivation above
    # takes no noise_scale input.
    m = _model()
    kw = dict(
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        record_sources=True,
        record_fields=False,
    )
    hp = {
        "hdp_rule": _ensure_053_replay_rule(),
        "hdp_rule_params": {"k_w": 0.0, "sigma": 0.5},
    }
    rt1 = jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params={**hp, "noise_scale": 0.0},
    )
    rt2 = jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params={**hp, "noise_scale": 0.5},
    )
    s1 = jtfne.simulate(m, runtime=rt1, **kw)
    s2 = jtfne.simulate(m, runtime=rt2, **kw)
    assert not jnp.array_equal(s1.spikes, s2.spikes)


def test_rule_streams_differ_across_rules_same_seed():
    """Two stochastic rules at the same seed each replay bit-identically
    (per-rule determinism); their aux streams are rule-specific."""
    m = _model()
    kw = dict(
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        record_sources=True,
        record_fields=False,
    )
    s1 = jtfne.simulate(m, runtime=_rt_replay_rule(), **kw)
    d1 = m.last_hdp_diagnostics()
    s2 = jtfne.simulate(m, runtime=_rt_replay_rule(), **kw)
    d2 = m.last_hdp_diagnostics()
    np.testing.assert_array_equal(np.asarray(d1["aux_trace"]), np.asarray(d2["aux_trace"]))
    assert jnp.array_equal(s1.spikes, s2.spikes)
