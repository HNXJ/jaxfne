"""0.5.3 ENGINE item 1: H/W/K ownership — configured -> realized -> executed.

Doctrine: Atlas source 8 + docs/doctrine/rbs_rbd_hdp.md.
H is RBS (relative biophysical state); HDP is hidden-state dependent
plasticity (persistent-parameter dynamics). H != HDP: RBD remains
meaningful with fixed weights (dW = 0).

For each of H, W, K this battery pins the inspection surface at all
three stages — configured (declared params), realized (model.params /
DynamicState), executed (traces/finals/draws) — including which rule
mutates which state. Read-only: no state containers are restructured;
the small named helpers below only read existing surfaces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne import _pipeline
from jaxfne.hdp_rule import get_hdp_rule, is_registered_hdp_rule


def _model():
    cfg = jtfne.suite2_net1_config(seed=11, n=8, duration_ms=20.0, dt_ms=0.5)
    return jtfne.construct(cfg)


def _hdp_runtime(**over):
    hp = {
        "K_HDP": 0.01,
        "K_ctrl": 0.15,
        "K_w_ctrl": 0.001,
        "tau_0_ms": 20.0,
        "alpha": 0.01,
        "barrier_c": 0.01,
        "barrier_d": 0.01,
        "noise_scale": 0.0,
    }
    hp.update(over)
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params=hp,
    )


def ownership_snapshot(model, signals, state, runtime):
    """Small named read-only ownership map: configured -> realized -> executed.

    Returns nested dict with H/W/K triples; every leaf is observed from an
    existing surface (no new state, no mutation).
    """
    diag = model.last_hdp_diagnostics()
    edges = model.params["edge_list"]
    return {
        "H": {
            "configured": dict(runtime.hdp_params),
            "realized": {
                "hdp_initial_H": (
                    None
                    if model.params.get("hdp_initial_H") is None
                    else np.asarray(model.params["hdp_initial_H"]).shape
                ),
                "dynamic_H_shape": tuple(state.dynamic.H.shape),
            },
            "executed": {
                "H_trace_shape": tuple(np.asarray(diag["H_trace"]).shape),
                "H_final": np.asarray(diag["H_final"]),
            },
        },
        "W": {
            "configured": {
                "K_HDP": runtime.hdp_params.get("K_HDP"),
                "K_w_ctrl": runtime.hdp_params.get("K_w_ctrl"),
                "hdp_rule": runtime.hdp_params.get("hdp_rule", "signed_linear"),
            },
            "realized": {
                "edge_weight_shape": (int(edges.n_edges),),
                "dynamic_w_shape": tuple(state.dynamic.w.shape),
            },
            "executed": {
                "w_final": np.asarray(diag["w_final"]),
            },
        },
        "K": {
            "configured": {"seed": 17},
            "realized": {
                "prng_key": np.asarray(state.prng_key),
                "step_index": int(state.step_index),
            },
            "executed": {"noise_scale": runtime.hdp_params.get("noise_scale")},
        },
    }


def _run(model, runtime, seed=17, duration_ms=6.0):
    return jtfne.simulate(
        model,
        duration_ms=duration_ms,
        dt_ms=0.5,
        seed=seed,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
        return_state=True,
    )


# --- H ownership -----------------------------------------------------------


def test_h_configured_realized_executed_chain():
    model = _model()
    runtime = _hdp_runtime()
    signals, state = _run(model, runtime)
    snap = ownership_snapshot(model, signals, state, runtime)
    assert snap["H"]["configured"]["K_HDP"] == 0.01
    assert snap["H"]["realized"]["dynamic_H_shape"] == (8,)
    assert snap["H"]["executed"]["H_trace_shape"] == (12, 8)
    # Nonzero gains move H away from the 1.0 equilibrium.
    assert not np.allclose(snap["H"]["executed"]["H_final"], 1.0)


def test_h_null_pins_at_reference():
    """K_HDP=0 + zero gains at the kernel level: H stays at 1.0 (null control).

    Mirrors tests/test_phaseC_H_carry_resume.py (kernel-level null); at the
    Model level these identity params route to the baseline kernel instead
    (see test_identity_params_route_to_baseline below).
    """
    import jax

    from jaxfne.emitters import (
        EdgeList,
        IzhikevichParams,
        simulate_edge_recurrent_izhikevich_hdp as hdp_kernel,
    )

    n = 8
    rng = np.random.default_rng(0)
    p = IzhikevichParams(
        a=jnp.full((n,), 0.02),
        b=jnp.full((n,), 0.2),
        c=jnp.full((n,), -65.0),
        d=jnp.full((n,), 8.0),
        drive=jnp.full((n,), 6.0),
        sign=jnp.ones((n,)),
        W=jnp.zeros((n, n)),
        v0=jnp.full((n,), -65.0),
        u0=jnp.full((n,), -13.0),
        source_scale=jnp.ones((n,)),
        labels=tuple("E" for _ in range(n)),
        layer_labels=tuple("L4" for _ in range(n)),
        source_calibration_status="x",
    )
    ne = 32
    ri = rng.integers(0, 2, ne)
    wm = np.abs(rng.normal(0, 0.3, ne)).astype(np.float32)
    w = np.where(ri == 0, wm, -wm).astype(np.float32)
    edges = EdgeList(
        pre=jnp.asarray(rng.integers(0, n, ne), jnp.int32),
        post=jnp.asarray(rng.integers(0, n, ne), jnp.int32),
        weight=jnp.asarray(w),
        receptor_index=jnp.asarray(ri, jnp.int32),
        tau_ms=jnp.full((ne,), 5.0),
        source_calibration_status="x",
    )
    _, _, _, d = hdp_kernel(
        p,
        edges,
        200,
        0.5,
        jax.random.PRNGKey(5),
        K_HDP=0.0,
        noise_scale=0.0,
    )
    assert bool(jnp.allclose(d["H_trace"], 1.0, atol=1e-6))
    assert jnp.array_equal(d["w_final"], edges.weight)


def test_identity_params_route_to_baseline():
    """Ownership routing fact: identity HDP params disengage HDP at Model
    level — HDP diagnostics stay None and the run is the fixed-W path."""
    from jaxfne.hdp_rule import hdp_is_engaged

    model = _model()
    null_hp = {
        "K_HDP": 0.0,
        "K_ctrl": 0.0,
        "K_w_ctrl": 0.0,
        "alpha": 0.0,
        "barrier_c": 0.0,
        "barrier_d": 0.0,
        "noise_scale": 0.0,
    }
    assert not hdp_is_engaged(null_hp, model.params, enable_hdp=True)
    runtime = _hdp_runtime(**null_hp)
    signals = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
    )
    assert model.last_hdp_diagnostics() is None
    ref = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=jtfne.RuntimeConfig(
            dtype="float32",
            recurrent_backend="edge_list",
            enable_hdp=False,
            hdp_params={"noise_scale": 0.0},
        ),
        record_sources=True,
        record_fields=False,
    )
    assert jnp.array_equal(signals.V_m, ref.V_m)
    assert jnp.array_equal(signals.spikes, ref.spikes)


def test_h_moves_while_w_fixed_proves_h_neq_hdp():
    """H != HDP: with K_HDP=0 but live H gains, H evolves while W is untouched.

    RBD without plasticity: state memory does not require weight change.
    """
    model = _model()
    runtime = _hdp_runtime(K_HDP=0.0, K_w_ctrl=0.0)
    _, state = _run(model, runtime)
    diag = model.last_hdp_diagnostics()
    assert not np.allclose(np.asarray(diag["H_trace"]), 1.0)
    np.testing.assert_array_equal(
        np.asarray(diag["w_final"]),
        np.asarray(model.params["edge_list"].weight),
    )


# --- W ownership -----------------------------------------------------------


def test_w_configured_realized_executed_chain():
    model = _model()
    runtime = _hdp_runtime()
    _, state = _run(model, runtime)
    diag = model.last_hdp_diagnostics()
    edges = model.params["edge_list"]
    assert tuple(state.dynamic.w.shape) == (int(edges.n_edges),)
    assert np.asarray(diag["w_final"]).shape == (int(edges.n_edges),)
    # Live plasticity moves weights off their realized values.
    assert not np.allclose(np.asarray(diag["w_final"]), np.asarray(edges.weight))


def test_w_plasticity_off_leaves_weights_untouched():
    """Disable path: enable_hdp=False never engages the weight update."""
    model = _model()
    runtime = jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list", enable_hdp=False)
    signals = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
    )
    assert model.last_hdp_diagnostics() is None
    assert signals.V_m.shape == (12, 8)


def test_w_k_hdp_zero_null_is_fixed_w():
    model = _model()
    runtime = _hdp_runtime(K_HDP=0.0, K_w_ctrl=0.0)
    _run(model, runtime)
    diag = model.last_hdp_diagnostics()
    np.testing.assert_array_equal(
        np.asarray(diag["w_final"]),
        np.asarray(model.params["edge_list"].weight),
    )


def test_edge_table_exposes_realized_w():
    """Realized W is inspectable per edge (pre/post/weight)."""
    rows = _model().edge_table()
    assert len(rows) > 0
    assert {"pre", "post", "weight"} <= set(rows[0])


# --- K ownership -----------------------------------------------------------


def test_k_seed_threads_to_prng_key_and_step_index():
    model = _model()
    _, state = _run(model, _hdp_runtime(), seed=17)
    assert tuple(np.asarray(state.prng_key).shape) == (2,)
    assert int(state.step_index) == 12
    # Same seed re-derives the same continuation key chain head.
    _, state2 = _run(model, _hdp_runtime(), seed=17)
    np.testing.assert_array_equal(np.asarray(state.prng_key), np.asarray(state2.prng_key))


def test_k_same_seed_bit_identical_deterministic():
    model = _model()
    s1, _ = _run(model, _hdp_runtime(), seed=17)
    s2, _ = _run(model, _hdp_runtime(), seed=17)
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(s1.spikes, s2.spikes)


def test_k_continuation_key_chain_advances():
    """K ownership across chunks: the carried key advances, segments chain."""
    model = _model()
    runtime = _hdp_runtime()
    _, s1 = _run(model, runtime, seed=17, duration_ms=6.0)
    _, s2 = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=999,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
        continuation=s1,
        return_state=True,
    )
    assert int(s2.step_index) == 24
    assert not np.array_equal(np.asarray(s1.prng_key), np.asarray(s2.prng_key))


# --- which rule mutates which state ---------------------------------------


def test_legacy_rule_mutation_map():
    """Legacy kernel: K_HDP gates the H-difference weight term; K_w_ctrl an
    independent weight-restoration term; neither gates the H equation."""
    active = _hdp_runtime(K_HDP=0.01)
    h_only = _hdp_runtime(K_HDP=0.0, K_w_ctrl=0.0)
    assert active.hdp_params["K_HDP"] != h_only.hdp_params["K_HDP"]
    # H moves under both (H equation independent of K_HDP); W only under active.
    m = _model()
    _, _ = _run(m, h_only)
    h_trace_null = np.asarray(m.last_hdp_diagnostics()["H_trace"])
    w_null = np.asarray(m.last_hdp_diagnostics()["w_final"])
    _, _ = _run(m, active)
    w_active = np.asarray(m.last_hdp_diagnostics()["w_final"])
    assert not np.allclose(h_trace_null, 1.0)  # H equation live
    np.testing.assert_array_equal(w_null, np.asarray(m.params["edge_list"].weight))
    assert not np.allclose(w_active, w_null)  # K_HDP>0 mutates W


def test_registered_rule_declares_mutated_targets():
    """Registered rules declare mutated state via theta_targets (read-only)."""
    from jaxfne import hdp_rule as _hr

    names = [n for n in _hr._REGISTRY]
    assert names, "no registered HDP rules to inspect"
    for name in names:
        desc, _ = get_hdp_rule(name)
        assert desc.theta_targets, f"{name} declares no mutated targets"
        for t in desc.theta_targets:
            assert t in ("edge_weight", "drive_bias"), (name, t)
    assert is_registered_hdp_rule(names[0])


def test_dynamic_state_carries_full_mutable_set():
    """Realized mutable set: v, u, prev_spikes, syn_state, H, w (+theta/aux/b)."""
    model = _model()
    dyn = _pipeline.dynamic_state_from_model(model)
    for field in ("v", "u", "prev_spikes", "syn_state", "H", "w"):
        assert hasattr(dyn, field), field
    assert tuple(dyn.H.shape) == (8,)
    assert tuple(dyn.v.shape) == (8,)


def test_hdp_initial_state_seeds_realized_h_w():
    model = _model()
    runtime = _hdp_runtime()
    _, state = _run(model, runtime)
    seeded = model.with_hdp_initial_state(
        H0=np.asarray(state.dynamic.H), w0=np.asarray(state.dynamic.w)
    )
    np.testing.assert_array_equal(
        np.asarray(seeded.params["hdp_initial_H"]), np.asarray(state.dynamic.H)
    )
    np.testing.assert_array_equal(
        np.asarray(seeded.params["hdp_initial_w"]), np.asarray(state.dynamic.w)
    )
