"""Review findings F1-F3: seed-0 handling, proxy float64 contract, HDP rule key.

F1: ``seed=0`` must stay 0 through the default builders, never collapse to
    42; seed 0 vs seed 42 realize different connectivity; the default
    (``seed=None``) path still records 42.
F2: ``population_rate``, ``cable_filter_sources`` and
    ``project_laminar_sources`` follow the input when ``dtype`` is not
    given: float64 input with x64 enabled computes in float64, otherwise
    float32 (with x64 off a float64 input reads float32, which is JAX's
    own canonicalization). An explicit ``dtype="float64"`` still
    requires x64 and is refused by name without it.
F3: a stochastic registered HDP rule with no external drive draws its
    per-step noise from ``seed``: seeds 1 vs 2 differ, the same seed
    repeats exactly, and an explicit ``seed=0`` never equals the unset
    key (the adex ``simulate()`` route runs ``simulate_adex`` and
    refuses HDP inputs, pinned by
    ``tests/test_adex_route.py::test_construct_refusals_name_the_argument[runtime enable_hdp]``;
    so no model path reaches this kernel).
F1 residual: the tensor bridge refuses a declared ``n_neurons <= 0``
    by name instead of promoting it to one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters_adex import make_adex_params, simulate_adex_hdp_registered
from jaxfne.fields import (
    cable_filter_sources,
    cable_filter_tau,
    population_rate,
    project_laminar_sources,
)
from jaxfne.hdp_rule import (
    HDPRuleDescriptor,
    HDPRuleUpdate,
    list_registered_hdp_rules,
    register_hdp_rule,
)


def test_f1_seed_zero_is_recorded_not_42():
    cfg = jtfne.default_cortical_column_config(n=20, seed=0)
    assert cfg.metadata["seed"] == 0
    assert cfg.metadata["connectivity"]["edge_seed"] == 0


def test_f1_complete_config_seed_zero_is_recorded_not_42():
    cfg = jtfne.default_complete_configuration(seed=0)
    assert cfg.metadata["seed"] == 0
    assert cfg.metadata["connectivity"]["edge_seed"] == 0


def test_f1_seed_zero_vs_42_realize_different_connectivity():
    m0 = jtfne.construct(jtfne.default_cortical_column_config(n=20, seed=0))
    m42 = jtfne.construct(jtfne.default_cortical_column_config(n=20, seed=42))
    w0 = np.asarray(m0.params["emitter"].W)
    w42 = np.asarray(m42.params["emitter"].W)
    assert w0.shape == (20, 20) and w42.shape == (20, 20)
    assert not np.array_equal(w0, w42), "seed=0 realized the seed=42 connectivity"
    m0_again = jtfne.construct(jtfne.default_cortical_column_config(n=20, seed=0))
    assert np.array_equal(w0, np.asarray(m0_again.params["emitter"].W))


def test_f1_default_seed_path_still_records_42():
    cfg = jtfne.default_cortical_column_config(n=20)
    assert cfg.metadata["seed"] == 42
    assert cfg.metadata["connectivity"]["edge_seed"] == 42


def _enable_x64():
    jax.config.update("jax_enable_x64", True)


def _disable_x64():
    jax.config.update("jax_enable_x64", False)


def test_f2_population_rate_float64_with_x64():
    _enable_x64()
    try:
        out = population_rate(np.ones((50, 4), dtype=np.float64), 0.5, dtype="float64")
        assert np.asarray(out).dtype == np.float64
        assert np.allclose(np.asarray(out), 1000.0 / 0.5)
        out32 = population_rate(np.ones((50, 4), dtype=np.float32), 0.5)
        assert np.asarray(out32).dtype == np.float32
        assert np.allclose(np.asarray(out32), 1000.0 / 0.5)
    finally:
        _disable_x64()


def test_f2_population_rate_float64_refused_without_x64():
    assert not bool(jax.config.read("jax_enable_x64"))
    with pytest.raises(ValueError, match="x64"):
        population_rate(np.ones((10, 4), dtype=np.float64), 0.5, dtype="float64")


def test_f2_cable_filter_float64_with_x64():
    _enable_x64()
    try:
        sources = np.linspace(-1.0, 1.0, 256 * 4).reshape(256, 4).astype(np.float64)
        tau64 = np.asarray(cable_filter_tau(["E"] * 4, [0.5] * 4), dtype=np.float64)
        out = cable_filter_sources(sources, tau64, 0.5, order=2, dtype="float64")
        assert np.asarray(out).dtype == np.float64
        assert bool(jnp.all(jnp.isfinite(out)))
        out32 = cable_filter_sources(
            sources.astype(np.float32),
            np.asarray(tau64, dtype=np.float32),
            0.5,
            order=2,
        )
        assert np.asarray(out32).dtype == np.float32
    finally:
        _disable_x64()


def test_f2_cable_filter_float64_refused_without_x64():
    assert not bool(jax.config.read("jax_enable_x64"))
    sources = np.zeros((32, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="x64"):
        cable_filter_sources(sources, np.full(2, 0.005), 0.5, dtype="float64")


def _ensure_key_noise_rule():
    if "f3_key_noise" not in list_registered_hdp_rules():
        def _step(ctx):
            noise = (
                jax.random.normal(ctx.key, shape=ctx.w.shape, dtype=ctx.H.dtype) * 0.05
            )
            return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), d_theta={"edge_weight": noise})

        register_hdp_rule(
            HDPRuleDescriptor(name="f3_key_noise", theta_targets=("edge_weight",)),
            _step,
        )


def _run_key_noise(*, seed, n_steps=50):
    _ensure_key_noise_rule()
    return simulate_adex_hdp_registered(
        make_adex_params(("E", "PV")),
        n_steps=n_steps,
        dt_ms=0.1,
        edge_pre=[0],
        edge_post=[1],
        edge_weight_nS=[3.0],
        edge_receptor=[0],
        edge_delay_steps=[1],
        receptor_reversal_mV=(0.0,),
        receptor_tau_ms=(2.0,),
        decay="exp",
        hdp_rule="f3_key_noise",
        seed=seed,
    )


def test_f3_seeds_differ_and_repeat():
    out1 = _run_key_noise(seed=1)
    out2 = _run_key_noise(seed=2)
    w1 = np.asarray(out1["w_final_nS"])
    w2 = np.asarray(out2["w_final_nS"])
    assert w1.shape == (1,) and w2.shape == (1,)
    assert not np.array_equal(w1, w2), "seeds 1 and 2 drew the same rule noise"
    assert np.array_equal(w1, np.asarray(_run_key_noise(seed=1)["w_final_nS"]))


def test_f3_non_int_seed_refused_by_name():
    with pytest.raises(ValueError, match="seed"):
        _run_key_noise(seed="1")


def test_f2_proxy_dtype_follows_float64_input_with_x64():
    _enable_x64()
    try:
        spikes64 = np.ones((50, 4), dtype=np.float64)
        out = population_rate(spikes64, 0.5)
        assert np.asarray(out).dtype == np.float64
        assert np.allclose(np.asarray(out), 1000.0 / 0.5)
        filt = cable_filter_sources(
            spikes64, np.full(4, 0.005, dtype=np.float64), 0.5, order=2
        )
        assert np.asarray(filt).dtype == np.float64
        field = project_laminar_sources(
            spikes64, np.zeros((4, 3), dtype=np.float64), n_contacts=4
        )
        assert np.asarray(field.source_proxy).dtype == np.float64
        out32 = population_rate(np.ones((50, 4), dtype=np.float32), 0.5)
        assert np.asarray(out32).dtype == np.float32
    finally:
        _disable_x64()


def test_f2_proxy_float64_input_without_x64_stays_float32():
    assert not bool(jax.config.read("jax_enable_x64"))
    out = population_rate(np.ones((50, 4), dtype=np.float64), 0.5)
    assert np.asarray(out).dtype == np.float32


def test_f3_seed_zero_differs_from_unset():
    w0 = np.asarray(_run_key_noise(seed=0)["w_final_nS"])
    assert not np.allclose(w0, 3.0), "rule applied no noise (precondition)"
    w_unset = np.asarray(_run_key_noise(seed=None)["w_final_nS"])
    assert not np.array_equal(w0, w_unset), "seed=0 collapsed to the unset key"
    assert np.array_equal(w0, np.asarray(_run_key_noise(seed=0)["w_final_nS"]))


def test_f1_bridge_refuses_zero_neuron_layer_by_name():
    from jaxfne.neuronal_tensor import (
        Area,
        Layer,
        NeuronalTensor,
        neuronal_tensor_to_configuration,
    )

    tensor = NeuronalTensor(
        areas=[Area(name="V1", layers=[Layer(name="L4", n_neurons=0)])]
    )
    with pytest.raises(ValueError, match="n_neurons"):
        neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)


def test_f2_proxy_refuses_bfloat16_by_name():
    spikes = np.ones((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="bfloat16"):
        population_rate(spikes, 0.5, dtype="bfloat16")
    with pytest.raises(ValueError, match="bfloat16"):
        cable_filter_sources(spikes, np.full(4, 0.005), 0.5, dtype="bfloat16")
    with pytest.raises(ValueError, match="bfloat16"):
        project_laminar_sources(spikes, np.zeros((4, 3)), n_contacts=4, dtype="bfloat16")
