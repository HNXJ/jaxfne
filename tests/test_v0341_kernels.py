import pytest
import time
import jax
import jax.numpy as jnp
import numpy as np
from jaxfne import (
    spectrolaminar_psd_jax,
    bandpower_jax,
    spectrolaminar_readout_kernel_jax,
    spectrolaminar_similarity_kernel_jax,
    spectrolaminar_similarity_candidates_jax,
    spectrolaminar_similarity_candidates_seeds_jax,
    compile_connection_rules_jax,
    update_stdp_weights_jax,
    solve_volume_conductor_experimental,
    Model,
    Simulation,
    RuntimeConfig,
    stimulus_schedule,
)

def test_spectral_functions():
    # Setup mock signal: shape (n_trials, n_steps, n_contacts)
    n_trials = 3
    n_steps = 100
    n_contacts = 5
    fs = 1000.0
    freqs = jnp.array([10.0, 20.0, 30.0])
    
    # Random key
    key = jax.random.PRNGKey(42)
    signal = jax.random.normal(key, (n_trials, n_steps, n_contacts))
    
    # 1. PSD
    psd = spectrolaminar_psd_jax(signal, fs, freqs)
    assert psd.shape == (len(freqs), n_contacts)
    assert psd.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(psd))
    
    # 2. Bandpower
    band_range = jnp.array([10.0, 25.0])
    profile = bandpower_jax(psd, freqs, band_range)
    assert profile.shape == (n_contacts,)
    assert jnp.all(profile >= 0.0)
    assert jnp.all(profile <= 1.0)
    
    # 3. Readout
    ab_range = jnp.array([10.0, 25.0])
    gamma_range = jnp.array([30.0, 80.0])
    readout = spectrolaminar_readout_kernel_jax(psd, freqs, ab_range, gamma_range)
    assert "relative_power" in readout
    assert "alpha_beta" in readout
    assert "gamma" in readout
    assert readout["relative_power"].shape == psd.shape
    assert readout["alpha_beta"].shape == (n_contacts,)
    assert readout["gamma"].shape == (n_contacts,)
    
    # 4. Similarity kernels
    target_ab = jnp.ones((n_contacts,))
    target_gamma = jnp.ones((n_contacts,))
    sim_score = spectrolaminar_similarity_kernel_jax(
        readout["alpha_beta"], readout["gamma"], target_ab, target_gamma
    )
    assert sim_score.shape == ()
    assert 0.0 <= float(sim_score) <= 100.0
    
    # Candidates vectorizations
    batch_ab = jnp.ones((2, n_contacts))
    batch_gamma = jnp.ones((2, n_contacts))
    scores_cand = spectrolaminar_similarity_candidates_jax(
        batch_ab, batch_gamma, target_ab, target_gamma
    )
    assert scores_cand.shape == (2,)
    
    scores_seeds = spectrolaminar_similarity_candidates_seeds_jax(
        jnp.ones((3, 2, n_contacts)), jnp.ones((3, 2, n_contacts)), target_ab, target_gamma
    )
    assert scores_seeds.shape == (3, 2)


def test_compile_connection_rules_jax():
    pre_indices = jnp.array([0, 1, 2])
    post_indices = jnp.array([3, 4])
    probability = 0.5
    key = jax.random.PRNGKey(123)
    max_edges = 4
    
    edge_pre, edge_post, edge_weight = compile_connection_rules_jax(
        pre_indices, post_indices, probability, key, max_edges, weight_val=1.5
    )
    
    assert edge_pre.shape == (max_edges,)
    assert edge_post.shape == (max_edges,)
    assert edge_weight.shape == (max_edges,)
    
    # Assert values are either within indices or padded with -1
    for p in edge_pre:
        assert p == -1 or p in pre_indices
    for p in edge_post:
        assert p == -1 or p in post_indices
    for w in edge_weight:
        assert w == 0.0 or w == 1.5


def test_update_stdp_weights_jax():
    n_neurons = 4
    W = jnp.array([
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.0]
    ])
    trace_pre = jnp.array([0.1, 0.2, 0.3, 0.4])
    trace_post = jnp.array([0.4, 0.3, 0.2, 0.1])
    spiked = jnp.array([True, False, True, False])
    exc_mask = jnp.array([True, True, False, False]) # Excitatory neurons
    
    W_next = update_stdp_weights_jax(
        W, trace_pre, trace_post, spiked, exc_mask,
        A_plus=0.01, A_minus=0.012, plasticity_scale=1.0,
        w_min=0.0, w_max=1.5
    )
    
    assert W_next.shape == (n_neurons, n_neurons)
    # Exclude self-connections
    assert float(W_next[0, 0]) == 0.0
    assert float(W_next[1, 1]) == 0.0
    # Inhibitory (non-excitatory) weights must remain unchanged
    # W[:, 2] and W[:, 3] are synapses from pre-synaptic neurons 2 and 3.
    # update_mask restricts updates to exc_mask[None, :] which maps to presynaptic index!
    # Wait, in the update_stdp_weights_jax, update_mask is exc_mask[None, :]
    # Let's verify if non-excitatory presynaptic synapses remained unchanged.
    # W[:, 2] (from neuron 2) should remain unchanged because neuron 2 is inhibitory.
    assert jnp.allclose(W_next[:, 2], W[:, 2])
    assert jnp.allclose(W_next[:, 3], W[:, 3])


def test_solve_volume_conductor_experimental():
    with pytest.raises(NotImplementedError) as excinfo:
        solve_volume_conductor_experimental()
    assert "experimental solver skeleton" in str(excinfo.value)


def test_stimulus_schedule_to_array_jax():
    sched = stimulus_schedule([
        {"onset_ms": 10.0, "duration_ms": 20.0, "amplitude": 1.5, "target_indices": [0, 2]}
    ], n_neurons=4)
    
    arr_np = sched.to_array(n_steps=100, dt_ms=1.0)
    arr_jax = sched.to_array_jax(n_steps=100, dt_ms=1.0)
    
    assert arr_jax.shape == (100, 4)
    assert jnp.allclose(arr_np, arr_jax)


def test_jit_warmup_timing():
    # Verify that JIT warmups record timings under Model._warmup_times
    # Set up a small test model using suite presets
    from jaxfne import construct, default_cortical_column_config
    cfg = default_cortical_column_config(column_name="V1", n=20)
    model = construct(cfg)
    
    # Set JIT to True
    sim = Simulation(
        duration_ms=10.0,
        dt_ms=0.5,
        runtime=RuntimeConfig(jit=True, recompilation_guard="warning")
    )
    
    # Run once - should warm up and record
    assert not hasattr(model, "_warmup_times")
    signals = model.simulate(sim)
    assert hasattr(model, "_warmup_times")
    assert len(model._warmup_times) > 0
    first_len = len(model._warmup_times)
    
    # Run again - should use compiled cache, so no new warmup times are added
    signals2 = model.simulate(sim)
    assert len(model._warmup_times) == first_len
