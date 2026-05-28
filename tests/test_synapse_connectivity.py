"""Tests for synapse and connectivity layers.

Validates connectivity shapes, E/I presynaptic weight signs, zero weights,
JAX compile compatibility, and metadata reports.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_synapse_shape_contract():
    """Verify SynapseLayer timing and dimension contracts."""
    n = 10
    W = jnp.ones((n, n), dtype=jnp.float32)
    synapse = jtfne.emitters.SynapseLayer(n=n, W=W, tau_ms=5.0)
    
    state = synapse.initial_state()
    assert state.s.shape == (n,)
    
    spikes = jnp.zeros((n,), dtype=jnp.float32)
    next_state, I_syn = synapse.step(state, spikes, dt_ms=0.1)
    
    assert next_state.s.shape == (n,)
    assert I_syn.shape == (n,)


def test_synapse_ei_weight_convention():
    """Verify E/I Presynaptic weight sign policies."""
    n = 2
    # Connection: neuron 0 to neuron 1
    # We set W[1, 0] = 5.0 (excitatory presynaptic weight)
    # and W[1, 1] = -5.0 (inhibitory presynaptic weight)
    W = jnp.array([[0.0, 0.0], [5.0, -5.0]], dtype=jnp.float32)
    synapse = jtfne.emitters.SynapseLayer(n=n, W=W, tau_ms=5.0)
    
    # Send a spike from excitatory neuron 0
    state = synapse.initial_state()
    spikes = jnp.array([1.0, 0.0], dtype=jnp.float32)
    _, I_syn_e = synapse.step(state, spikes, dt_ms=0.1)
    
    # Excitatory spike should increase postsynaptic drive (positive current)
    assert float(I_syn_e[1]) > 0.0
    
    # Send a spike from inhibitory neuron 1
    spikes = jnp.array([0.0, 1.0], dtype=jnp.float32)
    _, I_syn_i = synapse.step(state, spikes, dt_ms=0.1)
    
    # Inhibitory spike should decrease postsynaptic drive (negative current)
    assert float(I_syn_i[1]) < 0.0


def test_synapse_zero_weight_policy():
    """Verify that zero weights produce no postsynaptic effect."""
    n = 2
    W = jnp.zeros((n, n), dtype=jnp.float32)
    synapse = jtfne.emitters.SynapseLayer(n=n, W=W, tau_ms=5.0)
    
    state = synapse.initial_state()
    spikes = jnp.array([1.0, 1.0], dtype=jnp.float32)
    _, I_syn = synapse.step(state, spikes, dt_ms=0.1)
    
    assert np.allclose(I_syn, 0.0)


def test_synapse_jit_and_scan():
    """Verify that SynapseLayer step compiles under JIT and can roll out with scan."""
    n = 3
    W = jnp.ones((n, n), dtype=jnp.float32)
    synapse = jtfne.emitters.SynapseLayer(n=n, W=W, tau_ms=5.0)
    state = synapse.initial_state()
    
    @jax.jit
    def compile_step(s, spk):
        return synapse.step(s, spk, dt_ms=0.1)
        
    spikes = jnp.ones((n,), dtype=jnp.float32)
    next_state, I_syn = compile_step(state, spikes)
    assert next_state.s.shape == (n,)
    assert I_syn.shape == (n,)
    
    # Scan test
    spikes_seq = jnp.zeros((50, n), dtype=jnp.float32)
    
    def body_fn(carry, spk):
        next_s, I = synapse.step(carry, spk, dt_ms=0.1)
        return next_s, I
        
    @jax.jit
    def run_scan(s, seq):
        return jax.lax.scan(body_fn, s, seq)
        
    final_state, I_syn_seq = run_scan(state, spikes_seq)
    assert final_state.s.shape == (n,)
    assert I_syn_seq.shape == (50, n)


def test_synapse_metadata_report():
    """Verify metadata annotations and kernel parameters."""
    n = 2
    W = jnp.ones((n, n), dtype=jnp.float32)
    synapse = jtfne.emitters.SynapseLayer(n=n, W=W, tau_ms=10.0, kernel="exponential")
    report = synapse.report()
    
    assert report["synapse_kernel"] == "exponential"
    assert report["tau_ms"] == 10.0
    assert report["weight_sign_convention"] == "presynaptic_cell_type"
    
    # Verify placeholder failures
    with pytest.raises(NotImplementedError):
        bad_synapse1 = jtfne.emitters.SynapseLayer(n=n, W=W, kernel="alpha")
        bad_synapse1.step(synapse.initial_state(), jnp.ones(n), 0.1)
        
    with pytest.raises(NotImplementedError):
        bad_synapse2 = jtfne.emitters.SynapseLayer(n=n, W=W, kernel="double_exponential")
        bad_synapse2.step(synapse.initial_state(), jnp.ones(n), 0.1)
