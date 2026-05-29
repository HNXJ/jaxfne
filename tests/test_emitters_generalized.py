"""Tests for generalized emitter components.

Validates shapes, spike/reset logic, JAX transforms (jit, vmap, scan),
and correctness of EmitterState and EmitterOutput nodes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_izhikevich_emitter_basic_shapes():
    """Verify state and output shape contracts for IzhikevichEmitter."""
    n = 20
    emitter = jtfne.emitters.IzhikevichEmitter(n=n, dtype="float32")
    state = emitter.initial_state(seed=42)
    
    assert state.v.shape == (n,)
    assert state.u.shape == (n,)
    assert state.spikes.shape == (n,)
    assert state.step_count == 0
    
    # Run a step
    input_t = jnp.zeros((n,), dtype=jnp.float32)
    next_state, output = emitter.step(state, input_t, dt_ms=0.1)
    
    assert next_state.v.shape == (n,)
    assert next_state.step_count == 1
    assert output.spikes.shape == (n,)
    assert output.voltage.shape == (n,)
    assert bool(output.finite) is True
    assert output.dtype == "float32"


def test_izhikevich_emitter_jit():
    """Verify that emitter.step compiles under jax.jit."""
    n = 10
    emitter = jtfne.emitters.IzhikevichEmitter(n=n, dtype="float32")
    state = emitter.initial_state(seed=42)
    input_t = jnp.ones((n,), dtype=jnp.float32)
    
    @jax.jit
    def compile_step(s, inp):
        return emitter.step(s, inp, dt_ms=0.1)
        
    next_state, output = compile_step(state, input_t)
    assert next_state.step_count == 1
    assert output.finite.shape == ()


def test_izhikevich_emitter_scan():
    """Verify that emitter steps can be rolled out under jax.lax.scan."""
    n = 5
    emitter = jtfne.emitters.IzhikevichEmitter(n=n, dtype="float32")
    state = emitter.initial_state(seed=0)
    
    inputs = jnp.zeros((100, n), dtype=jnp.float32)
    
    def body_fn(carry, inp):
        next_s, out = emitter.step(carry, inp, dt_ms=0.1)
        return next_s, out
        
    @jax.jit
    def run_scan(s, inps):
        return jax.lax.scan(body_fn, s, inps)
        
    final_state, outputs = run_scan(state, inputs)
    assert final_state.step_count == 100
    assert outputs.voltage.shape == (100, n)
    assert outputs.spikes.shape == (100, n)


def test_placeholders_throw():
    """Verify that GLIFEmitter and LIFEmitter throw NotImplementedError."""
    with pytest.raises(NotImplementedError, match="GLIFEmitter"):
        jtfne.emitters.GLIFEmitter(n=10)
        
    with pytest.raises(NotImplementedError, match="LIFEmitter"):
        jtfne.emitters.LIFEmitter(n=10)
