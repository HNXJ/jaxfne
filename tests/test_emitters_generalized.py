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


def _realized_noise_coef(dt_ms):
    """Noise coefficient realized by one IzhikevichEmitter.step at ``dt_ms``.

    Two initial states differ only in their RNG key (seed), so v_next differs
    only by dt * coef * (z1 - z2), with z the step's own normal draw.
    """
    n = 64
    emitter = jtfne.emitters.IzhikevichEmitter(n=n, dtype="float32")
    zero = jnp.zeros((n,), dtype=jnp.float32)
    v_next, z = [], []
    for seed in (1, 2):
        state = emitter.initial_state(seed=seed)
        k = state.key
        nxt, out = emitter.step(state, zero, dt_ms=dt_ms)
        assert not bool(jnp.any(out.spikes)), "fixture must not spike (reset hides noise)"
        v_next.append(np.asarray(nxt.v, dtype=np.float64))
        z.append(np.asarray(jax.random.normal(jax.random.split(k)[1], shape=(n,)), dtype=np.float64))
    return float(np.median((v_next[0] - v_next[1]) / (dt_ms * (z[0] - z[1]))))


def test_izhikevich_emitter_step_noise_is_dt_invariant():
    """Noise coefficient is 0.5 at dt 0.5 ms and scales by sqrt(0.5 / dt) (D1)."""
    np.testing.assert_allclose(_realized_noise_coef(0.5), 0.5, rtol=1e-3)
    np.testing.assert_allclose(_realized_noise_coef(0.125), 1.0, rtol=1e-3)
