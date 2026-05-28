"""Tests for source bookkeeping and construct_source_tensor double-count guards.

Validates that Mode A and Mode B operate under strict non-double-counting constraints,
throwing correct ValueError triggers on violation, and compile under JAX JIT.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_mode_a_total_membrane_current():
    """Verify that Mode A constructs from total membrane and rejects extra synaptic current."""
    total_membrane = jnp.array([1.5, -2.0, 3.0], dtype=jnp.float32)
    synaptic = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
    
    # Correct usage
    source, report = jtfne.construct_source_tensor(
        mode="total_membrane_current_proxy",
        total_membrane_current=total_membrane,
        synaptic_current=None
    )
    assert np.allclose(source, total_membrane)
    assert report["source_mode"] == "total_membrane_current_proxy"
    assert report["double_count_guard"] == "passed"
    
    # Violate and expect ValueError
    with pytest.raises(ValueError, match="Double-counting detected"):
        jtfne.construct_source_tensor(
            mode="total_membrane_current_proxy",
            total_membrane_current=total_membrane,
            synaptic_current=synaptic
        )


def test_mode_b_decomposed_cap_ion_synaptic():
    """Verify that Mode B constructs source by sum and rejects if inputs missing."""
    cap_ion = jnp.array([1.0, -1.5, 2.0], dtype=jnp.float32)
    synaptic = jnp.array([0.5, -0.5, 1.0], dtype=jnp.float32)
    
    # Correct usage
    source, report = jtfne.construct_source_tensor(
        mode="decomposed_cap_ion_plus_synaptic_proxy",
        total_membrane_current=jnp.zeros_like(cap_ion),
        decomposed_cap_ion=cap_ion,
        synaptic_current=synaptic
    )
    assert np.allclose(source, cap_ion + synaptic)
    assert report["source_mode"] == "decomposed_cap_ion_plus_synaptic_proxy"
    assert report["double_count_guard"] == "passed"
    
    # Missing cap_ion
    with pytest.raises(ValueError, match="requires both decomposed_cap_ion and synaptic_current"):
        jtfne.construct_source_tensor(
            mode="decomposed_cap_ion_plus_synaptic_proxy",
            total_membrane_current=jnp.zeros_like(cap_ion),
            decomposed_cap_ion=None,
            synaptic_current=synaptic
        )


def test_invalid_double_count_mode():
    """Verify that explicit invalid double count mode fails loudly."""
    total_membrane = jnp.array([1.0], dtype=jnp.float32)
    with pytest.raises(ValueError, match="Double-counting detected: total_membrane_current_proxy \\+ synaptic_current_proxy"):
        jtfne.construct_source_tensor(
            mode="invalid_double_count_mode",
            total_membrane_current=total_membrane
        )


def test_source_construction_jit():
    """Verify that construct_source_tensor compiles under JIT when inputs are JAX arrays."""
    cap_ion = jnp.array([1.0, 2.0], dtype=jnp.float32)
    synaptic = jnp.array([0.5, 0.5], dtype=jnp.float32)
    
    @jax.jit
    def compile_mode_b(ci, syn):
        # We pass static string argument mode
        source, _ = jtfne.construct_source_tensor(
            mode="decomposed_cap_ion_plus_synaptic_proxy",
            total_membrane_current=jnp.zeros_like(ci),
            decomposed_cap_ion=ci,
            synaptic_current=syn
        )
        return source
        
    source = compile_mode_b(cap_ion, synaptic)
    assert np.allclose(source, cap_ion + synaptic)
