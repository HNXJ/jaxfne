"""Tests for Jaxley-focused biophysical emitter bridge (JaxleyBridge).

Validates source extraction on mock compartmental signals, unimplemented exception guards,
and bridge reporting metadata.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_jaxley_bridge_creation_and_report():
    """Verify JaxleyBridge parameters and reporting metadata."""
    bridge = jtfne.bridges.JaxleyBridge(
        model="mock_model",
        source_mode="transmembrane_current",
        compartment_axis="last"
    )
    
    assert bridge.source_mode == "transmembrane_current"
    assert bridge.compartment_axis == "last"
    
    report = bridge.report()
    assert report["bridge_name"] == "jaxley_bridge"
    assert report["source_mode"] == "transmembrane_current"
    assert report["physical_amplitude_claim_allowed"] is False
    assert report["source_calibration_status"] == "uncalibrated_jaxley_bridge"


def test_jaxley_bridge_extract_sources():
    """Verify that extract_sources works correctly on mock compartment fixtures.
    
    Fixture shape: [T, N, C] or dynamic membrane traces.
    """
    bridge = jtfne.bridges.JaxleyBridge(model="mock_model")
    
    # 1. From signals-like object with sources
    class MockSignals:
        def __init__(self, sources, V_m=None):
            self.sources = sources
            self.V_m = V_m
            
    sources_in = jnp.ones((100, 2, 3), dtype=jnp.float32)
    sig = MockSignals(sources=sources_in)
    
    extracted = bridge.extract_sources(sig)
    assert np.allclose(extracted, sources_in)
    
    # 2. From signals-like object fallback to V_m
    vm_in = jnp.ones((100, 2, 3), dtype=jnp.float32) * -65.0
    sig_vm = MockSignals(sources=None, V_m=vm_in)
    
    extracted_vm = bridge.extract_sources(sig_vm)
    assert np.allclose(extracted_vm, vm_in)
    
    # 3. Direct array extraction fallback
    arr_in = jnp.ones((50, 4), dtype=jnp.float32)
    assert np.allclose(bridge.extract_sources(arr_in), arr_in)


def test_jaxley_bridge_unimplemented_fail_loudly(monkeypatch):
    """Verify that unimplemented bridge methods (like simulate) fail loudly."""
    import jaxfne.bridges
    monkeypatch.setattr(jaxfne.bridges, "require_jaxley", lambda: None)
    
    bridge = jtfne.bridges.JaxleyBridge(model="mock_model")
    
    with pytest.raises(NotImplementedError, match="JaxleyBridge.simulate"):
        bridge.simulate()
