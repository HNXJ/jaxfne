"""Targeted unit tests for the v0.3.10 EEG/MEG/EMM proxy operators.

Validates that:
- Distinct readout operators are accessible under the public API.
- Operators generate separate, distinct output signals and reports.
- Reports conform to the required JSON contracts.
- Strictly adheres to the non-negotiable scientific scope boundary.
"""

import pytest
import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne.fields import (
    eeg_proxy_probe,
    meg_proxy_probe,
    emm_proxy_probe,
)

def test_v0310_operator_separation():
    """Verify that EEG, MEG, and EMM-proxy are distinct operators producing separate outputs."""
    T = 100
    n_sensors = 4
    
    # Declare dummy LFP field signals [T, space]
    lfp_dummy = jnp.arange(T * n_sensors, dtype=jnp.float32).reshape(T, n_sensors)
    
    # 1. EEG-proxy
    eeg_readout = eeg_proxy_probe(lfp_dummy, leadfield_status="toy_or_declared_proxy", n_sensors=n_sensors)
    assert eeg_readout.name == "eeg_proxy"
    assert eeg_readout.kind == "eeg_proxy"
    assert eeg_readout.data.shape == (T, n_sensors)
    
    # 2. MEG-proxy
    meg_readout = meg_proxy_probe(lfp_dummy, leadfield_status="toy_or_declared_proxy", orientation_convention="declared", n_sensors=n_sensors)
    assert meg_readout.name == "meg_proxy"
    assert meg_readout.kind == "meg_proxy"
    assert meg_readout.data.shape == (T, n_sensors)
    
    # 3. EMM-proxy
    # Typically EMM is computed as a 1D timeline of metabolic activity costs
    emm_dummy = jnp.mean(jnp.abs(lfp_dummy), axis=1)
    emm_readout = emm_proxy_probe(emm_dummy, method="normalized_activity_field_source_cost_proxy")
    assert emm_readout.name == "emm_proxy"
    assert emm_readout.kind == "emm_proxy"
    assert emm_readout.data.shape == (T,)

    # Verify distinctness: reports have different structures and assumptions
    assert eeg_readout.report["kind"] == "eeg_proxy"
    assert meg_readout.report["kind"] == "meg_proxy"
    assert emm_readout.report["kind"] == "emm_proxy"
    
    assert "linear_leadfield" in eeg_readout.report["method"]
    assert "current_orientation" in meg_readout.report["method"]
    assert "activity_field" in emm_readout.report["method"]


def test_v0310_non_negotiable_boundaries():
    """Verify that all three operators strictly honor the non-negotiable scope limits."""
    lfp_dummy = jnp.ones((50, 2))
    emm_dummy = jnp.ones((50,))
    
    readouts = [
        eeg_proxy_probe(lfp_dummy),
        meg_proxy_probe(lfp_dummy),
        emm_proxy_probe(emm_dummy),
    ]
    
    for r in readouts:
        report = r.report
        
        # 1. No physical amplitude claims are allowed
        assert report["physical_amplitude_claim_allowed"] is False
        
        # 2. Field solver is laminar proxy (no PDE)
        assert report["field_solver_status"] == "laminar_proxy_no_pde"
        
        # 3. Explicit uncalibrated/proxy status in assumptions
        assumptions = report["assumptions"]
        assert any("not_empirically_calibrated" in a or "uncalibrated_proxy" in a or "not_biological_metabolism" in a for a in assumptions)
