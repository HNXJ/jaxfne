"""Tests for global superposition and scaling of linear readout operators.

Verifies the mathematical contract: F(S1 + S2) ≈ F(S1) + F(S2) and F(aS) ≈ aF(S)
for all linear readout types.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_operator_superposition_and_scaling():
    """Verify linear superposition and scaling on EEG, MEG, LFP, and CSD maps."""
    c = 4
    n = 10
    
    rng = np.random.default_rng(100)
    W = rng.standard_normal((c, n)).astype(np.float32)
    s1 = rng.standard_normal((50, n)).astype(np.float32)
    s2 = rng.standard_normal((50, n)).astype(np.float32)
    
    # 1. Standard LinearReadout operator
    readout = jtfne.fields.LinearReadout(name="generic_lfp", W=W)
    
    y_s1 = readout.apply(s1)
    y_s2 = readout.apply(s2)
    y_sum = readout.apply(s1 + s2)
    
    assert np.allclose(y_sum, y_s1 + y_s2, rtol=1e-5, atol=1e-5)
    
    a = 3.14
    y_scale = readout.apply(a * s1)
    assert np.allclose(y_scale, a * y_s1, rtol=1e-5, atol=1e-5)
    
    # 2. Verify that EEG, MEG proxy transformations satisfy superposition
    # eeg_proxy_transform(source, leadfield)
    leadfield = W
    eeg1 = jtfne.fields.eeg_proxy_transform(s1, leadfield)
    eeg2 = jtfne.fields.eeg_proxy_transform(s2, leadfield)
    eeg_sum = jtfne.fields.eeg_proxy_transform(s1 + s2, leadfield)
    
    assert np.allclose(eeg_sum, eeg1 + eeg2, rtol=1e-5, atol=1e-5)
    
    # MEG proxy transform
    meg1 = jtfne.fields.meg_proxy_transform(s1, leadfield)
    meg2 = jtfne.fields.meg_proxy_transform(s2, leadfield)
    meg_sum = jtfne.fields.meg_proxy_transform(s1 + s2, leadfield)
    
    assert np.allclose(meg_sum, meg1 + meg2, rtol=1e-5, atol=1e-5)
