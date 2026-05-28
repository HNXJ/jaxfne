"""Smoke example for the generalized LinearReadout TFNE operators.

Demonstrates LFP/EEG/MEG proxy transforms, custom LinearReadout maps, and superposition checks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os

import jaxfne as jtfne


def main():
    print("Running generalized LinearReadout TFNE E2E smoke...")
    
    n_neurons = 10
    n_contacts = 4
    n_steps = 500
    
    # 1. Generate simulated source traces S[t, n]
    rng = np.random.default_rng(100)
    s1 = rng.standard_normal((n_steps, n_neurons)).astype(np.float32)
    s2 = rng.standard_normal((n_steps, n_neurons)).astype(np.float32)
    
    # 2. Build LinearReadout operator
    W_lfp = rng.standard_normal((n_contacts, n_neurons)).astype(np.float32)
    readout_lfp = jtfne.fields.LinearReadout(
        name="lfp_like",
        W=jnp.asarray(W_lfp),
        input_key="source"
    )
    
    # Apply readouts
    lfp1 = readout_lfp.apply(jnp.asarray(s1))
    lfp2 = readout_lfp.apply(jnp.asarray(s2))
    lfp_sum = readout_lfp.apply(jnp.asarray(s1 + s2))
    
    # 3. Superposition validation: F(S1 + S2) == F(S1) + F(S2)
    superposition_holds = bool(np.allclose(lfp_sum, lfp1 + lfp2, rtol=1e-5, atol=1e-5))
    print(f"LFP linear superposition holds: {superposition_holds}")
    assert superposition_holds
    
    # 4. EEG and MEG proxy transforms
    W_eeg = rng.standard_normal((n_contacts, n_neurons)).astype(np.float32)
    eeg1 = jtfne.fields.eeg_proxy_transform(jnp.asarray(s1), jnp.asarray(W_eeg))
    
    W_meg = rng.standard_normal((n_contacts, n_neurons)).astype(np.float32)
    meg1 = jtfne.fields.meg_proxy_transform(jnp.asarray(s1), jnp.asarray(W_meg))
    
    print(f"EEG output shape: {eeg1.shape}, MEG output shape: {meg1.shape}")
    
    # Verify outputs are finite
    assert jnp.all(jnp.isfinite(lfp1))
    assert jnp.all(jnp.isfinite(eeg1))
    assert jnp.all(jnp.isfinite(meg1))
    
    # 5. Export JSON validation report
    validation_report = {
        "readout_operator": "LinearReadout",
        "n_contacts": n_contacts,
        "n_neurons": n_neurons,
        "superposition_test": "passed" if superposition_holds else "failed",
        "physical_amplitude_claim_allowed": False,
        "operator_status": readout_lfp.report()
    }
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/generalized_readout_report.json", "w") as f:
        json.dump(validation_report, f, indent=2)
    print("Report exported to outputs/generalized_readout_report.json!")


if __name__ == "__main__":
    main()
