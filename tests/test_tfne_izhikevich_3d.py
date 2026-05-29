"""E2E Integration test for the generalized 3D Izhikevich TFNE model.

Verifies end-to-end simulation rollout, connectivity, source projection,
and strict JSON-safe manifest serialization.
"""

import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_tfne_izhikevich_3d_integration():
    """Verify end-to-end integration rollout of 3D Izhikevich and Synapse connectivity."""
    n_neurons = 10
    duration_ms = 100.0
    dt_ms = 0.1
    n_steps = int(duration_ms / dt_ms)
    
    # Emitters Setup
    emitter = jtfne.emitters.IzhikevichEmitter(n=n_neurons, dtype="float32")
    state = emitter.initial_state(seed=42)
    
    # 3D declared positions and Connectivity Setup
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 1, (n_neurons, 3)).astype(np.float32)
    
    # Distance-dependent weights W[i, j] = exp(-d_ij)
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    W = np.exp(-dists) * (1.0 - np.eye(n_neurons))  # no self connections
    
    # Scale inhibitory weights
    sign = np.array([1.0 if t == "E" else -1.0 for t in emitter.params.labels])
    W = W * sign[None, :]
    W_jax = jnp.asarray(W, dtype=jnp.float32)
    
    synapse = jtfne.emitters.SynapseLayer(n=n_neurons, W=W_jax, tau_ms=5.0)
    syn_state = synapse.initial_state()
    
    # Step function coupling synapse currents to emitters
    def body_fn(carry, _):
        emit_s, syn_s = carry
        next_syn_s, I_syn = synapse.step(syn_s, emit_s.spikes, dt_ms=dt_ms)
        next_emit_s, out = emitter.step(emit_s, I_syn, dt_ms=dt_ms)
        return (next_emit_s, next_syn_s), (out.voltage, out.spikes, I_syn)

    init_state = (state, syn_state)
    
    # Rollout simulation
    _, (voltages, spikes, I_syn_seq) = jax.lax.scan(body_fn, init_state, xs=None, length=n_steps)
    
    assert voltages.shape == (n_steps, n_neurons)
    assert spikes.shape == (n_steps, n_neurons)
    assert np.all(np.isfinite(voltages))
    assert np.all(np.isfinite(spikes))
    
    # 4. Expose separate LFP linear operator projection
    W_lfp = jnp.ones((2, n_neurons), dtype=jnp.float32)  # 2 contacts LFP
    readout = jtfne.fields.LinearReadout(name="lfp_like", W=W_lfp)
    
    lfp = readout.apply(I_syn_seq)
    assert lfp.shape == (n_steps, 2)
    assert np.all(np.isfinite(lfp))
    
    # 5. Verify JSON manifest serialization safety
    source, report = jtfne.construct_source_tensor(
        mode="total_membrane_current_proxy",
        total_membrane_current=I_syn_seq
    )
    
    manifest_data = {
        "run_id": "test_tfne_izhikevich_3d_integration",
        "jaxfne_version": jtfne.__version__,
        "truth_mode": "truth_safe_unverified",
        "seed": 42,
        "n_steps": n_steps,
        "n_neurons": n_neurons,
        "dt_ms": dt_ms,
        "source_bookkeeping": report,
        "field_solver_status": "laminar_proxy_no_pde",
        "physical_amplitude_claim_allowed": False
    }
    
    # Strict JSON parsing
    json_str = json.dumps(manifest_data, allow_nan=False)
    parsed = json.loads(json_str)
    assert parsed["run_id"] == "test_tfne_izhikevich_3d_integration"
