"""Smoke example for the generalized 3D Izhikevich TFNE model.

Demonstrates simulation setup, JAX step rollouts, SynapseLayer, and manifest creation.
"""

import os
import json
import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne


def main():
    print("Running generalized 3D Izhikevich TFNE E2E smoke...")
    
    n_neurons = 20
    duration_ms = 1000.0
    dt_ms = 0.1
    n_steps = int(duration_ms / dt_ms)
    
    # 1. Emitters Setup
    cell_types = {"E": 0.8, "PV": 0.2}
    emitter = jtfne.emitters.IzhikevichEmitter(n=n_neurons, dtype="float32")
    emitter_state = emitter.initial_state(seed=42)
    
    # 2. Synapse & Connectivity Setup
    rng = np.random.default_rng(42)
    # Excitatory weight signs (+), Inhibitory (-), zero diagonal (no self-connections)
    sign = jnp.array([1.0 if t == "E" else -1.0 for t in emitter.params.labels], dtype=jnp.float32)
    W = jnp.asarray(rng.uniform(0.1, 0.5, (n_neurons, n_neurons)).astype(np.float32)) * sign[None, :]
    W = W * (1.0 - jnp.eye(n_neurons, dtype=jnp.float32))
    
    synapse = jtfne.emitters.SynapseLayer(n=n_neurons, W=W, tau_ms=5.0)
    synapse_state = synapse.initial_state()
    
    # 3. Simulate Rollout via jax.lax.scan
    def body_fn(carry, _):
        emit_s, syn_s = carry
        
        # 3.1 Get synaptic input current
        # For simplicity in this E2E smoke, input current drives the emitters
        next_syn_s, I_syn = synapse.step(syn_s, emit_s.spikes, dt_ms=dt_ms)
        
        # 3.2 Update local emitter state
        next_emit_s, out = emitter.step(emit_s, I_syn, dt_ms=dt_ms)
        
        return (next_emit_s, next_syn_s), (out.voltage, out.spikes, I_syn)

    init_state = (emitter_state, synapse_state)
    
    @jax.jit
    def run_simulation():
        return jax.lax.scan(body_fn, init_state, xs=None, length=n_steps)
        
    _, (voltages, spikes, I_syn_seq) = run_simulation()
    
    print(f"Simulation completed successfully over {n_steps} steps!")
    print(f"Voltages shape: {voltages.shape}, Spikes shape: {spikes.shape}")
    
    # 4. Source & manifest bookkeeping
    source, source_report = jtfne.construct_source_tensor(
        mode="total_membrane_current_proxy",
        total_membrane_current=I_syn_seq
    )
    
    # Verify outputs are finite
    assert jnp.all(jnp.isfinite(voltages))
    assert jnp.all(jnp.isfinite(source))
    
    # 5. Export JSON manifest
    manifest_data = {
        "run_id": "generalized_tfne_3d_smoke_00",
        "jaxfne_version": jtfne.__version__,
        "truth_mode": "truth_safe_unverified",
        "seed": 42,
        "n_steps": n_steps,
        "n_neurons": n_neurons,
        "dt_ms": dt_ms,
        "source_bookkeeping": source_report,
        "field_solver_status": "laminar_proxy_no_pde",
        "physical_amplitude_claim_allowed": False
    }
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/generalized_tfne_manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)
    print("Manifest exported to outputs/generalized_tfne_manifest.json!")


if __name__ == "__main__":
    main()
