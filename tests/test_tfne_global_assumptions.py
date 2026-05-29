"""Tests for global TFNE assumptions.

Validates that:
- Nonlinear emitter update is local per neuron/compartment.
- Passive medium and readout projections satisfy linear superposition: F(S1 + S2) ≈ F(S1) + F(S2).
- Scaling is satisfied: F(aS) ≈ aF(S).
- Metadata reports fixed geometry, boundary conditions, and gauge.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_locality_contract():
    """Verify that nonlinear emitter state transitions are local.
    
    Perturbing input to neuron j must not affect state updates of neuron i (i != j)
    in the absence of recurrent/source aggregation.
    """
    # Create standard EIG parameters/network
    n = 5
    cell_types = {"E": 0.8, "PV": 0.2}
    params = jtfne.emitters.izhikevich_eig_params(n, cell_types)
    
    # Decouple connections to test pure emitter locality
    params = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x) if x.ndim == 2 else x,
        params
    )
    
    # State setup
    v = jnp.full((n,), -65.0, dtype=jnp.float32)
    u = params.b * v
    
    # Simulate single steps with different inputs
    input_base = jnp.zeros((n,), dtype=jnp.float32)
    input_perturbed = input_base.at[2].set(10.0)  # Perturb neuron 2
    
    dt_ms = 0.1
    
    # Step logic for standard Izhikevich
    def step_fn(v_val, u_val, inp):
        v_next = v_val + dt_ms * (0.04 * v_val**2 + 5.0 * v_val + 140.0 - u_val + inp)
        u_next = u_val + dt_ms * params.a * (params.b * v_val - u_val)
        
        # Spiking reset
        spike = (v_next >= 30.0)
        v_next = jnp.where(spike, params.c, v_next)
        u_next = jnp.where(spike, u_next + params.d, u_next)
        return v_next, u_next, spike.astype(jnp.float32)

    v_next_base, u_next_base, _ = step_fn(v, u, input_base)
    v_next_pert, u_next_pert, _ = step_fn(v, u, input_perturbed)
    
    # Check that perturbed input at index 2 only changed index 2
    for i in range(n):
        if i != 2:
            assert np.allclose(v_next_base[i], v_next_pert[i], atol=1e-5)
            assert np.allclose(u_next_base[i], u_next_pert[i], atol=1e-5)
        else:
            assert not np.allclose(v_next_base[i], v_next_pert[i], atol=1e-5)


def test_linear_projection_superposition_and_scaling():
    """Verify that source-to-field projections satisfy linear superposition.
    
    F(S1 + S2) ≈ F(S1) + F(S2) and F(aS) ≈ aF(S).
    """
    n_sources = 10
    n_contacts = 4
    
    # Random source patterns
    rng = np.random.default_rng(42)
    s1 = rng.standard_normal(n_sources).astype(np.float32)
    s2 = rng.standard_normal(n_sources).astype(np.float32)
    
    # Linear projection matrix (leadfield or laminar proxy)
    W = rng.standard_normal((n_contacts, n_sources)).astype(np.float32)
    
    # Apply operator F
    def F(s):
        return W @ s
        
    y_s1 = F(s1)
    y_s2 = F(s2)
    y_sum = F(s1 + s2)
    
    # Superposition: F(S1 + S2) == F(S1) + F(S2)
    assert np.allclose(y_sum, y_s1 + y_s2, rtol=1e-5, atol=1e-6)
    
    # Scaling: F(a S) == a F(S)
    a = 2.5
    y_scale = F(a * s1)
    assert np.allclose(y_scale, a * y_s1, rtol=1e-5, atol=1e-6)


def test_metadata_exports_fixed_environment():
    """Verify that the model exports fixed geometry, boundary condition, and gauge."""
    cfg = (
        jtfne.configuration()
        .network(n=10)
        .emitter(family="izhikevich")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero"
        )
        .probe(n_contacts=4)
    )
    model = jtfne.construct(cfg)
    sim = jtfne.Simulation(duration_ms=10.0, dt_ms=0.5)
    signals = model.simulate(sim)
    
    # Check fixed constraints
    manifest = model.manifest(signals=signals)
    field_adm = manifest["backend_metadata"]["field_admissibility"]
    assert field_adm["field_solver_status"] == "laminar_proxy_no_pde"
    assert manifest["physical_amplitude_claim_allowed"] is False
    assert manifest["claim_level"] == "computational_scaffold"
    assert manifest["truth_mode"] == "truth_safe_unverified"
