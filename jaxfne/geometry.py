"""Geometry helper functions for jaxfne."""

from __future__ import annotations
from typing import Tuple
import jax.numpy as jnp
import jax.random as jr

def make_ei_cloud_network(n_neurons: int = 100, seed: int = 42) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generates geometry and initial weights for a 100-neuron E-I cloud network.
    
    Returns:
        positions: (n_neurons, 3) positions inside a unit sphere.
        exc_mask: (n_neurons,) boolean mask for excitatory neurons.
        inh_mask: (n_neurons,) boolean mask for inhibitory neurons.
        W: (n_neurons, n_neurons) initial signed weight matrix W[post, pre] (W[dst, src]).
    """
    key = jr.PRNGKey(seed)
    pos_key, w_key = jr.split(key)
    phi_key, costheta_key, u_key = jr.split(pos_key, 3)

    # Generate positions inside a unit sphere
    phi = jr.uniform(phi_key, (n_neurons,), minval=0.0, maxval=2 * jnp.pi)
    costheta = jr.uniform(costheta_key, (n_neurons,), minval=-1.0, maxval=1.0)
    u = jr.uniform(u_key, (n_neurons,), minval=0.0, maxval=1.0)
    
    theta = jnp.arccos(costheta)
    r = u ** (1.0 / 3.0)
    
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    
    positions = jnp.stack([x, y, z], axis=1)
    
    # 70% Excitatory / 30% Inhibitory split
    n_exc = int(0.7 * n_neurons)
    exc_mask = jnp.arange(n_neurons) < n_exc
    inh_mask = ~exc_mask
    
    # Generate initial weights W[post, pre]
    W = jr.uniform(w_key, (n_neurons, n_neurons), minval=0.01, maxval=0.1)
    
    # Apply signs based on presynaptic source type (pre-synaptic is column index j)
    W_signed = jnp.where(exc_mask[None, :], W, -W)
    
    # Enforce no autapses/self-connections
    W_signed = W_signed * (1.0 - jnp.eye(n_neurons))
    
    return positions, exc_mask, inh_mask, W_signed
