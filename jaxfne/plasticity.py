"""STDP plasticity rules, state models, and analysis utilities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import jax
import jax.numpy as jnp
import numpy as np

@dataclass
class STDPPlasticityConfig:
    """Configuration class for STDP activity-dependent plasticity."""
    A_plus: float = 0.01
    A_minus: float = 0.012
    tau_plus: float = 20.0  # ms
    tau_minus: float = 20.0 # ms
    w_min: float = 0.0
    w_max: float = 1.5

@dataclass
class STDPState:
    """Container for the state variables of the STDP synapse model."""
    W: jnp.ndarray          # shape (n_neurons, n_neurons)
    trace_pre: jnp.ndarray   # shape (n_neurons,)
    trace_post: jnp.ndarray  # shape (n_neurons,)

def summarize_stdp_adaptation(W_before: np.ndarray, W_after: np.ndarray) -> Dict[str, Any]:
    """Computes synapse-by-synapse adaptation statistics.
    
    Args:
        W_before: Weight matrix before adaptation.
        W_after: Weight matrix after adaptation.
        
    Returns:
        dict: Summary metrics.
    """
    delta_W = W_after - W_before
    
    ltp_count = int(np.sum(delta_W > 1e-6))
    ltd_count = int(np.sum(delta_W < -1e-6))
    unchanged_count = int(np.sum(np.abs(delta_W) <= 1e-6))
    
    W_before_mean = float(np.mean(W_before))
    W_after_mean = float(np.mean(W_after))
    
    # Sparsity = fraction of zero weights
    W_after_sparsity = float(np.mean(np.abs(W_after) < 1e-9))
    
    delta_W_min = float(np.min(delta_W))
    delta_W_max = float(np.max(delta_W))
    
    finite_checks = bool(np.all(np.isfinite(W_after)))
    
    # Sign preservation check: W_before and W_after must have the same sign (excluding zeros)
    sign_preservation = bool(np.all(W_before * W_after >= -1e-9))
    
    return {
        "ltp_count": ltp_count,
        "ltd_count": ltd_count,
        "unchanged_count": unchanged_count,
        "W_before_mean": W_before_mean,
        "W_after_mean": W_after_mean,
        "W_after_sparsity": W_after_sparsity,
        "delta_W_min": delta_W_min,
        "delta_W_max": delta_W_max,
        "finite_checks": finite_checks,
        "sign_preservation": sign_preservation
    }



@jax.jit
def update_stdp_weights_jax(
    W: jax.Array,
    trace_pre: jax.Array,
    trace_post: jax.Array,
    spiked: jax.Array,
    exc_mask: jax.Array,
    A_plus: float,
    A_minus: float,
    plasticity_scale: float,
    w_min: float,
    w_max: float,
) -> jax.Array:
    """JAX-native plasticity weight update kernel (STDP).

    Parameters
    ----------
    W : jax.Array
        Synaptic weight matrix of shape (n_neurons, n_neurons).
    trace_pre : jax.Array
        Presynaptic traces of shape (n_neurons,).
    trace_post : jax.Array
        Postsynaptic traces of shape (n_neurons,).
    spiked : jax.Array
        Boolean spike indicator array of shape (n_neurons,).
    exc_mask : jax.Array
        Excitatory cell mask of shape (n_neurons,).
    A_plus, A_minus : float
        LTP and LTD rate parameters.
    plasticity_scale : float
        Global scaling factor.
    w_min, w_max : float
        Synaptic weight limits.

    Returns
    -------
    jax.Array
        Updated weight matrix of shape (n_neurons, n_neurons).
    """
    post_spike = spiked[:, None]
    pre_spike = spiked[None, :]
    
    # LTP: pre active, then post spikes (potentiate)
    dW_ltp = post_spike * trace_pre[None, :] * A_plus
    # LTD: post active, then pre spikes (depress)
    dW_ltd = pre_spike * trace_post[:, None] * A_minus
    dW = plasticity_scale * (dW_ltp - dW_ltd)
    
    # Enforce E/I sign preservation and exclude self-connections
    update_mask = exc_mask[None, :] & (~jnp.eye(W.shape[0], dtype=bool))
    W_next = W + jnp.where(update_mask, dW, 0.0)
    
    # Clip excitatory weights to [w_min, w_max]
    W_next = jnp.where(exc_mask[None, :], jnp.clip(W_next, w_min, w_max), W_next)
    
    # Enforce no self-connections
    W_next = W_next * (1.0 - jnp.eye(W.shape[0]))
    return W_next

