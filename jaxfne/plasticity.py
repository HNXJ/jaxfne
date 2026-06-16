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

def plot_stdp_adaptation_suite(
    trajectories: Dict[str, Any],
    W_before: np.ndarray,
    W_after: np.ndarray,
    stimulus: np.ndarray,
    fig_dir: str,
    prefix: str = ""
) -> None:
    """Generates and saves the standard STDP adaptation visualization figures.
    
    All figures are saved with size >= 1000x500.
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    
    vm = trajectories["vm"]
    spk = trajectories["spk"]
    dt_ms = 0.5 * 10  # Accounts for downsampling of 10
    times = np.arange(spk.shape[0]) * dt_ms / 1000.0
    
    # 1. Raster
    fig, ax = plt.subplots(figsize=(11, 6))
    spk_idx, neuron_idx = np.where(spk > 0.0)
    ax.scatter(times[spk_idx], neuron_idx, s=1, color="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron ID")
    ax.set_title("100-Neuron Spike Raster (STDP Proxy Readout)")
    plt.savefig(os.path.join(fig_dir, f"{prefix}raster.png"), dpi=100)
    plt.close()
    
    # 2. Population rate
    fig, ax = plt.subplots(figsize=(11, 6))
    bin_size_ms = 100.0
    bin_steps = int(bin_size_ms / dt_ms)
    rates = []
    for i in range(0, spk.shape[0], bin_steps):
        rates.append(np.mean(spk[i:i+bin_steps]) * 1000.0 / dt_ms)
    ax.plot(np.arange(len(rates)) * bin_size_ms / 1000.0, rates, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_title("Population Firing Rate (100 ms bins)")
    plt.savefig(os.path.join(fig_dir, f"{prefix}population_rates.png"), dpi=100)
    plt.close()
    
    # 3. Stimulus Trace (First 2000 points of downsampled, equivalent to first 10s)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(times[:2000], stimulus[:2000 * 10:10], color="orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drive Current (a.u.)")
    ax.set_title("Triangular Stimulus Drive")
    plt.savefig(os.path.join(fig_dir, f"{prefix}stimulus_trace.png"), dpi=100)
    plt.close()
    
    # 4. W before/after/delta heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    delta_W = W_after - W_before
    
    im0 = axes[0].imshow(W_before, aspect="auto", cmap="viridis")
    axes[0].set_title("W Before")
    axes[0].set_xlabel("Presynaptic ID")
    axes[0].set_ylabel("Postsynaptic ID")
    fig.colorbar(im0, ax=axes[0], label="Weight")
    
    im1 = axes[1].imshow(W_after, aspect="auto", cmap="viridis")
    axes[1].set_title("W After")
    axes[1].set_xlabel("Presynaptic ID")
    axes[1].set_ylabel("Postsynaptic ID")
    fig.colorbar(im1, ax=axes[1], label="Weight")
    
    im2 = axes[2].imshow(delta_W, aspect="auto", cmap="bwr")
    axes[2].set_title("Delta W")
    axes[2].set_xlabel("Presynaptic ID")
    axes[2].set_ylabel("Postsynaptic ID")
    fig.colorbar(im2, ax=axes[2], label="Weight Delta")
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}W_heatmaps.png"), dpi=100)
    plt.close()


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

