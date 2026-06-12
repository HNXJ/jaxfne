"""STDP activity-dependent plasticity and chunked simulation runner.

Provides the 100-neuron E-I network geometry, STDP adaptation rule, and
streaming simulation functionality under computational_scaffold gates.
"""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

def build_stdp_network_geometry(
    n_neurons: int = 100,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generates random 3D positions inside a unit sphere.

    Returns:
        positions: shape (n_neurons, 3)
        exc_mask: shape (n_neurons,) boolean array (70% Excitatory)
        inh_mask: shape (n_neurons,) boolean array (30% Inhibitory)
    """
    key = jr.PRNGKey(seed)
    pos_key, _ = jr.split(key)
    
    # Generate points inside unit sphere
    phi = jr.uniform(pos_key, (n_neurons,), minval=0.0, maxval=2 * jnp.pi)
    costheta = jr.uniform(pos_key, (n_neurons,), minval=-1.0, maxval=1.0)
    u = jr.uniform(pos_key, (n_neurons,), minval=0.0, maxval=1.0)
    
    theta = jnp.arccos(costheta)
    r = u ** (1.0 / 3.0)
    
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    
    positions = jnp.stack([x, y, z], axis=1)
    
    # 70% Excitatory, 30% Inhibitory
    n_exc = int(0.7 * n_neurons)
    exc_mask = jnp.arange(n_neurons) < n_exc
    inh_mask = ~exc_mask
    
    return positions, exc_mask, inh_mask

def build_initial_stdp_weights(
    n_neurons: int = 100,
    exc_mask: jnp.ndarray = None,
    seed: int = 42
) -> jnp.ndarray:
    """Generates the initial weight matrix W[i,j] where E-to-all is positive and I-to-all is negative.

    Each connection weight resides in [0, 1] for E connections and [-1, 0] for I connections.
    """
    if exc_mask is None:
        n_exc = int(0.7 * n_neurons)
        exc_mask = jnp.arange(n_neurons) < n_exc
        
    key = jr.PRNGKey(seed)
    W = jr.uniform(key, (n_neurons, n_neurons), minval=0.01, maxval=0.1)
    
    # Apply signs based on source neuron type (columns of W represent presynaptic source)
    # W[i, j] represents connection from j (source) to i (destination)
    # If source j is inhibitory, weight must be negative.
    W_signed = jnp.where(exc_mask[None, :], W, -W)
    
    # Remove self-connections
    W_signed = W_signed * (1.0 - jnp.eye(n_neurons))
    return W_signed

def generate_triangular_drive(
    duration_ms: float,
    dt_ms: float,
    freq_hz: float = 6.0,
    amplitude: float = 5.0
) -> jnp.ndarray:
    """Generates a 6 Hz triangular drive trace."""
    n_steps = int(duration_ms / dt_ms)
    t = np.arange(n_steps) * dt_ms / 1000.0  # seconds
    period = 1.0 / freq_hz
    # Triangular wave formula: 2 * |2 * (t/period - floor(t/period + 0.5))|
    phase = t / period - np.floor(t / period + 0.5)
    wave = 2.0 * np.abs(2.0 * phase)
    return jnp.array(amplitude * (wave - 1.0))

def simulate_stdp_step(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    inputs: Tuple[float, float, float, float],
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
    exc_mask: jnp.ndarray,
    inh_mask: jnp.ndarray,
    dt_ms: float,
    plasticity_scale: float
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Single forward Euler step with STDP updates using lax.scan compatibility."""
    v, u, s, trace_pre, W = state
    stim_val, noise_val, A_plus, A_minus = inputs
    
    # Dynamics (Izhikevich)
    # Total input: stimulus drive + background noise + synaptic input (s)
    I = stim_val + noise_val + s
    dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
    v_next = v + dt_ms * dv
    
    du = a * (b * v - u)
    u_next = u + dt_ms * du
    
    # Spike detection
    spiked = v_next >= 30.0
    v_next = jnp.where(spiked, c, v_next)
    u_next = jnp.where(spiked, u_next + d, u_next)
    
    # Update synaptic trace
    # Synapse decay time constant = 5.0 ms
    s_next = s * (1.0 - dt_ms / 5.0) + jnp.dot(W, spiked.astype(jnp.float32))
    
    # STDP activity-dependent adaptation trace
    # pre-synaptic trace decay time constant = 20.0 ms
    trace_pre_next = trace_pre * (1.0 - dt_ms / 20.0) + spiked.astype(jnp.float32)
    
    # STDP Update:
    # dW_ij: j (pre) to i (post)
    # Potentiation (post-synaptic spike i): dW_ij += plasticity_scale * A_plus * trace_pre_j
    # Depression (pre-synaptic spike j): dW_ij -= plasticity_scale * A_minus * trace_post_i
    post_spike = spiked[:, None]  # shape (n_neurons, 1)
    pre_spike = spiked[None, :]   # shape (1, n_neurons)
    
    dW_ltp = post_spike * trace_pre[None, :] * A_plus
    trace_post = trace_pre  # symmetric trace for simplicity
    dW_ltd = pre_spike * trace_post[:, None] * A_minus
    
    dW = plasticity_scale * (dW_ltp - dW_ltd)
    
    # Apply updates only to excitatory synapses to preserve signs
    update_mask = exc_mask[None, :] & (~jnp.eye(v.shape[0], dtype=bool))
    W_next = W + jnp.where(update_mask, dW, 0.0)
    
    # Clip excitatory weights to [0.0, 1.5]
    W_next = jnp.where(exc_mask[None, :], jnp.clip(W_next, 0.0, 1.5), W_next)
    
    # Self-connections remain zero
    W_next = W_next * (1.0 - jnp.eye(v.shape[0]))
    
    return (v_next, u_next, s_next, trace_pre_next, W_next), (v_next, spiked)

def run_stdp_simulation_chunk(
    v_init: jnp.ndarray,
    u_init: jnp.ndarray,
    s_init: jnp.ndarray,
    trace_pre_init: jnp.ndarray,
    W_init: jnp.ndarray,
    stim_drive: jnp.ndarray,
    noise: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
    exc_mask: jnp.ndarray,
    inh_mask: jnp.ndarray,
    dt_ms: float,
    plasticity_scale: float,
    A_plus: float = 0.01,
    A_minus: float = 0.012
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Runs a single chunk of simulation using lax.scan."""
    n_steps = stim_drive.shape[0]
    
    # Pack inputs: stim, noise, A_plus, A_minus
    inputs_in = (
        stim_drive,
        noise,
        jnp.full((n_steps,), A_plus),
        jnp.full((n_steps,), A_minus)
    )
    
    state_init = (v_init, u_init, s_init, trace_pre_init, W_init)
    
    def step_wrapper(state, inputs):
        return simulate_stdp_step(
            state, inputs, a, b, c, d, exc_mask, inh_mask, dt_ms, plasticity_scale
        )
        
    state_final, (vm_traj, spk_traj) = jax.lax.scan(
        step_wrapper, state_init, inputs_in
    )
    return state_final, (vm_traj, spk_traj)
