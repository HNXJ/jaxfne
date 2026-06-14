"""Memory-safe chunked streaming simulation runner for jaxfne."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from .solvers import SolverConfig
from .plasticity import STDPPlasticityConfig, STDPState

def simulate_stdp_euler_step(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    inputs: Tuple[float, float, float, float],
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
    exc_mask: jnp.ndarray,
    inh_mask: jnp.ndarray,
    dt_ms: float,
    plasticity_scale: float,
    w_min: float,
    w_max: float,
    tau_plus: float,
    tau_minus: float
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Single ODE Euler step for Izhikevich neurons and STDP trace/weight updates."""
    v, u, s, trace_pre, trace_post, W = state
    stim_val, noise_val, A_plus, A_minus = inputs
    
    # ODE Euler solver updates for Izhikevich state variables
    I = stim_val + noise_val + s
    dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
    v_next = v + dt_ms * dv
    
    du = a * (b * v - u)
    u_next = u + dt_ms * du
    
    # Spike detection & reset
    spiked = v_next >= 30.0
    v_next = jnp.where(spiked, c, v_next)
    u_next = jnp.where(spiked, u_next + d, u_next)
    
    # Post-synaptic current decay
    s_next = s * (1.0 - dt_ms / 5.0) + jnp.dot(W, spiked.astype(jnp.float32))
    
    # STDP continuous decay
    trace_pre_next = trace_pre * (1.0 - dt_ms / tau_plus) + spiked.astype(jnp.float32)
    trace_post_next = trace_post * (1.0 - dt_ms / tau_minus) + spiked.astype(jnp.float32)
    
    # Synapse-by-synapse STDP weight update:
    # W[i, j] represents pre-synaptic j connected to post-synaptic i.
    post_spike = spiked[:, None]
    pre_spike = spiked[None, :]
    
    # LTP: pre active, then post spikes (potentiate)
    dW_ltp = post_spike * trace_pre[None, :] * A_plus
    # LTD: post active, then pre spikes (depress)
    dW_ltd = pre_spike * trace_post[:, None] * A_minus
    
    dW = plasticity_scale * (dW_ltp - dW_ltd)
    
    # Enforce E/I sign preservation: update only excitatory synapses
    # exc_mask[None, :] masks presynaptic (columns)
    update_mask = exc_mask[None, :] & (~jnp.eye(v.shape[0], dtype=bool))
    W_next = W + jnp.where(update_mask, dW, 0.0)
    
    # Clip excitatory weights to [w_min, w_max]
    W_next = jnp.where(exc_mask[None, :], jnp.clip(W_next, w_min, w_max), W_next)
    
    # Enforce no self-connections
    W_next = W_next * (1.0 - jnp.eye(v.shape[0]))
    
    return (v_next, u_next, s_next, trace_pre_next, trace_post_next, W_next), (v_next, spiked)

def run_stdp_stream(
    v_init: jnp.ndarray,
    u_init: jnp.ndarray,
    s_init: jnp.ndarray,
    stdp_state: STDPState,
    stim_drive: jnp.ndarray,
    noise: jnp.ndarray,
    solver_config: SolverConfig,
    plasticity_config: STDPPlasticityConfig,
    plasticity_scale: float,
    exc_mask: jnp.ndarray,
    inh_mask: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
    chunk_size_ms: float = 10000.0,
    downsample_factor: int = 10
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, STDPState], Dict[str, Any]]:
    """Runs simulation in a chunked, streaming fashion to avoid memory explosion.
    
    Returns:
        tuple: (v_final, u_final, s_final, final_stdp_state)
        dict: traj_dict containing downsampled trajectories and chunk statistics.
    """
    dt_ms = solver_config.dt
    total_steps = stim_drive.shape[0]
    chunk_steps = int(chunk_size_ms / dt_ms)
    n_chunks = int(np.ceil(total_steps / chunk_steps))
    
    v, u, s = v_init, u_init, s_init
    trace_pre = stdp_state.trace_pre
    trace_post = stdp_state.trace_post
    W = stdp_state.W
    
    vm_collected = []
    spk_collected = []
    chunk_summaries = []
    
    for chunk_idx in range(n_chunks):
        start_step = chunk_idx * chunk_steps
        end_step = min(start_step + chunk_steps, total_steps)
        n_steps_curr = end_step - start_step
        if n_steps_curr <= 0:
            break
            
        stim_chunk = stim_drive[start_step:end_step]
        noise_chunk = noise[start_step:end_step]
        
        # Prepare inputs for scan
        inputs_in = (
            stim_chunk,
            noise_chunk,
            jnp.full((n_steps_curr,), plasticity_config.A_plus),
            jnp.full((n_steps_curr,), plasticity_config.A_minus)
        )
        
        state_init = (v, u, s, trace_pre, trace_post, W)
        
        # JIT-compiled scanner loop for the current chunk
        @jax.jit
        def run_chunk_scan(st, ip):
            """Documented public function `run_chunk_scan`."""
            def step_wrapper(state, inputs):
                """Documented public function `step_wrapper`."""
                return simulate_stdp_euler_step(
                    state, inputs, a, b, c, d, exc_mask, inh_mask, dt_ms,
                    plasticity_scale, plasticity_config.w_min, plasticity_config.w_max,
                    plasticity_config.tau_plus, plasticity_config.tau_minus
                )
            return jax.lax.scan(step_wrapper, st, ip)
            
        state_final, (vm_traj, spk_traj) = run_chunk_scan(state_init, inputs_in)
        v, u, s, trace_pre, trace_post, W = state_final
        
        # Compute chunk summaries (strictly finite checks)
        chunk_spk_rate = float(jnp.mean(spk_traj) * 1000.0 / dt_ms)
        chunk_w_mean = float(jnp.mean(W))
        chunk_summaries.append({
            "chunk_index": chunk_idx,
            "mean_firing_rate_hz": chunk_spk_rate,
            "mean_weight": chunk_w_mean,
            "finite_status": bool(jnp.all(jnp.isfinite(W)))
        })
        
        # Collect downsampled trajectories to save memory
        vm_collected.append(np.array(vm_traj[::downsample_factor]))
        spk_collected.append(np.array(spk_traj[::downsample_factor]))
        
    final_state = STDPState(W=W, trace_pre=trace_pre, trace_post=trace_post)
    
    trajectories = {
        "vm": np.concatenate(vm_collected, axis=0),
        "spk": np.concatenate(spk_collected, axis=0),
        "chunk_summaries": chunk_summaries
    }
    
    return (v, u, s, final_state), trajectories
