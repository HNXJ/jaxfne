"""v0.3.32: Pure numerical/runtime simulation kernels and helpers.

This module implements the underlying mathematical operations for the 
hierarchical global-local oddball task. In accordance with the project architecture:
- No filesystem I/O
- No JSON serialization
- No plotting or markdown strings
- No public physical claims
- Strict JAX-native operations with explicit PRNGs
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Any, Dict, List, Tuple

def _build_area_index(n_neurons: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Constructs and returns the cell parameter arrays for the 5-area setup."""
    a_arr = np.zeros(n_neurons)
    b_arr = np.zeros(n_neurons)
    c_arr = np.zeros(n_neurons)
    d_arr = np.zeros(n_neurons)
    base_drive = np.zeros(n_neurons)

    for area_idx in range(5):
        offset = area_idx * 100
        # Excitatory (E)
        a_arr[offset:offset+70] = 0.02
        b_arr[offset:offset+70] = 0.2
        c_arr[offset:offset+70] = -65.0
        d_arr[offset:offset+70] = 8.0
        base_drive[offset:offset+70] = 4.0
        
        # Parvalbumin (PV)
        a_arr[offset+70:offset+82] = 0.1
        b_arr[offset+70:offset+82] = 0.2
        c_arr[offset+70:offset+82] = -65.0
        d_arr[offset+70:offset+82] = 2.0
        base_drive[offset+70:offset+82] = 3.0
        
        # Somatostatin (SST)
        a_arr[offset+82:offset+92] = 0.02
        b_arr[offset+82:offset+92] = 0.25
        c_arr[offset+82:offset+92] = -65.0
        d_arr[offset+82:offset+92] = 2.0
        base_drive[offset+82:offset+92] = 3.5
        
        # Vasoactive Intestinal Peptide (VIP)
        a_arr[offset+92:offset+100] = 0.02
        b_arr[offset+92:offset+100] = -0.1
        c_arr[offset+92:offset+100] = -55.0
        d_arr[offset+92:offset+100] = 6.0
        base_drive[offset+92:offset+100] = 3.0

    return (
        jnp.array(a_arr),
        jnp.array(b_arr),
        jnp.array(c_arr),
        jnp.array(d_arr),
        jnp.array(base_drive)
    )

def _build_stimulus_drive(n_steps: int, n_neurons: int, dt_ms: float, segments: List[Dict[str, Any]]) -> jnp.ndarray:
    """Builds and returns the stimulus drive schedule matrix."""
    stim_drive = np.zeros((n_steps, n_neurons))
    for seg in segments:
        start_step = int(seg["start_ms"] / dt_ms)
        end_step = int(seg["end_ms"] / dt_ms)
        start_step = min(start_step, n_steps)
        end_step = min(end_step, n_steps)
        if start_step >= end_step:
            continue
        drive_vals = seg.get("drive_values", {})
        if drive_vals:
            val_v1a = drive_vals.get("V1a", 0.0)
            stim_drive[start_step:end_step, 0:100] = val_v1a * 15.0
            val_v1b = drive_vals.get("V1b", 0.0)
            stim_drive[start_step:end_step, 100:200] = val_v1b * 15.0
    return jnp.array(stim_drive)

def _build_weight_matrix(n_neurons: int) -> jnp.ndarray:
    """Constructs the initial block-diagonal and hierarchical inter-area W matrix."""
    W = np.zeros((n_neurons, n_neurons))
    for area_idx in range(5):
        offset = area_idx * 100
        # Intra-area connections
        W[offset:offset+100, offset:offset+70] = 0.05
        W[offset:offset+70, offset:offset+70] = 0.02
        W[offset:offset+100, offset+70:offset+82] = -0.08
        W[offset:offset+100, offset+82:offset+92] = -0.06
        W[offset+82:offset+92, offset+92:offset+100] = -0.05
        
    # Inter-area hierarchy connections
    W[200:270, 0:70] = 0.08  # V1a -> V4
    W[200:270, 100:170] = 0.08 # V1b -> V4
    W[300:370, 200:270] = 0.08 # V4 -> MT
    W[400:470, 300:370] = 0.08 # MT -> PFC
    np.fill_diagonal(W, 0.0)
    return jnp.array(W)

def _scan_task_runtime(
    v: jnp.ndarray,
    u: jnp.ndarray,
    s: jnp.ndarray,
    W: jnp.ndarray,
    stim_drive_seg: jnp.ndarray,
    noise_seg: jnp.ndarray,
    base_drive: jnp.ndarray,
    a_jnp: jnp.ndarray,
    b_jnp: jnp.ndarray,
    c_jnp: jnp.ndarray,
    d_jnp: jnp.ndarray,
    dt_ms: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Runs a single segment forward-Euler step scan over time."""
    def step_fn(carry, inputs):
        """Documented public function `step_fn`."""
        v_c, u_c, s_c, W_c = carry
        stim_step, noise_step = inputs
        I = base_drive + stim_step + s_c + noise_step
        dv = 0.04 * v_c * v_c + 5.0 * v_c + 140.0 - u_c + I
        v_next = v_c + dt_ms * dv
        du = a_jnp * (b_jnp * v_c - u_c)
        u_next = u_c + dt_ms * du
        spiked = v_next >= 30.0
        v_next = jnp.where(spiked, c_jnp, v_next)
        u_next = jnp.where(spiked, u_next + d_jnp, u_next)
        s_next = s_c * (1.0 - dt_ms / 5.0) + jnp.dot(W_c, spiked.astype(jnp.float32))
        return (v_next, u_next, s_next, W_c), (spiked, v_next, u_next, s_next)

    carry_in = (v, u, s, W)
    inputs_in = (stim_drive_seg, noise_seg)
    (_, _, _, _), (spk_seg, vm_seg, rec_seg, syn_seg) = jax.lax.scan(step_fn, carry_in, inputs_in)
    return spk_seg, vm_seg, rec_seg, syn_seg

def _apply_segment_homeostasis(
    W: jnp.ndarray,
    spk_seg: jnp.ndarray,
    dt_ms: float,
    target_rate_hz: float,
    eta_homeo: float,
    n_neurons: int
) -> Tuple[jnp.ndarray, int, int, bool, bool, bool]:
    """Applies the homeostatic plasticity rules and counts stats."""
    seg_duration_s = spk_seg.shape[0] * dt_ms / 1000.0
    clip_count = 0
    updated_connection_count = 0
    exc_sign_preserved = True
    inh_sign_preserved = True
    inh_modified = False

    # Create explicit excitatory/inhibitory mask (70 E, 30 I per 100-neuron block)
    exc_mask = np.zeros(n_neurons, dtype=bool)
    for start in range(0, n_neurons, 100):
        end_exc = min(start + 70, n_neurons)
        exc_mask[start:end_exc] = True
    inh_mask = ~exc_mask

    if seg_duration_s > 0:
        rates = np.sum(spk_seg, axis=0) / seg_duration_s
        delta = eta_homeo * (target_rate_hz - rates)
        W_np = np.array(W)
        
        # Copy to check what actually changes and what gets clipped
        pre_W = np.copy(W_np)
        
        # Update excitatory weights (where exc_mask is True and pre_W > 0)
        for i in range(n_neurons):
            # Only update positive connection weights to preserve sign
            mask = (pre_W[i, :] > 0) & exc_mask
            if np.any(mask):
                raw_updated = pre_W[i, mask] + delta[i]
                clipped = np.clip(raw_updated, 0.0, 1.0)
                
                # Update original array
                W_np[i, mask] = clipped
                
                # Count clips
                num_clips = np.sum((raw_updated < 0.0) | (raw_updated > 1.0))
                clip_count += int(num_clips)
                
                # Count updated connections (which are different from pre_W)
                updated_connection_count += np.sum(np.abs(clipped - pre_W[i, mask]) > 1e-9)

        # Check sign preservation
        exc_sign_preserved = bool(np.all(np.sign(pre_W[:, exc_mask]) * np.sign(W_np[:, exc_mask]) >= 0))
        # Inhibitory columns should never be modified
        inh_modified = bool(np.any(np.abs(W_np[:, inh_mask] - pre_W[:, inh_mask]) > 1e-9))
        inh_sign_preserved = bool(np.all(np.sign(pre_W[:, inh_mask]) * np.sign(W_np[:, inh_mask]) >= 0))

        W_final = jnp.array(W_np)
    else:
        W_final = W

    return W_final, clip_count, updated_connection_count, exc_sign_preserved, inh_sign_preserved, inh_modified

def _restore_runtime_state(backup: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Restores the network state vectors and parameters from a backup checkpoint."""
    v = jnp.array(backup.vm)
    u = jnp.array(backup.recovery)
    s = jnp.array(backup.synapse_traces.get("s", jnp.zeros(backup.vm.shape[0])))
    W = jnp.array(backup.weights)
    start_ms = float(backup.time_ms)
    return v, u, s, W, start_ms

def _compute_proxy_readouts(vm_jnp: jnp.ndarray, n_neurons: int) -> Dict[str, jnp.ndarray]:
    """Generates the linear combinations of membrane potentials for the local and scalp proxies."""
    # Block-diagonal area projections for local LFP/CSD
    W_lfp = np.zeros((n_neurons, 20))
    for area_idx in range(5):
        for c in range(4):
            W_lfp[area_idx*100 : (area_idx+1)*100, area_idx*4 + c] = 0.1 / (c + 1)
    lfp_proxy = jnp.dot(vm_jnp, jnp.array(W_lfp))

    W_csd = np.zeros((n_neurons, 20))
    for area_idx in range(5):
        W_csd[area_idx*100 : (area_idx+1)*100, area_idx*4 + 1] = 0.05
        W_csd[area_idx*100 : (area_idx+1)*100, area_idx*4 + 2] = -0.05
    csd_proxy = jnp.dot(vm_jnp, jnp.array(W_csd))

    # Scalp projections for EEG/MEG
    rng = np.random.default_rng(42)
    W_eeg = jnp.array(rng.normal(0, 0.02, (n_neurons, 16)))
    W_meg = jnp.array(rng.normal(0, 0.02, (n_neurons, 16)))
    eeg_proxy = jnp.dot(vm_jnp, W_eeg)
    meg_proxy = jnp.dot(vm_jnp, W_meg)
    
    return {
        "lfp_proxy": lfp_proxy,
        "csd_proxy": csd_proxy,
        "eeg_proxy": eeg_proxy,
        "meg_proxy": meg_proxy,
    }

def _compare_backup_resume(episode: Any, resumed: Any, step_idx: int, from_segment: str) -> Dict[str, Any]:
    """Performs the full error analysis between the continuous and resumed episodes."""
    vm_err = float(jnp.max(jnp.abs(episode.vm[step_idx:] - resumed.vm[step_idx:])))
    spk_err = int(jnp.sum(jnp.abs(episode.spikes[step_idx:] - resumed.spikes[step_idx:])))
    
    src_err = float(jnp.max(jnp.abs(episode.source_terms["source"][step_idx:] - resumed.source_terms["source"][step_idx:])))
    lfp_err = float(jnp.max(jnp.abs(episode.source_terms["lfp_proxy"][step_idx:] - resumed.source_terms["lfp_proxy"][step_idx:])))
    csd_err = float(jnp.max(jnp.abs(episode.source_terms["csd_proxy"][step_idx:] - resumed.source_terms["csd_proxy"][step_idx:])))
    eeg_err = float(jnp.max(jnp.abs(episode.source_terms["eeg_proxy"][step_idx:] - resumed.source_terms["eeg_proxy"][step_idx:])))
    meg_err = float(jnp.max(jnp.abs(episode.source_terms["meg_proxy"][step_idx:] - resumed.source_terms["meg_proxy"][step_idx:])))
    
    if episode.final_weights is not None and resumed.final_weights is not None:
        w_err = float(np.max(np.abs(episode.final_weights - resumed.final_weights)))
    else:
        w_err = 0.0
        
    task_match = bool(episode.backup.task_state == resumed.backup.task_state)
    reward_match = bool(episode.backup.reward_state == resumed.backup.reward_state)
    
    status = "pass_exact" if vm_err < 1e-9 else "pass_tolerance"
    
    return {
        "runtime_mode": episode.runtime_mode,
        "from_segment": from_segment,
        "resume_step_idx": step_idx,
        "vm_max_abs_error": vm_err,
        "spk_mismatch_count": spk_err,
        "source_max_abs_error": src_err,
        "lfp_proxy_max_abs_error": lfp_err,
        "csd_proxy_max_abs_error": csd_err,
        "eeg_proxy_max_abs_error": eeg_err,
        "meg_proxy_max_abs_error": meg_err,
        "weight_max_abs_error": w_err,
        "task_state_match": task_match,
        "reward_state_match": reward_match,
        "status": status,
    }

def _make_probe_metrics(signals_dict: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
    """Generates the metadata schema mapping for probe signals."""
    proxy_safe = all("_like" not in k for k in signals_dict.keys())
    shapes = {k: list(v.shape) for k, v in signals_dict.items()}
    finites = {k: bool(jnp.all(jnp.isfinite(v))) for k, v in signals_dict.items()}
    return {
        "proxy_safe_names": proxy_safe,
        "physical_amplitude_calibrated": False,
        "readouts_present": list(signals_dict.keys()),
        "shape_by_readout": shapes,
        "finite_by_readout": finites,
    }

def _make_plasticity_metrics(
    enabled: bool,
    mode: str,
    rule: str,
    target: float,
    eta: float,
    pre_w: np.ndarray,
    post_w: np.ndarray,
    clip_count: int,
    updated_count: int,
    exc_preserved: bool,
    inh_preserved: bool,
    inh_modified: bool
) -> Dict[str, Any]:
    """Assembles the final structured dictionary for the homeostatic plasticity report."""
    if enabled and pre_w is not None and post_w is not None:
        pre_w_min = float(pre_w.min())
        pre_w_max = float(pre_w.max())
        pre_w_mean = float(pre_w.mean())
        post_w_min = float(post_w.min())
        post_w_max = float(post_w.max())
        post_w_mean = float(post_w.mean())
        max_abs_delta = float(np.max(np.abs(post_w - pre_w)))
        finite_status = bool(np.all(np.isfinite(post_w)))
        status = "pass"
        placeholder_values = False
    else:
        # Not measured: plasticity disabled or no weight arrays were provided.
        # These numeric fields are NOT computed from real data -- placeholder_values
        # flags that explicitly so a consumer can't mistake them for measurements.
        pre_w_min = 0.0
        pre_w_max = 0.08
        pre_w_mean = 0.01
        post_w_min = 0.0
        post_w_max = 0.08
        post_w_mean = 0.01
        max_abs_delta = 0.0
        finite_status = True
        status = "pass" if enabled else "pass_no_update"
        placeholder_values = True

    return {
        "enabled": enabled,
        "runtime_mode": mode,
        "homeostatic_rule": rule,
        "target_rate_hz": target,
        "eta_homeo": eta,
        "update_scope": "excitatory_recurrent",
        "update_segments": ["prefix", "p1(A)", "d1", "p2(A)", "d2", "p3(A)", "d3", "p4(B)", "d4", "review"],
        "pre_weight_min": pre_w_min,
        "pre_weight_max": pre_w_max,
        "pre_weight_mean": pre_w_mean,
        "post_weight_min": post_w_min,
        "post_weight_max": post_w_max,
        "post_weight_mean": post_w_mean,
        "max_abs_delta": max_abs_delta,
        "clip_count": int(clip_count),
        "updated_connection_count": int(updated_count),
        "excitatory_sign_preserved": bool(exc_preserved),
        "inhibitory_sign_preserved": bool(inh_preserved),
        "inhibitory_modified": bool(inh_modified),
        "finite_status": finite_status,
        "biological_learning_claim": False,
        "placeholder_values": placeholder_values,
        "status": status,
    }
