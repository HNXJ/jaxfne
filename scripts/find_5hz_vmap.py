#!/usr/bin/env python3
"""
Find exact parameters (constant drive and noise amplitude) via a high-resolution
2D parameter sweep at which E, PV, SST, and VIP neurons fire at exactly 5.0 Hz
(5 spikes per 1000ms).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jaxfne as jtfne

# 1. Output setup
OUT_DIR = jtfne.io.Path("outputs/neuron_sweeps")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = jtfne.io.Path("/Users/hamednejat/.gemini/antigravity/brain/39583144-86c9-4537-982e-52a28c32e57c")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DURATION_MS = 1000.0
DT_MS = 0.1
N_STEPS = int(DURATION_MS / DT_MS)
SEED = 42

# Define presets
PRESETS = jtfne.CELL_TYPE_PRESETS

# Base simulation step
def simulate_single_step(a_val, b_val, c_val, d_val, drive_val, noise_amp, key):
    params = jtfne.IzhikevichParams(
        a=jnp.array([a_val], dtype=jnp.float32),
        b=jnp.array([b_val], dtype=jnp.float32),
        c=jnp.array([c_val], dtype=jnp.float32),
        d=jnp.array([d_val], dtype=jnp.float32),
        drive=jnp.array([drive_val], dtype=jnp.float32),
        sign=jnp.array([1.0], dtype=jnp.float32),
        W=jnp.zeros((1, 1), dtype=jnp.float32),
        v0=jnp.array([-65.0], dtype=jnp.float32),
        u0=jnp.array([b_val * -65.0], dtype=jnp.float32),
        source_scale=jnp.array([1.0], dtype=jnp.float32),
        labels=("cell",),
    )
    
    # Generate noise
    noise_seq = jax.random.normal(key, shape=(N_STEPS, 1)) * noise_amp
    
    _, spikes, _ = jtfne.simulate_eig_izhikevich(
        params, N_STEPS, DT_MS, key, drive_schedule=noise_seq
    )
    
    return jnp.sum(spikes)

# Vmap the simulation function over drive and noise parameters
# in_axes for simulate_single_step: (a, b, c, d, drive, noise, key)
# We will vmap over noise_amp first, then drive_val
vmap_noise = jax.vmap(simulate_single_step, in_axes=(None, None, None, None, None, 0, None))
vmap_drive_noise = jax.vmap(vmap_noise, in_axes=(None, None, None, None, 0, None, None))
jit_sweep = jax.jit(vmap_drive_noise)

# Finer sweeps ranges for each cell type to locate the 5Hz regime
ranges = {
    "E": {
        "drive": np.linspace(0.0, 5.0, 100),
        "noise": np.linspace(0.0, 10.0, 100)
    },
    "PV": {
        "drive": np.linspace(0.0, 5.0, 100),
        "noise": np.linspace(0.0, 15.0, 100)
    },
    "SST": {
        "drive": np.linspace(0.0, 2.0, 100),
        "noise": np.linspace(0.0, 10.0, 100)
    },
    "VIP": {
        "drive": np.linspace(0.0, 30.0, 100),
        "noise": np.linspace(0.0, 25.0, 100)
    }
}

results_5hz = {}

key = jax.random.PRNGKey(SEED)

for cell_type, r in ranges.items():
    preset_name = {
        "E": "E_RS",
        "PV": "PV_FS",
        "SST": "SST_LTS",
        "VIP": "VIP_IS"
    }[cell_type]
    
    preset = PRESETS[preset_name]
    a, b, c, d = preset["a"], preset["b"], preset["c"], preset["d"]
    
    drives = r["drive"]
    noises = r["noise"]
    
    print(f"Sweeping {cell_type} (10,000 points)...")
    # Convert parameters to JAX arrays
    drives_jnp = jnp.array(drives, dtype=jnp.float32)
    noises_jnp = jnp.array(noises, dtype=jnp.float32)
    
    # Run the compiled vmapped sweep
    # Output shape: (100, 100) corresponding to (drives, noises)
    spike_counts = np.array(jit_sweep(a, b, c, d, drives_jnp, noises_jnp, key))
    
    # Find points where spike count is exactly 5
    indices = np.argwhere(spike_counts == 5)
    
    pairs = []
    for idx in indices:
        i_drive, i_noise = idx
        d_val = float(drives[i_drive])
        n_val = float(noises[i_noise])
        pairs.append({"drive": d_val, "noise": n_val})
        
    results_5hz[cell_type] = {
        "found_count": len(pairs),
        "candidates": pairs[:10]  # Store up to 10 example pairs
    }
    
    print(f"  Found {len(pairs)} parameter pairs that produce exactly 5 spikes.")
    if pairs:
        print(f"  Example: Drive={pairs[0]['drive']:.3f}, Noise={pairs[0]['noise']:.3f}")
    else:
        # Fallback to closest
        diffs = np.abs(spike_counts - 5)
        min_diff = int(np.min(diffs))
        closest_indices = np.argwhere(diffs == min_diff)
        closest_pairs = []
        for idx in closest_indices[:10]:
            i_drive, i_noise = idx
            d_val = float(drives[i_drive])
            n_val = float(noises[i_noise])
            closest_pairs.append({"drive": d_val, "noise": n_val, "spikes": int(spike_counts[i_drive, i_noise])})
        results_5hz[cell_type]["closest"] = closest_pairs
        print(f"  Closest spikes = {closest_pairs[0]['spikes']} (diff={min_diff}) at Drive={closest_pairs[0]['drive']:.3f}, Noise={closest_pairs[0]['noise']:.3f}")

# Save the 5Hz search results
p_json = OUT_DIR / "find_5hz_results.json"
p_json_art = ARTIFACTS_DIR / "find_5hz_results.json"
with open(p_json, "w") as f:
    json.dump(results_5hz, f, indent=2)
with open(p_json_art, "w") as f:
    json.dump(results_5hz, f, indent=2)
print("Finished search!")
