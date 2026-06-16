#!/usr/bin/env python3
"""
Parameter sweep script for E, PV, SST, and VIP single neurons.
Evaluates spike counts over 1000ms under noise and constant drive sweeps.
Generates beautiful annotated 2D heatmaps.
"""

import os
import json
import hashlib
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
ARTIFACTS_DIR = jtfne.io.Path("artifacts/neuron_sweeps")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Simulation parameters
DURATION_MS = 1000.0
DT_MS = 0.1
N_STEPS = int(DURATION_MS / DT_MS)
SEED = 42

# Define presets
PRESETS = jtfne.CELL_TYPE_PRESETS

# Let's write a fast JAX function to run a single sweep step
# We can compile it with jit to make the sweep run in milliseconds
@jax.jit
def simulate_single_step(a_val, b_val, c_val, d_val, drive_val, noise_amp, key):
    # Setup params
    params = jtfne.IzhikevichParams(
        a=jnp.array([a_val], dtype=jnp.float32),
        b=jnp.array([b_val], dtype=jnp.float32),
        c=jnp.array([c_val], dtype=jnp.float32),
        d=jnp.array([d_val], dtype=jnp.float32),
        drive=jnp.array([drive_val], dtype=jnp.float32),
        sign=jnp.array([1.0], dtype=jnp.float32), # sign doesn't affect single neuron dynamics
        W=jnp.zeros((1, 1), dtype=jnp.float32),
        v0=jnp.array([-65.0], dtype=jnp.float32),
        u0=jnp.array([b_val * -65.0], dtype=jnp.float32),
        source_scale=jnp.array([1.0], dtype=jnp.float32),
        labels=("cell",),
    )
    
    # Generate custom noise sequence as drive_schedule
    noise_seq = jax.random.normal(key, shape=(N_STEPS, 1)) * noise_amp
    
    # Simulate
    _, spikes, _ = jtfne.simulate_eig_izhikevich(
        params, N_STEPS, DT_MS, key, drive_schedule=noise_seq
    )
    
    return jnp.sum(spikes)

# Grid setup (11x11)
grid_values = np.linspace(0.0, 1.0, 11)

def run_sweep(cell_type, get_drive, get_noise, get_key):
    preset_name = {
        "E": "E_RS",
        "PV": "PV_FS",
        "SST": "SST_LTS",
        "VIP": "VIP_IS"
    }[cell_type]
    
    preset = PRESETS[preset_name]
    a_val, b_val, c_val, d_val = preset["a"], preset["b"], preset["c"], preset["d"]
    
    grid = np.zeros((11, 11), dtype=int)
    
    for i, u_noise in enumerate(grid_values):
        for j, u_drive in enumerate(grid_values):
            drive_val = get_drive(u_drive, preset)
            noise_amp = get_noise(u_noise)
            # Use deterministic key based on indices for reproducible noise
            key = jax.random.PRNGKey(SEED + int(i * 100 + j))
            spike_count = int(simulate_single_step(a_val, b_val, c_val, d_val, drive_val, noise_amp, key))
            grid[i, j] = spike_count
            
    return grid

def plot_heatmaps(grids, title, ylabel, xlabel, xticklabels, yticklabels, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    axes = axes.flatten()
    cell_types = ["E", "PV", "SST", "VIP"]
    
    # Custom vibrant color palettes for each cell type
    colormaps = {
        "E": "Blues",
        "PV": "Oranges",
        "SST": "Greens",
        "VIP": "Purples"
    }
    
    for idx, cell_type in enumerate(cell_types):
        ax = axes[idx]
        grid = grids[cell_type]
        cmap = colormaps[cell_type]
        
        im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto")
        
        # Add labels and titles
        ax.set_title(f"{cell_type} Neuron Spikes (1000 ms)", fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        
        # Set ticks
        ax.set_xticks(np.arange(11))
        ax.set_yticks(np.arange(11))
        ax.set_xticklabels([f"{x:.1f}" for x in xticklabels], fontsize=10)
        ax.set_yticklabels([f"{y:.1f}" for y in yticklabels], fontsize=10)
        
        # Annotate each cell with the spike count
        for i in range(11):
            for j in range(11):
                val = grid[i, j]
                # Determine text color based on background intensity
                text_color = "white" if val > np.max(grid) * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")
                
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spike Count")
        
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save to outputs and artifacts
    p1 = OUT_DIR / filename
    p2 = ARTIFACTS_DIR / filename
    plt.savefig(p1, bbox_inches="tight")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {p1} and {p2}")

# --- SWEEP A: Absolute Native Sweep ---
# Both axes are absolute; constant drive is scaled by 10x (0 to 10.0)
print("Running Sweep A (Absolute Sweep)...")
grids_a = {}
for cell_type in ["E", "PV", "SST", "VIP"]:
    grids_a[cell_type] = run_sweep(
        cell_type,
        get_drive=lambda u, preset: float(u * 10.0),
        get_noise=lambda u: float(u),
        get_key=lambda i, j: SEED + i * 100 + j
    )
plot_heatmaps(
    grids=grids_a,
    title="Sweep A: Absolute Native Sweep (Drive 0-10, Noise 0-1 a.u.)\nTruth Mode:  | Status: computational_scaffold",
    ylabel="Noise Amplitude (a.u.)",
    xlabel="Constant Drive (a.u.)",
    xticklabels=grid_values * 10.0,
    yticklabels=grid_values,
    filename="sweep_a_absolute.png"
)

# --- SWEEP B: Preset-Relative Drive + Gaussian Noise Sweep ---
# Drive scales from 0.0 to 10x baseline preset drive
# Noise amplitude scales from 0.0 to 10.0
print("Running Sweep B (Preset-Relative Sweep)...")
grids_b = {}
for cell_type in ["E", "PV", "SST", "VIP"]:
    grids_b[cell_type] = run_sweep(
        cell_type,
        get_drive=lambda u, preset: float(u * preset["drive"] * 10.0),
        get_noise=lambda u: float(u * 10.0),
        get_key=lambda i, j: SEED + i * 100 + j
    )
plot_heatmaps(
    grids=grids_b,
    title="Sweep B: Preset-Relative Sweep (Drive 0 to 10x Baseline, Noise 0-10 a.u.)\nTruth Mode:  | Status: computational_scaffold",
    ylabel="Noise Amplitude (a.u.)",
    xlabel="Constant Drive (Relative to 10x Preset)",
    xticklabels=grid_values * 10.0, # relative labels 0.0 to 10.0
    yticklabels=grid_values * 10.0, # absolute noise values 0.0 to 10.0
    filename="sweep_b_preset_relative.png"
)

# --- SWEEP C: Threshold-Spanning Drive + Gaussian Noise Sweep ---
# Drive scales from 0.0 to 10x MaxDrive
# Noise amplitude scales from 0.0 to 10.0
print("Running Sweep C (Threshold-Spanning Sweep)...")
grids_c = {}
max_drives = {
    "E": 6.0,
    "PV": 6.0,
    "SST": 6.0,
    "VIP": 35.0
}
for cell_type in ["E", "PV", "SST", "VIP"]:
    grids_c[cell_type] = run_sweep(
        cell_type,
        get_drive=lambda u, preset: float(u * max_drives[cell_type] * 10.0),
        get_noise=lambda u: float(u * 10.0),
        get_key=lambda i, j: SEED + i * 100 + j
    )
plot_heatmaps(
    grids=grids_c,
    title="Sweep C: Threshold-Spanning Sweep (Drive 0 to 10x MaxDrive, Noise 0-10 a.u.)\nTruth Mode:  | Status: computational_scaffold",
    ylabel="Noise Amplitude (a.u.)",
    xlabel="Constant Drive (Relative to 10x Threshold-Spanning Max)",
    xticklabels=grid_values * 10.0, # relative labels 0.0 to 10.0
    yticklabels=grid_values * 10.0, # absolute noise values 0.0 to 10.0
    filename="sweep_c_threshold_spanning.png"
)

# Save raw sweep results to JSON for verification
raw_data = {
    "metadata": {
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "seed": SEED,
        "claim_level": "computational_scaffold",
        "field_solver_status": "linear_solver",
        "physical_amplitude_calibrated": False
    },
    "sweep_a_absolute": {k: v.tolist() for k, v in grids_a.items()},
    "sweep_b_preset_relative": {k: v.tolist() for k, v in grids_b.items()},
    "sweep_c_threshold_spanning": {k: v.tolist() for k, v in grids_c.items()}
}

p_json = OUT_DIR / "sweep_results.json"
p_json_art = ARTIFACTS_DIR / "sweep_results.json"
with open(p_json, "w") as f:
    json.dump(raw_data, f, indent=2)
with open(p_json_art, "w") as f:
    json.dump(raw_data, f, indent=2)
print("Sweep analysis complete!")
