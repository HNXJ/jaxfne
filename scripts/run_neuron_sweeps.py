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
import jax
import jax.numpy as jnp
import jaxfne as jtfne
import functools

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _neuron_sweep_common import simulate_single_step as _simulate_single_step_shared

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

# Fast JAX single-neuron-in-isolation sweep step, jitted with n_steps/dt_ms
# bound as static Python values (shared with find_5hz_vmap.py, see
# _neuron_sweep_common.py).
simulate_single_step = jax.jit(functools.partial(
    _simulate_single_step_shared, n_steps=N_STEPS, dt_ms=DT_MS
))

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
    fig = jtfne.vis.celltype_sweep_heatmap_grid(
        grids, title=title, ylabel=ylabel, xlabel=xlabel,
        xticklabels=xticklabels, yticklabels=yticklabels,
    )

    # Save to outputs and artifacts
    p1 = OUT_DIR / filename
    p2 = ARTIFACTS_DIR / filename
    fig.savefig(p1, bbox_inches="tight")
    fig.savefig(p2, bbox_inches="tight")
    jtfne.vis.close_all()
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
