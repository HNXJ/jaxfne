#!/usr/bin/env python3
"""
Simulate and verify the 5 Hz (5 spikes / 1000ms) minimal-noise parameter sets.
Plots the voltage traces for the four cell types.
"""

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

PRESETS = jtfne.CELL_TYPE_PRESETS

# Identified minimal-noise parameters for 5 spikes (5.0 Hz):
params_5hz = {
    "E": {"drive": 3.737, "noise": 0.202},
    "PV": {"drive": 3.737, "noise": 0.950},
    "SST": {"drive": 0.606, "noise": 0.000},
    "VIP": {"drive": 23.030, "noise": 0.000}
}

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, dpi=150)
time_ms = np.arange(N_STEPS) * DT_MS

colors = {
    "E": "#1f77b4",   # Blue
    "PV": "#ff7f0e",  # Orange
    "SST": "#2ca02c", # Green
    "VIP": "#9467bd"  # Purple
}

print("Running verification simulations:")

for idx, (cell_type, p) in enumerate(params_5hz.items()):
    preset_name = {
        "E": "E_RS",
        "PV": "PV_FS",
        "SST": "SST_LTS",
        "VIP": "VIP_IS"
    }[cell_type]
    
    preset = PRESETS[preset_name]
    a_val, b_val, c_val, d_val = preset["a"], preset["b"], preset["c"], preset["d"]
    
    # Manually configure IzhikevichParams
    params = jtfne.IzhikevichParams(
        a=jnp.array([a_val], dtype=jnp.float32),
        b=jnp.array([b_val], dtype=jnp.float32),
        c=jnp.array([c_val], dtype=jnp.float32),
        d=jnp.array([d_val], dtype=jnp.float32),
        drive=jnp.array([p["drive"]], dtype=jnp.float32),
        sign=jnp.array([1.0], dtype=jnp.float32),
        W=jnp.zeros((1, 1), dtype=jnp.float32),
        v0=jnp.array([-65.0], dtype=jnp.float32),
        u0=jnp.array([b_val * -65.0], dtype=jnp.float32),
        source_scale=jnp.array([1.0], dtype=jnp.float32),
        labels=(cell_type,),
    )
    
    key = jax.random.PRNGKey(SEED)
    noise_seq = jax.random.normal(key, shape=(N_STEPS, 1)) * float(p["noise"])
    
    voltages, spikes, _ = jtfne.simulate_eig_izhikevich(
        params, N_STEPS, DT_MS, key, drive_schedule=noise_seq
    )
    
    voltages_np = np.array(voltages[:, 0])
    spikes_np = np.array(spikes[:, 0])
    spike_count = int(np.sum(spikes_np))
    
    print(f"  {cell_type}: Drive={p['drive']:.3f}, Noise={p['noise']:.3f} -> Spikes={spike_count}")
    
    # Plot voltage trace
    ax = axes[idx]
    ax.plot(time_ms, voltages_np, color=colors[cell_type], linewidth=1.5, label=f"{cell_type} Voltage")
    
    # Draw spikes on top as vertical ticks or dots
    spike_times = time_ms[spikes_np > 0.5]
    ax.plot(spike_times, np.ones_like(spike_times) * 35.0, "|", color="red", markersize=12, markeredgewidth=2, label="Spikes")
    
    ax.set_ylabel("Voltage (mV)", fontsize=11)
    ax.set_title(f"{cell_type} Neuron (Drive = {p['drive']:.3f}, Noise = {p['noise']:.3f}) | Spikes: {spike_count} (5.0 Hz)", fontsize=12, fontweight="bold", pad=5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim([-90, 45])

axes[-1].set_xlabel("Time (ms)", fontsize=12)
plt.suptitle("Verification of 5.0 Hz (5 spikes / 1000ms) Firing Rates across Cell Types\nTruth Mode: truth_safe_unverified | Status: computational_scaffold", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

p1 = OUT_DIR / "sweep_d_5hz_verification.png"
p2 = ARTIFACTS_DIR / "sweep_d_5hz_verification.png"
plt.savefig(p1, bbox_inches="tight")
plt.savefig(p2, bbox_inches="tight")
plt.close()

print("Verification plots saved successfully!")
