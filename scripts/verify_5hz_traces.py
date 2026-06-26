#!/usr/bin/env python3
"""
Simulate and verify the 5 Hz (5 spikes / 1000ms) minimal-noise parameter sets.
Plots the voltage traces for the four cell types.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jaxfne as jtfne

# 1. Output setup
OUT_DIR = jtfne.io.Path("outputs/neuron_sweeps")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = jtfne.io.Path("artifacts/verify_5hz_traces")
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

time_ms = np.arange(N_STEPS) * DT_MS

colors = {
    "E": "#1f77b4",   # Blue
    "PV": "#ff7f0e",  # Orange
    "SST": "#2ca02c", # Green
    "VIP": "#9467bd"  # Purple
}

print("Running verification simulations:")

voltages_by_celltype = {}
spikes_by_celltype = {}
params_by_celltype = {}

for cell_type, p in params_5hz.items():
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

    voltages_by_celltype[cell_type] = voltages_np
    spikes_by_celltype[cell_type] = spikes_np
    params_by_celltype[cell_type] = {"drive": p["drive"], "noise": p["noise"], "spike_count": spike_count}

fig = jtfne.vis.voltage_trace_with_spikes_by_celltype(
    time_ms, voltages_by_celltype, spikes_by_celltype, params_by_celltype, colors=colors,
)

p1 = OUT_DIR / "sweep_d_5hz_verification.png"
p2 = ARTIFACTS_DIR / "sweep_d_5hz_verification.png"
fig.savefig(p1, bbox_inches="tight")
fig.savefig(p2, bbox_inches="tight")
jtfne.vis.close_all()

print("Verification plots saved successfully!")
