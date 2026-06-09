#!/usr/bin/env python3
"""Characterize Izhikevich neuron input-output curves (DC current → firing rate).

Direct ODE integration to map baseline drive → steady-state firing rate.
Produces reference table for eliminating silent-neuron populations.

Output: neuron_io_reference.json with DC drive values for target firing rates.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit for numerical stability
jax.config.update('jax_enable_x64', True)


# Izhikevich parameters from presets
IZHIKEVICH_PARAMS = {
    "E": {"a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0},
    "PV": {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0},
    "SST": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},
    "VIP": {"a": 0.02, "b": -0.10, "c": -55.0, "d": 6.0},
}


def izhikevich_step(v, u, t, dt, a, b, c, d, I_input):
    """Single forward Euler step for Izhikevich model.

    dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)

    Args:
        v: membrane potential (mV)
        u: recovery variable
        t: current time (ms) — unused but kept for consistency
        dt: time step (ms)
        a, b, c, d: Izhikevich parameters
        I_input: injected current

    Returns:
        (v_next, u_next)
    """
    # Izhikevich ODE
    dv_dt = 0.04 * v**2 + 5 * v + 140 - u + I_input
    du_dt = a * (b * v - u)

    v_next = v + dt * dv_dt
    u_next = u + dt * du_dt

    # Spike reset: if v >= 30 mV, reset
    spiked = v_next >= 30.0
    v_next = jnp.where(spiked, c, v_next)
    u_next = jnp.where(spiked, u_next + d, u_next)

    return v_next, u_next, float(spiked)


def simulate_neuron(
    cell_type: str,
    dc_input: float,
    duration_ms: float = 5000.0,
    dt_ms: float = 0.1,
) -> tuple[float, np.ndarray]:
    """Simulate single Izhikevich neuron and return firing rate.

    Args:
        cell_type: One of "E", "PV", "SST", "VIP"
        dc_input: Constant DC input (drive amplitude)
        duration_ms: Simulation duration
        dt_ms: Time step

    Returns:
        (firing_rate_hz, spike_times_ms)
    """
    params = IZHIKEVICH_PARAMS[cell_type]
    a, b, c, d = params["a"], params["b"], params["c"], params["d"]

    # Initialize rest state (approximate rest potential)
    v = -65.0  # mV
    u = 0.0

    n_steps = int(duration_ms / dt_ms)
    spikes = []
    times = []

    for step in range(n_steps):
        t = step * dt_ms
        v, u, is_spike = izhikevich_step(v, u, t, dt_ms, a, b, c, d, dc_input)

        if is_spike:
            spikes.append(1)
            times.append(t)
        else:
            spikes.append(0)

    # Steady-state firing rate (discard first 1 second transient)
    transient_steps = int(1000.0 / dt_ms)
    spikes_steady = np.array(spikes[transient_steps:])
    steady_state_duration = (duration_ms - 1000.0) / 1000.0

    if steady_state_duration > 0:
        firing_rate_hz = np.sum(spikes_steady) / steady_state_duration
    else:
        firing_rate_hz = np.sum(spikes) / (duration_ms / 1000.0)

    spike_times = np.array(times)
    return float(firing_rate_hz), spike_times


def characterize_cell_type(
    cell_type: str,
    dc_range: tuple[float, float] = (0.0, 20.0),
    n_samples: int = 20,
    duration_ms: float = 5000.0,
    dt_ms: float = 0.1,
) -> dict:
    """Characterize input-output curve for a cell type.

    Args:
        cell_type: One of "E", "PV", "SST", "VIP"
        dc_range: Tuple (min_dc, max_dc)
        n_samples: Number of DC values to test
        duration_ms: Simulation duration per sample
        dt_ms: Time step

    Returns:
        Dictionary with DC → firing rate mapping
    """
    print(f"\nCharacterizing {cell_type} neuron (duration={duration_ms} ms)...")

    dc_values = np.linspace(dc_range[0], dc_range[1], n_samples)
    firing_rates = []

    for dc in dc_values:
        fr, _ = simulate_neuron(cell_type, dc, duration_ms, dt_ms)
        firing_rates.append(float(fr))
        print(f"  DC={dc:6.2f}: {fr:6.2f} Hz", flush=True)

    # Interpolate to find DC for target firing rates
    target_rates_hz = [1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 50.0]
    dc_for_target = {}

    for target_rate in target_rates_hz:
        rates_array = np.array(firing_rates)
        idx = np.argmin(np.abs(rates_array - target_rate))
        dc_for_target[target_rate] = float(dc_values[idx])

    return {
        "cell_type": cell_type,
        "dc_range": (float(dc_range[0]), float(dc_range[1])),
        "dc_values": [float(x) for x in dc_values],
        "firing_rates": firing_rates,
        "target_rates_hz": target_rates_hz,
        "dc_for_target_rate": {float(k): v for k, v in dc_for_target.items()},
        "noise_amplitude_for_2hz": dc_for_target.get(2.0, None),
    }


def main():
    """Run characterization for all cell types."""
    print("=" * 80)
    print("IZHIKEVICH NEURON INPUT-OUTPUT CHARACTERIZATION")
    print("=" * 80)
    print("\nMapping: DC input (drive) → steady-state firing rate")
    print("(Transient first 1 second discarded for steady-state estimate)")
    print("(VIP: burst-like neuron; uses longer observation window)")

    cell_types = ["E", "PV", "SST", "VIP"]
    results = {}

    # Cell-type specific characterization parameters
    # E, PV, SST: standard 0-20 nA, 5 sec
    # VIP: requires higher DC range (0-100 nA) and longer observation (10 sec for bursts)
    params_by_type = {
        "E": {"dc_range": (0.0, 20.0), "n_samples": 25, "duration_ms": 5000.0},
        "PV": {"dc_range": (0.0, 20.0), "n_samples": 25, "duration_ms": 5000.0},
        "SST": {"dc_range": (0.0, 20.0), "n_samples": 25, "duration_ms": 5000.0},
        "VIP": {"dc_range": (0.0, 100.0), "n_samples": 30, "duration_ms": 10000.0},  # Burst-like; extended
    }

    # Characterize each cell type with appropriate parameters
    for cell_type in cell_types:
        params = params_by_type[cell_type]
        result = characterize_cell_type(
            cell_type,
            dc_range=params["dc_range"],
            n_samples=params["n_samples"],
            duration_ms=params["duration_ms"],
            dt_ms=0.1,
        )
        results[cell_type] = result

    # Create reference table
    print("\n" + "=" * 80)
    print("REFERENCE TABLE: DC Drive for Target Firing Rates")
    print("=" * 80)
    print("\n(These are baseline input values to inject into network construction)")

    target_rates = [1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 50.0]
    print("\nFiring Rate (Hz) | E (nA) | PV(nA) | SST(nA)| VIP(nA)")
    print("-" * 62)

    for target_rate in target_rates:
        row = f"{target_rate:6.1f}          |"
        for cell_type in cell_types:
            dc = results[cell_type]["dc_for_target_rate"].get(target_rate, None)
            if dc is not None:
                row += f" {dc:6.2f} |"
            else:
                row += " ------ |"
        print(row)

    # Save to JSON
    output_path = Path("outputs/neuron_io_reference.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Reference table saved: {output_path}")

    # Print recommended baseline drive (for 2 Hz firing rate)
    print("\n" + "=" * 80)
    print("RECOMMENDED BASELINE DRIVE (for 2 Hz expected firing rate)")
    print("=" * 80)
    print("\nUse these values in laminar_cortex_config to eliminate silent neurons:")
    print()

    baseline_drive = {}
    for cell_type in cell_types:
        noise_amp = results[cell_type].get("noise_amplitude_for_2hz", None)
        if noise_amp is not None:
            baseline_drive[cell_type] = round(noise_amp, 2)
            print(f"  {cell_type:>5}: {noise_amp:6.2f} nA")

    print(f"\nThen set noise_amplitude to same values for ~2 Hz expected rate.")
    print()

    # Print implementation guidance
    print("=" * 80)
    print("IMPLEMENTATION GUIDANCE")
    print("=" * 80)
    print(f"""
1. In laminar_cortex_config, add external DC drive per cell type:

   baseline_drive_dict = {baseline_drive}
   # Pass to simulation or network construction

2. In network, inject constant current before simulation:

   for cell_type, dc_value in baseline_drive_dict.items():
       # Add dc_value to I_input for neurons of cell_type

3. Set noise amplitude to same values (for 2 Hz baseline):

   noise_amplitude = {baseline_drive}

4. Rerun AGSDR with TFNE_SMOKE=0:

   TFNE_SMOKE=0 python3 scripts/run_delta_notebook_01.py

5. Expected outcome:
   - Baseline mean rate: ~7-9 Hz (from 2 Hz per cell + network activity)
   - Min neuron rate: >= 1.0 Hz (from baseline DC drive)
   - Both gates PASS → 100/100 delta-test acceptance

6. If rates still wrong, iterate by:
   - Adjusting baseline_drive values (proportional scaling)
   - Adjusting noise_amplitude (±20% band)
   - Keep knobs declared in GLOBAL config (single source of truth)
""")

    print("=" * 80)


if __name__ == "__main__":
    main()
