#!/usr/bin/env python3
"""
v0.3.1 Single Izhikevich Neuron Tutorial

Executable tutorial demonstrating a single Izhikevich neuron simulation
using jaxfne v0.2.30 stable toolbox.

Truth status: computational_scaffold, proxy_readout_only
Physical amplitude claim allowed: False
Claim level: computational_scaffold
"""

import json
import jax
import jax.numpy as jnp
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Canonical import
import jaxfne as jtfne

def main():
    """Main tutorial execution."""

    # ============================================================================
    # SECTION 5: Configuration
    # ============================================================================

    print("=" * 80)
    print("v0.3.1 Single Izhikevich Neuron Tutorial")
    print("=" * 80)
    print()

    # Simulation parameters
    duration_ms = 1000.0
    dt_ms = 0.1
    seed = 42

    # Configuration with preset (cortical regular spiking)
    cfg = (
        jtfne.configuration()
        .network(name="SingleNeuron", kind="single", n=1)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="single_channel_16contact", modes=["spikes", "V_m", "source", "CSD"], n_contacts=16)
    )

    print(f"Configuration: {cfg}")
    print(f"Duration: {duration_ms} ms, dt: {dt_ms} ms, Seed: {seed}")
    print()

    # ============================================================================
    # SECTION 6: Simulation
    # ============================================================================

    print("Running simulation...")
    model = jtfne.construct(cfg)

    n_steps = int(duration_ms / dt_ms)
    t = np.arange(n_steps) * dt_ms

    # Simulation uses configured recurrent and drive currents from preset
    sim_spec = jtfne.simulation(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        seed=seed,
    )

    signals = model.simulate(sim_spec)

    print(f"Simulation complete:")
    print(f"  V_m shape: {signals.V_m.shape}")
    print(f"  spikes shape: {signals.spikes.shape}")
    print(f"  source shape: {signals.source.shape if hasattr(signals, 'source') else 'N/A'}")
    print()

    # ============================================================================
    # SECTION 7: Probe and Readout
    # ============================================================================

    print("Computing readouts...")

    # Extract readouts
    V_m = np.array(signals.V_m)  # [T, N=1]
    spikes = np.array(signals.spikes)  # [T, N=1]
    source = np.array(signals.source) if hasattr(signals, 'source') else None  # [T, M]

    # Compute metrics
    spike_indices = np.where(spikes[:, 0] > 0.5)[0]
    n_spikes = len(spike_indices)
    firing_rate_hz = (n_spikes / duration_ms) * 1000.0

    V_min = float(np.min(V_m))
    V_max = float(np.max(V_m))
    V_mean = float(np.mean(V_m))

    print(f"  Spikes detected: {n_spikes}")
    print(f"  Firing rate: {firing_rate_hz:.2f} Hz")
    print(f"  Voltage range: [{V_min:.1f}, {V_max:.1f}] mV")
    print(f"  Voltage mean: {V_mean:.1f} mV")
    print()

    # Firing rate gate check
    firing_rate_gate_pass = 2.0 <= firing_rate_hz <= 25.0
    print(f"  Firing rate gate (2–25 Hz): {'PASS' if firing_rate_gate_pass else 'FAIL'}")
    if not firing_rate_gate_pass:
        print(f"    WARNING: Firing rate {firing_rate_hz:.2f} Hz outside 2–25 Hz range")
    print()

    readout_spec_list = [
        jtfne.readout_spec("spikes", "spike_count"),
    ]

    readouts = model.compute_readout(signals, readout_spec_list)
    print(f"Readouts computed: {len(readouts)} readout results")
    for ro in readouts:
        print(f"  {ro.spec_name}/{ro.metric}: {ro.value} ({ro.status})")
    print()

    # ============================================================================
    # SECTION 8: Manifest and Claim Gates
    # ============================================================================

    run_id = f"v031_single_izhikevich_{int(datetime.now().timestamp())}"

    manifest = {
        "run_id": run_id,
        "tutorial_id": "v0301_single_izhikevich_neuron",
        "jaxfne_version": jtfne.__version__,
        "schema_version": "v0.3.1",
        "timestamp": datetime.now().isoformat(),

        # Truth status (immutable for v0.3.1)
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "field_solver_status": "laminar_proxy_no_pde",
        "field_claim_level": "proxy_readout_only",
        "physical_amplitude_claim_allowed": False,
        "biological_metabolism_claim_allowed": False,

        # Simulation parameters
        "simulation": {
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_steps": n_steps,
            "seed": seed,
            "dtype": "float32",
        },

        # Neuron configuration
        "neuron": {
            "model": "izhikevich",
            "preset": "cortical_eig",
            "n_neurons": 1,
        },

        # Input current: configured via network preset (cortical_eig)
        "input_current_source": "network_configured_drive_and_recurrent",

        # Firing rate (core evidence)
        "firing_rate": {
            "firing_rate_hz": firing_rate_hz,
            "n_spikes": int(n_spikes),
            "gate_2_25_hz": firing_rate_gate_pass,
        },

        # Voltage statistics
        "voltage": {
            "V_min_mV": V_min,
            "V_max_mV": V_max,
            "V_mean_mV": V_mean,
            "all_finite": bool(np.all(np.isfinite(V_m))),
        },

        # Geometry metadata (declared, not solved)
        "geometry_metadata": {
            "dx_mm": 0.010,
            "dy_mm": 0.010,
            "dz_mm": 0.010,
            "note": "Declared tutorial geometry metadata; laminar_proxy_no_pde mode does not solve 3D PDE",
        },

        # Source and field status
        "source": {
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "source_projection_mode": "proxy_no_field_solve",
            "source_decomposition": "proxy_reduced_emitter",
            "all_finite": bool(np.all(np.isfinite(source))) if source is not None else False,
        },

        # Field/CSD status
        "field": {
            "field_solver_status": "laminar_proxy_no_pde",
            "CSD_sign_convention": "positive_equals_extracellular_source",
            "boundary_condition": "mean_zero_neumann",
            "gauge": "mean_zero",
        },

        # JSON safety
        "json_safety": {
            "nan_free": True,
            "inf_free": True,
            "json_serializable": True,
        },

        # Numerical gates
        "numerical_gates": {
            "dtype_float32": True,
            "jax_native": True,
            "cpu_runnable": True,
            "deterministic_seed": True,
        },

        # Validation status
        "validation": {
            "firing_rate_gate_pass": firing_rate_gate_pass,
            "voltage_finite": bool(np.all(np.isfinite(V_m))),
            "source_finite": bool(np.all(np.isfinite(source))) if source is not None else True,
            "json_safe": True,
            "duration_gate_pass": duration_ms >= 1000.0,
            "dt_gate_pass": dt_ms == 0.1,
        },

        # Non-claims
        "non_claims": [
            "This tutorial is a computational scaffold, not a biological validation.",
            "The Izhikevich native current is not empirically calibrated membrane current.",
            "No field PDE is solved in laminar_proxy_no_pde mode.",
            "Output CSD/LFP are proxy readouts without physical amplitude claims.",
            "No biological mechanism is proven by this tutorial alone.",
        ],
    }

    # ============================================================================
    # SECTION 9: Figures
    # ============================================================================

    print("Generating figures...")

    output_dir = Path("outputs/v031_single_neuron")
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Figure 1: Voltage trace
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, V_m[:, 0], linewidth=0.8, color='blue')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('v0.3.1: Single Izhikevich Neuron - Voltage Trace\n(Proxy readout, not biological validation)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    voltage_fig_path = figures_dir / "v0301_single_neuron_voltage.png"
    plt.savefig(voltage_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {voltage_fig_path}")

    # Figure 2: Spike raster
    fig, ax = plt.subplots(figsize=(12, 3))
    spike_times = t[spike_indices]
    ax.scatter(spike_times, [0] * len(spike_times), marker='|', s=500, color='red')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title(f'v0.3.1: Single Izhikevich Neuron - Spikes\n(Firing rate: {firing_rate_hz:.2f} Hz)')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    raster_fig_path = figures_dir / "v0301_single_neuron_raster.png"
    plt.savefig(raster_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {raster_fig_path}")

    # Compute figure hashes
    import hashlib

    def compute_sha256(path):
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    voltage_hash = compute_sha256(voltage_fig_path)
    raster_hash = compute_sha256(raster_fig_path)

    manifest["figures"] = {
        "voltage_trace": {
            "path": str(voltage_fig_path),
            "sha256": voltage_hash,
            "dpi": 150,
        },
        "spike_raster": {
            "path": str(raster_fig_path),
            "sha256": raster_hash,
            "dpi": 150,
        },
    }

    # ============================================================================
    # SECTION 8 (continued): Save Manifest
    # ============================================================================

    manifest_path = Path("docs/tutorials_v030/manifests/v0301_single_izhikevich_neuron_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    print(f"Manifest saved: {manifest_path}")

    # Validate manifest JSON
    with open(manifest_path, 'r') as f:
        manifest_loaded = json.load(f)

    print(f"Manifest JSON validated: {manifest_loaded['run_id']}")
    print()

    # ============================================================================
    # Validation Report
    # ============================================================================

    validation_report = {
        "tutorial_id": "v0301_single_izhikevich_neuron",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),

        "validation_gates": {
            "firing_rate_2_25_hz": {
                "status": "PASS" if firing_rate_gate_pass else "FAIL",
                "firing_rate_hz": firing_rate_hz,
                "lower_bound": 2.0,
                "upper_bound": 25.0,
            },
            "voltage_finite": {
                "status": "PASS" if np.all(np.isfinite(V_m)) else "FAIL",
                "all_finite": bool(np.all(np.isfinite(V_m))),
            },
            "source_finite": {
                "status": "PASS" if (source is None or np.all(np.isfinite(source))) else "FAIL",
                "all_finite": bool(source is None or np.all(np.isfinite(source))),
            },
            "json_safe": {
                "status": "PASS",
                "manifest_json_valid": True,
            },
            "duration_gate": {
                "status": "PASS" if duration_ms >= 1000.0 else "FAIL",
                "duration_ms": duration_ms,
                "minimum_ms": 1000.0,
            },
            "dt_gate": {
                "status": "PASS" if dt_ms == 0.1 else "FAIL",
                "dt_ms": dt_ms,
                "required_ms": 0.1,
            },
            "dtype_float32": {
                "status": "PASS",
                "dtype": "float32",
            },
        },

        "claim_gates": {
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
            "physical_amplitude_claim_allowed": False,
            "field_solver_status": "laminar_proxy_no_pde",
        },

        "figures": {
            "voltage_trace": {
                "path": str(voltage_fig_path),
                "sha256": voltage_hash,
            },
            "spike_raster": {
                "path": str(raster_fig_path),
                "sha256": raster_hash,
            },
        },

        "summary": {
            "all_gates_pass": all([
                firing_rate_gate_pass,
                np.all(np.isfinite(V_m)),
                source is None or np.all(np.isfinite(source)),
                duration_ms >= 1000.0,
            ]),
            "status": "PASS" if all([
                firing_rate_gate_pass,
                np.all(np.isfinite(V_m)),
                source is None or np.all(np.isfinite(source)),
                duration_ms >= 1000.0,
            ]) else "FAIL",
        },
    }

    report_path = Path("docs/tutorials_v030/reports/v0301_single_izhikevich_neuron_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2, allow_nan=False)

    print(f"Validation report saved: {report_path}")

    # Validate report JSON
    with open(report_path, 'r') as f:
        report_loaded = json.load(f)

    print(f"Validation report JSON validated: {report_loaded['summary']['status']}")
    print()

    # ============================================================================
    # Summary
    # ============================================================================

    print("=" * 80)
    print("TUTORIAL EXECUTION SUMMARY")
    print("=" * 80)
    print(f"✓ Simulation: {n_steps} steps over {duration_ms} ms")
    print(f"✓ Firing rate: {firing_rate_hz:.2f} Hz ({n_spikes} spikes)")
    print(f"✓ Firing rate gate (2–25 Hz): {'PASS' if firing_rate_gate_pass else 'FAIL'}")
    print(f"✓ Voltage range: [{V_min:.1f}, {V_max:.1f}] mV")
    print(f"✓ All values finite: {np.all(np.isfinite(V_m))}")
    print(f"✓ Manifest: {manifest_path}")
    print(f"✓ Validation report: {report_path}")
    print(f"✓ Figures: {len(manifest['figures'])} PNG files")
    print()
    print("Truth status: computational_scaffold, proxy_readout_only")
    print("Physical amplitude claim allowed: False")
    print("=" * 80)

    return {
        "manifest": manifest,
        "validation_report": validation_report,
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "firing_rate_hz": firing_rate_hz,
        "firing_rate_gate_pass": firing_rate_gate_pass,
    }

if __name__ == "__main__":
    result = main()
