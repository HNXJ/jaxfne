#!/usr/bin/env python3
"""
v0.3.3 Two-Neuron E/I Multimodal Tutorial

Demonstrates coupled excitatory/inhibitory neuron dynamics using Izhikevich models
with jaxfne v0.2.30 stable toolbox. Includes source aggregation, proxy field readouts,
and all eight multimodal operators (SPK, Vm, source, LFP, CSD, EEG, MEG, EMM).

Writes atlas-compatible manifest to outputs/v030_03_two_neuron_ei_multimodal/
for v0.3 collector validation.

Computational question: How does E→I excitatory drive and I→E inhibitory feedback
shape the spike timing and voltage dynamics of a minimal coupled network?

Truth status: computational_scaffold, proxy_readout_only
Physical amplitude claim allowed: False
Claim level: computational_scaffold
Scope: Tutorial demonstrating jaxfne TFNE pipeline; not biological validation.
"""

import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Canonical import
import jaxfne as jtfne
from jaxfne.fields import (
    csd_proxy_probe,
    eeg_proxy_probe,
    emm_proxy_probe,
    lfp_proxy_probe,
    meg_proxy_probe,
    source_probe,
    spk_probe,
    vm_probe,
)

# Collector-visible output directory (v030_03 prefix required by collector)
OUT = Path("outputs/v030_03_two_neuron_ei_multimodal")
# Docs-stable figures directory
STATIC_FIGS = Path("docs/tutorials_v030/_static/figures")

# Simulation parameters
DURATION_MS = 1000.0  # hard gate: >= 1000 ms
DT_MS = 0.1
SEED = 42

# Network parameters
N_NEURONS = 2
E_INDEX = 0
I_INDEX = 1
E_FRACTION = 0.5  # One E neuron out of two
I_FRACTION = 0.5  # One I neuron out of two

# Connectivity: E→I weight and I→E weight
# Note: These are illustrative coupling strengths. The actual effect depends on
# whether the coupling is implemented in the emitter or field layers.
# For v0.3.3, we document the intended coupling but note that it may not affect
# firing dynamics without explicit synaptic current implementation.
E_TO_I_WEIGHT = 2.0  # Excitatory drive to inhibitory neuron
I_TO_E_WEIGHT = -1.5  # Inhibitory drive to excitatory neuron (negative = inhibition)

# Rate gate bounds (matching v0.3.1 and v0.3.2)
RATE_GATE_LOW_HZ = 2.0
RATE_GATE_HIGH_HZ = 25.0


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    """Write JSON with strict allow_nan=False."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, allow_nan=False, indent=2, sort_keys=True)


def classify_neuron_regime(firing_rate_hz: float, finite: bool) -> str:
    """Assign per-neuron regime label."""
    if not finite:
        return "nonfinite_failure"
    if firing_rate_hz > RATE_GATE_HIGH_HZ:
        return "high_rate_out_of_target_regime"
    if firing_rate_hz < RATE_GATE_LOW_HZ:
        return "low_or_silent_out_of_target_regime"
    return "target_regime"


def main():
    """Main tutorial execution."""

    print("=" * 80)
    print("v0.3.3 Two-Neuron E/I Multimodal Tutorial")
    print("=" * 80)
    print()

    # ============================================================================
    # SECTION 1: Runtime and Configuration
    # ============================================================================

    run = jtfne.runtime(device_type="auto", dtype="float32", x64_enabled=False, seed=SEED)

    cfg = (
        jtfne.configuration()
        .network(
            name="v030_03_two_neuron_ei",
            kind="coupled_neurons",
            n=N_NEURONS,
            cell_types={"E": E_FRACTION, "I": I_FRACTION},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(
            name="two_channel_16contact_ei",
            modes=["spikes", "V_m", "source", "LFP", "CSD"],
            n_contacts=16,
        )
        .update_metadata(
            dx_mm=0.010,
            dy_mm=0.010,
            dz_mm=0.010,
            geometry_mode="declared_tutorial_metadata_not_solved_3d_grid",
            tutorial_id="v0.3.3",
            coupling_scenario="E_to_I_excitatory_I_to_E_inhibitory",
        )
    )

    print(f"Duration: {DURATION_MS} ms, dt: {DT_MS} ms, Seed: {SEED}")
    print(f"Network: {N_NEURONS} neurons (E={E_FRACTION}, I={I_FRACTION})")
    print(f"Coupling: E→I weight={E_TO_I_WEIGHT}, I→E weight={I_TO_E_WEIGHT}")
    print()

    # ============================================================================
    # SECTION 2: Simulation
    # ============================================================================

    print("Running simulation...")
    model = jtfne.construct(cfg)

    n_steps = int(DURATION_MS / DT_MS)
    t = np.arange(n_steps) * DT_MS

    sim_spec = jtfne.simulation(duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED, runtime=run)
    signals = model.simulate(sim_spec)

    print(f"  V_m shape: {signals.V_m.shape}")
    print(f"  spikes shape: {signals.spikes.shape}")
    field = signals.field
    if field is None:
        raise RuntimeError("Expected laminar proxy field output; field is None")
    print(f"  sources shape: {signals.sources.shape}")
    print()

    # ============================================================================
    # SECTION 3: Probe and Readout (all 8 operators)
    # ============================================================================

    print("Computing probe readouts...")
    V_m = np.array(signals.V_m)
    spikes_arr = np.array(signals.spikes)

    # Per-neuron firing rates
    e_spike_indices = np.where(spikes_arr[:, E_INDEX] > 0.5)[0]
    i_spike_indices = np.where(spikes_arr[:, I_INDEX] > 0.5)[0]

    e_n_spikes = len(e_spike_indices)
    i_n_spikes = len(i_spike_indices)

    e_firing_rate_hz = float((e_n_spikes / DURATION_MS) * 1000.0)
    i_firing_rate_hz = float((i_n_spikes / DURATION_MS) * 1000.0)

    # Gate checks per neuron
    e_firing_rate_gate_pass = RATE_GATE_LOW_HZ <= e_firing_rate_hz <= RATE_GATE_HIGH_HZ
    i_firing_rate_gate_pass = RATE_GATE_LOW_HZ <= i_firing_rate_hz <= RATE_GATE_HIGH_HZ

    # Overall gates
    e_voltage_finite = bool(np.all(np.isfinite(V_m[:, E_INDEX])))
    i_voltage_finite = bool(np.all(np.isfinite(V_m[:, I_INDEX])))

    # Voltage statistics
    e_v_min = float(np.min(V_m[:, E_INDEX]))
    e_v_max = float(np.max(V_m[:, E_INDEX]))
    e_v_mean = float(np.mean(V_m[:, E_INDEX]))

    i_v_min = float(np.min(V_m[:, I_INDEX]))
    i_v_max = float(np.max(V_m[:, I_INDEX]))
    i_v_mean = float(np.mean(V_m[:, I_INDEX]))

    print(f"  E neuron (idx={E_INDEX}):")
    print(f"    Spikes: {e_n_spikes}, Firing rate: {e_firing_rate_hz:.2f} Hz")
    print(f"    Voltage range: [{e_v_min:.1f}, {e_v_max:.1f}] mV")
    print(f"    Firing rate gate (2-25 Hz): {'PASS' if e_firing_rate_gate_pass else 'FAIL'}")
    print(f"  I neuron (idx={I_INDEX}):")
    print(f"    Spikes: {i_n_spikes}, Firing rate: {i_firing_rate_hz:.2f} Hz")
    print(f"    Voltage range: [{i_v_min:.1f}, {i_v_max:.1f}] mV")
    print(f"    Firing rate gate (2-25 Hz): {'PASS' if i_firing_rate_gate_pass else 'FAIL'}")
    print()

    # Source finite status
    sources_finite: bool | None
    if signals.sources is not None:
        sources_finite = bool(jnp.all(jnp.isfinite(signals.sources)))
    else:
        sources_finite = None  # not_generated

    # Generate probe reports using collector-required key names
    probe_report = {
        "spikes": spk_probe(signals.spikes).report,
        "V_m": vm_probe(signals.V_m).report,
        "source": source_probe(signals.sources).report,
        "lfp_proxy": lfp_proxy_probe(field.lfp_proxy).report,
        "csd_proxy": csd_proxy_probe(field.csd_proxy).report,
        "eeg_proxy": eeg_proxy_probe(field.lfp_proxy).report,
        "meg_proxy": meg_proxy_probe(field.lfp_proxy).report,
        "emm_proxy": emm_proxy_probe(jnp.mean(jnp.abs(field.lfp_proxy), axis=1)).report,
    }

    # ============================================================================
    # SECTION 4: Atlas Manifest (collector-compatible)
    # ============================================================================

    # Get conservation_proxy_diagnostics from model manifest
    raw_manifest = model.manifest(signals=signals)
    diag = dict(raw_manifest.get("conservation_proxy_diagnostics", {}))
    diag["e_mean_firing_rate_hz"] = e_firing_rate_hz
    diag["i_mean_firing_rate_hz"] = i_firing_rate_hz

    # Plotly status
    try:
        import plotly  # noqa: F401
        plotly_available = True
    except ImportError:
        plotly_available = False

    run_id = f"v033_two_neuron_ei_{int(datetime.now().timestamp())}"

    # Coupling scenario control comparison: ideally we would run an uncoupled version,
    # but for v0.3.3 we just document the intended coupling weights.
    # NOTE: In v0.3.3, coupling is specified but may not be fully instantiated without
    # explicit synaptic current implementation in the emitter layer. The two neurons
    # are configured to coexist in the network, but E/I feedback depends on whether
    # the configuration is passed to lower-level simulator APIs.
    coupling_scenario = {
        "scenario_kind": "intended_coupled_e_to_i_and_i_to_e",
        "e_to_i_weight": E_TO_I_WEIGHT,
        "i_to_e_weight": I_TO_E_WEIGHT,
        "implementation_status": "weights_specified_but_synaptic_coupling_not_verified_in_v033",
        "control_variant_planned": "uncoupled_baseline_and_synaptic_implementation_future_extension",
        "control_variant_status": "not_implemented_in_v033",
        "note": "This tutorial demonstrates the jaxfne API for multi-neuron configuration and readout infrastructure. Actual E/I feedback mechanisms require synaptic current implementation, which is marked as a future extension.",
    }

    # Atlas-compatible manifest with all 4 embedded blocks
    atlas_manifest = {
        "run_id": run_id,
        "tutorial_id": "v0303_two_neuron_ei_multimodal",
        "scenario_id": "v030_03_two_neuron_ei_multimodal",
        "jaxfne_version": jtfne.__version__,
        "schema_version": "v0.3.3",
        "timestamp": datetime.now().isoformat(),

        # === Gate 1: Required embedded blocks ===

        # basis block — immutable claim gates (collector checks these)
        "basis": {
            "truth_mode": "truth_safe_unverified",
            "claim_level": "computational_scaffold",
            "field_solver_status": "laminar_proxy_no_pde",
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
            "biological_metabolism_claim_allowed": False,
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "source_projection_mode": "proxy_no_field_solve",
        },

        # probe_report block — all 8 operators (collector checks keys)
        "probe_report": probe_report,

        # validation_report block — embedded (collector checks presence)
        "validation_report": {
            "e_firing_rate_gate_2_25_hz": e_firing_rate_gate_pass,
            "i_firing_rate_gate_2_25_hz": i_firing_rate_gate_pass,
            "e_voltage_finite": e_voltage_finite,
            "i_voltage_finite": i_voltage_finite,
            "source_finite": sources_finite,  # None if not generated
            "json_safe": True,
            "duration_gate": DURATION_MS >= 1000.0,
            "dt_gate": DT_MS == 0.1,
            "dtype_gate": str(signals.V_m.dtype) == "float32",
            "all_gates_pass": all([
                e_firing_rate_gate_pass,
                i_firing_rate_gate_pass,
                e_voltage_finite,
                i_voltage_finite,
                (sources_finite if sources_finite is not None else True),
                DURATION_MS >= 1000.0,
                DT_MS == 0.1,
            ]),
            "status": "PASS" if all([
                e_firing_rate_gate_pass,
                i_firing_rate_gate_pass,
                e_voltage_finite,
                i_voltage_finite,
                DURATION_MS >= 1000.0,
            ]) else "FAIL",
        },

        # conservation_proxy_diagnostics block — from model + firing rates
        "conservation_proxy_diagnostics": diag,

        # === Additional fields ===

        "simulation": {
            "duration_ms": DURATION_MS,
            "dt_ms": DT_MS,
            "n_steps": n_steps,
            "seed": SEED,
            "dtype": str(signals.V_m.dtype),
        },

        "network": {
            "n_neurons": N_NEURONS,
            "e_fraction": E_FRACTION,
            "i_fraction": I_FRACTION,
            "e_index": E_INDEX,
            "i_index": I_INDEX,
        },

        "neuron": {
            "model": "izhikevich",
            "preset": "cortical_eig",
            "n_neurons": int(signals.V_m.shape[1]),
        },

        "coupling": coupling_scenario,

        "e_neuron": {
            "index": E_INDEX,
            "cell_type": "excitatory",
            "firing_rate_hz": e_firing_rate_hz,
            "n_spikes": int(e_n_spikes),
            "gate_2_25_hz": e_firing_rate_gate_pass,
            "regime": classify_neuron_regime(e_firing_rate_hz, e_voltage_finite),
            "voltage": {
                "V_min_mV": e_v_min,
                "V_max_mV": e_v_max,
                "V_mean_mV": e_v_mean,
                "all_finite": e_voltage_finite,
            },
        },

        "i_neuron": {
            "index": I_INDEX,
            "cell_type": "inhibitory",
            "firing_rate_hz": i_firing_rate_hz,
            "n_spikes": int(i_n_spikes),
            "gate_2_25_hz": i_firing_rate_gate_pass,
            "regime": classify_neuron_regime(i_firing_rate_hz, i_voltage_finite),
            "voltage": {
                "V_min_mV": i_v_min,
                "V_max_mV": i_v_max,
                "V_mean_mV": i_v_mean,
                "all_finite": i_voltage_finite,
            },
        },

        "source": {
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "source_projection_mode": "proxy_no_field_solve",
            "source_finite": sources_finite,
            "source_proxy_status": "generated_proxy" if sources_finite is not None else "not_generated",
        },

        "geometry_metadata": {
            "dx_mm": 0.010,
            "dy_mm": 0.010,
            "dz_mm": 0.010,
            "note": "Declared tutorial geometry; laminar_proxy_no_pde mode does not solve 3D PDE",
        },

        "plotly": {
            "plotly_available": plotly_available,
            "plotly_html_generated": False,
            "plotly_status_reason": "static_png_baseline_for_v0303",
        },

        "non_claims": [
            "This tutorial is a computational scaffold, not a biological validation.",
            "The Izhikevich native current is not empirically calibrated membrane current.",
            "No field PDE is solved in laminar_proxy_no_pde mode.",
            "Output CSD/LFP are proxy readouts without physical amplitude claims.",
            "Coupling weights are specified but may not be implemented in lower-level APIs.",
            "Each neuron's dynamics reflect the Izhikevich preset applied independently.",
            "Actual E/I feedback dynamics (cross-neuron synaptic currents) are not implemented in v0.3.3.",
            "Spike timing patterns reflect model presets, not coupled network mechanisms.",
            "No mechanism of E/I balance or cortical function is proven by this tutorial.",
            "This tutorial validates the jaxfne API for multi-neuron configuration; real coupling requires synaptic implementation.",
        ],
    }

    # ============================================================================
    # SECTION 5: Figures
    # ============================================================================

    print("Generating figures...")
    OUT.mkdir(parents=True, exist_ok=True)
    STATIC_FIGS.mkdir(parents=True, exist_ok=True)

    figures_dir = OUT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Figure 1: Voltage traces for E and I neurons (two-row panel)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # E neuron voltage
    axes[0].plot(t, V_m[:, E_INDEX], linewidth=0.6, color='blue', label='E neuron')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_title(
        f'v0.3.3: Excitatory Neuron Voltage Trace\n'
        f'(Firing rate: {e_firing_rate_hz:.2f} Hz, Proxy readout)'
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # I neuron voltage
    axes[1].plot(t, V_m[:, I_INDEX], linewidth=0.6, color='red', label='I neuron')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Voltage (mV)')
    axes[1].set_title(
        f'v0.3.3: Inhibitory Neuron Voltage Trace\n'
        f'(Firing rate: {i_firing_rate_hz:.2f} Hz, Proxy readout)'
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    voltage_fig_path = figures_dir / "v0303_two_neuron_ei_voltage.png"
    plt.savefig(voltage_fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Spike raster (two neurons)
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot E spikes
    e_spike_times = t[e_spike_indices]
    ax.scatter(e_spike_times, [0] * len(e_spike_times), marker='|', s=500, color='blue',
               linewidth=2, label=f'E spikes (n={e_n_spikes})')

    # Plot I spikes
    i_spike_times = t[i_spike_indices]
    ax.scatter(i_spike_times, [1] * len(i_spike_times), marker='|', s=500, color='red',
               linewidth=2, label=f'I spikes (n={i_n_spikes})')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['E (blue)', 'I (red)'])
    ax.set_title(
        f'v0.3.3: Two-Neuron E/I Spike Raster\n'
        f'(E→I weight={E_TO_I_WEIGHT}, I→E weight={I_TO_E_WEIGHT})'
    )
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right')

    plt.tight_layout()
    raster_fig_path = figures_dir / "v0303_two_neuron_ei_raster.png"
    plt.savefig(raster_fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Source aggregation (E vs I contribution)
    if signals.sources is not None:
        sources = np.array(signals.sources)
        # sources shape: (time, neurons, n_locations) or (time, neurons, n_components)
        # Sum across space/components to get per-neuron source time series
        e_source_ts = np.sum(sources[:, E_INDEX, :], axis=1) if sources.ndim == 3 else sources[:, E_INDEX]
        i_source_ts = np.sum(sources[:, I_INDEX, :], axis=1) if sources.ndim == 3 else sources[:, I_INDEX]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, e_source_ts, linewidth=0.6, color='blue', label='E source (aggregated)', alpha=0.7)
        ax.plot(t, i_source_ts, linewidth=0.6, color='red', label='I source (aggregated)', alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Source (proxy units)')
        ax.set_title(
            'v0.3.3: E/I Source Aggregation\n'
            '(Proxy readout, uncalibrated Izhikevich native current)'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        plt.tight_layout()
        source_fig_path = figures_dir / "v0303_two_neuron_ei_source.png"
        plt.savefig(source_fig_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        source_fig_path = None

    # Figure 4: LFP-like proxy
    lfp_proxy = np.array(field.lfp_proxy)  # shape: (time, n_contacts)
    if lfp_proxy.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        # Show first 4 contacts
        for contact_idx in range(min(4, lfp_proxy.shape[1])):
            ax.plot(t, lfp_proxy[:, contact_idx], linewidth=0.5, label=f'Contact {contact_idx}', alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('LFP-like proxy')
        ax.set_title(
            'v0.3.3: Simulated LFP-like Proxy (first 4 contacts)\n'
            '(Proxy readout, no physical amplitude claim)'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        lfp_fig_path = figures_dir / "v0303_two_neuron_ei_lfp_proxy.png"
        plt.savefig(lfp_fig_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        lfp_fig_path = None

    # Figure 5: CSD-like proxy
    csd_proxy = np.array(field.csd_proxy)  # shape: (time, n_contacts-1) or similar
    if csd_proxy.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        # Show first 4 contacts
        for contact_idx in range(min(4, csd_proxy.shape[1])):
            ax.plot(t, csd_proxy[:, contact_idx], linewidth=0.5, label=f'Layer {contact_idx}', alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('CSD-like proxy')
        ax.set_title(
            'v0.3.3: Simulated CSD-like Proxy (first 4 layers)\n'
            '(Proxy readout, no PDE solve)'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        csd_fig_path = figures_dir / "v0303_two_neuron_ei_csd_proxy.png"
        plt.savefig(csd_fig_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        csd_fig_path = None

    print(f"  Saved: {voltage_fig_path}")
    print(f"  Saved: {raster_fig_path}")
    if source_fig_path and source_fig_path.exists():
        print(f"  Saved: {source_fig_path}")
    if lfp_fig_path and lfp_fig_path.exists():
        print(f"  Saved: {lfp_fig_path}")
    if csd_fig_path and csd_fig_path.exists():
        print(f"  Saved: {csd_fig_path}")

    # Copy to docs-stable _static/figures (committed paths referenced in docs)
    static_voltage = STATIC_FIGS / "v0303_two_neuron_ei_voltage.png"
    static_raster = STATIC_FIGS / "v0303_two_neuron_ei_raster.png"
    shutil.copy2(voltage_fig_path, static_voltage)
    shutil.copy2(raster_fig_path, static_raster)
    print(f"  Copied to _static: {static_voltage}")
    print(f"  Copied to _static: {static_raster}")

    # SHA256 hashes from docs-stable paths (canonical tracked paths)
    voltage_hash = sha256_file(static_voltage)
    raster_hash = sha256_file(static_raster)

    source_hash = None
    lfp_hash = None
    csd_hash = None

    if source_fig_path and source_fig_path.exists():
        static_source = STATIC_FIGS / "v0303_two_neuron_ei_source.png"
        shutil.copy2(source_fig_path, static_source)
        source_hash = sha256_file(static_source)
        print(f"  Copied to _static: {static_source}")

    if lfp_fig_path and lfp_fig_path.exists():
        static_lfp = STATIC_FIGS / "v0303_two_neuron_ei_lfp_proxy.png"
        shutil.copy2(lfp_fig_path, static_lfp)
        lfp_hash = sha256_file(static_lfp)
        print(f"  Copied to _static: {static_lfp}")

    if csd_fig_path and csd_fig_path.exists():
        static_csd = STATIC_FIGS / "v0303_two_neuron_ei_csd_proxy.png"
        shutil.copy2(csd_fig_path, static_csd)
        csd_hash = sha256_file(static_csd)
        print(f"  Copied to _static: {static_csd}")

    atlas_manifest["figures"] = {
        "voltage_traces": {
            "docs_stable_path": str(static_voltage),
            "runtime_path": str(voltage_fig_path),
            "sha256": voltage_hash,
            "dpi": 150,
            "description": "E and I neuron voltage traces (two-panel)",
        },
        "spike_raster": {
            "docs_stable_path": str(static_raster),
            "runtime_path": str(raster_fig_path),
            "sha256": raster_hash,
            "dpi": 150,
            "description": "E (blue) and I (red) spike raster",
        },
    }

    if source_hash:
        atlas_manifest["figures"]["source_aggregation"] = {
            "docs_stable_path": str(STATIC_FIGS / "v0303_two_neuron_ei_source.png"),
            "runtime_path": str(source_fig_path),
            "sha256": source_hash,
            "dpi": 150,
            "description": "E/I source aggregation (proxy units)",
        }

    if lfp_hash:
        atlas_manifest["figures"]["lfp_proxy"] = {
            "docs_stable_path": str(STATIC_FIGS / "v0303_two_neuron_ei_lfp_proxy.png"),
            "runtime_path": str(lfp_fig_path),
            "sha256": lfp_hash,
            "dpi": 150,
            "description": "LFP-like proxy (first 4 contacts)",
        }

    if csd_hash:
        atlas_manifest["figures"]["csd_proxy"] = {
            "docs_stable_path": str(STATIC_FIGS / "v0303_two_neuron_ei_csd_proxy.png"),
            "runtime_path": str(csd_fig_path),
            "sha256": csd_hash,
            "dpi": 150,
            "description": "CSD-like proxy (first 4 layers)",
        }

    # ============================================================================
    # Save atlas manifest and reports
    # ============================================================================

    manifest_path = OUT / "manifest.json"
    write_json(manifest_path, atlas_manifest)
    print(f"\nAtlas manifest saved: {manifest_path}")

    # Verify JSON round-trip
    with open(manifest_path) as f:
        loaded = json.load(f)
    assert loaded["basis"]["physical_amplitude_claim_allowed"] is False
    assert loaded["basis"]["claim_level"] == "computational_scaffold"
    assert set(loaded["probe_report"].keys()) == {
        "spikes", "V_m", "source", "lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy", "emm_proxy"
    }
    print("  Manifest round-trip check: PASS")

    # Also write probe_report.json as a separate file for reference
    write_json(OUT / "probe_report.json", probe_report)

    # Write separate validation_report for reference
    write_json(OUT / "validation_report.json", atlas_manifest["validation_report"])

    # Metrics file
    write_json(OUT / "metrics.json", {
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "seed": SEED,
        "dtype": str(signals.V_m.dtype),
        "n_steps": n_steps,
        "n_neurons": int(signals.V_m.shape[1]),
        "e_firing_rate_hz": e_firing_rate_hz,
        "i_firing_rate_hz": i_firing_rate_hz,
        "e_n_spikes": int(e_n_spikes),
        "i_n_spikes": int(i_n_spikes),
        "e_Vm_mean": e_v_mean,
        "e_Vm_min": e_v_min,
        "e_Vm_max": e_v_max,
        "i_Vm_mean": i_v_mean,
        "i_Vm_min": i_v_min,
        "i_Vm_max": i_v_max,
        "e_voltage_finite": e_voltage_finite,
        "i_voltage_finite": i_voltage_finite,
    })

    # Asset hashes
    json_files = [manifest_path, OUT / "probe_report.json",
                  OUT / "validation_report.json", OUT / "metrics.json"]
    hashes = {p.name: sha256_file(p) for p in json_files}
    hashes["figures/v0303_two_neuron_ei_voltage.png"] = voltage_hash
    hashes["figures/v0303_two_neuron_ei_raster.png"] = raster_hash
    if source_hash:
        hashes["figures/v0303_two_neuron_ei_source.png"] = source_hash
    if lfp_hash:
        hashes["figures/v0303_two_neuron_ei_lfp_proxy.png"] = lfp_hash
    if csd_hash:
        hashes["figures/v0303_two_neuron_ei_csd_proxy.png"] = csd_hash
    write_json(OUT / "asset_hashes.json", hashes)

    # Also update canonical docs manifest (for v0303 naming)
    canonical_manifest_path = Path("docs/tutorials_v030/manifests/v0303_two_neuron_ei_multimodal_manifest.json")
    canonical_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(canonical_manifest_path, atlas_manifest)

    canonical_report_path = Path("docs/tutorials_v030/reports/v0303_two_neuron_ei_multimodal_validation_report.json")
    canonical_report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(canonical_report_path, atlas_manifest["validation_report"])

    print(f"Canonical manifest: {canonical_manifest_path}")
    print(f"Canonical report: {canonical_report_path}")

    # ============================================================================
    # Summary
    # ============================================================================

    print()
    print("=" * 80)
    print("TUTORIAL EXECUTION SUMMARY")
    print("=" * 80)
    print(f"✓ Simulation: {n_steps} steps over {DURATION_MS} ms")
    print()
    print(f"✓ E neuron (idx={E_INDEX}):")
    print(f"  - Firing rate: {e_firing_rate_hz:.2f} Hz ({e_n_spikes} spikes)")
    print(f"  - Gate (2-25 Hz): {'PASS' if e_firing_rate_gate_pass else 'FAIL'}")
    print(f"  - Voltage range: [{e_v_min:.1f}, {e_v_max:.1f}] mV")
    print(f"  - All finite: {e_voltage_finite}")
    print()
    print(f"✓ I neuron (idx={I_INDEX}):")
    print(f"  - Firing rate: {i_firing_rate_hz:.2f} Hz ({i_n_spikes} spikes)")
    print(f"  - Gate (2-25 Hz): {'PASS' if i_firing_rate_gate_pass else 'FAIL'}")
    print(f"  - Voltage range: [{i_v_min:.1f}, {i_v_max:.1f}] mV")
    print(f"  - All finite: {i_voltage_finite}")
    print()
    print(f"✓ Source finite: {sources_finite}")
    print(f"✓ Manifest (collector-visible): {manifest_path}")
    print(f"✓ Docs-stable figures: {len(atlas_manifest['figures'])} generated")
    print(f"✓ Plotly available: {plotly_available}")
    print()
    print("Truth status: computational_scaffold, proxy_readout_only")
    print("Physical amplitude claim allowed: False")
    print("=" * 80)

    return {
        "atlas_manifest": atlas_manifest,
        "manifest_path": str(manifest_path),
        "e_firing_rate_hz": e_firing_rate_hz,
        "i_firing_rate_hz": i_firing_rate_hz,
        "e_firing_rate_gate_pass": e_firing_rate_gate_pass,
        "i_firing_rate_gate_pass": i_firing_rate_gate_pass,
        "figures": atlas_manifest["figures"],
    }


if __name__ == "__main__":
    main()
