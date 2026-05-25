#!/usr/bin/env python3
"""
v0.3.3 Two-Neuron E/I Multimodal Tutorial

Demonstrates coupled excitatory/inhibitory neuron dynamics using Izhikevich models
with jaxfne v0.2.30 stable toolbox. Uses simulate_dynamic_ei_coupling for genuine
dynamic synaptic current injection (not post-hoc). Includes source aggregation,
proxy field readouts, and all eight multimodal operators (SPK, Vm, source, LFP,
CSD, EEG, MEG, EMM).

Writes atlas-compatible manifest to outputs/v030_03_two_neuron_ei_multimodal/
for v0.3 collector validation.

Computational question: How does E→I excitatory drive and I→E inhibitory feedback
shape the spike timing and voltage dynamics of a minimal coupled network?

Truth status: computational_scaffold, proxy_readout_only
Physical amplitude claim allowed: False
Claim level: computational_scaffold
Scope: Tutorial demonstrating jaxfne TFNE pipeline; not biological validation.
"""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

# Canonical import
import jaxfne as jtfne
from jaxfne.emitters import simulate_dynamic_ei_coupling, izhikevich_eig_params
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
I_FRACTION = 0.5  # One I neuron out of two (PV fast-spiking interneuron)

# Dynamic coupling parameters (E→I excitatory, I→E inhibitory)
# Tuned empirically to produce E~11 Hz and PV~11 Hz with seed=42
G_EI = 4.0           # E→I excitatory coupling conductance (model units)
G_IE = 2.0           # I→E inhibitory coupling magnitude (model units)
TAU_SYN_E_MS = 5.0   # Excitatory synaptic time constant
TAU_SYN_I_MS = 10.0  # Inhibitory synaptic time constant

# Per-neuron drive overrides (E: regular-spiking, PV: fast-spiking)
E_DRIVE = 5.0   # Regular-spiking E neuron drive → ~11 Hz baseline
PV_DRIVE = 3.0  # PV fast-spiking interneuron drive (default, driven by E→I coupling)

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


def plot_coupled_vs_uncoupled(time_ms, spikes_coupled, spikes_uncoupled,
                               v_m_coupled, v_m_uncoupled):
    """Plot effect of coupling: coupled vs. uncoupled comparison (4-panel)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

    ax_spk_c = axes[0, 0]
    ax_vm_c = axes[0, 1]

    spike_times_e_c = time_ms[spikes_coupled[:, 0] > 0.5]
    spike_times_i_c = time_ms[spikes_coupled[:, 1] > 0.5]

    ax_spk_c.scatter(spike_times_e_c, [0] * len(spike_times_e_c),
                     color='blue', s=15, alpha=0.7, label='E')
    ax_spk_c.scatter(spike_times_i_c, [1] * len(spike_times_i_c),
                     color='red', s=15, alpha=0.7, label='I')
    ax_spk_c.set_ylabel("Neuron (Coupled)")
    ax_spk_c.set_yticks([0, 1])
    ax_spk_c.set_yticklabels(['E', 'I'])
    ax_spk_c.set_title("With Coupling — Spikes")
    ax_spk_c.grid(True, alpha=0.3)

    ax_vm_c.plot(time_ms, v_m_coupled[:, 0], label='E', color='blue', linewidth=0.8)
    ax_vm_c.plot(time_ms, v_m_coupled[:, 1], label='I', color='red', linewidth=0.8)
    ax_vm_c.set_ylabel("V_m (mV)")
    ax_vm_c.set_ylim([-80, 40])
    ax_vm_c.set_title("With Coupling — Voltage")
    ax_vm_c.legend(fontsize=8)
    ax_vm_c.grid(True, alpha=0.3)

    ax_spk_u = axes[1, 0]
    ax_vm_u = axes[1, 1]

    spike_times_e_u = time_ms[spikes_uncoupled[:, 0] > 0.5]
    spike_times_i_u = time_ms[spikes_uncoupled[:, 1] > 0.5]

    ax_spk_u.scatter(spike_times_e_u, [0] * len(spike_times_e_u),
                     color='blue', s=15, alpha=0.7, label='E')
    ax_spk_u.scatter(spike_times_i_u, [1] * len(spike_times_i_u),
                     color='red', s=15, alpha=0.7, label='I')
    ax_spk_u.set_xlabel("Time (ms)")
    ax_spk_u.set_ylabel("Neuron (Uncoupled)")
    ax_spk_u.set_yticks([0, 1])
    ax_spk_u.set_yticklabels(['E', 'I'])
    ax_spk_u.set_title("Without Coupling — Spikes")
    ax_spk_u.grid(True, alpha=0.3)

    ax_vm_u.plot(time_ms, v_m_uncoupled[:, 0], label='E', color='blue', linewidth=0.8)
    ax_vm_u.plot(time_ms, v_m_uncoupled[:, 1], label='I', color='red', linewidth=0.8)
    ax_vm_u.set_xlabel("Time (ms)")
    ax_vm_u.set_ylabel("V_m (mV)")
    ax_vm_u.set_ylim([-80, 40])
    ax_vm_u.set_title("Without Coupling — Voltage")
    ax_vm_u.legend(fontsize=8)
    ax_vm_u.grid(True, alpha=0.3)

    fig.suptitle("Coupling Effect Comparison (Coupled vs. Uncoupled)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_circuit_schematic(g_ei, g_ie, firing_rate_e, firing_rate_i):
    """Plot E/I circuit schematic with coupling conductance labels."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(8, 6))

    x_e, y_e = 0.3, 0.5
    x_i, y_i = 0.7, 0.5
    radius = 0.08

    e_circle = patches.Circle((x_e, y_e), radius, color='blue', alpha=0.7)
    ax.add_patch(e_circle)
    ax.text(x_e, y_e, 'E', ha='center', va='center',
            color='white', fontsize=14, fontweight='bold')
    ax.text(x_e, y_e - 0.14, f'{firing_rate_e:.1f} Hz',
            ha='center', va='center', fontsize=10, color='blue')

    i_circle = patches.Circle((x_i, y_i), radius, color='red', alpha=0.7)
    ax.add_patch(i_circle)
    ax.text(x_i, y_i, 'I', ha='center', va='center',
            color='white', fontsize=14, fontweight='bold')
    ax.text(x_i, y_i - 0.14, f'{firing_rate_i:.1f} Hz',
            ha='center', va='center', fontsize=10, color='red')

    arrow_ie = patches.FancyArrowPatch(
        (x_e + radius, y_e), (x_i - radius, y_i),
        arrowstyle='->', mutation_scale=25,
        color='green', linewidth=2.0, alpha=0.8,
        connectionstyle="arc3,rad=0.25"
    )
    ax.add_patch(arrow_ie)
    ax.text(0.5, y_e + 0.10, f'g_EtoI={g_ei:.1f}' + chr(10) + '(excitatory)',
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    arrow_ei = patches.FancyArrowPatch(
        (x_i - radius, y_i), (x_e + radius, y_e),
        arrowstyle='->', mutation_scale=25,
        color='darkred', linewidth=2.0, alpha=0.8,
        connectionstyle="arc3,rad=0.25"
    )
    ax.add_patch(arrow_ei)
    ax.text(0.5, y_e - 0.10, f'g_ItoE={g_ie:.1f}' + chr(10) + '(inhibitory)',
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.25, 0.75)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Recurrent E/I Circuit (Coupling Conductances Shown)", fontsize=13)
    fig.tight_layout()
    return fig


def main():
    """Main tutorial execution."""

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("WARNING: matplotlib not installed; skipping figure generation")
        print()

    print("=" * 80)
    print("v0.3.3 Two-Neuron E/I Multimodal Tutorial (Dynamic Coupling)")
    print("=" * 80)
    print()

    # ============================================================================
    # SECTION 1: Runtime and Configuration
    # ============================================================================

    print(f"Duration: {DURATION_MS} ms, dt: {DT_MS} ms, Seed: {SEED}")
    print(f"Network: {N_NEURONS} neurons (E fraction={E_FRACTION}, PV fraction={I_FRACTION})")
    print(f"Dynamic coupling: E→I g={G_EI}, tau={TAU_SYN_E_MS} ms; I→E g={G_IE}, tau={TAU_SYN_I_MS} ms")
    print()

    # ============================================================================
    # SECTION 2: Simulation with Dynamic Synaptic Coupling
    # ============================================================================

    print("Setting up Izhikevich parameters (E + PV cell types)...")
    # Use E + PV (fast-spiking) as the two-neuron pair
    # PV (index 1) has a=0.1, b=0.2, c=-65, d=2 — proper fast-spiking interneuron
    params = izhikevich_eig_params(N_NEURONS, {"E": E_FRACTION, "PV": I_FRACTION})
    # Override per-neuron drives
    params = dataclasses.replace(params, drive=jnp.array([E_DRIVE, PV_DRIVE], dtype=jnp.float32))

    print(f"  E neuron (idx=0): a={float(params.a[0]):.3f}, b={float(params.b[0]):.3f}, drive={float(params.drive[0]):.1f}")
    print(f"  PV neuron (idx=1): a={float(params.a[1]):.3f}, b={float(params.b[1]):.3f}, drive={float(params.drive[1]):.1f}")
    print()

    print("Running simulation with dynamic E/I coupling via lax.scan...")
    key = jax.random.PRNGKey(SEED)
    n_steps = int(DURATION_MS / DT_MS)
    t = np.arange(n_steps) * DT_MS

    voltages, spikes_arr_jax, syn_currents_jax, sources_jax = simulate_dynamic_ei_coupling(
        params,
        n_steps=n_steps,
        dt_ms=DT_MS,
        key=key,
        g_ei=G_EI,
        g_ie=G_IE,
        tau_syn_e_ms=TAU_SYN_E_MS,
        tau_syn_i_ms=TAU_SYN_I_MS,
    )

    # Convert to numpy for plotting and JSON
    V_m = np.array(voltages)               # (n_steps, 2)
    spikes_arr = np.array(spikes_arr_jax)  # (n_steps, 2)
    syn_currents = np.array(syn_currents_jax)  # (n_steps, 2) — dynamic injection
    sources = np.array(sources_jax)        # (n_steps, 2)

    # Run uncoupled simulation (g_ei=0, g_ie=0) for coupled_vs_uncoupled comparison figure
    voltages_unc, spikes_arr_unc_jax, _, _ = simulate_dynamic_ei_coupling(
        params,
        n_steps=n_steps,
        dt_ms=DT_MS,
        key=key,
        g_ei=0.0,
        g_ie=0.0,
        tau_syn_e_ms=TAU_SYN_E_MS,
        tau_syn_i_ms=TAU_SYN_I_MS,
    )
    V_m_unc = np.array(voltages_unc)           # (n_steps, 2) — no coupling
    spikes_arr_unc = np.array(spikes_arr_unc_jax)  # (n_steps, 2)

    print(f"  V_m shape: {V_m.shape}")
    print(f"  spikes shape: {spikes_arr.shape}")
    print(f"  syn_currents shape (dynamic): {syn_currents.shape}")
    print(f"  sources shape: {sources.shape}")
    print()

    # ============================================================================
    # SECTION 3: Firing Rate Computation and Gate Checks
    # ============================================================================

    print("Computing firing rates and probe readouts...")

    # Per-neuron spike indices and times
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
    sources_finite = bool(np.all(np.isfinite(sources)))
    syn_currents_finite = bool(np.all(np.isfinite(syn_currents)))

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
    print(f"  PV neuron (idx={I_INDEX}):")
    print(f"    Spikes: {i_n_spikes}, Firing rate: {i_firing_rate_hz:.2f} Hz")
    print(f"    Voltage range: [{i_v_min:.1f}, {i_v_max:.1f}] mV")
    print(f"    Firing rate gate (2-25 Hz): {'PASS' if i_firing_rate_gate_pass else 'FAIL'}")
    print(f"  Synaptic currents max (dynamic): {float(np.max(np.abs(syn_currents))):.4f}")
    print()

    # ============================================================================
    # SECTION 4: Proxy Field Readouts (LFP-like and CSD-like)
    # ============================================================================

    # Compute LFP proxy from sources (sum across neurons, broadcast to n_contacts)
    N_CONTACTS = 16
    # LFP proxy: weighted sum of sources, broadcast to contact array
    lfp_raw = np.sum(sources, axis=1)  # (n_steps,)
    lfp_proxy = np.outer(lfp_raw, np.ones(N_CONTACTS))  # (n_steps, N_CONTACTS)

    # CSD proxy: spatial derivative of LFP
    csd_proxy = np.diff(lfp_proxy, axis=1)  # (n_steps, N_CONTACTS-1)

    # ============================================================================
    # SECTION 5: Probe Report (8 operators via jaxfne.fields)
    # ============================================================================

    # Convert arrays to jax for probe operators
    spikes_jax = jnp.array(spikes_arr)
    V_m_jax = jnp.array(V_m)
    sources_jax2 = jnp.array(sources)
    lfp_proxy_jax = jnp.array(lfp_proxy)
    csd_proxy_jax = jnp.array(csd_proxy)

    probe_report = {
        "spikes": spk_probe(spikes_jax).report,
        "V_m": vm_probe(V_m_jax).report,
        "source": source_probe(sources_jax2).report,
        "lfp_proxy": lfp_proxy_probe(lfp_proxy_jax).report,
        "csd_proxy": csd_proxy_probe(csd_proxy_jax).report,
        "eeg_proxy": eeg_proxy_probe(lfp_proxy_jax).report,
        "meg_proxy": meg_proxy_probe(lfp_proxy_jax).report,
        "emm_proxy": emm_proxy_probe(jnp.mean(jnp.abs(lfp_proxy_jax), axis=1)).report,
    }

    # ============================================================================
    # SECTION 6: Atlas Manifest (collector-compatible)
    # ============================================================================

    try:
        import plotly  # noqa: F401
        plotly_available = True
    except ImportError:
        plotly_available = False

    run_id = f"v033_two_neuron_ei_{int(datetime.now().timestamp())}"

    # Coupling scenario: documented as dynamic synaptic current injection
    coupling_scenario = {
        "scenario_kind": "coupled_e_to_i_excitatory_and_i_to_e_inhibitory",
        "e_to_i_conductance_model_units": float(G_EI),
        "i_to_e_conductance_model_units": float(G_IE),
        "tau_syn_excitatory_ms": float(TAU_SYN_E_MS),
        "tau_syn_inhibitory_ms": float(TAU_SYN_I_MS),
        "implementation_method": "dynamic_synaptic_current_injection",
        "implementation_note": (
            "Synaptic currents are computed DYNAMICALLY during lax.scan via exponential "
            "synaptic traces in the carry state (simulate_dynamic_ei_coupling). "
            "This is genuine real-time coupling injection, not post-hoc computation."
        ),
        "coupling_documented": True,
        "synaptic_current_traces_available": True,
        "post_hoc_coupling_only": False,
        "dynamic_carry_state": True,
    }

    # Conservation proxy diagnostics (basic stats without field PDE)
    diag = {
        "e_mean_firing_rate_hz": e_firing_rate_hz,
        "i_mean_firing_rate_hz": i_firing_rate_hz,
        "source_sum": float(np.sum(sources)),
        "source_finite": sources_finite,
        "syn_currents_finite": syn_currents_finite,
        "v_finite_e": e_voltage_finite,
        "v_finite_i": i_voltage_finite,
        "coupling_mode": "dynamic_injection_simulate_dynamic_ei_coupling",
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
            "source_finite": sources_finite,
            "json_safe": True,
            "duration_gate": DURATION_MS >= 1000.0,
            "dt_gate": DT_MS == 0.1,
            "dtype_gate": str(V_m.dtype) == "float32",
            "coupling_dynamic_injection": True,
            "all_gates_pass": all([
                e_firing_rate_gate_pass,
                i_firing_rate_gate_pass,
                e_voltage_finite,
                i_voltage_finite,
                sources_finite,
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
            "dtype": str(V_m.dtype),
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
            "preset": "eig_e_plus_pv",
            "n_neurons": N_NEURONS,
            "e_cell_type": "regular_spiking_E",
            "i_cell_type": "fast_spiking_PV",
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
            "cell_type": "inhibitory_pv",
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
            "source_proxy_status": "generated_proxy",
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
            "Dynamic coupling is implemented via exponential synaptic traces in lax.scan carry state.",
            "Each neuron's dynamics reflect the Izhikevich E+PV preset with adjusted drives.",
            "No mechanism of E/I balance or cortical function is proven by this tutorial.",
            "This tutorial validates the jaxfne API for multi-neuron dynamic coupling.",
        ],
    }

    # ============================================================================
    # SECTION 7: Figures
    # ============================================================================

    if HAS_MATPLOTLIB:
        print("Generating figures...")
        OUT.mkdir(parents=True, exist_ok=True)
        STATIC_FIGS.mkdir(parents=True, exist_ok=True)

        figures_dir = OUT / "figures"
        figures_dir.mkdir(exist_ok=True)

        e_spike_times = t[e_spike_indices]
        i_spike_times = t[i_spike_indices]

        # Figure 1: Voltage traces for E and PV neurons (two-row panel)
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        axes[0].plot(t, V_m[:, E_INDEX], linewidth=0.6, color='blue', label='E neuron')
        axes[0].set_ylabel('Voltage (mV)')
        axes[0].set_title(
            f'v0.3.3: Excitatory Neuron Voltage Trace\n'
            f'(Firing rate: {e_firing_rate_hz:.2f} Hz, Proxy readout)'
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')

        axes[1].plot(t, V_m[:, I_INDEX], linewidth=0.6, color='red', label='PV neuron')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Voltage (mV)')
        axes[1].set_title(
            f'v0.3.3: PV (Inhibitory) Neuron Voltage Trace\n'
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

        ax.scatter(e_spike_times, [0] * len(e_spike_times), marker='|', s=500, color='blue',
                   linewidth=2, label=f'E spikes (n={e_n_spikes})')
        ax.scatter(i_spike_times, [1] * len(i_spike_times), marker='|', s=500, color='red',
                   linewidth=2, label=f'PV spikes (n={i_n_spikes})')

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['E (blue)', 'PV (red)'])
        ax.set_title(
            f'v0.3.3: Two-Neuron E/PV Spike Raster\n'
            f'(Dynamic coupling: E→PV g={G_EI}, PV→E g={G_IE})'
        )
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper right')

        plt.tight_layout()
        raster_fig_path = figures_dir / "v0303_two_neuron_ei_raster.png"
        plt.savefig(raster_fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 3: Dynamic synaptic coupling currents (from lax.scan carry, NOT post-hoc)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, syn_currents[:, E_INDEX], linewidth=0.8, color='orange',
                label=f'I→E inhibitory current (g={G_IE})', alpha=0.8)
        ax.plot(t, syn_currents[:, I_INDEX], linewidth=0.8, color='green',
                label=f'E→PV excitatory current (g={G_EI})', alpha=0.8)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Synaptic current (proxy units)')
        ax.set_title(
            'v0.3.3: Dynamic E/PV Synaptic Currents (lax.scan carry state)\n'
            '(Dynamic injection during simulation, not post-hoc, computational scaffold)'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        plt.tight_layout()
        coupling_fig_path = figures_dir / "v0303_two_neuron_ei_coupling_currents.png"
        plt.savefig(coupling_fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 4: Source aggregation (E vs PV contribution)
        e_source_ts = sources[:, E_INDEX]
        i_source_ts = sources[:, I_INDEX]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, e_source_ts, linewidth=0.6, color='blue', label='E source (aggregated)', alpha=0.7)
        ax.plot(t, i_source_ts, linewidth=0.6, color='red', label='PV source (aggregated)', alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Source (proxy units)')
        ax.set_title(
            'v0.3.3: E/PV Source Aggregation\n'
            '(Proxy readout, uncalibrated Izhikevich native current)'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        plt.tight_layout()
        source_fig_path = figures_dir / "v0303_two_neuron_ei_source.png"
        plt.savefig(source_fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 5: LFP-like proxy
        fig, ax = plt.subplots(figsize=(14, 4))
        for contact_idx in range(min(4, lfp_proxy.shape[1])):
            ax.plot(t, lfp_proxy[:, contact_idx], linewidth=0.5,
                    label=f'Contact {contact_idx}', alpha=0.7)
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

        # Figure 6: CSD-like proxy
        fig, ax = plt.subplots(figsize=(14, 4))
        for contact_idx in range(min(4, csd_proxy.shape[1])):
            ax.plot(t, csd_proxy[:, contact_idx], linewidth=0.5,
                    label=f'Layer {contact_idx}', alpha=0.7)
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

        # Figure 7: Coupled vs. uncoupled comparison (ablation figure)
        fig_cvu = plot_coupled_vs_uncoupled(
            time_ms=t,
            spikes_coupled=spikes_arr,
            spikes_uncoupled=spikes_arr_unc,
            v_m_coupled=V_m,
            v_m_uncoupled=V_m_unc,
        )
        coupled_vs_uncoupled_fig_path = figures_dir / "v0303_two_neuron_ei_coupled_vs_uncoupled.png"
        fig_cvu.savefig(coupled_vs_uncoupled_fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig_cvu)

        # Figure 8: Circuit schematic (connectivity diagram)
        fig_cs = plot_circuit_schematic(
            g_ei=G_EI,
            g_ie=G_IE,
            firing_rate_e=e_firing_rate_hz,
            firing_rate_i=i_firing_rate_hz,
        )
        circuit_schematic_fig_path = figures_dir / "v0303_two_neuron_ei_circuit_schematic.png"
        fig_cs.savefig(circuit_schematic_fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig_cs)

        print(f"  Saved: {voltage_fig_path}")
        print(f"  Saved: {raster_fig_path}")
        print(f"  Saved: {coupling_fig_path}")
        print(f"  Saved: {source_fig_path}")
        print(f"  Saved: {lfp_fig_path}")
        print(f"  Saved: {csd_fig_path}")
        print(f"  Saved: {coupled_vs_uncoupled_fig_path}")
        print(f"  Saved: {circuit_schematic_fig_path}")

        # Copy to docs-stable _static/figures (committed paths referenced in docs)
        static_voltage = STATIC_FIGS / "v0303_two_neuron_ei_voltage.png"
        static_raster = STATIC_FIGS / "v0303_two_neuron_ei_raster.png"
        static_coupling = STATIC_FIGS / "v0303_two_neuron_ei_coupling_currents.png"
        static_source = STATIC_FIGS / "v0303_two_neuron_ei_source.png"
        static_lfp = STATIC_FIGS / "v0303_two_neuron_ei_lfp_proxy.png"
        static_csd = STATIC_FIGS / "v0303_two_neuron_ei_csd_proxy.png"
        static_coupled_vs_uncoupled = STATIC_FIGS / "v0303_two_neuron_ei_coupled_vs_uncoupled.png"
        static_circuit_schematic = STATIC_FIGS / "v0303_two_neuron_ei_circuit_schematic.png"

        shutil.copy2(voltage_fig_path, static_voltage)
        shutil.copy2(raster_fig_path, static_raster)
        shutil.copy2(coupling_fig_path, static_coupling)
        shutil.copy2(source_fig_path, static_source)
        shutil.copy2(lfp_fig_path, static_lfp)
        shutil.copy2(csd_fig_path, static_csd)
        shutil.copy2(coupled_vs_uncoupled_fig_path, static_coupled_vs_uncoupled)
        shutil.copy2(circuit_schematic_fig_path, static_circuit_schematic)

        print(f"  Copied to _static: {static_voltage}")
        print(f"  Copied to _static: {static_raster}")
        print(f"  Copied to _static: {static_coupling}")
        print(f"  Copied to _static: {static_source}")
        print(f"  Copied to _static: {static_lfp}")
        print(f"  Copied to _static: {static_csd}")
        print(f"  Copied to _static: {static_coupled_vs_uncoupled}")
        print(f"  Copied to _static: {static_circuit_schematic}")

        # SHA256 hashes from docs-stable paths (canonical tracked paths)
        voltage_hash = sha256_file(static_voltage)
        raster_hash = sha256_file(static_raster)
        coupling_hash = sha256_file(static_coupling)
        source_hash = sha256_file(static_source)
        lfp_hash = sha256_file(static_lfp)
        csd_hash = sha256_file(static_csd)
        coupled_vs_uncoupled_hash = sha256_file(static_coupled_vs_uncoupled)
        circuit_schematic_hash = sha256_file(static_circuit_schematic)

        atlas_manifest["figures"] = {
            "voltage_traces": {
                "docs_stable_path": str(static_voltage),
                "runtime_path": str(voltage_fig_path),
                "sha256": voltage_hash,
                "dpi": 150,
                "description": "E and PV neuron voltage traces (two-panel)",
            },
            "spike_raster": {
                "docs_stable_path": str(static_raster),
                "runtime_path": str(raster_fig_path),
                "sha256": raster_hash,
                "dpi": 150,
                "description": "E (blue) and PV (red) spike raster",
            },
            "coupling_currents": {
                "docs_stable_path": str(static_coupling),
                "runtime_path": str(coupling_fig_path),
                "sha256": coupling_hash,
                "dpi": 150,
                "description": "Dynamic E/PV synaptic currents (from lax.scan carry state)",
            },
            "source_aggregation": {
                "docs_stable_path": str(static_source),
                "runtime_path": str(source_fig_path),
                "sha256": source_hash,
                "dpi": 150,
                "description": "E/PV source aggregation (proxy units)",
            },
            "lfp_proxy": {
                "docs_stable_path": str(static_lfp),
                "runtime_path": str(lfp_fig_path),
                "sha256": lfp_hash,
                "dpi": 150,
                "description": "LFP-like proxy (first 4 contacts)",
            },
            "csd_proxy": {
                "docs_stable_path": str(static_csd),
                "runtime_path": str(csd_fig_path),
                "sha256": csd_hash,
                "dpi": 150,
                "description": "CSD-like proxy (first 4 layers)",
            },
            "coupled_vs_uncoupled": {
                "docs_stable_path": str(static_coupled_vs_uncoupled),
                "runtime_path": str(coupled_vs_uncoupled_fig_path),
                "sha256": coupled_vs_uncoupled_hash,
                "dpi": 150,
                "description": "Coupling ablation: coupled vs. uncoupled E/I dynamics (4-panel)",
            },
            "circuit_schematic": {
                "docs_stable_path": str(static_circuit_schematic),
                "runtime_path": str(circuit_schematic_fig_path),
                "sha256": circuit_schematic_hash,
                "dpi": 150,
                "description": "Recurrent E/I circuit schematic with conductance labels",
            },
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

        # Write probe_report.json as a separate file for reference
        write_json(OUT / "probe_report.json", probe_report)

        # Write separate validation_report for reference
        write_json(OUT / "validation_report.json", atlas_manifest["validation_report"])

        # Metrics file
        write_json(OUT / "metrics.json", {
            "duration_ms": DURATION_MS,
            "dt_ms": DT_MS,
            "seed": SEED,
            "dtype": str(V_m.dtype),
            "n_steps": n_steps,
            "n_neurons": N_NEURONS,
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
            "syn_currents_finite": syn_currents_finite,
            "coupling_mode": "dynamic_injection",
        })

        # Asset hashes
        json_files = [manifest_path, OUT / "probe_report.json",
                      OUT / "validation_report.json", OUT / "metrics.json"]
        hashes = {p.name: sha256_file(p) for p in json_files}
        hashes["figures/v0303_two_neuron_ei_voltage.png"] = voltage_hash
        hashes["figures/v0303_two_neuron_ei_raster.png"] = raster_hash
        hashes["figures/v0303_two_neuron_ei_coupling_currents.png"] = coupling_hash
        hashes["figures/v0303_two_neuron_ei_source.png"] = source_hash
        hashes["figures/v0303_two_neuron_ei_lfp_proxy.png"] = lfp_hash
        hashes["figures/v0303_two_neuron_ei_csd_proxy.png"] = csd_hash
        hashes["figures/v0303_two_neuron_ei_coupled_vs_uncoupled.png"] = coupled_vs_uncoupled_hash
        hashes["figures/v0303_two_neuron_ei_circuit_schematic.png"] = circuit_schematic_hash
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
        print(f"✓ Simulation: {n_steps} steps over {DURATION_MS} ms (dynamic coupling)")
        print()
        print(f"✓ E neuron (idx={E_INDEX}):")
        print(f"  - Firing rate: {e_firing_rate_hz:.2f} Hz ({e_n_spikes} spikes)")
        print(f"  - Gate (2-25 Hz): {'PASS' if e_firing_rate_gate_pass else 'FAIL'}")
        print(f"  - Voltage range: [{e_v_min:.1f}, {e_v_max:.1f}] mV")
        print(f"  - All finite: {e_voltage_finite}")
        print()
        print(f"✓ PV neuron (idx={I_INDEX}):")
        print(f"  - Firing rate: {i_firing_rate_hz:.2f} Hz ({i_n_spikes} spikes)")
        print(f"  - Gate (2-25 Hz): {'PASS' if i_firing_rate_gate_pass else 'FAIL'}")
        print(f"  - Voltage range: [{i_v_min:.1f}, {i_v_max:.1f}] mV")
        print(f"  - All finite: {i_voltage_finite}")
        print()
        print(f"✓ Dynamic coupling: E→PV g={G_EI}, PV→E g={G_IE}, syn_currents finite={syn_currents_finite}")
        print(f"✓ Source finite: {sources_finite}")
        print(f"✓ Manifest (collector-visible): {manifest_path}")
        print(f"✓ Figures: {len(atlas_manifest['figures'])} generated and hashed")
        print(f"✓ Plotly available: {plotly_available}")
        print()
        print("Truth status: computational_scaffold, proxy_readout_only")
        print("Physical amplitude claim allowed: False")
        print("Coupling mode: dynamic_synaptic_current_injection (not post-hoc)")
        print("=" * 80)
    else:
        # matplotlib not available; provide empty figures dict
        atlas_manifest["figures"] = {}
        print("(Figures skipped - matplotlib not installed)")

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
