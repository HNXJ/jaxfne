#!/usr/bin/env python3
"""
v0.3.1 Single Izhikevich Neuron Tutorial

Executable tutorial demonstrating a single Izhikevich neuron simulation
using jaxfne v0.2.30 stable toolbox.

Writes atlas-compatible manifest to outputs/v030_01_single_neuron_izhikevich/
for v0.3 collector validation.

Truth status: computational_scaffold, proxy_readout_only
Physical amplitude claim allowed: False
Claim level: computational_scaffold
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

# Collector-visible output directory (v030_01 prefix required by collector)
OUT = Path("outputs/v030_01_single_neuron_izhikevich")
# Docs-stable figures directory
STATIC_FIGS = Path("docs/tutorials_v030/_static/figures")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, allow_nan=False, indent=2, sort_keys=True)


def main():
    """Main tutorial execution."""

    print("=" * 80)
    print("v0.3.1 Single Izhikevich Neuron Tutorial")
    print("=" * 80)
    print()

    # ============================================================================
    # SECTION 5: Configuration
    # ============================================================================

    duration_ms = 1000.0
    dt_ms = 0.1
    seed = 42

    run = jtfne.runtime(device_type="auto", dtype="float32", x64_enabled=False, seed=seed)

    cfg = (
        jtfne.configuration()
        .network(name="v030_01_single_neuron", kind="isolated_neuron", n=1, cell_types={"E": 1.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(
            name="single_channel_16contact",
            modes=["spikes", "V_m", "source", "LFP", "CSD"],
            n_contacts=16,
        )
        .update_metadata(
            dx_mm=0.010,
            dy_mm=0.010,
            dz_mm=0.010,
            geometry_mode="declared_tutorial_metadata_not_solved_3d_grid",
            tutorial_id="v0.3.1",
        )
    )

    print(f"Duration: {duration_ms} ms, dt: {dt_ms} ms, Seed: {seed}")

    # ============================================================================
    # SECTION 6: Simulation
    # ============================================================================

    print("Running simulation...")
    model = jtfne.construct(cfg)

    n_steps = int(duration_ms / dt_ms)
    t = np.arange(n_steps) * dt_ms

    sim_spec = jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed, runtime=run)
    signals = model.simulate(sim_spec)

    print(f"  V_m shape: {signals.V_m.shape}")
    print(f"  spikes shape: {signals.spikes.shape}")
    field = signals.field
    if field is None:
        raise RuntimeError("Expected laminar proxy field output; field is None")
    print(f"  sources shape: {signals.sources.shape}")
    print()

    # ============================================================================
    # SECTION 7: Probe and Readout (all 8 operators)
    # ============================================================================

    print("Computing probe readouts...")
    V_m = np.array(signals.V_m)
    spikes_arr = np.array(signals.spikes)

    # Firing rate
    spike_indices = np.where(spikes_arr[:, 0] > 0.5)[0]
    n_spikes = len(spike_indices)
    firing_rate_hz = float((n_spikes / duration_ms) * 1000.0)
    firing_rate_gate_pass = 2.0 <= firing_rate_hz <= 25.0

    # Voltage statistics
    V_min = float(np.min(V_m))
    V_max = float(np.max(V_m))
    V_mean = float(np.mean(V_m))

    print(f"  Spikes detected: {n_spikes}")
    print(f"  Firing rate: {firing_rate_hz:.2f} Hz")
    print(f"  Voltage range: [{V_min:.1f}, {V_max:.1f}] mV")
    print(f"  Firing rate gate (2-25 Hz): {'PASS' if firing_rate_gate_pass else 'FAIL'}")
    print()

    # Source finite status — None-safe
    sources_finite: bool | None
    if signals.sources is not None:
        sources_finite = bool(jnp.all(jnp.isfinite(signals.sources)))
    else:
        sources_finite = None  # not_generated

    # Generate probe reports using collector-required key names:
    # spikes, V_m, source, lfp_proxy, csd_proxy, eeg_proxy, meg_proxy, emm_proxy
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
    # SECTION 8: Atlas Manifest (collector-compatible)
    # ============================================================================

    # Get conservation_proxy_diagnostics from model manifest
    raw_manifest = model.manifest(signals=signals)
    diag = dict(raw_manifest.get("conservation_proxy_diagnostics", {}))
    diag["mean_firing_rate_hz"] = firing_rate_hz  # Required by collector Gate 5

    # Plotly status
    try:
        import plotly  # noqa: F401
        plotly_available = True
    except ImportError:
        plotly_available = False

    run_id = f"v031_single_izhikevich_{int(datetime.now().timestamp())}"

    # Atlas-compatible manifest with all 4 embedded blocks
    atlas_manifest = {
        "run_id": run_id,
        "tutorial_id": "v0301_single_izhikevich_neuron",
        "scenario_id": "v030_01_single_neuron_izhikevich",
        "jaxfne_version": jtfne.__version__,
        "schema_version": "v0.3.1",
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
            "firing_rate_gate_2_25_hz": firing_rate_gate_pass,
            "voltage_finite": bool(np.all(np.isfinite(V_m))),
            "source_finite": sources_finite,  # None if not generated
            "json_safe": True,
            "duration_gate": duration_ms >= 1000.0,
            "dt_gate": dt_ms == 0.1,
            "dtype_gate": str(signals.V_m.dtype) == "float32",
            "all_gates_pass": all([
                firing_rate_gate_pass,
                bool(np.all(np.isfinite(V_m))),
                (sources_finite if sources_finite is not None else True),
                duration_ms >= 1000.0,
                dt_ms == 0.1,
            ]),
            "status": "PASS" if all([
                firing_rate_gate_pass,
                bool(np.all(np.isfinite(V_m))),
                duration_ms >= 1000.0,
            ]) else "FAIL",
        },

        # conservation_proxy_diagnostics block — from model + firing rate
        "conservation_proxy_diagnostics": diag,

        # === Additional fields ===

        "simulation": {
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_steps": n_steps,
            "seed": seed,
            "dtype": str(signals.V_m.dtype),
        },

        "neuron": {
            "model": "izhikevich",
            "preset": "cortical_eig",
            "n_neurons": int(signals.V_m.shape[1]),
        },

        "firing_rate": {
            "firing_rate_hz": firing_rate_hz,
            "n_spikes": int(n_spikes),
            "gate_2_25_hz": firing_rate_gate_pass,
        },

        "voltage": {
            "V_min_mV": V_min,
            "V_max_mV": V_max,
            "V_mean_mV": V_mean,
            "all_finite": bool(np.all(np.isfinite(V_m))),
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
            "plotly_status_reason": "static_png_baseline_for_v0301",
        },

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
    OUT.mkdir(parents=True, exist_ok=True)
    STATIC_FIGS.mkdir(parents=True, exist_ok=True)

    figures_dir = OUT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Figure 1: Voltage trace
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, V_m[:, 0], linewidth=0.8, color='blue')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title(
        'v0.3.1: Single Izhikevich Neuron - Voltage Trace\n'
        '(Proxy readout, not biological validation)'
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    voltage_fig_path = figures_dir / "v0301_single_neuron_voltage.png"
    plt.savefig(voltage_fig_path, dpi=150, bbox_inches='tight')
    plt.close()

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

    print(f"  Saved: {voltage_fig_path}")
    print(f"  Saved: {raster_fig_path}")

    # Copy to docs-stable _static/figures (committed paths referenced in docs)
    static_voltage = STATIC_FIGS / "v0301_single_neuron_voltage.png"
    static_raster = STATIC_FIGS / "v0301_single_neuron_raster.png"
    shutil.copy2(voltage_fig_path, static_voltage)
    shutil.copy2(raster_fig_path, static_raster)
    print(f"  Copied to _static: {static_voltage}")
    print(f"  Copied to _static: {static_raster}")

    # SHA256 hashes from docs-stable paths (canonical tracked paths)
    voltage_hash = sha256_file(static_voltage)
    raster_hash = sha256_file(static_raster)

    atlas_manifest["figures"] = {
        "voltage_trace": {
            "docs_stable_path": str(static_voltage),
            "runtime_path": str(voltage_fig_path),
            "sha256": voltage_hash,
            "dpi": 150,
        },
        "spike_raster": {
            "docs_stable_path": str(static_raster),
            "runtime_path": str(raster_fig_path),
            "sha256": raster_hash,
            "dpi": 150,
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

    # Also write probe_report.json as a separate file for reference
    write_json(OUT / "probe_report.json", probe_report)

    # Write separate validation_report for reference
    write_json(OUT / "validation_report.json", atlas_manifest["validation_report"])

    # Metrics file
    write_json(OUT / "metrics.json", {
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
        "dtype": str(signals.V_m.dtype),
        "n_steps": n_steps,
        "n_neurons": int(signals.V_m.shape[1]),
        "mean_firing_rate_hz": firing_rate_hz,
        "n_spikes": int(n_spikes),
        "Vm_mean": V_mean,
        "Vm_min": V_min,
        "Vm_max": V_max,
        "finite_all_core": bool(np.all(np.isfinite(V_m))),
    })

    # Asset hashes
    json_files = [manifest_path, OUT / "probe_report.json",
                  OUT / "validation_report.json", OUT / "metrics.json"]
    hashes = {p.name: sha256_file(p) for p in json_files}
    hashes["figures/v0301_single_neuron_voltage.png"] = voltage_hash
    hashes["figures/v0301_single_neuron_raster.png"] = raster_hash
    write_json(OUT / "asset_hashes.json", hashes)

    # Also update canonical docs manifest (for v0301 naming)
    canonical_manifest_path = Path("docs/tutorials_v030/manifests/v0301_single_izhikevich_neuron_manifest.json")
    canonical_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(canonical_manifest_path, atlas_manifest)

    canonical_report_path = Path("docs/tutorials_v030/reports/v0301_single_izhikevich_neuron_validation_report.json")
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
    print(f"✓ Simulation: {n_steps} steps over {duration_ms} ms")
    print(f"✓ Firing rate: {firing_rate_hz:.2f} Hz ({n_spikes} spikes)")
    print(f"✓ Firing rate gate (2-25 Hz): {'PASS' if firing_rate_gate_pass else 'FAIL'}")
    print(f"✓ Voltage range: [{V_min:.1f}, {V_max:.1f}] mV")
    print(f"✓ All V_m finite: {bool(np.all(np.isfinite(V_m)))}")
    print(f"✓ Source finite: {sources_finite}")
    print(f"✓ Manifest (collector-visible): {manifest_path}")
    print(f"✓ Docs-stable figures: {static_voltage}, {static_raster}")
    print(f"✓ Plotly available: {plotly_available}")
    print()
    print("Truth status: computational_scaffold, proxy_readout_only")
    print("Physical amplitude claim allowed: False")
    print("=" * 80)

    return {
        "atlas_manifest": atlas_manifest,
        "manifest_path": str(manifest_path),
        "firing_rate_hz": firing_rate_hz,
        "firing_rate_gate_pass": firing_rate_gate_pass,
        "figures": atlas_manifest["figures"],
    }


if __name__ == "__main__":
    main()
