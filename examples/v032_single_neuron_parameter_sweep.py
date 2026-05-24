#!/usr/bin/env python3
"""
v0.3.2 Single Neuron Parameter Sweep Tutorial

Sweeps Izhikevich a and d parameters to show regime transitions
(regular spiking → adaptation → fast spiking proxies).
Uses jtfne.with_emitter_parameters() on the stable jaxfne==0.2.30 toolbox.

Writes atlas-compatible manifest to outputs/v030_02_single_neuron_parameter_sweep/
for v0.3 collector validation (gate: 2 PASS, 0 FAIL alongside v0.3.1).

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

# Collector-visible output directory (v030_02 prefix required by collector)
OUT = Path("outputs/v030_02_single_neuron_parameter_sweep")
# Docs-stable figures directory
STATIC_FIGS = Path("docs/tutorials_v030/_static/figures")

# Sweep grid — kept small for deterministic, fast execution
A_VALUES = [0.02, 0.05, 0.10]          # recovery time scale (slow → fast)
D_VALUES = [2.0, 6.0, 8.0, 12.0]       # after-spike reset (weak → strong adaptation)
DURATION_MS = 500.0                      # shorter than v0.3.1; sufficient to measure rate
DT_MS = 0.1
SEED = 42


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


def build_base_model(run):
    """Build the shared base model (same config as v0.3.1, runtime injected)."""
    cfg = (
        jtfne.configuration()
        .network(
            name="v030_02_parameter_sweep",
            kind="isolated_neuron",
            n=1,
            cell_types={"E": 1.0},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
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
            tutorial_id="v0.3.2",
        )
    )
    return jtfne.construct(cfg)


def run_sweep_point(base_model, a_val, d_val, sim_spec):
    """Run one sweep point; return firing_rate_hz and basic stats."""
    model = jtfne.with_emitter_parameters(base_model, a=a_val, d=d_val)
    signals = model.simulate(sim_spec)
    spikes_arr = np.array(signals.spikes)
    spike_indices = np.where(spikes_arr[:, 0] > 0.5)[0]
    n_spikes = len(spike_indices)
    firing_rate_hz = float((n_spikes / DURATION_MS) * 1000.0)
    V_m = np.array(signals.V_m)
    return {
        "a": float(a_val),
        "d": float(d_val),
        "firing_rate_hz": firing_rate_hz,
        "n_spikes": int(n_spikes),
        "V_min": float(np.min(V_m)),
        "V_max": float(np.max(V_m)),
        "V_mean": float(np.mean(V_m)),
        "all_finite": bool(np.all(np.isfinite(V_m))),
    }


def main():
    print("=" * 80)
    print("v0.3.2 Single Neuron Parameter Sweep Tutorial")
    print("=" * 80)
    print()

    OUT.mkdir(parents=True, exist_ok=True)
    STATIC_FIGS.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    # =========================================================================
    # SECTION 5: Configuration and base model
    # =========================================================================

    run = jtfne.runtime(
        device_type="auto", dtype="float32", x64_enabled=False, seed=SEED
    )
    base_model = build_base_model(run)
    sim_spec = jtfne.simulation(
        duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED, runtime=run
    )

    print(f"Sweep grid: a={A_VALUES}, d={D_VALUES}")
    print(f"Points: {len(A_VALUES) * len(D_VALUES)}")
    print(f"Duration per point: {DURATION_MS} ms, dt: {DT_MS} ms")
    print()

    # =========================================================================
    # SECTION 6: Sweep execution
    # =========================================================================

    print("Running sweep...")
    sweep_results = []
    firing_grid = np.zeros((len(A_VALUES), len(D_VALUES)), dtype=float)

    for i, a_val in enumerate(A_VALUES):
        for j, d_val in enumerate(D_VALUES):
            result = run_sweep_point(base_model, a_val, d_val, sim_spec)
            sweep_results.append(result)
            firing_grid[i, j] = result["firing_rate_hz"]
            print(
                f"  a={a_val:.2f}, d={d_val:.1f} → "
                f"{result['firing_rate_hz']:.1f} Hz  ({result['n_spikes']} spikes)"
            )

    print()

    # Gate: at least one point must be in 2–25 Hz
    rates = [r["firing_rate_hz"] for r in sweep_results]
    any_in_gate = any(2.0 <= hz <= 25.0 for hz in rates)
    rate_min = float(np.min(rates))
    rate_max = float(np.max(rates))
    all_finite = all(r["all_finite"] for r in sweep_results)

    print(f"Firing rate range across sweep: [{rate_min:.1f}, {rate_max:.1f}] Hz")
    print(f"Any point in 2–25 Hz gate: {'PASS' if any_in_gate else 'FAIL'}")
    print(f"All V_m finite: {all_finite}")

    # Baseline (a=0.02, d=8 ~ cortical_eig default) firing rate for gate
    baseline_result = next(
        r for r in sweep_results if abs(r["a"] - 0.02) < 0.001 and abs(r["d"] - 8.0) < 0.1
    )
    baseline_rate = baseline_result["firing_rate_hz"]
    baseline_gate = 2.0 <= baseline_rate <= 25.0
    print(f"Baseline (a=0.02, d=8): {baseline_rate:.1f} Hz, gate: {'PASS' if baseline_gate else 'FAIL'}")
    print()

    # =========================================================================
    # SECTION 9: Figures
    # =========================================================================

    print("Generating figures...")

    # Figure 1: Firing rate heatmap (a vs d)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        firing_grid,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        vmin=0,
        vmax=max(rate_max, 25.0),
    )
    ax.set_xticks(range(len(D_VALUES)))
    ax.set_xticklabels([f"{d:.1f}" for d in D_VALUES])
    ax.set_yticks(range(len(A_VALUES)))
    ax.set_yticklabels([f"{a:.2f}" for a in A_VALUES])
    ax.set_xlabel("d — after-spike reset (adaptation strength, proxy units)")
    ax.set_ylabel("a — recovery time scale (proxy units)")
    ax.set_title(
        "v0.3.2: Izhikevich Parameter Sweep — Firing Rate (Hz)\n"
        "(Proxy readout, not biological validation)"
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Firing rate (Hz)")
    # Annotate cells
    for i in range(len(A_VALUES)):
        for j in range(len(D_VALUES)):
            ax.text(
                j, i, f"{firing_grid[i, j]:.1f}",
                ha="center", va="center", fontsize=9,
                color="white" if firing_grid[i, j] < rate_max * 0.6 else "black",
            )
    plt.tight_layout()
    heatmap_path = OUT / "figures" / "v0302_sweep_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {heatmap_path}")

    # Figure 2: Regime line plot — firing rate vs d for each a
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, a_val in enumerate(A_VALUES):
        rates_row = firing_grid[i, :]
        ax.plot(D_VALUES, rates_row, marker="o", label=f"a={a_val:.2f}")
    ax.axhline(2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Gate low (2 Hz)")
    ax.axhline(25.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7, label="Gate high (25 Hz)")
    ax.set_xlabel("d — after-spike reset (proxy units)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(
        "v0.3.2: Firing Rate vs. Adaptation Parameter d\n"
        "(Proxy readout, not biological validation)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    regime_path = OUT / "figures" / "v0302_regime_lines.png"
    plt.savefig(regime_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {regime_path}")

    # Copy to docs-stable _static/figures
    static_heatmap = STATIC_FIGS / "v0302_sweep_heatmap.png"
    static_regime = STATIC_FIGS / "v0302_regime_lines.png"
    shutil.copy2(heatmap_path, static_heatmap)
    shutil.copy2(regime_path, static_regime)
    print(f"  Copied to _static: {static_heatmap}")
    print(f"  Copied to _static: {static_regime}")

    heatmap_hash = sha256_file(static_heatmap)
    regime_hash = sha256_file(static_regime)

    # =========================================================================
    # SECTION 8: Atlas manifest (collector-compatible)
    # =========================================================================

    # Probe report: use baseline point for embedded 8-key block
    # (sweep manifest embeds baseline probe readout as representative)
    from jaxfne.fields import (
        csd_proxy_probe, eeg_proxy_probe, emm_proxy_probe,
        lfp_proxy_probe, meg_proxy_probe, source_probe, spk_probe, vm_probe,
    )

    baseline_model = jtfne.with_emitter_parameters(base_model, a=0.02, d=8.0)
    baseline_signals = baseline_model.simulate(sim_spec)
    field = baseline_signals.field

    probe_report = {
        "spikes": spk_probe(baseline_signals.spikes).report,
        "V_m": vm_probe(baseline_signals.V_m).report,
        "source": source_probe(baseline_signals.sources).report,
        "lfp_proxy": lfp_proxy_probe(field.lfp_proxy).report,
        "csd_proxy": csd_proxy_probe(field.csd_proxy).report,
        "eeg_proxy": eeg_proxy_probe(field.lfp_proxy).report,
        "meg_proxy": meg_proxy_probe(field.lfp_proxy).report,
        "emm_proxy": emm_proxy_probe(
            jnp.mean(jnp.abs(field.lfp_proxy), axis=1)
        ).report,
    }

    raw_manifest = base_model.manifest(signals=baseline_signals)
    diag = dict(raw_manifest.get("conservation_proxy_diagnostics", {}))
    diag["mean_firing_rate_hz"] = baseline_rate  # collector Gate 5

    try:
        import plotly  # noqa: F401
        plotly_available = True
    except ImportError:
        plotly_available = False

    run_id = f"v032_parameter_sweep_{int(datetime.now().timestamp())}"

    firing_rate_gate_pass = 2.0 <= baseline_rate <= 25.0

    atlas_manifest = {
        "run_id": run_id,
        "tutorial_id": "v0302_single_neuron_parameter_sweep",
        "scenario_id": "v030_02_single_neuron_parameter_sweep",
        "jaxfne_version": jtfne.__version__,
        "schema_version": "v0.3.2",
        "timestamp": datetime.now().isoformat(),

        # === Required embedded blocks ===

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

        "probe_report": probe_report,

        "validation_report": {
            "firing_rate_gate_2_25_hz": firing_rate_gate_pass,
            "baseline_firing_rate_hz": baseline_rate,
            "any_sweep_point_in_gate": any_in_gate,
            "sweep_rate_min_hz": rate_min,
            "sweep_rate_max_hz": rate_max,
            "voltage_finite": all_finite,
            "json_safe": True,
            "duration_gate": DURATION_MS >= 500.0,
            "dt_gate": DT_MS == 0.1,
            "all_gates_pass": all([
                firing_rate_gate_pass,
                all_finite,
                any_in_gate,
            ]),
            "status": "PASS" if all([firing_rate_gate_pass, all_finite, any_in_gate]) else "FAIL",
        },

        "conservation_proxy_diagnostics": diag,

        # === Sweep-specific fields ===

        "sweep": {
            "a_values": A_VALUES,
            "d_values": D_VALUES,
            "n_points": len(sweep_results),
            "duration_ms": DURATION_MS,
            "dt_ms": DT_MS,
            "seed": SEED,
            "results": sweep_results,
            "firing_grid": firing_grid.tolist(),
            "tool_used": "jtfne.with_emitter_parameters",
        },

        "simulation": {
            "duration_ms": DURATION_MS,
            "dt_ms": DT_MS,
            "seed": SEED,
            "dtype": str(baseline_signals.V_m.dtype),
        },

        "neuron": {
            "model": "izhikevich",
            "preset": "cortical_eig",
            "n_neurons": 1,
        },

        "plotly": {
            "plotly_available": plotly_available,
            "plotly_html_generated": False,
            "plotly_status_reason": "static_png_baseline_for_v0302",
        },

        "figures": {
            "sweep_heatmap": {
                "docs_stable_path": str(static_heatmap),
                "runtime_path": str(heatmap_path),
                "sha256": heatmap_hash,
                "dpi": 150,
            },
            "regime_lines": {
                "docs_stable_path": str(static_regime),
                "runtime_path": str(regime_path),
                "sha256": regime_hash,
                "dpi": 150,
            },
        },

        "non_claims": [
            "This tutorial is a computational scaffold, not a biological validation.",
            "Parameter regimes are proxy computational ranges, not calibrated biophysical ranges.",
            "Firing rate changes across the sweep reflect model dynamics, not measured neuron behavior.",
            "No field PDE is solved in laminar_proxy_no_pde mode.",
            "No biological mechanism is proven by this tutorial alone.",
        ],
    }

    # =========================================================================
    # Save outputs
    # =========================================================================

    manifest_path = OUT / "manifest.json"
    write_json(manifest_path, atlas_manifest)
    print(f"\nAtlas manifest saved: {manifest_path}")

    # Round-trip check
    with open(manifest_path) as f:
        loaded = json.load(f)
    assert loaded["basis"]["physical_amplitude_claim_allowed"] is False
    assert loaded["basis"]["claim_level"] == "computational_scaffold"
    assert set(loaded["probe_report"].keys()) == {
        "spikes", "V_m", "source", "lfp_proxy", "csd_proxy",
        "eeg_proxy", "meg_proxy", "emm_proxy",
    }
    print("  Manifest round-trip check: PASS")

    write_json(OUT / "probe_report.json", probe_report)
    write_json(OUT / "validation_report.json", atlas_manifest["validation_report"])
    write_json(OUT / "metrics.json", {
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "seed": SEED,
        "n_sweep_points": len(sweep_results),
        "a_values": A_VALUES,
        "d_values": D_VALUES,
        "baseline_firing_rate_hz": baseline_rate,
        "sweep_rate_min_hz": rate_min,
        "sweep_rate_max_hz": rate_max,
        "all_finite": all_finite,
    })

    json_files = [
        manifest_path, OUT / "probe_report.json",
        OUT / "validation_report.json", OUT / "metrics.json",
    ]
    hashes = {p.name: sha256_file(p) for p in json_files}
    hashes["figures/v0302_sweep_heatmap.png"] = heatmap_hash
    hashes["figures/v0302_regime_lines.png"] = regime_hash
    write_json(OUT / "asset_hashes.json", hashes)

    # Canonical docs manifest
    canonical_path = Path(
        "docs/tutorials_v030/manifests/v0302_single_neuron_parameter_sweep_manifest.json"
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(canonical_path, atlas_manifest)

    canonical_report = Path(
        "docs/tutorials_v030/reports/v0302_single_neuron_parameter_sweep_validation_report.json"
    )
    canonical_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(canonical_report, atlas_manifest["validation_report"])

    print(f"Canonical manifest: {canonical_path}")
    print(f"Canonical report:   {canonical_report}")

    # =========================================================================
    # Summary
    # =========================================================================

    print()
    print("=" * 80)
    print("TUTORIAL EXECUTION SUMMARY")
    print("=" * 80)
    print(f"✓ Sweep: {len(sweep_results)} points ({len(A_VALUES)} a × {len(D_VALUES)} d)")
    print(f"✓ Firing rate range: [{rate_min:.1f}, {rate_max:.1f}] Hz")
    print(f"✓ Baseline (a=0.02, d=8): {baseline_rate:.1f} Hz")
    print(f"✓ Firing rate gate (baseline in 2-25 Hz): {'PASS' if firing_rate_gate_pass else 'FAIL'}")
    print(f"✓ Any sweep point in gate: {'PASS' if any_in_gate else 'FAIL'}")
    print(f"✓ All V_m finite: {all_finite}")
    print(f"✓ Manifest: {manifest_path}")
    print(f"✓ Docs-stable figures: {static_heatmap}, {static_regime}")
    print(f"✓ Plotly available: {plotly_available}")
    print()
    print("Truth status: computational_scaffold, proxy_readout_only")
    print("Physical amplitude claim allowed: False")
    print("=" * 80)

    return atlas_manifest


if __name__ == "__main__":
    main()
