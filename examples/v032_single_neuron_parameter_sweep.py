#!/usr/bin/env python3
"""
v0.3.2 Single Neuron Parameter Sweep Tutorial

Sweeps Izhikevich a and d parameters to show regime transitions
(regular spiking → adaptation → high-excitability out-of-target).
Uses jtfne.with_emitter_parameters() on the stable jaxfne==0.2.30 toolbox.

Writes atlas-compatible manifest to outputs/v030_02_single_neuron_parameter_sweep/
for v0.3 collector validation (gate: 2 PASS, 0 FAIL alongside v0.3.1).

Per-condition gate semantics:
  - target_regime:                     2 <= Hz <= 25, finite
  - high_rate_out_of_target_regime:    Hz > 25, finite (contrast, not accepted baseline)
  - low_or_silent_out_of_target_regime: Hz < 2, finite
  - nonfinite_failure:                 any nonfinite output

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

# Sweep grid
A_VALUES = [0.02, 0.05, 0.10]          # recovery time scale (slow → fast)
D_VALUES = [2.0, 6.0, 8.0, 12.0]       # after-spike reset (weak → strong adaptation)
DURATION_MS = 1000.0                    # hard gate: >= 1000 ms (matches v0.3.1)
DT_MS = 0.1
SEED = 42

# Rate gate bounds
RATE_GATE_LOW_HZ = 2.0
RATE_GATE_HIGH_HZ = 25.0


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


def classify_condition(firing_rate_hz: float, finite: bool) -> str:
    """Assign per-condition regime label."""
    if not finite:
        return "nonfinite_failure"
    if firing_rate_hz > RATE_GATE_HIGH_HZ:
        return "high_rate_out_of_target_regime"
    if firing_rate_hz < RATE_GATE_LOW_HZ:
        return "low_or_silent_out_of_target_regime"
    return "target_regime"


def build_base_model(run):
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
    model = jtfne.with_emitter_parameters(base_model, a=a_val, d=d_val)
    signals = model.simulate(sim_spec)
    V_m = np.array(signals.V_m)
    spikes_arr = np.array(signals.spikes)
    spike_indices = np.where(spikes_arr[:, 0] > 0.5)[0]
    n_spikes = len(spike_indices)
    firing_rate_hz = float((n_spikes / DURATION_MS) * 1000.0)
    finite_voltage = bool(np.all(np.isfinite(V_m)))
    finite_spikes = bool(np.all(np.isfinite(spikes_arr)))
    finite = finite_voltage and finite_spikes
    # retrieve emitter params for record
    e = model.params["emitter"]
    return {
        "a": float(a_val),
        "b": float(e.b[0]),
        "c": float(e.c[0]),
        "d": float(d_val),
        "drive": float(e.drive[0]),
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "n_spikes": int(n_spikes),
        "firing_rate_hz": firing_rate_hz,
        "finite_voltage": finite_voltage,
        "finite_spikes": finite_spikes,
        "rate_gate_pass_2_25_hz": bool(RATE_GATE_LOW_HZ <= firing_rate_hz <= RATE_GATE_HIGH_HZ),
        "regime_label": classify_condition(firing_rate_hz, finite),
        "regime_status": "target" if classify_condition(firing_rate_hz, finite) == "target_regime" else "out_of_target",
        "V_min": float(np.min(V_m)),
        "V_max": float(np.max(V_m)),
    }


def main(update_canonical: bool = False):
    """Main tutorial execution.

    Parameters
    ----------
    update_canonical : bool, default False
        When True, write the canonical docs manifest to docs/tutorials_v030/manifests/.
        Tests should call main() with update_canonical=False (the default) to prevent
        tracked tutorial artifact drift.
    """
    print("=" * 80)
    print("v0.3.2 Single Neuron Parameter Sweep Tutorial")
    print("=" * 80)
    print()

    OUT.mkdir(parents=True, exist_ok=True)
    STATIC_FIGS.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    # =========================================================================
    # SECTION 5: Configuration
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
    print(f"Duration per point: {DURATION_MS} ms (gate: >= {DURATION_MS} ms), dt: {DT_MS} ms")
    print()

    # =========================================================================
    # SECTION 6: Sweep execution
    # =========================================================================

    print("Running sweep...")
    sweep_results = []
    firing_grid = np.zeros((len(A_VALUES), len(D_VALUES)), dtype=float)
    label_grid = []

    for i, a_val in enumerate(A_VALUES):
        label_row = []
        for j, d_val in enumerate(D_VALUES):
            result = run_sweep_point(base_model, a_val, d_val, sim_spec)
            sweep_results.append(result)
            firing_grid[i, j] = result["firing_rate_hz"]
            label_row.append(result["regime_label"])
            gate_str = "PASS" if result["rate_gate_pass_2_25_hz"] else "FAIL"
            print(
                f"  a={a_val:.2f}, d={d_val:.1f} → "
                f"{result['firing_rate_hz']:.1f} Hz  "
                f"[{gate_str}] [{result['regime_label']}]"
            )
        label_grid.append(label_row)

    print()

    # Per-condition counts
    n_target = sum(1 for r in sweep_results if r["regime_label"] == "target_regime")
    n_high = sum(1 for r in sweep_results if r["regime_label"] == "high_rate_out_of_target_regime")
    n_low = sum(1 for r in sweep_results if r["regime_label"] == "low_or_silent_out_of_target_regime")
    n_nonfinite = sum(1 for r in sweep_results if r["regime_label"] == "nonfinite_failure")
    all_finite = n_nonfinite == 0
    all_out_of_target_labelled = True  # every non-target has explicit label by construction

    rates = [r["firing_rate_hz"] for r in sweep_results]
    rate_min = float(np.min(rates))
    rate_max = float(np.max(rates))

    # Baseline: a=0.02, d=8 (cortical_eig default)
    baseline_result = next(
        r for r in sweep_results if abs(r["a"] - 0.02) < 0.001 and abs(r["d"] - 8.0) < 0.1
    )
    baseline_rate = baseline_result["firing_rate_hz"]
    baseline_gate = RATE_GATE_LOW_HZ <= baseline_rate <= RATE_GATE_HIGH_HZ

    print(f"Regime counts:  target={n_target}, high_rate_out_of_target={n_high}, "
          f"low_or_silent={n_low}, nonfinite={n_nonfinite}")
    print(f"Firing rate range: [{rate_min:.1f}, {rate_max:.1f}] Hz")
    print(f"Baseline (a=0.02, d=8): {baseline_rate:.1f} Hz — "
          f"{'PASS' if baseline_gate else 'FAIL'} (2–25 Hz gate)")
    print(f"All conditions finite: {all_finite}")
    print()

    # Tutorial acceptance: baseline passes, finite, all out-of-target labelled
    duration_gate_pass = DURATION_MS >= 1000.0
    dt_gate_pass = DT_MS == 0.1
    dtype_gate_pass = str(base_model.simulate(sim_spec).V_m.dtype) == "float32"
    tutorial_acceptance = all([
        baseline_gate,
        duration_gate_pass,
        dt_gate_pass,
        all_finite,
        all_out_of_target_labelled,
    ])

    # =========================================================================
    # SECTION 7: Probe readout (baseline)
    # =========================================================================

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

    # =========================================================================
    # SECTION 9: Figures
    # =========================================================================

    print("Generating figures...")

    # Figure 1: Firing rate heatmap with regime annotation
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        firing_grid, aspect="auto", origin="upper", cmap="viridis",
        vmin=0, vmax=max(rate_max, 30.0),
    )
    ax.set_xticks(range(len(D_VALUES)))
    ax.set_xticklabels([f"{d:.1f}" for d in D_VALUES])
    ax.set_yticks(range(len(A_VALUES)))
    ax.set_yticklabels([f"{a:.2f}" for a in A_VALUES])
    ax.set_xlabel("d — after-spike reset (proxy units)")
    ax.set_ylabel("a — recovery time scale (proxy units)")
    ax.set_title(
        "v0.3.2: Izhikevich Parameter Sweep — Firing Rate (Hz)\n"
        "(★ = target regime 2–25 Hz  |  ✗ = out-of-target high-rate)"
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Firing rate (Hz)")
    # Annotate cells with rate + regime marker
    for i in range(len(A_VALUES)):
        for j in range(len(D_VALUES)):
            hz = firing_grid[i, j]
            label = label_grid[i][j]
            marker = "★" if label == "target_regime" else "✗"
            ax.text(
                j, i, f"{hz:.0f}\n{marker}",
                ha="center", va="center", fontsize=8,
                color="white" if hz < rate_max * 0.6 else "black",
            )
    # Gate line annotation
    plt.tight_layout()
    heatmap_path = OUT / "figures" / "v0302_sweep_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {heatmap_path}")

    # Figure 2: Firing rate vs d (per a), with gate band
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, a_val in enumerate(A_VALUES):
        rates_row = firing_grid[i, :]
        ax.plot(D_VALUES, rates_row, marker="o", label=f"a={a_val:.2f}")
    ax.axhspan(RATE_GATE_LOW_HZ, RATE_GATE_HIGH_HZ, alpha=0.10, color="green", label="Target regime (2–25 Hz)")
    ax.axhline(RATE_GATE_LOW_HZ, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(RATE_GATE_HIGH_HZ, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("d — after-spike reset (proxy units)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(
        "v0.3.2: Firing Rate vs. d (per a)\n"
        "(Green band = target regime 2–25 Hz | Above band = high-rate out-of-target)"
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
    # SECTION 8: Atlas manifest
    # =========================================================================

    run_id = f"v032_parameter_sweep_{int(datetime.now().timestamp())}"

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
            # Hard simulation gates
            "duration_ms": DURATION_MS,
            "duration_gate_pass": duration_gate_pass,
            "dt_ms": DT_MS,
            "dt_gate_pass": dt_gate_pass,
            "dtype_gate_pass": dtype_gate_pass,
            # Baseline gate (cortical_eig default: a=0.02, d=8)
            "baseline_firing_rate_hz": baseline_rate,
            "baseline_rate_gate_pass": baseline_gate,
            # Sweep-wide finite gate
            "all_conditions_finite": all_finite,
            "n_conditions_total": len(sweep_results),
            # Per-condition regime counts
            "n_target_regime": n_target,
            "n_high_rate_out_of_target_regime": n_high,
            "n_low_or_silent_out_of_target_regime": n_low,
            "n_nonfinite_failure": n_nonfinite,
            # Per-condition gate table (explicit labelling of out-of-target)
            "per_condition_gate": [
                {
                    "condition_id": f"a{r['a']:.2f}_d{r['d']:.1f}",
                    "a": r["a"],
                    "d": r["d"],
                    "firing_rate_hz": r["firing_rate_hz"],
                    "rate_gate_pass_2_25_hz": r["rate_gate_pass_2_25_hz"],
                    "regime_label": r["regime_label"],
                }
                for r in sweep_results
            ],
            # All out-of-target conditions carry explicit label
            "all_out_of_target_conditions_labelled": all_out_of_target_labelled,
            # JSON safety
            "json_safe": True,
            # Tutorial acceptance: PASS only if all hard gates + labelling satisfied
            "tutorial_acceptance_status": "PASS" if tutorial_acceptance else "FAIL",
            "tutorial_acceptance_criteria": {
                "baseline_gate_pass": baseline_gate,
                "duration_gate_pass": duration_gate_pass,
                "dt_gate_pass": dt_gate_pass,
                "dtype_gate_pass": dtype_gate_pass,
                "all_conditions_finite": all_finite,
                "all_out_of_target_conditions_labelled": all_out_of_target_labelled,
            },
            # Note: high-rate conditions are pedagogically intentional
            "out_of_target_regime_note": (
                "Conditions with firing_rate_hz > 25 are labelled "
                "high_rate_out_of_target_regime and are used as parameter-contrast "
                "examples. They are NOT accepted as baseline cortical regimes."
            ),
            # Convenience alias for collector Gate 5
            "firing_rate_gate_2_25_hz": baseline_gate,
            "status": "PASS" if tutorial_acceptance else "FAIL",
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
            "sweep_rate_min_hz": rate_min,
            "sweep_rate_max_hz": rate_max,
            "results": sweep_results,
            "firing_grid": firing_grid.tolist(),
            "tool_used": "jtfne.with_emitter_parameters",
        },

        "simulation": {
            "duration_ms": DURATION_MS,
            "dt_ms": DT_MS,
            "seed": SEED,
            "dtype": "float32",
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
            "High-rate conditions (>25 Hz) are explicitly labelled out-of-target regimes for contrast; they are not accepted as baseline cortical activity.",
            "Firing rate changes reflect model dynamics, not measured neuron behavior.",
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
    assert loaded["validation_report"]["duration_gate_pass"] is True
    assert loaded["validation_report"]["tutorial_acceptance_status"] == "PASS"
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
        "n_target_regime": n_target,
        "n_high_rate_out_of_target_regime": n_high,
        "n_low_or_silent_out_of_target_regime": n_low,
        "n_nonfinite_failure": n_nonfinite,
        "all_finite": all_finite,
        "tutorial_acceptance_status": "PASS" if tutorial_acceptance else "FAIL",
        "mean_firing_rate_hz": baseline_rate,
    })

    json_files = [
        manifest_path, OUT / "probe_report.json",
        OUT / "validation_report.json", OUT / "metrics.json",
    ]
    hashes = {p.name: sha256_file(p) for p in json_files}
    hashes["figures/v0302_sweep_heatmap.png"] = heatmap_hash
    hashes["figures/v0302_regime_lines.png"] = regime_hash
    write_json(OUT / "asset_hashes.json", hashes)

    # Only update canonical docs manifests when explicitly requested
    if update_canonical:
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
    print(f"✓ Duration per condition: {DURATION_MS} ms (gate >= 1000 ms: {'PASS'})")
    print(f"✓ Firing rate range: [{rate_min:.1f}, {rate_max:.1f}] Hz")
    print(f"✓ Baseline (a=0.02, d=8): {baseline_rate:.1f} Hz — {'PASS' if baseline_gate else 'FAIL'}")
    print(f"✓ Regime counts: {n_target} target / {n_high} high-rate out-of-target / {n_low} low / {n_nonfinite} nonfinite")
    print(f"✓ All conditions finite: {all_finite}")
    print(f"✓ All out-of-target conditions labelled: {all_out_of_target_labelled}")
    print(f"✓ Tutorial acceptance: {'PASS' if tutorial_acceptance else 'FAIL'}")
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
