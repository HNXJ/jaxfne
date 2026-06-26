#!/usr/bin/env python3
"""
Generate PNG tutorial figures from jaxfne simulations.

Generates 12 tutorial figures with proxy-safe titles and visual confirmation.
Target: >= 10 real-data figures (no placeholders).

Usage:
  python scripts/generate_tutorial_figures.py [--output-dir docs/_static/tutorial_figures]
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np

import jaxfne as jtfne


def safe_to_numpy(arr):
    """Convert JAX array to NumPy safely."""
    try:
        return np.asarray(arr)
    except Exception as e:
        print(f"Warning: Failed to convert array: {e}", file=sys.stderr)
        return np.array([])


def build_config():
    """Build cortical column configuration (observed API)."""
    cfg = (
        jtfne.configuration()
        .network(
            name="V1_tutorial",
            kind="cortical_column",
            n=50,
            layers=["L2/3", "L4", "L5", "L6"],
            cell_types={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="declared_proxy",
            gauge="mean_zero",
        )
        .probe(
            name="laminar_probe",
            modes=["spikes", "V_m", "source", "phi_e", "J_e", "CSD", "LFP"],
        )
    )
    cfg = cfg.update_metadata(
        run_status="tutorial_scaffold",
        model_status="computational_scaffold",
    )
    return cfg


def simulate():
    """Run a deterministic simulation and return signals and manifest."""
    cfg = build_config()
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=500.0, dt_ms=0.1, plasticity=0.0, seed=0)
    signals = model.simulate(sim)
    manifest = model.manifest(signals)
    return signals, manifest


def gen_spike_raster(signals, output_dir):
    """Spike raster figure."""
    spikes = safe_to_numpy(signals.spikes)
    if spikes.size == 0:
        return None

    fig = jtfne.vis.tutorial_spike_raster(spikes, title="Spike Raster (Izhikevich Simulation)")

    path = output_dir / "01_spike_raster.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "01_spike_raster.png", "title": "Spike Raster", "type": "behavioral", "uses_real_data": True}


def gen_voltage_traces(signals, output_dir):
    """Membrane voltage traces (subsample units for clarity)."""
    v_m = safe_to_numpy(signals.V_m)
    if v_m.size == 0:
        return None

    fig = jtfne.vis.tutorial_voltage_traces(
        v_m, n_display=6, title="Membrane Voltage Traces (Izhikevich Native)"
    )

    path = output_dir / "02_voltage_traces.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "02_voltage_traces.png", "title": "Voltage Traces", "type": "state", "uses_real_data": True}


def gen_source_proxy_heatmap(signals, output_dir):
    """Source proxy heatmap."""
    sources = safe_to_numpy(signals.sources)
    if sources.size == 0:
        return None

    fig = jtfne.vis.tutorial_matrix_heatmap(
        sources,
        cmap="RdBu_r",
        title="Source Proxy (Synaptic Current Model)",
        colorbar_label="Proxy amplitude (nA)",
        figsize=(12, 5),
    )

    path = output_dir / "03_source_proxy_heatmap.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "03_source_proxy_heatmap.png", "title": "Source Proxy Heatmap", "type": "field_source", "uses_real_data": True}


def gen_lfp_proxy_trace(signals, output_dir):
    """LFP proxy trace (mean across contacts)."""
    try:
        lfp_proxy = safe_to_numpy(signals.field.lfp_proxy)
        if lfp_proxy.size == 0:
            return None
    except:
        return None

    time_steps, n_contacts = lfp_proxy.shape
    lfp_mean = np.mean(lfp_proxy, axis=1)

    fig = jtfne.vis.tutorial_lfp_proxy_trace(lfp_mean, title="LFP Proxy (Averaged Across Contacts)")

    path = output_dir / "04_lfp_proxy_trace.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "04_lfp_proxy_trace.png", "title": "LFP Proxy Trace", "type": "readout_scalar", "uses_real_data": True}


def gen_csd_proxy_heatmap(signals, output_dir):
    """CSD proxy heatmap (spatial proxy)."""
    try:
        csd_proxy = safe_to_numpy(signals.field.csd_proxy)
        if csd_proxy.size == 0:
            return None
    except:
        return None

    fig = jtfne.vis.tutorial_matrix_heatmap(
        csd_proxy,
        cmap="seismic",
        xlabel="Time step",
        ylabel="Contact index",
        title="CSD Proxy (Spatial Derivative Proxy)",
        colorbar_label="Proxy amplitude",
        figsize=(12, 4),
    )

    path = output_dir / "05_csd_proxy_heatmap.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "05_csd_proxy_heatmap.png", "title": "CSD Proxy Heatmap", "type": "readout_spatial", "uses_real_data": True}


def gen_phi_e_proxy_heatmap(signals, output_dir):
    """Extracellular potential proxy heatmap."""
    try:
        phi_e_proxy = safe_to_numpy(signals.field.phi_e_proxy)
        if phi_e_proxy.size == 0:
            return None
    except:
        return None

    fig = jtfne.vis.tutorial_matrix_heatmap(
        phi_e_proxy,
        cmap="viridis",
        xlabel="Time step",
        ylabel="Contact index",
        title="Extracellular Potential Proxy (φ_e Proxy)",
        colorbar_label="Proxy amplitude (mV)",
        figsize=(12, 4),
    )

    path = output_dir / "06_phi_e_proxy_heatmap.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "06_phi_e_proxy_heatmap.png", "title": "φ_e Proxy Heatmap", "type": "field_potential", "uses_real_data": True}


def gen_source_proxy_spatial(signals, output_dir):
    """Source proxy in space (contact-averaged source)."""
    try:
        source_proxy = safe_to_numpy(signals.field.source_proxy)
        if source_proxy.size == 0:
            return None
    except:
        return None

    fig = jtfne.vis.tutorial_matrix_heatmap(
        source_proxy,
        cmap="RdBu_r",
        xlabel="Time step",
        ylabel="Contact index",
        title="Source Proxy Spatial (Kernel-Weighted Source)",
        colorbar_label="Proxy amplitude",
        figsize=(12, 4),
    )

    path = output_dir / "07_source_proxy_spatial.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "07_source_proxy_spatial.png", "title": "Source Proxy Spatial", "type": "field_source", "uses_real_data": True}


def gen_conservation_diagnostics(manifest, output_dir):
    """Conservation proxy diagnostics bar chart."""
    try:
        diag = manifest.get("conservation_proxy_diagnostics", {})
        if not diag:
            return None
    except:
        return None

    metrics = {
        "L1 norm": float(diag.get("source_norm_L1", 0.0)),
        "L2 norm": float(diag.get("source_norm_L2", 0.0)),
        "Field grad": float(diag.get("field_gradient_proxy_L2", 0.0)),
        "Conserv. res.": abs(float(diag.get("conservation_residual", 0.0))),
    }

    fig = jtfne.vis.tutorial_conservation_bar(metrics, title="Conservation Proxy Diagnostics (Laminar Field)")

    path = output_dir / "08_conservation_diagnostics.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "08_conservation_diagnostics.png", "title": "Conservation Diagnostics", "type": "diagnostics", "uses_real_data": True}


def gen_contact_depths_profile(signals, output_dir):
    """Contact depths (laminar profile axis)."""
    try:
        contact_depths = safe_to_numpy(signals.field.contact_depths)
        if contact_depths.size == 0:
            return None
    except:
        return None

    fig = jtfne.vis.tutorial_contact_depths_barh(contact_depths, title="Laminar Profile (Contact Depths)")

    path = output_dir / "09_laminar_profile_depths.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "09_laminar_profile_depths.png", "title": "Laminar Profile Depths", "type": "geometry", "uses_real_data": True}


def gen_firing_rate_raster(signals, output_dir):
    """Firing rate over time (smoothed spikes)."""
    spikes = safe_to_numpy(signals.spikes)
    if spikes.size == 0:
        return None

    # Smooth spikes over 50-step windows
    window = 50
    n_units = spikes.shape[1]
    firing_rate = []
    for unit_idx in range(n_units):
        rate = np.convolve(spikes[:, unit_idx], np.ones(window) / window, mode="same")
        firing_rate.append(rate)
    firing_rate = np.array(firing_rate)

    fig = jtfne.vis.tutorial_matrix_heatmap(
        firing_rate,
        transpose=True,
        cmap="hot",
        title="Firing Rate Proxy (Smoothed Spike Count)",
        colorbar_label="Smoothed rate",
        figsize=(12, 5),
    )

    path = output_dir / "10_firing_rate_raster.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "10_firing_rate_raster.png", "title": "Firing Rate Proxy", "type": "behavioral", "uses_real_data": True}


def gen_status_summary(manifest, output_dir):
    """Status checks and status status (text-based figure)."""
    gates = [
        ("run_status", manifest.get("run_status", "N/A")),
        ("model_status", manifest.get("model_status", "N/A")),
        ("field_solver_status", manifest.get("field_solver_status", "N/A")),
        ("amplitude_status", manifest.get("amplitude_status", "N/A")),
        ("source_calibration_status", manifest.get("source_calibration_status", "N/A")),
        ("metabolism_status", manifest.get("metabolism_status", "N/A")),
    ]

    text_lines = ["Statement Gates and Status Status", "=" * 50]
    for gate_name, gate_value in gates:
        text_lines.append(f"{gate_name}: {gate_value}")

    text_content = "\n".join(text_lines)
    fig = jtfne.vis.tutorial_status_text_panel(text_content, suptitle="v0.2.27 Statement Gates Summary")

    path = output_dir / "11_status_summary.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "11_status_summary.png", "title": "Statement Gates Summary", "type": "metadata", "uses_real_data": False}


def gen_spectral_summary(signals, output_dir):
    """Spectral summary (FFT-based proxy)."""
    spikes = safe_to_numpy(signals.spikes)
    if spikes.size == 0:
        return None

    # Compute mean power spectrum (log scale)
    spike_mean = np.mean(spikes, axis=1)
    fft = np.fft.fft(spike_mean)
    power = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(len(power))

    # Keep positive frequencies only
    positive_idx = freqs > 0
    freqs_pos = freqs[positive_idx]
    power_pos = power[positive_idx]

    fig = jtfne.vis.tutorial_spectral_summary(
        freqs_pos, power_pos, title="Spectral Summary (Network Activity FFT)"
    )

    path = output_dir / "12_spectral_summary.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  ✓ {path.name}")
    return {"filename": "12_spectral_summary.png", "title": "Spectral Summary", "type": "analysis", "uses_real_data": True}


def main():
    parser = argparse.ArgumentParser(description="Generate tutorial figures for jaxfne v0.2.28")
    parser.add_argument("--output-dir", default="docs/_static/tutorial_figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating tutorial figures to {output_dir}")
    print()

    print("[1/3] Building and simulating model...")
    signals, manifest = simulate()
    print("      ✓ Simulation complete")
    print()

    print("[2/3] Generating figures...")
    figures_metadata = []

    generators = [
        ("spike_raster", gen_spike_raster),
        ("voltage_traces", gen_voltage_traces),
        ("source_proxy_heatmap", gen_source_proxy_heatmap),
        ("lfp_proxy_trace", gen_lfp_proxy_trace),
        ("csd_proxy_heatmap", gen_csd_proxy_heatmap),
        ("phi_e_proxy_heatmap", gen_phi_e_proxy_heatmap),
        ("source_proxy_spatial", gen_source_proxy_spatial),
        ("conservation_diagnostics", gen_conservation_diagnostics),
        ("contact_depths_profile", gen_contact_depths_profile),
        ("firing_rate_raster", gen_firing_rate_raster),
        ("status_checks_summary", gen_status_summary),
        ("spectral_summary", gen_spectral_summary),
    ]

    for figure_name, generator_func in generators:
        try:
            if figure_name in ["conservation_diagnostics", "status_checks_summary"]:
                result = generator_func(manifest, output_dir)
            else:
                result = generator_func(signals, output_dir)

            if result:
                figures_metadata.append(result)
        except Exception as e:
            print(f"  ✗ {figure_name}: {e}")
            continue

    print()
    print("[3/3] Writing manifest...")

    # Count real-data figures
    real_data_count = sum(1 for f in figures_metadata if f.get("uses_real_data", False))

    manifest_dict = {
        "figure_count": len(figures_metadata),
        "real_data_figure_count": real_data_count,
        "min_required": 10,
        "jaxfne_version": jtfne.__version__,
        "run_status": manifest.get("run_status", "tutorial_scaffold"),
        "model_status": manifest.get("model_status", "computational_scaffold"),
        "field_solver_status": manifest.get("field_solver_status", "linear_solver"),
        "amplitude_status": manifest.get("amplitude_status", False),
        "metabolism_status": manifest.get("metabolism_status", False),
        "source_script": "scripts/generate_tutorial_figures.py",
        "visual_confirmation_method": "manual_inspection_and_image_nonblank_check",
        "figures": [
            {
                **fig,
                "path": f"docs/_static/tutorial_figures/{fig['filename']}",
                "visually_confirmed": False,  # To be updated in Phase E
                "visual_status": "pending",
                "readout_status": "simulated_proxy",
            }
            for fig in figures_metadata
        ],
    }

    manifest_path = output_dir / "figure_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)

    print(f"✓ Manifest: {manifest_path}")
    print()
    print(f"Summary: {len(figures_metadata)} figures ({real_data_count} with real data)")
    print(f"Status: {'PASS (>= 10 real data)' if real_data_count >= 10 else 'FAIL (< 10 real data)'}")


if __name__ == "__main__":
    main()
