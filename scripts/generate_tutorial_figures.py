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
import matplotlib
matplotlib.use("Agg")  # CPU-safe, non-interactive
import matplotlib.pyplot as plt

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
        truth_mode="truth_safe_unverified",
        claim_level="computational_scaffold",
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

    time_steps, n_units = spikes.shape
    fig, ax = plt.subplots(figsize=(12, 4))

    for unit_idx in range(n_units):
        spike_times = np.where(spikes[:, unit_idx] > 0.5)[0]
        ax.vlines(spike_times, unit_idx - 0.4, unit_idx + 0.4, colors="black", linewidth=0.5)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Unit index")
    ax.set_title("Spike Raster (Izhikevich Simulation)")
    ax.set_ylim(-1, n_units)
    ax.set_xlim(0, time_steps)
    fig.tight_layout()

    path = output_dir / "01_spike_raster.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return {"filename": "01_spike_raster.png", "title": "Spike Raster", "type": "behavioral", "uses_real_data": True}


def gen_voltage_traces(signals, output_dir):
    """Membrane voltage traces (subsample units for clarity)."""
    v_m = safe_to_numpy(signals.V_m)
    if v_m.size == 0:
        return None

    time_steps, n_units = v_m.shape
    n_display = min(6, n_units)  # Show up to 6 units

    fig, axes = plt.subplots(n_display, 1, figsize=(12, 2 * n_display), sharex=True)
    if n_display == 1:
        axes = [axes]

    unit_indices = np.linspace(0, n_units - 1, n_display, dtype=int)
    for i, ax in enumerate(axes):
        unit_idx = unit_indices[i]
        ax.plot(v_m[:, unit_idx], linewidth=0.5, color="steelblue")
        ax.set_ylabel(f"Unit {unit_idx}\n(mV)")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Membrane Voltage Traces (Izhikevich Native)")
    fig.tight_layout()

    path = output_dir / "02_voltage_traces.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return {"filename": "02_voltage_traces.png", "title": "Voltage Traces", "type": "state", "uses_real_data": True}


def gen_source_proxy_heatmap(signals, output_dir):
    """Source proxy heatmap."""
    sources = safe_to_numpy(signals.sources)
    if sources.size == 0:
        return None

    time_steps, n_units = sources.shape
    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(sources.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Unit index")
    ax.set_title("Source Proxy (Synaptic Current Model)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proxy amplitude (nA)")
    fig.tight_layout()

    path = output_dir / "03_source_proxy_heatmap.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(lfp_mean, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Proxy amplitude")
    ax.set_title("LFP Proxy (Averaged Across Contacts)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = output_dir / "04_lfp_proxy_trace.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(csd_proxy.T, aspect="auto", cmap="seismic", interpolation="nearest")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Contact index")
    ax.set_title("CSD Proxy (Spatial Derivative Proxy)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proxy amplitude")
    fig.tight_layout()

    path = output_dir / "05_csd_proxy_heatmap.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(phi_e_proxy.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Contact index")
    ax.set_title("Extracellular Potential Proxy (φ_e Proxy)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proxy amplitude (mV)")
    fig.tight_layout()

    path = output_dir / "06_phi_e_proxy_heatmap.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(source_proxy.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Contact index")
    ax.set_title("Source Proxy Spatial (Kernel-Weighted Source)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proxy amplitude")
    fig.tight_layout()

    path = output_dir / "07_source_proxy_spatial.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax.bar(names, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
    ax.set_ylabel("Magnitude")
    ax.set_title("Conservation Proxy Diagnostics (Laminar Field)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "08_conservation_diagnostics.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
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

    n_contacts = len(contact_depths)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.arange(n_contacts), contact_depths, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_ylabel("Contact index")
    ax.set_xlabel("Depth (μm, proxy)")
    ax.set_title("Laminar Profile (Contact Depths)")
    ax.invert_yaxis()
    fig.tight_layout()

    path = output_dir / "09_laminar_profile_depths.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(firing_rate.T, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Unit index")
    ax.set_title("Firing Rate Proxy (Smoothed Spike Count)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Smoothed rate")
    fig.tight_layout()

    path = output_dir / "10_firing_rate_raster.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return {"filename": "10_firing_rate_raster.png", "title": "Firing Rate Proxy", "type": "behavioral", "uses_real_data": True}


def gen_claim_gates_summary(manifest, output_dir):
    """Claim gates and truth status (text-based figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    gates = [
        ("truth_mode", manifest.get("truth_mode", "N/A")),
        ("claim_level", manifest.get("claim_level", "N/A")),
        ("field_solver_status", manifest.get("field_solver_status", "N/A")),
        ("physical_amplitude_claim_allowed", manifest.get("physical_amplitude_claim_allowed", "N/A")),
        ("source_calibration_status", manifest.get("source_calibration_status", "N/A")),
        ("biological_metabolism_claim_allowed", manifest.get("biological_metabolism_claim_allowed", "N/A")),
    ]

    text_lines = ["Claim Gates and Truth Status", "=" * 50]
    for gate_name, gate_value in gates:
        text_lines.append(f"{gate_name}: {gate_value}")

    text_content = "\n".join(text_lines)
    ax.text(0.1, 0.9, text_content, transform=ax.transAxes, fontfamily="monospace",
            fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("v0.2.27 Claim Gates Summary")
    fig.tight_layout()

    path = output_dir / "11_claim_gates_summary.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return {"filename": "11_claim_gates_summary.png", "title": "Claim Gates Summary", "type": "metadata", "uses_real_data": False}


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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freqs_pos, power_pos + 1e-10, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Frequency (normalized)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Spectral Summary (Network Activity FFT)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = output_dir / "12_spectral_summary.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
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
        ("claim_gates_summary", gen_claim_gates_summary),
        ("spectral_summary", gen_spectral_summary),
    ]

    for figure_name, generator_func in generators:
        try:
            if figure_name in ["conservation_diagnostics", "claim_gates_summary"]:
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
        "truth_mode": manifest.get("truth_mode", "truth_safe_unverified"),
        "claim_level": manifest.get("claim_level", "computational_scaffold"),
        "field_solver_status": manifest.get("field_solver_status", "laminar_proxy_no_pde"),
        "physical_amplitude_claim_allowed": manifest.get("physical_amplitude_claim_allowed", False),
        "biological_metabolism_claim_allowed": manifest.get("biological_metabolism_claim_allowed", False),
        "source_script": "scripts/generate_tutorial_figures.py",
        "visual_confirmation_method": "manual_inspection_and_image_nonblank_check",
        "figures": [
            {
                **fig,
                "path": f"docs/_static/tutorial_figures/{fig['filename']}",
                "visually_confirmed": False,  # To be updated in Phase E
                "visual_status": "pending",
                "claim_status": "simulated_proxy",
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
