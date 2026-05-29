"""E2E Smoke example for generalized visualization namespace.

Demonstrates creating mock signals, plotting them with the visualizer namespace,
saving figures to disk, and exporting a JSON-safe validation manifest.
"""

import os
import json
import numpy as np
import jaxfne as jtfne

def main():
    print("Running generalized visualization smoke test...")
    np.random.seed(42)
    
    # 1. Config
    dt_ms = 0.1
    duration_ms = 1000.0
    t_steps = int(duration_ms / dt_ms)
    n_neurons = 20
    
    # 2. Mock signals
    spikes = (np.random.rand(t_steps, n_neurons) > 0.98).astype(np.float32)
    V_m = np.random.randn(t_steps, n_neurons).astype(np.float32) * 8 - 65.0
    sources = np.random.randn(t_steps, n_neurons).astype(np.float32) * 0.05
    lfp_data = np.random.randn(t_steps, 8).astype(np.float32) * 0.2
    
    signals = {
        "spikes": spikes,
        "V_m": V_m,
        "sources": sources,
        "lfp_proxy": lfp_data,
        "csd_proxy": lfp_data * -1.5,
        "time_ms": np.arange(t_steps) * dt_ms
    }
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # 3. Render and save visualizations
    print("Generating raster plot...")
    fig_raster = jtfne.vis.raster(signals, dt_ms=dt_ms)
    fig_raster.savefig("outputs/raster_smoke.png")
    
    print("Generating Vm plot...")
    fig_vm = jtfne.vis.vm(signals, dt_ms=dt_ms)
    fig_vm.savefig("outputs/vm_smoke.png")
    
    print("Generating source plot...")
    fig_source = jtfne.vis.source(signals, dt_ms=dt_ms)
    fig_source.savefig("outputs/source_smoke.png")
    
    print("Generating LFP plot...")
    fig_lfp = jtfne.vis.lfp(signals, dt_ms=dt_ms)
    fig_lfp.savefig("outputs/lfp_smoke.png")
    
    print("Generating CSD plot...")
    fig_csd = jtfne.vis.csd(signals, dt_ms=dt_ms)
    fig_csd.savefig("outputs/csd_smoke.png")
    
    print("Generating PSD plot...")
    fig_psd = jtfne.vis.psd(signals, dt_ms=dt_ms)
    fig_psd.savefig("outputs/psd_smoke.png")
    
    print("Generating Spectrogram plot...")
    fig_spec = jtfne.vis.spectrogram(signals, dt_ms=dt_ms)
    fig_spec.savefig("outputs/spectrogram_smoke.png")
    
    print("Generating Summary plot...")
    fig_summary = jtfne.vis.summary(signals, dt_ms=dt_ms)
    fig_summary.savefig("outputs/summary_smoke.png")
    
    # Clean up plots
    import matplotlib.pyplot as plt
    plt.close("all")
    
    # 4. Generate JSON-safe manifest
    manifest = {
        "manifest_version": "1.0",
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "t_steps": t_steps,
        "n_neurons": n_neurons,
        "mean_firing_rate_hz": float(np.mean(spikes) * (1000.0 / dt_ms)),
        "artifacts": [
            "outputs/raster_smoke.png",
            "outputs/vm_smoke.png",
            "outputs/source_smoke.png",
            "outputs/lfp_smoke.png",
            "outputs/csd_smoke.png",
            "outputs/psd_smoke.png",
            "outputs/spectrogram_smoke.png",
            "outputs/summary_smoke.png"
        ],
        "metadata": {
            "operator_status": "simulated_proxy",
            "units_or_status": "proxy_units_or_declared",
            "physical_amplitude_claim_allowed": False
        }
    }
    
    with open("outputs/vis_smoke_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Visualization smoke manifest exported successfully!")

if __name__ == "__main__":
    main()
