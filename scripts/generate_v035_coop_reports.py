#!/usr/bin/env python3
"""Simulation sweep, stability optimization, and report generation for v0.3.35 COOP paradigm gate.

Uses only public package APIs.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxfne as jtfne

def calculate_stability_score(spikes, target_rate=10.0, dt_ms=0.5):
    """Calculate the stability score based on the RMSE of firing rates relative to a target rate."""
    n_steps, n_neurons = spikes.shape
    total_time_sec = (n_steps * dt_ms) / 1000.0
    rates = np.sum(spikes, axis=0) / total_time_sec
    rmse = np.sqrt(np.mean((rates - target_rate)**2))
    # Score is bounded between 0 and 100
    score = max(0.0, 100.0 - 5.0 * rmse)
    return float(score), float(np.mean(rates))

def main():
    print("=== STARTING v0.3.35 COOP PARADIGM STABILITY REPORT ===")
    out_dir = Path("outputs/v035_coop")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Build and configure 10-neuron dummy model
    cfg_dummy = (
        jtfne.Configuration()
        .network(name="dummy_10", kind="cortical_column", n=10, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model_dummy = jtfne.construct(cfg_dummy)
    
    # 2. Build and configure duplicated/connected two-column model
    cfg_connected = jtfne.build_multi_area_columns(["ColA", "ColB"], n_per_area=10, connectivity_mode="all_to_all")
    # Add probes
    cfg_connected = cfg_connected.set_probes(["spikes", "V_m", "CSD", "LFP"])
    model_connected = jtfne.construct(cfg_connected)
    
    # 3. Create COOP paradigm sequence (2000 ms, 6 Hz, 20% omission probability)
    duration_ms = 2000.0
    dt_ms = 0.5
    target_idx = [0, 1, 2, 3] # Explicit target indices
    
    p = jtfne.coop_omission_oddball_paradigm(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        freq_hz=6.0,
        omission_prob=0.2,
        standard_amplitude=8.0,
        pre_stimulus_buffer_ms=200.0,
        target_indices=target_idx,
        seed=42
    )
    
    # 4. Simulation settings
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=42)
    
    # Run baseline on dummy model
    print("Running baseline on 10-neuron dummy model (suboptimal drive_scale=0.5)...")
    suboptimal_model = jtfne.with_emitter_parameters(model_dummy, drive_scale=0.5)
    signals_dummy_base = suboptimal_model.simulate(sim, paradigm=p)
    readout_dummy_base = model_dummy.probe(signals_dummy_base, modes=["spikes", "V_m"])
    spikes_dummy_base = np.array(readout_dummy_base["spikes"])
    score_dummy_base, rate_dummy_base = calculate_stability_score(spikes_dummy_base, target_rate=10.0, dt_ms=dt_ms)
    print(f"Dummy model baseline stability: {score_dummy_base:.2f} (mean rate: {rate_dummy_base:.2f} Hz)")
    
    # PID parameter tuning loop to adjust drive_scale on dummy model to improve stability
    print("Running PID-style control adaptation loop on dummy model...")
    current_drive_scale = 0.5
    target_rate = 10.0
    K_p = 0.08
    best_model = suboptimal_model
    best_score = score_dummy_base
    best_rate = rate_dummy_base
    best_drive_scale = current_drive_scale
    
    tuning_steps = []
    
    for step in range(4):
        # Update model parameters
        adapted_model = jtfne.with_emitter_parameters(model_dummy, drive_scale=float(current_drive_scale))
        signals_dummy = adapted_model.simulate(sim, paradigm=p)
        readout_dummy = adapted_model.probe(signals_dummy, modes=["spikes", "V_m"])
        spikes_dummy = np.array(readout_dummy["spikes"])
        score_dummy, rate_dummy = calculate_stability_score(spikes_dummy, target_rate=target_rate, dt_ms=dt_ms)
        
        tuning_steps.append({
            "step": step,
            "drive_scale": float(current_drive_scale),
            "stability_score": score_dummy,
            "mean_rate_hz": rate_dummy
        })
        
        print(f"Step {step}: drive_scale={current_drive_scale:.3f} -> Stability: {score_dummy:.2f}, Mean Rate: {rate_dummy:.2f} Hz")
        
        if score_dummy > best_score:
            best_score = score_dummy
            best_model = adapted_model
            best_rate = rate_dummy
            best_drive_scale = current_drive_scale
            
        # PID update step
        error = target_rate - rate_dummy
        current_drive_scale = max(0.1, current_drive_scale + K_p * error)
        
    print(f"Tuning finished. Best drive_scale: {best_drive_scale:.3f}, Stability: {best_score:.2f}")
    
    # Simulate adapted model for plotting
    signals_dummy_final = best_model.simulate(sim, paradigm=p)
    readout_dummy_final = best_model.probe(signals_dummy_final, modes=["spikes", "V_m", "CSD", "LFP"])
    
    # 5. Run COOP paradigm on Connected Model
    print("Running COOP paradigm on duplicated/connected two-column model...")
    signals_conn = model_connected.simulate(sim, paradigm=p)
    readout_conn = model_connected.probe(signals_conn, modes=["spikes", "V_m", "CSD", "LFP"])
    spikes_conn = np.array(readout_conn["spikes"])
    score_conn, rate_conn = calculate_stability_score(spikes_conn, target_rate=10.0, dt_ms=dt_ms)
    print(f"Connected model stability: {score_conn:.2f} (mean rate: {rate_conn:.2f} Hz)")
    
    # 6. Generate figures
    time_points = np.arange(spikes_dummy_base.shape[0]) * dt_ms / 1000.0
    
    # Fig 1: Raster Comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].imshow(spikes_dummy_base.T, aspect="auto", cmap="binary", origin="lower", extent=[0, duration_ms/1000.0, 0, 10])
    axes[0].set_title("10-Neuron Dummy Model Raster (Baseline)")
    axes[0].set_ylabel("Neuron ID")
    
    axes[1].imshow(spikes_conn.T, aspect="auto", cmap="binary", origin="lower", extent=[0, duration_ms/1000.0, 0, 20])
    axes[1].set_title("Connected Model Raster")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Neuron ID")
    plt.tight_layout()
    plt.savefig(fig_dir / "raster.png", dpi=100)
    plt.close()
    
    # Fig 2: Stability score comparison / adaptation steps
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [s["step"] for s in tuning_steps]
    scores = [s["stability_score"] for s in tuning_steps]
    ax.plot(steps, scores, "o-", color="purple", label="Stability Score")
    ax.set_xlabel("Tuning Step")
    ax.set_ylabel("Stability Score")
    ax.set_title("PID-style Stability Optimization Progress")
    ax.grid(True)
    plt.savefig(fig_dir / "stability_adaptation.png", dpi=100)
    plt.close()
    
    # Write JSON report
    report = {
        "gate_version": "v0.3.35",
        "paradigm_name": "coop_omission_oddball",
        "n_neurons_dummy": 10,
        "n_neurons_connected": 20,
        "dummy_baseline": {
            "stability_score": score_dummy_base,
            "mean_rate_hz": rate_dummy_base
        },
        "dummy_optimized": {
            "stability_score": best_score,
            "mean_rate_hz": best_rate,
            "drive_scale": float(best_drive_scale)
        },
        "connected_model": {
            "stability_score": score_conn,
            "mean_rate_hz": rate_conn
        },
        "adaptation_steps": tuning_steps,
        "finite_checks": {
            "dummy_voltages_finite": bool(np.all(np.isfinite(signals_dummy_final.V_m))),
            "connected_voltages_finite": bool(np.all(np.isfinite(signals_conn.V_m)))
        },
        "objective_improved": bool(best_score > score_dummy_base)
    }
    
    with open(out_dir / "agsdr_gain_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Report saved to {out_dir / 'agsdr_gain_report.json'}")
    
    # Also copy figures and report to the artifacts directory as required
    artifact_path = Path("/Users/hamednejat/.gemini/antigravity/brain/761087d9-8c96-406d-b914-b61190a913a9")
    artifact_path.mkdir(parents=True, exist_ok=True)
    
    # Copy JSON report
    import shutil
    shutil.copy(out_dir / "agsdr_gain_report.json", artifact_path / "agsdr_gain_report.json")
    
    # Copy figures
    art_fig_dir = artifact_path / "figures"
    art_fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_dir / "raster.png", art_fig_dir / "raster.png")
    shutil.copy(fig_dir / "stability_adaptation.png", art_fig_dir / "stability_adaptation.png")
    
    print("=== v0.3.35 COOP PARADIGM STABILITY REPORT COMPLETED ===")

if __name__ == "__main__":
    main()
