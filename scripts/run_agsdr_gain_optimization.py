#!/usr/bin/env python3
"""Structured AGSDR gain optimization script.

Optimizes the inter-area gain matrix G[src_area, dst_area] to drive area-wise
firing rates toward a target rate of 7.5 Hz.
"""

import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne

def run_simulation_with_gain(G, duration_ms=500.0):
    cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball(
        runtime_mode="full",
        seed=0,
        duration_ms=duration_ms,
        dt_ms=0.1,
    )
    model = cfg.construct()
    W_base = np.array(model.model_state["weights"])
    
    # Apply G
    W_tuned = np.copy(W_base)
    for src in range(5):
        for dst in range(5):
            if src != dst:
                W_tuned[dst*100:(dst+1)*100, src*100:(src+1)*100] = (
                    W_base[dst*100:(dst+1)*100, src*100:(src+1)*100] * G[src, dst]
                )
                
    model.model_state["weights"] = jnp.array(W_tuned)
    paradigm = cfg.make_paradigm()
    gate = paradigm.make_fixation_gate()
    backup = model.initialize_backup(paradigm=paradigm, history_ms=100.0)
    
    episode = model.run_task(paradigm=paradigm, gate=gate, backup=backup, runtime_mode="full")
    episode = episode.probe(readouts=("spk",))
    spk = np.array(episode.signals.get("spk"))
    
    # Compute rates per area
    duration_sec = duration_ms / 1000.0
    area_rates = []
    for i in range(5):
        rates = np.sum(spk[:, i*100:(i+1)*100]) / (100.0 * duration_sec)
        area_rates.append(float(rates))
        
    return area_rates, spk

def main():
    print("Starting structured gain optimization...")
    target_rate = 7.5
    
    # Initial G (all 1.0)
    G_init = np.ones((5, 5))
    rates_init, spk_init = run_simulation_with_gain(G_init)
    err_init = np.mean((np.array(rates_init) - target_rate) ** 2)
    print(f"Initial rates: {rates_init}")
    print(f"Initial loss: {err_init:.4f}")
    
    # We optimize only the active inter-area projection gains:
    # (0, 2): V1a -> V4
    # (1, 2): V1b -> V4
    # (2, 3): V4 -> MT
    # (3, 4): MT -> PFC
    active_edges = [(0, 2), (1, 2), (2, 3), (3, 4)]
    
    # Simple coordinate search/perturbation to optimize
    best_G = np.copy(G_init)
    best_loss = err_init
    best_rates = rates_init
    
    # We will try 8 candidates with random adjustments to find a better gain matrix
    rng = np.random.default_rng(42)
    for step in range(8):
        G_candidate = np.copy(best_G)
        for (src, dst) in active_edges:
            # Perturb active edge gains slightly
            G_candidate[src, dst] = np.clip(G_candidate[src, dst] + rng.normal(0, 0.25), 0.1, 4.0)
            
        rates, _ = run_simulation_with_gain(G_candidate)
        loss = np.mean((np.array(rates) - target_rate) ** 2)
        print(f"Candidate {step}: rates={rates}, loss={loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_G = G_candidate
            best_rates = rates
            
    print(f"\nOptimized rates: {best_rates}")
    print(f"Optimized loss: {best_loss:.4f}")
    
    # Create final report
    out_dir = Path("outputs/v0333_final_patch")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "mode": "structured_gain_optimization",
        "gain_shape": list(best_G.shape),
        "initial_gain": G_init.tolist(),
        "optimized_gain": best_G.tolist(),
        "gain_delta": (best_G - G_init).tolist(),
        "nonuniform_gain_count": int(np.sum(np.abs(best_G - 1.0) > 1e-5)),
        "objective_before": float(err_init),
        "objective_after": float(best_loss),
        "target_rate_error_before": float(np.mean(np.abs(np.array(rates_init) - target_rate))),
        "target_rate_error_after": float(np.mean(np.abs(np.array(best_rates) - target_rate))),
        "finite_outputs": bool(np.all(np.isfinite(best_G))),
        "bounded_gains": bool(np.all(best_G >= 0.0) and np.all(best_G <= 5.0)),
        "interpretation": "Inter-area synaptic pathway gains G[src, dst] tuned using Coordinate Stochastic search to match target mean rates of 7.5 Hz."
    }
    
    with open(out_dir / "agsdr_gain_report.json", "w") as f:
        json.dump(report, f, allow_nan=False, indent=2)
        
    print(f"Saved AGSDR report to: {out_dir / 'agsdr_gain_report.json'}")

    # Plot the 5x5 structured gain matrix heatmap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(best_G.T, cmap="viridis", vmin=0.0, vmax=3.0)
    cbar = fig.colorbar(im, ax=ax, label="Synaptic Gain Scale")
    
    # Label areas
    areas = ["V1a", "V1b", "V4", "MT", "PFC"]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(areas)
    ax.set_yticklabels(areas)
    ax.set_xlabel("Presynaptic Source Area")
    ax.set_ylabel("Postsynaptic Destination Area")
    ax.set_title("Optimized Structured Gain Matrix G.T[dst, src]")
    
    # Annotate non-diagonal active gain values
    for i in range(5):
        for j in range(5):
            if i != j:
                val = best_G.T[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 1.5 else "black")
                
    plt.tight_layout()
    fig_dir = Path("outputs/v0333_final_patch/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "agsdr_rate_tuning.png", dpi=120)
    plt.close()
    print(f"Saved optimized structured gain figure to: {fig_dir / 'agsdr_rate_tuning.png'}")

if __name__ == "__main__":
    main()
