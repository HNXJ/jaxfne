"""STDP/plasticity adaptation visualization suite.

Moved from ``jaxfne.plasticity`` — keeps the legacy direct-to-disk save
behavior (writes 4 PNGs, returns ``None``) since downstream call sites pass
``fig_dir``/``prefix`` rather than expecting figure objects back.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def plot_stdp_adaptation_suite(
    trajectories: Dict[str, Any],
    W_before: np.ndarray,
    W_after: np.ndarray,
    stimulus: np.ndarray,
    fig_dir: str,
    prefix: str = "",
) -> None:
    """Generate and save the standard STDP adaptation visualization figures.

    All figures are saved with size >= 1000x500.
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)

    _ = trajectories["vm"]  # not plotted here, but required to be present in trajectories
    spk = trajectories["spk"]
    dt_ms = 0.5 * 10  # Accounts for downsampling of 10
    times = np.arange(spk.shape[0]) * dt_ms / 1000.0

    # 1. Raster
    fig, ax = plt.subplots(figsize=(11, 6))
    spk_idx, neuron_idx = np.where(spk > 0.0)
    ax.scatter(times[spk_idx], neuron_idx, s=1, color="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron ID")
    ax.set_title("100-Neuron Spike Raster (STDP Proxy Readout)")
    plt.savefig(os.path.join(fig_dir, f"{prefix}raster.png"), dpi=100)
    plt.close()

    # 2. Population rate
    fig, ax = plt.subplots(figsize=(11, 6))
    bin_size_ms = 100.0
    bin_steps = int(bin_size_ms / dt_ms)
    rates = []
    for i in range(0, spk.shape[0], bin_steps):
        rates.append(np.mean(spk[i:i + bin_steps]) * 1000.0 / dt_ms)
    ax.plot(np.arange(len(rates)) * bin_size_ms / 1000.0, rates, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_title("Population Firing Rate (100 ms bins)")
    plt.savefig(os.path.join(fig_dir, f"{prefix}population_rates.png"), dpi=100)
    plt.close()

    # 3. Stimulus Trace (First 2000 points of downsampled, equivalent to first 10s)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(times[:2000], stimulus[:2000 * 10:10], color="orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drive Current (a.u.)")
    ax.set_title("Triangular Stimulus Drive")
    plt.savefig(os.path.join(fig_dir, f"{prefix}stimulus_trace.png"), dpi=100)
    plt.close()

    # 4. W before/after/delta heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    delta_W = W_after - W_before

    im0 = axes[0].imshow(W_before, aspect="auto", cmap="viridis")
    axes[0].set_title("W Before")
    axes[0].set_xlabel("Presynaptic ID")
    axes[0].set_ylabel("Postsynaptic ID")
    fig.colorbar(im0, ax=axes[0], label="Weight")

    im1 = axes[1].imshow(W_after, aspect="auto", cmap="viridis")
    axes[1].set_title("W After")
    axes[1].set_xlabel("Presynaptic ID")
    axes[1].set_ylabel("Postsynaptic ID")
    fig.colorbar(im1, ax=axes[1], label="Weight")

    im2 = axes[2].imshow(delta_W, aspect="auto", cmap="bwr")
    axes[2].set_title("Delta W")
    axes[2].set_xlabel("Presynaptic ID")
    axes[2].set_ylabel("Postsynaptic ID")
    fig.colorbar(im2, ax=axes[2], label="Weight Delta")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}W_heatmaps.png"), dpi=100)
    plt.close()
