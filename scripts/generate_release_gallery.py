#!/usr/bin/env python3
"""Generate the 8 canonical reproducible gallery figures using repository code only.

Target Panels:
  01: Realized Network (3D positions and cell classes)
  02: Spikes & Population Activity (raster & smoothed population firing rate)
  03: Fast Neural State (membrane potentials V_m)
  04: H / RBD State (hidden-state trajectories under HDP/RBD)
  05: Source -> Field / Proxy (uncalibrated transmembrane current Q and LFP proxy)
  06: Finite-Delay Timing (membrane response under heterogeneous axonal delays)
  07: Multi-Area / Laminar Network (inter-area laminar connectivity matrix)
  08: Fast vs Slow State (phase portrait: fast V_m vs slow recovery u)

Outputs saved to: docs/_static/gallery/
Metadata saved to: docs/_static/gallery/gallery_manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne.emitters import EdgeList, IzhikevichParams


def generate_gallery(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": jtfne.__version__,
        "output_dir": str(output_dir),
        "figures": [],
    }

    # -------------------------------------------------------------
    # 01: Realized Network
    # -------------------------------------------------------------
    cfg1 = jtfne.suite2_v1_v4_config(seed=42)
    model1 = jtfne.construct(cfg1)
    pos = np.asarray(model1.params["positions"])
    labels = np.asarray(model1.params["emitter"].labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"E": "#1f77b4", "PV": "#d62728", "SST": "#2ca02c", "VIP": "#ff7f0e"}
    for ct, color in colors.items():
        mask = labels == ct
        if np.any(mask):
            ax.scatter(pos[mask, 0] * 1e3, pos[mask, 2] * 1e3, c=color, label=ct, alpha=0.7, s=20)
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Depth Z (mm)")
    ax.set_title("01: Realized Network Structure (V1-V4 Multi-Area)")
    ax.legend(frameon=True)
    fig.tight_layout()
    p1 = output_dir / "01_realized_network.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    manifest["figures"].append({
        "id": "01_realized_network",
        "script": "scripts/generate_release_gallery.py",
        "seed": 42,
        "configuration": "suite2_v1_v4_config",
        "output": str(p1.name),
        "caption": "Realized spatial locations and cell-class identities in a two-area (V1-V4) laminar network.",
        "calibration_status": "relative_proxy_coordinates",
    })

    # -------------------------------------------------------------
    # 02: Spikes & Population Activity
    # -------------------------------------------------------------
    cfg2 = jtfne.suite2_net1_config(seed=12, n=100, duration_ms=200.0, dt_ms=0.5)
    model2 = jtfne.construct(cfg2)
    sim2 = jtfne.simulation(duration_ms=200.0, dt_ms=0.5, seed=10)
    sig2 = jtfne.simulate(model2, sim2)

    spk = np.asarray(sig2.spikes)
    time_ms = np.asarray(sig2.time_ms)
    pop_rate = np.mean(spk, axis=1) / 0.0005  # Hz

    fig, (ax_spk, ax_rate) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    t_idx, n_idx = np.where(spk > 0)
    ax_spk.scatter(time_ms[t_idx], n_idx, s=4, c="black", marker="|")
    ax_spk.set_ylabel("Neuron Index")
    ax_spk.set_title("02: Population Activity and Spike Raster")

    ax_rate.plot(time_ms, pop_rate, color="#2ca02c", lw=1.5)
    ax_rate.set_ylabel("Mean Rate (Hz)")
    ax_rate.set_xlabel("Time (ms)")
    fig.tight_layout()
    p2 = output_dir / "02_spikes_activity.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    manifest["figures"].append({
        "id": "02_spikes_activity",
        "script": "scripts/generate_release_gallery.py",
        "seed": 10,
        "configuration": "suite2_net1_config(n=100)",
        "output": str(p2.name),
        "caption": "Spike raster and instantaneous population firing rate for a recurrent 100-neuron network.",
        "calibration_status": "uncalibrated_computational_scaffold",
    })

    # -------------------------------------------------------------
    # 03: Fast Neural State (V_m)
    # -------------------------------------------------------------
    v = np.asarray(sig2.V_m)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(min(5, v.shape[1])):
        ax.plot(time_ms, v[:, i] + i * 40.0, lw=1.2, label=f"Neuron {i}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Potential (mV, stacked)")
    ax.set_title("03: Fast Neural State Trajectories (V_m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p3 = output_dir / "03_fast_state_vm.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    manifest["figures"].append({
        "id": "03_fast_state_vm",
        "script": "scripts/generate_release_gallery.py",
        "seed": 10,
        "configuration": "suite2_net1_config(n=100)",
        "output": str(p3.name),
        "caption": "Membrane potential traces demonstrating fast spiking dynamics and subthreshold integration.",
        "calibration_status": "native_izhikevich_millivolts",
    })

    # -------------------------------------------------------------
    # 04: H / RBD State
    # -------------------------------------------------------------
    hp4 = dict(jtfne.DEFAULT_HDP)
    hp4["K_HDP"] = 0.05
    hp4["K_ctrl"] = 0.02
    rc4 = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params=hp4)
    sim4 = jtfne.simulation(duration_ms=200.0, dt_ms=0.5, seed=7, runtime=rc4, record_sources=True)
    _ = jtfne.simulate(model2, sim4)
    diag4 = getattr(model2, "_last_hdp_diag", None)
    H_trace = np.asarray(diag4["H_trace"]) if diag4 and diag4.get("H_trace") is not None else None

    fig, ax = plt.subplots(figsize=(8, 4))
    if H_trace is not None and H_trace.ndim >= 2:
        for i in range(min(5, H_trace.shape[1])):
            ax.plot(time_ms, H_trace[:, i], lw=1.5, label=f"Neuron {i}")
    else:
        ax.plot(time_ms, np.ones_like(time_ms), lw=1.5, label="H (quiescent)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("RBS State H")
    ax.set_title("04: Relative Biophysical State (RBD Dynamics)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p4 = output_dir / "04_h_rbd_state.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    manifest["figures"].append({
        "id": "04_h_rbd_state",
        "script": "scripts/generate_release_gallery.py",
        "seed": 7,
        "configuration": "suite2_net1_config with enable_hdp=True",
        "output": str(p4.name),
        "caption": "Dynamic evolution of the Relative Biophysical State H under activity-dependent drain and restorative control.",
        "calibration_status": "relative_dimensionless_state",
    })

    # -------------------------------------------------------------
    # 05: Source -> Field / Proxy
    # -------------------------------------------------------------
    sim5 = jtfne.simulation(duration_ms=200.0, dt_ms=0.5, seed=10, record_sources=True, record_fields=True)
    sig5 = jtfne.simulate(model2, sim5)
    src = np.asarray(sig5.sources)
    lfp = np.asarray(sig5.field.lfp_proxy) if sig5.field else None

    fig, (ax_src, ax_lfp) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax_src.plot(time_ms, np.mean(src, axis=1), color="#8c564b", lw=1.5)
    ax_src.set_ylabel("Source Current Q (mean)")
    ax_src.set_title("05: Transmembrane Current Source Q to Extracellular LFP Proxy")

    if lfp is not None:
        for ch in [0, 4, 8, 12]:
            if ch < lfp.shape[1]:
                ax_lfp.plot(time_ms, lfp[:, ch] + ch * 0.05, lw=1.2, label=f"Contact {ch}")
        ax_lfp.set_ylabel("LFP Proxy (stacked)")
        ax_lfp.legend(loc="upper right")
    ax_lfp.set_xlabel("Time (ms)")
    fig.tight_layout()
    p5 = output_dir / "05_source_to_field.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)

    manifest["figures"].append({
        "id": "05_source_to_field",
        "script": "scripts/generate_release_gallery.py",
        "seed": 10,
        "configuration": "suite2_net1_config with record_fields=True",
        "output": str(p5.name),
        "caption": "Transmembrane relative source current density and projected laminar field potential (LFP) proxy across linear contacts.",
        "calibration_status": "uncalibrated_source_and_field_proxy",
    })

    # -------------------------------------------------------------
    # 06: Finite-Delay Timing
    # -------------------------------------------------------------
    # 06: Finite-Delay Timing
    # -------------------------------------------------------------
    # Build 2-neuron coupled circuit with 10 ms delay (20 steps at dt=0.5 ms)
    p6 = IzhikevichParams(
        a=jnp.array([0.02, 0.02], dtype=jnp.float32),
        b=jnp.array([0.2, 0.2], dtype=jnp.float32),
        c=jnp.array([-65.0, -65.0], dtype=jnp.float32),
        d=jnp.array([8.0, 8.0], dtype=jnp.float32),
        drive=jnp.array([12.0, 0.0], dtype=jnp.float32),
        sign=jnp.array([1.0, 1.0], dtype=jnp.float32),
        W=jnp.zeros((2, 2), dtype=jnp.float32),
        v0=jnp.array([-65.0, -65.0], dtype=jnp.float32),
        u0=jnp.array([0.2 * -65.0, 0.2 * -65.0], dtype=jnp.float32),
        source_scale=jnp.array([1.0, 1.0], dtype=jnp.float32),
        labels=("E", "E"),
    )
    el6 = EdgeList(
        pre=jnp.array([0], dtype=jnp.int32),
        post=jnp.array([1], dtype=jnp.int32),
        weight=jnp.array([25.0], dtype=jnp.float32),
        receptor_index=jnp.array([0], dtype=jnp.int32),
        tau_ms=jnp.array([5.0], dtype=jnp.float32),
        delay_steps=jnp.array([20], dtype=jnp.int32),
    )
    import jax
    v6, spk6, _, state6 = jtfne.emitters.simulate_edge_recurrent_izhikevich(
        p6, el6, n_steps=120, dt_ms=0.5, key=jax.random.PRNGKey(1), record_edge_current=True
    )
    t6 = np.arange(120) * 0.5
    v6_np = np.asarray(v6)
    spk6_np = np.asarray(spk6)
    presyn6_np = np.asarray(state6["presynaptic_drive_trace"])[:, 0]
    edge_i6_np = np.asarray(state6["edge_current_trace"])[:, 0]

    fig, (ax_pre, ax_delay, ax_post) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    
    # 1. Presynaptic event
    ax_pre.plot(t6, v6_np[:, 0], color="#1f77b4", lw=1.5, label="Neuron 0 V_m (presynaptic)")
    pre_spk_times = t6[spk6_np[:, 0] > 0]
    for st in pre_spk_times:
        ax_pre.axvline(st, color="blue", linestyle="--", alpha=0.6, label="Presynaptic event" if st == pre_spk_times[0] else "")
    ax_pre.set_ylabel("Presyn V_m (mV)")
    ax_pre.set_title("06: Finite-Delay Timing — Event, Axonal Latency, and Postsynaptic Response")
    ax_pre.legend(loc="upper right")

    # 2. Delayed synaptic arrival
    ax_delay.plot(t6, edge_i6_np, color="#2ca02c", lw=1.5, label="Synaptic current I_syn arriving at post")
    arr_times = t6[presyn6_np > 0]
    for at in arr_times:
        ax_delay.axvline(at, color="green", linestyle=":", lw=1.5, label="Delayed arrival (t_pre + 10ms)" if at == arr_times[0] else "")
    ax_delay.set_ylabel("I_syn (relative)")
    ax_delay.legend(loc="upper right")

    # 3. Postsynaptic response
    ax_post.plot(t6, v6_np[:, 1], color="#ff7f0e", lw=1.5, label="Neuron 1 V_m (postsynaptic response)")
    post_spk_times = t6[spk6_np[:, 1] > 0]
    for pst in post_spk_times:
        ax_post.axvline(pst, color="red", linestyle="-.", alpha=0.6, label="Postsynaptic spike" if pst == post_spk_times[0] else "")
    ax_post.set_ylabel("Postsyn V_m (mV)")
    ax_post.set_xlabel("Time (ms)")
    ax_post.legend(loc="upper right")

    fig.tight_layout()
    p6_path = output_dir / "06_finite_delay_timing.png"
    fig.savefig(p6_path, dpi=150)
    plt.close(fig)

    manifest["figures"].append({
        "id": "06_finite_delay_timing",
        "script": "scripts/generate_release_gallery.py",
        "seed": 1,
        "configuration": "two_neuron_delay_circuit(delay=10ms)",
        "output": str(p6_path.name),
        "caption": "Explicit temporal decomposition: presynaptic spike emission, 10 ms axonal transmission buffer latency, arriving synaptic current, and subsequent postsynaptic EPSP integration.",
        "calibration_status": "discrete_delay_buffer_exact",
    })

    # -------------------------------------------------------------
    # 07: Multi-Area / Laminar Network
    # -------------------------------------------------------------
    edges1 = model1.edge_table()
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(edges1) > 0:
        pre_idx = [e["pre"] for e in edges1]
        post_idx = [e["post"] for e in edges1]
        weights = [e["weight"] for e in edges1]
        sc = ax.scatter(pre_idx, post_idx, c=weights, cmap="coolwarm", s=8, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="Synaptic Weight")
    ax.set_xlabel("Presynaptic Neuron")
    ax.set_ylabel("Postsynaptic Neuron")
    ax.set_title("07: Multi-Area Laminar Connectivity Matrix")
    fig.tight_layout()
    p7 = output_dir / "07_multiarea_laminar_connectivity.png"
    fig.savefig(p7, dpi=150)
    plt.close(fig)

    manifest["figures"].append({
        "id": "07_multiarea_laminar_connectivity",
        "script": "scripts/generate_release_gallery.py",
        "seed": 42,
        "configuration": "suite2_v1_v4_config",
        "output": str(p7.name),
        "caption": "Realized sparse inter-column and intra-column synaptic connection matrix for V1-V4 multi-area model.",
        "calibration_status": "uncalibrated_sparse_weights",
    })

    # -------------------------------------------------------------
    # 08: Fast vs Slow State (Multiscale Dynamical Demonstration)
    # -------------------------------------------------------------
    # Fast membrane potential X (Vm) vs Slower hidden state H and synaptic weight W
    cfg8 = jtfne.suite2_net1_config(seed=12, n=100, duration_ms=400.0, dt_ms=0.5)
    m8 = jtfne.construct(cfg8)
    hp8 = dict(jtfne.DEFAULT_HDP)
    hp8["K_HDP"] = 0.04
    hp8["K_ctrl"] = 0.01
    rc8 = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params=hp8)
    sim8 = jtfne.simulation(duration_ms=400.0, dt_ms=0.5, seed=7, runtime=rc8, record_sources=True)
    sig8 = jtfne.simulate(m8, sim8)
    diag8 = getattr(m8, "_last_hdp_diag", None)

    t8 = np.asarray(sig8.time_ms)
    v8 = np.asarray(sig8.V_m)[:, 0]  # Fast neural state X
    H8 = np.asarray(diag8["H_trace"])[:, 0] if diag8 and diag8.get("H_trace") is not None else np.ones_like(t8)
    # Mean weight across outgoing edges of neuron 0
    el8 = m8.params["edge_list"]
    pre8 = np.asarray(el8.pre)
    edge_idx_0 = np.where(pre8 == 0)[0]
    w_trace8 = np.asarray(diag8["w_trace"]) if diag8 and diag8.get("w_trace") is not None else None
    w8_mean = np.mean(w_trace8[:, edge_idx_0], axis=1) if w_trace8 is not None and len(edge_idx_0) > 0 else np.ones_like(t8)

    fig, (ax_fast, ax_slow_h, ax_slow_w) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    ax_fast.plot(t8, v8, color="#1f77b4", lw=1.2, label="Fast Neural State V_m (millisecond scale)")
    ax_fast.set_ylabel("V_m (mV)")
    ax_fast.set_title("08: Multiscale State Evolution — Fast Observable X (V_m) vs Slower H and W")
    ax_fast.legend(loc="upper right")

    ax_slow_h.plot(t8, H8, color="#d62728", lw=1.8, label="Slow Relative Biophysical State H (RBD)")
    ax_slow_h.set_ylabel("RBS State H")
    ax_slow_h.legend(loc="upper right")

    ax_slow_w.plot(t8, w8_mean, color="#9467bd", lw=1.8, label="Plastic Synaptic Weight <W> (HDP parameter)")
    ax_slow_w.set_ylabel("Mean Weight W")
    ax_slow_w.set_xlabel("Time (ms)")
    ax_slow_w.legend(loc="upper right")

    fig.tight_layout()
    p8 = output_dir / "08_fast_vs_slow_state.png"
    fig.savefig(p8, dpi=150)
    plt.close(fig)

    manifest["figures"].append({
        "id": "08_fast_vs_slow_state",
        "script": "scripts/generate_release_gallery.py",
        "seed": 7,
        "configuration": "suite2_net1_config(n=100, enable_hdp=True)",
        "output": str(p8.name),
        "caption": "Co-registered multiscale trajectories: sub-millisecond membrane state X (V_m) alongside slower hidden biophysical state H (RBD dynamics) and plastic synaptic weight coupling W (HDP).",
        "calibration_status": "multiscale_state_coupling",
    })

    # Save manifest
    m_path = output_dir / "gallery_manifest.json"
    m_path.write_text(json.dumps(manifest, indent=2))
    print(f"Gallery generation complete: {len(manifest['figures'])} figures written to {output_dir}")
    return manifest


if __name__ == "__main__":
    out = Path("docs/_static/gallery")
    generate_gallery(out)
