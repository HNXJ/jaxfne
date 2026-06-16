#!/usr/bin/env python3
"""Reproducible figure/metric regeneration for the 10k V1 spectrolaminar report.

Builds a single V1 laminar column of N=10,000 reduced Izhikevich emitters, sweeps a
constant (DC) drive to all deep (L5/L6) cells over {0, 5, 10}, and for each level saves a
spike raster and the package spectrolaminar suite. Renders a true Plotly 3D snapshot of the
column via kaleido, and writes per-DC metrics (mean firing rate; relative deep alpha/beta
vs superficial gamma from the package laminar field) to metrics.json.

All field outputs are proxy readouts: field_solver_status=linear_solver,
field_claim_level=proxy_readout, physical_amplitude_calibrated=False. Not physical
EEG/MEG/LFP/CSD measurements.

Run as a fresh subprocess (10k build ~40 s, each sim ~90 s on CPU):
    python3 reports/v1_spectrolaminar_10k/regen_figs.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

N, DUR, DT = 10000, 1000.0, 0.1
LAYERS = ["L1", "L2/3", "L4", "L5", "L6"]
# deep layers thin (sparser) + positioned deep
LAYER_RANGES = {"L1": (0.0, 0.12), "L2/3": (0.12, 0.46), "L4": (0.46, 0.66),
                "L5": (0.66, 0.80), "L6": (0.80, 1.0)}
# deep layers more E; superficial more PV (fast inhibition -> gamma)
LAYER_CT = {"L1": {"E": 0.50, "PV": 0.20, "SST": 0.20, "VIP": 0.10},
            "L2/3": {"E": 0.72, "PV": 0.15, "SST": 0.09, "VIP": 0.04},
            "L4": {"E": 0.78, "PV": 0.13, "SST": 0.07, "VIP": 0.02},
            "L5": {"E": 0.90, "PV": 0.06, "SST": 0.03, "VIP": 0.01},
            "L6": {"E": 0.92, "PV": 0.05, "SST": 0.02, "VIP": 0.01}}
DEEP = {"L5", "L6"}
DC_LEVELS = [0.0, 5.0, 10.0]


def build(deep_dc: float):
    cfg = jtfne.laminar_cortex_config(areas=["V1"], layers=LAYERS, n=N,
                                      duration_ms=DUR, dt_ms=DT, seed=0)
    cfg = cfg.layer_fractions(layer_fractions=dict(LAYER_RANGES))
    cfg = cfg.area_layer_cell_types("V1", LAYER_CT)
    model = jtfne.construct(cfg)
    nt = model.neuron_table()
    deep_mask = np.array([r["layer"] in DEEP for r in nt], dtype="float32")
    emit = model.params["emitter"]
    drive = np.asarray(emit.drive) + deep_mask * deep_dc  # DC to all deep cells
    model = model.with_emitter_parameters(
        drive_per_neuron=jnp.asarray(drive, dtype=emit.drive.dtype))
    return model


def crossing_metric(sig):
    """Relative deep alpha/beta vs superficial gamma from the package laminar field."""
    lfp = np.asarray(sig.field.lfp_proxy)              # (T, C)
    depths = np.asarray(sig.field.contact_depths)      # (C,) 0=superficial .. 1=deep
    if not np.isfinite(lfp).all():
        return None
    fs = 1000.0 / DT
    freqs = jnp.arange(1.0, 101.0)
    psd = np.asarray(jtfne.spectrolaminar_psd_jax(lfp[None], fs=fs, freqs=freqs))  # (F, C)
    ro = jtfne.spectrolaminar_readout_kernel_jax(
        jnp.asarray(psd), freqs, jnp.array([8.0, 30.0]), jnp.array([30.0, 80.0]))
    ab = np.asarray(ro["alpha_beta"])                  # (C,)
    gm = np.asarray(ro["gamma"])                       # (C,)
    # normalize each band across depth, then split deep / superficial halves
    ab_n = ab / (ab.sum() + 1e-12)
    gm_n = gm / (gm.sum() + 1e-12)
    deep = depths >= 0.5
    sup = ~deep
    return {
        "deep_alpha_beta_frac": float(ab_n[deep].sum()),
        "sup_alpha_beta_frac": float(ab_n[sup].sum()),
        "deep_gamma_frac": float(gm_n[deep].sum()),
        "sup_gamma_frac": float(gm_n[sup].sum()),
        "n_contacts": int(lfp.shape[1]),
    }


def main():
    metrics = {"N": N, "duration_ms": DUR, "dt_ms": DT, "dc_levels": DC_LEVELS, "per_dc": {}}
    last_model = None
    for dc in DC_LEVELS:
        print(f"[regen] building + simulating deep_DC={dc} ...", flush=True)
        model = build(dc)
        last_model = model
        sig = jtfne.simulate(model, duration_ms=DUR, dt_ms=DT, seed=0)
        spk = np.asarray(sig.get("spk"))
        rate = float(spk.sum() / N / (DUR / 1000.0))
        vm_finite = bool(np.isfinite(np.asarray(sig.get("vm"))).all())
        tag = str(int(dc))
        jtfne.vis.raster(sig).savefig(
            os.path.join(FIGS, f"03_raster_dc{tag}.png"), dpi=150, bbox_inches="tight")
        jtfne.vis.spectrolaminar_suite(sig).savefig(
            os.path.join(FIGS, f"04_spectrolaminar_dc{tag}.png"), dpi=150, bbox_inches="tight")
        cm = crossing_metric(sig)
        metrics["per_dc"][tag] = {"deep_dc": dc, "mean_rate_hz": round(rate, 2),
                                  "vm_finite": vm_finite, "crossing": cm}
        print(f"[regen] deep_DC={dc}: rate={rate:.2f} Hz vm_finite={vm_finite} crossing={cm}",
              flush=True)

    # true Plotly 3D snapshot (kaleido) from the last (DC=10) model
    print("[regen] rendering 3D Plotly circuit snapshot via kaleido ...", flush=True)
    fig3d = jtfne.vis.visualize_network_3d(
        last_model.neuron_table(), title="V1 column — 10,000 neurons",
        show_layers=True, coordinate_unit="mm", display_unit="um",
        output_html=os.path.join(FIGS, "09_circuit3d_10k.html"))
    try:
        fig3d.update_layout(scene_camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)))
        fig3d.write_image(os.path.join(FIGS, "09_circuit3d_10k.png"),
                          width=1400, height=1100, scale=2)
        metrics["plotly_snapshot"] = "kaleido"
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[regen] WARN kaleido export failed: {type(e).__name__}: {e}", flush=True)
        metrics["plotly_snapshot"] = f"failed:{type(e).__name__}"

    with open(os.path.join(HERE, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, allow_nan=False)
    print("[regen] DONE. metrics.json written.", flush=True)
    print(json.dumps(metrics["per_dc"], indent=2))


if __name__ == "__main__":
    main()
