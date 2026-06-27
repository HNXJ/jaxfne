"""1000-neuron V1 column expressed as a NeuronalTensor (Area x Layer x
NeuronType x InterConnection), HDP-driven both synaptically (per-edge
dw_E/dw_I) and H-factor-wise (per-neuron H_i), with cube-law (size**3)
adaptation time constants.

This is the 0.4.7 tensor-first counterpart of
scripts/hdp_1000_laminar_column_boosted.py: same validated canonical
layer/cell-type composition and drive correction, but the circuit itself
-- areas, layers, cell types, and every inter-layer/intra-layer synapse --
is declared as data (a NeuronalTensor), not as Configuration builder calls.
Real edges (not just metadata) are compiled from InterConnection entries via
jaxfne.neuronal_tensor.neuronal_tensor_to_configuration's selector-based
Configuration.connections()/.mechanisms() bridge.

HDP is enabled via an explicit RuntimeConfig override at simulate() time
(RuntimeConfig.hdp_params is already a free-form dict -- no new public API),
with size_scale_by_cell_type matching the tensor's declared
NeuronType.relative_size (defaults from `DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE`:
E=5.0, PV=1.0, SST/VIP=1.5), so tau_i = tau_0_ms *
size_i**3 differentiates E's homeostatic adaptation rate from interneurons'
exactly as declared on the tensor.

Usage: PYTHONPATH=. python3 scripts/hdp_1000_neuronal_tensor_column.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.core import RuntimeConfig
from jaxfne.hdp_network import (
    BASE_DRIVE_BY_CELL_TYPE_DEFAULT, DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT,
    DEFAULT_HDP, BASE_HDP_KWARGS_DEFAULT,
)

OUTPUT_DIR = Path("outputs/hdp_1000_neuronal_tensor_column")

SEED = 0
DT_MS = 0.5
DURATION_MS = 1000.0
TAIL_FRACTION = 0.3
TARGET_RATE_HZ = 5.0
TARGET_RATE_TOL_HZ = 2.5

# Canonical 1000-neuron V1 column composition (validated in
# scripts/hdp_1000_laminar_column_boosted.py): deep-E/superficial-I gradient,
# dense L2/3, PV absent in L1.
LAYER_COUNTS = {
    "L1": {"E": 50, "SST": 15, "VIP": 35},
    "L2": {"E": 162, "PV": 50, "SST": 25, "VIP": 13},
    "L3": {"E": 160, "PV": 16, "SST": 16, "VIP": 8},
    "L4": {"E": 75, "PV": 18, "SST": 4, "VIP": 3},
    "L5": {"E": 176, "PV": 12, "SST": 8, "VIP": 4},
    "L6": {"E": 135, "PV": 8, "SST": 4, "VIP": 3},
}
LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6"]
ZBANDS = {
    "L1": (0.00, 0.10), "L2": (0.10, 0.35), "L3": (0.35, 0.55),
    "L4": (0.55, 0.65), "L5": (0.65, 0.85), "L6": (0.85, 1.00),
}

from jaxfne.emitters import DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE

# Relative sizes aligned with HDP kernel table (same as NeuronType.make defaults).
RELATIVE_SIZE = {ct: float(DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE.get(ct, 1.0))
                 for ct in ("E", "PV", "SST", "VIP")}

AMPA_TAU_MS, AMPA_REV_MV = 2.0, 0.0
GABA_TAU_MS, GABA_REV_MV = 5.0, -80.0
SYNAPTIC_GAIN = 0.45  # w_mech; matches the established within_gain=0.45 convention

# Intra-layer E/I microcircuit motif (applied per layer, only for type pairs
# the layer actually has -- L1 has no PV, so its PV-touching rules are
# skipped automatically).
INTRA_LAYER_RULES = [
    ("E", "PV", "AMPA"), ("E", "SST", "AMPA"), ("E", "VIP", "AMPA"),
    ("PV", "E", "GABA_A"), ("SST", "E", "GABA_A"), ("VIP", "SST", "GABA_A"),
    ("PV", "PV", "GABA_A"),
]

# Inter-laminar E->E feedforward/feedback motif (canonical-microcircuit-style:
# L4 granular input -> L2/3 supragranular -> L5 infragranular -> L6, with a
# L6->L4 corticothalamic-like feedback loop and L6->L1/L1->L2 closing the
# loop through the modulatory layer).
INTER_LAMINAR_EE = [
    ("L1", "L2"), ("L4", "L2"), ("L4", "L3"), ("L2", "L3"),
    ("L2", "L5"), ("L3", "L5"), ("L5", "L6"), ("L6", "L4"), ("L6", "L1"),
]


def _static(mechanism: str) -> nt.StaticParams:
    if mechanism == "AMPA":
        return nt.StaticParams(g_mech={"AMPA": 1.0},
                                reversal_potentials_mV={"AMPA": AMPA_REV_MV},
                                dT_ms=AMPA_TAU_MS)
    return nt.StaticParams(g_mech={"GABA_A": 1.0},
                            reversal_potentials_mV={"GABA_A": GABA_REV_MV},
                            dT_ms=GABA_TAU_MS)


def build_tensor() -> nt.NeuronalTensor:
    layers = []
    for layer_name in LAYER_ORDER:
        counts = LAYER_COUNTS[layer_name]
        total = sum(counts.values())
        neuron_types = [
            nt.NeuronType.make(ct, relative_size=RELATIVE_SIZE[ct], fraction=count / total)
            for ct, count in counts.items()
        ]
        geometry = nt.Geometry3D(distribution="uniform_random",
                                  x_range=(0.0, 1.0), y_range=(0.0, 1.0),
                                  z_range=ZBANDS[layer_name])
        layers.append(nt.Layer(name=layer_name, neuron_types=neuron_types,
                                geometry=geometry, n_neurons=total))

    inter_connections = []
    for layer_name in LAYER_ORDER:
        counts = LAYER_COUNTS[layer_name]
        for src_type, tgt_type, mechanism in INTRA_LAYER_RULES:
            if src_type not in counts or tgt_type not in counts:
                continue
            inter_connections.append(nt.InterConnection(
                source_layer=layer_name, source_neuron_type=src_type,
                target_layer=layer_name, target_neuron_type=tgt_type,
                mechanism=mechanism,
                static=_static(mechanism),
                plastic=nt.PlasticParams(w_mech=SYNAPTIC_GAIN, H=1.0),
            ))
    for src_layer, tgt_layer in INTER_LAMINAR_EE:
        inter_connections.append(nt.InterConnection(
            source_layer=src_layer, source_neuron_type="E",
            target_layer=tgt_layer, target_neuron_type="E",
            mechanism="AMPA",
            static=_static("AMPA"),
            plastic=nt.PlasticParams(w_mech=SYNAPTIC_GAIN, H=1.0),
        ))

    area = nt.Area(name="V1", layers=layers, inter_connections=inter_connections)
    return nt.NeuronalTensor(areas=[area], name="hdp_1000_v1_column")


def summarize(model: "jtfne.core.Model", sig: "jtfne.Signals", diag: dict) -> dict:
    """Per-layer/cell-type rates, kappa synchrony, HDP H stats, and edge-weight
    stats over the tail window (post-transient)."""
    rows = model.neuron_table()
    layers = np.asarray([r["layer"] for r in rows])
    cell_types = np.asarray([r["cell_type"] for r in rows])
    spikes_np = np.asarray(sig.spikes)
    n_steps = spikes_np.shape[0]
    tail_steps = int(round(n_steps * TAIL_FRACTION))
    tail = spikes_np[-tail_steps:]
    duration_s = tail.shape[0] * DT_MS / 1000.0

    rate_by_celltype = {
        ct: float(tail[:, cell_types == ct].sum() / ((cell_types == ct).sum() * duration_s))
        for ct in sorted(set(cell_types.tolist()))
    }
    rate_by_layer_celltype = {}
    for layer in LAYER_ORDER:
        for ct in sorted(set(cell_types.tolist())):
            mask = (layers == layer) & (cell_types == ct)
            if mask.any():
                rate_by_layer_celltype[f"{layer}/{ct}"] = float(
                    tail[:, mask].sum() / (mask.sum() * duration_s))
    overall_rate = float(tail.sum() / (tail.shape[1] * duration_s))
    kappa = float(jtfne.kappa_synchrony(spikes_np, dt_ms=DT_MS))

    H_trace = np.asarray(diag["H_trace"])
    H_tail = H_trace[-tail_steps:]
    H_stats_by_celltype = {}
    for ct in sorted(set(cell_types.tolist())):
        mask = cell_types == ct
        dev = H_tail[:, mask] - 1.0
        H_stats_by_celltype[ct] = {
            "H_mean": float(H_tail[:, mask].mean()),
            "H_var": float(H_tail[:, mask].var()),
            "H_max_abs_drift": float(np.abs(dev).max()),
        }

    w_trace = np.asarray(diag["w_trace"])
    w_final = np.asarray(diag["w_final"])
    edges = model.params["edge_list"]
    exc_mask = np.asarray(edges.receptor_index) == 0
    w_floor, w_ceiling = BASE_HDP_KWARGS_DEFAULT["w_floor"], BASE_HDP_KWARGS_DEFAULT["w_ceiling"]
    saturated = (np.abs(w_final) >= 0.999 * w_ceiling) | (np.abs(w_final) <= 1.001 * w_floor)

    return {
        "rate_by_celltype_hz": rate_by_celltype,
        "rate_by_layer_celltype_hz": rate_by_layer_celltype,
        "overall_rate_hz": overall_rate,
        "rate_in_target_band": all(
            abs(r - TARGET_RATE_HZ) <= TARGET_RATE_TOL_HZ for r in rate_by_celltype.values()),
        "kappa_synchrony": kappa,
        "H_stats_by_celltype": H_stats_by_celltype,
        "weight_saturation_fraction": float(saturated.mean()),
        "weight_min_abs": float(np.abs(w_final).min()),
        "weight_max_abs": float(np.abs(w_final).max()),
        "weight_mean_abs_excitatory": float(np.abs(w_final[exc_mask]).mean()),
        "weight_mean_abs_inhibitory": float(np.abs(w_final[~exc_mask]).mean()),
        "n_steps": n_steps,
        "tail_steps": tail_steps,
        "_w_trace_tail": w_trace[-tail_steps:],
        "_exc_mask": exc_mask,
        "_H_trace": H_trace,
        "_cell_types": cell_types,
        "_layers": layers,
    }


def visualize(model: "jtfne.core.Model", sig: "jtfne.Signals", diag: dict, summary: dict) -> None:
    """All rendering via jaxfne.vis (visualization isolation invariant)."""
    cell_types = summary["_cell_types"]
    H_trace = summary["_H_trace"]
    n_steps = summary["n_steps"]
    t_ms = np.arange(n_steps) * DT_MS

    fig = jtfne.vis.raster(sig, figsize=(10, 6))
    fig.savefig(OUTPUT_DIR / "raster_by_depth.png", dpi=150)
    jtfne.vis.close_all()

    per_type_H = {
        ct: {"t_ms": t_ms, "mean_H_t": H_trace[:, cell_types == ct].mean(axis=1)}
        for ct in sorted(set(cell_types.tolist()))
    }
    fig = jtfne.vis.hdp_state_trajectories(per_type_H, mode="H",
                                            title="HDP H_i over time, per cell type (cube-law tau)")
    fig.savefig(OUTPUT_DIR / "hdp_H_trajectories.png", dpi=150)
    jtfne.vis.close_all()

    w_trace_tail = summary["_w_trace_tail"]
    exc_mask = summary["_exc_mask"]
    t_tail_ms = t_ms[-w_trace_tail.shape[0]:]
    fig = jtfne.vis.multi_trace(
        t_tail_ms,
        {
            "mean |w| excitatory": np.abs(w_trace_tail[:, exc_mask]).mean(axis=1),
            "mean |w| inhibitory": np.abs(w_trace_tail[:, ~exc_mask]).mean(axis=1),
        },
        title="HDP synaptic weight trajectory (tail window)", ylabel="|w| (native units)",
    )
    fig.savefig(OUTPUT_DIR / "hdp_weight_trajectories.png", dpi=150)
    jtfne.vis.close_all()

    fig = jtfne.vis.rate_histogram_by_celltype(
        summary["rate_by_celltype_hz"], band=(TARGET_RATE_HZ - TARGET_RATE_TOL_HZ, TARGET_RATE_HZ + TARGET_RATE_TOL_HZ))
    fig.savefig(OUTPUT_DIR / "rate_by_celltype.png", dpi=150)
    jtfne.vis.close_all()

    html_path = OUTPUT_DIR / "network_3d.html"
    jtfne.vis.visualize_network_3d(model, output_html=str(html_path), show_layers=True)
    print(f"Wrote interactive 3D network: {html_path}")

    if sig.field is not None:
        fig = jtfne.vis.spectrolaminar_suite(sig)
        fig.savefig(OUTPUT_DIR / "spectrolaminar_suite.png", dpi=150)
        jtfne.vis.close_all()
        print("Wrote spectrolaminar_suite.png")
    else:
        print("sig.field is None -- skipping spectrolaminar_suite")


def apply_drive_correction(model: "jtfne.core.Model") -> "jtfne.core.Model":
    labels = np.asarray(model.params["emitter"].labels)
    drive = np.empty(len(labels), dtype=np.float32)
    for ct in BASE_DRIVE_BY_CELL_TYPE_DEFAULT:
        mask = labels == ct
        base = BASE_DRIVE_BY_CELL_TYPE_DEFAULT[ct]
        corr = DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT[ct]
        drive[mask] = base * corr
    import jax.numpy as jnp
    return model.with_emitter_parameters(drive_per_neuron=jnp.asarray(drive))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Building 1000-neuron V1 column as a NeuronalTensor ===")
    tensor = build_tensor()
    n_inter_connections = sum(len(a.inter_connections) for a in tensor.areas)
    n_neurons = sum(l.n_neurons for a in tensor.areas for l in a.layers)
    print(f"areas={len(tensor.areas)} layers={sum(len(a.layers) for a in tensor.areas)} "
          f"neurons={n_neurons} inter_connections={n_inter_connections}")
    nt.save_neuronal_tensor(tensor, OUTPUT_DIR / "hdp_1000_v1_column_tensor.json")

    print("\n=== Constructing Model (tensor-first: jtfne.construct(tensor, runtime)) ===")
    model = jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=SEED, duration_ms=DURATION_MS, dt_ms=DT_MS, emitter="izhikevich"))
    n_edges = int(model.params["edge_list"].weight.shape[0])
    print(f"constructed: n_neurons={model.params['emitter'].n_neurons} n_edges={n_edges}")

    model = apply_drive_correction(model)
    print(f"Applied per-cell-type drive correction: base={BASE_DRIVE_BY_CELL_TYPE_DEFAULT} "
          f"correction={DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT}")

    hdp_kwargs = dict(BASE_HDP_KWARGS_DEFAULT)
    hdp_kwargs.update(DEFAULT_HDP)
    hdp_kwargs["size_scale_by_cell_type"] = dict(RELATIVE_SIZE)
    runtime_hdp = RuntimeConfig(enable_hdp=True, hdp_params=hdp_kwargs)
    print(f"\n=== HDP run (size_scale_by_cell_type={RELATIVE_SIZE}, cube law) ===")
    print(json.dumps(hdp_kwargs, indent=2))

    sig = jtfne.simulate(model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED, runtime=runtime_hdp)
    diag = model.last_hdp_diagnostics()

    print("\n=== Statistics ===")
    summary = summarize(model, sig, diag)
    report = {k: v for k, v in summary.items() if not k.startswith("_")}
    print(json.dumps(report, indent=2))

    print("\n=== Visualizing (jaxfne.vis only) ===")
    visualize(model, sig, diag, summary)

    payload = {
        "n_neurons": n_neurons,
        "n_edges": n_edges,
        "n_inter_connections": n_inter_connections,
        "hdp_kwargs": {k: v for k, v in hdp_kwargs.items()},
        "drive_base": BASE_DRIVE_BY_CELL_TYPE_DEFAULT,
        "drive_correction": DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT,
        "summary": report,
    }
    with open(OUTPUT_DIR / "build_manifest.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"\nDone. Results dir: {OUTPUT_DIR.resolve()}")
    return model, sig, diag, tensor, summary


if __name__ == "__main__":
    main()
