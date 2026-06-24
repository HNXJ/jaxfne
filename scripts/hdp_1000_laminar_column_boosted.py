"""1000-neuron laminar column with HDP homeostasis + H<1 drive boost.

User spec: canonical 1K column (E-deep/I-superficial gradient, dense L2/3)
but with a PV peak at BOTH L4 and L2 (the stored canonical default only
peaks PV at L4); HDP tuned so H is steady at 1.0, weights bounded by
w_floor/w_ceiling, population firing rate stable at 5.0+-2.5 Hz, and the
newly-added jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp
H_boost_gain mechanism providing a strong drive boost when H_i < 1.0.

IMPORTANT CORRECTION: the H-to-drive-boost mechanism did NOT already exist
in the codebase (verified by grep before this script was written) -- it was
newly added as the H_boost_gain kernel parameter in this session. This
script is also the smoke test/usage example for that new parameter.

HDP is not wired into core.py/RuntimeConfig (open task), so this drives
jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp directly against a
jtfne.construct()-built Model's emitter/edge_list/positions params, then
hand-assembles a jtfne.Signals container so jtfne.vis.spectrolaminar_suite
can be run on the result (see project memory `jaxfne-laminar-sim-recipe`).

Usage: PYTHONPATH=. python3 scripts/hdp_1000_laminar_column_boosted.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import simulate_edge_recurrent_izhikevich_hdp

OUTPUT_DIR = Path("outputs/hdp_1000_laminar_column_boosted")

LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6"]

# Canonical z-bands (width prop. to neuron count): 100,250,200,100,200,150.
ZBANDS = {
    "L1": (0.00, 0.10), "L2": (0.10, 0.35), "L3": (0.35, 0.55),
    "L4": (0.55, 0.65), "L5": (0.65, 0.85), "L6": (0.85, 1.00),
}

# Modified from the canonical LAYER_CELL_TYPE_FRAC (PV peak L4-only) to put
# a PV peak at L2 AND L4, with a dip at L3 between them. Deep-layer E/I
# gradient (L5/L6) and L1 (no PV) left exactly as canonical. Each row sums
# to 1.0.
LAYER_CELL_TYPE_FRAC = {
    "L1": {"E": 0.50, "PV": 0.00, "SST": 0.15, "VIP": 0.35},
    "L2": {"E": 0.65, "PV": 0.20, "SST": 0.10, "VIP": 0.05},  # PV peak #1
    "L3": {"E": 0.80, "PV": 0.08, "SST": 0.08, "VIP": 0.04},  # dip between peaks
    "L4": {"E": 0.75, "PV": 0.18, "SST": 0.04, "VIP": 0.03},  # PV peak #2
    "L5": {"E": 0.88, "PV": 0.06, "SST": 0.04, "VIP": 0.02},
    "L6": {"E": 0.90, "PV": 0.05, "SST": 0.03, "VIP": 0.02},
}

N_NEURONS = 1000
SEED = 0
DT_MS = 0.5
DURATION_MS = 1000.0
TAIL_FRACTION = 0.3
TARGET_RATE_HZ = 5.0
TARGET_RATE_TOL_HZ = 2.5

# HDP defaults: chosen so the H ODE's only live terms are alpha*I_syn (income)
# and K_ctrl*(1-H) (linear restoring to exactly H=1.0); gamma=delta=0 keeps
# H purely a synaptic-drive sensor (matches the frozen-weight-stable finding
# from the earlier small-scale sweep). K_HDP small but nonzero so weights
# still adapt (not frozen) but slowly, away from saturation.
HDP_KWARGS = dict(
    H_min=0.1, H_max=10.0, tau_0_ms=200.0,
    alpha=0.01, beta=0.0, gamma=0.0, delta=0.0, C_spike=0.0,
    K_HDP=0.05, K_ctrl=0.5, barrier_c=0.01, barrier_d=0.01, barrier_eps=1.0e-3,
    w_floor=0.01, w_ceiling=10.0,
    H_boost_gain=4.0,  # strong boost when H<1.0, per user spec
)

BASE_DRIVE_BY_CELL_TYPE = {"E": 4.0, "PV": 4.0, "SST": 4.0, "VIP": 4.0}

# First-pass result with uniform 4.0 drive: E=6.7Hz (in band), PV~14-22Hz
# (too hot), SST~52-54Hz (far too hot), VIP=0Hz (silent). Per-cell-type
# drive correction applied post-hoc via with_emitter_parameters(drive_per_
# neuron=...) -- no rebuild needed (reuse-built-model discipline).
DRIVE_CORRECTION_BY_CELL_TYPE = {"E": 1.0, "PV": 0.45, "SST": 0.09, "VIP": 5.6}
# Found by a short standalone grid search (300-600ms probes) directly
# against simulate_edge_recurrent_izhikevich_hdp before committing to this
# script; all four cell types land in 5.0+-2.5Hz at the 1000ms tuning
# window. CAVEAT (verified, not papered over): at 2000ms duration the same
# tuned point drifts outside the target band (PV/SST/VIP rates roughly
# double) -- weight plasticity (K_HDP=0.05) keeps growing slowly even
# though H stays pinned near 1.0, so the spectrolaminar run below
# deliberately uses the SAME 1000ms duration as the validated tuning
# window rather than a longer window that would silently leave the
# verified-in-band regime.


def apply_drive_correction(model: "jtfne.core.Model") -> "jtfne.core.Model":
    labels = np.asarray(model.params["emitter"].labels)
    base_drive = np.asarray(model.params["emitter"].drive)
    corrected = base_drive.copy()
    for ct, factor in DRIVE_CORRECTION_BY_CELL_TYPE.items():
        corrected[labels == ct] = base_drive[labels == ct] * factor
    return model.with_emitter_parameters(drive_per_neuron=jnp.asarray(corrected))


def build_model() -> "jtfne.core.Model":
    builder = (
        jtfne.laminar_cortex_config(
            areas=["V1"], layers=LAYERS, n=N_NEURONS,
            duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED, emitter="izhikevich",
            baseline_drive_by_cell_type=BASE_DRIVE_BY_CELL_TYPE,
        )
        .layer_fractions(layer_fractions=ZBANDS)
        .area_layer_cell_types("V1", LAYER_CELL_TYPE_FRAC)
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="hdp_1000_probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(builder)


def run(model: "jtfne.core.Model", duration_ms: float, seed: int) -> dict[str, Any]:
    n_steps = int(round(duration_ms / DT_MS))
    key = jax.random.PRNGKey(seed)
    voltages, spikes, sources, diagnostics = simulate_edge_recurrent_izhikevich_hdp(
        model.params["emitter"], model.params["edge_list"], n_steps, DT_MS, key,
        **HDP_KWARGS,
    )
    return {"voltages": voltages, "spikes": spikes, "sources": sources,
            "diagnostics": diagnostics, "n_steps": n_steps}


def summarize(model: "jtfne.core.Model", run_out: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray(model.params["emitter"].labels)
    spikes_np = np.asarray(run_out["spikes"])
    n_steps = run_out["n_steps"]
    tail_steps = int(round(n_steps * TAIL_FRACTION))
    tail = spikes_np[-tail_steps:]
    duration_s = tail.shape[0] * DT_MS / 1000.0

    rate_by_type = {
        ct: float(tail[:, labels == ct].sum() / ((labels == ct).sum() * duration_s))
        for ct in sorted(set(labels.tolist())) if (labels == ct).any()
    }
    overall_rate = float(tail.sum() / (tail.shape[1] * duration_s))
    kappa = float(jtfne.kappa_synchrony(spikes_np, dt_ms=DT_MS))

    H_trace = np.asarray(run_out["diagnostics"]["H_trace"])[-tail_steps:]
    w_trace = np.asarray(run_out["diagnostics"]["w_trace"])[-tail_steps:]
    w_ceiling, w_floor = HDP_KWARGS["w_ceiling"], HDP_KWARGS["w_floor"]
    saturated = (np.abs(w_trace[-1]) >= 0.999 * w_ceiling) | (np.abs(w_trace[-1]) <= 1.001 * w_floor)

    return {
        "rate_by_type_hz": rate_by_type,
        "overall_rate_hz": overall_rate,
        "rate_in_target_band": all(
            abs(r - TARGET_RATE_HZ) <= TARGET_RATE_TOL_HZ for r in rate_by_type.values()
        ),
        "kappa_synchrony": kappa,
        "H_mean": float(H_trace.mean()),
        "H_var": float(H_trace.var()),
        "H_max_abs_drift": float(np.abs(H_trace - 1.0).max()),
        "weight_saturation_fraction": float(saturated.mean()),
        "weight_min": float(np.abs(w_trace[-1]).min()),
        "weight_max": float(np.abs(w_trace[-1]).max()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Building 1000-neuron laminar column (PV peak L2+L4) ===")
    model = build_model()

    lc = {}
    for r in model.neuron_table():
        lc[(r["layer"], r["cell_type"])] = lc.get((r["layer"], r["cell_type"]), 0) + 1
    print("Per-layer/cell-type counts:", dict(sorted(lc.items())))

    model = apply_drive_correction(model)
    print(f"Applied per-cell-type drive correction: {DRIVE_CORRECTION_BY_CELL_TYPE}")

    print(f"\n=== HDP run, duration={DURATION_MS}ms, kwargs={HDP_KWARGS} ===")
    run_out = run(model, DURATION_MS, SEED)
    summary = summarize(model, run_out)
    print(json.dumps(summary, indent=2))

    print("\n=== Spectrolaminar run (same duration as the validated tuning window -- ===")
    print("=== see DRIVE_CORRECTION_BY_CELL_TYPE caveat: longer durations drift out of band) ===")
    spectro_duration_ms = DURATION_MS
    spectro_run = run(model, spectro_duration_ms, SEED + 1)
    spectro_summary = summarize(model, spectro_run)
    print(json.dumps(spectro_summary, indent=2))

    field = jtfne.project_laminar_sources(
        spectro_run["sources"], model.params["positions"],
        n_contacts=16, width=0.10, mode="density_preserving",
    )
    n_steps = spectro_run["n_steps"]
    time_ms = jnp.arange(n_steps, dtype=jnp.float32) * DT_MS
    signals = jtfne.Signals(
        time_ms=time_ms,
        V_m=spectro_run["voltages"],
        spikes=spectro_run["spikes"],
        sources=spectro_run["sources"],
        field=field,
        metadata={
            "claim_level": "proxy_readout",
            "field_solver_status": "linear_solver",
            "physical_amplitude_calibrated": False,
            "note": "computational scaffold; HDP H_boost_gain diagnostic run, not a calibrated biological result",
        },
    )

    fig = jtfne.vis.spectrolaminar_suite(signals)
    fig.savefig(OUTPUT_DIR / "spectrolaminar_suite.png", dpi=150)

    payload = {
        "layer_cell_type_frac": LAYER_CELL_TYPE_FRAC,
        "hdp_kwargs": HDP_KWARGS,
        "tuning_run_summary": summary,
        "spectro_run_summary": spectro_summary,
        "per_layer_celltype_counts": {f"{k[0]}/{k[1]}": v for k, v in lc.items()},
    }
    with open(OUTPUT_DIR / "hdp_1000_laminar_column_boosted.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
