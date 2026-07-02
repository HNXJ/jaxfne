"""Laminar column with HDP homeostasis + H<1 drive boost (default N=1000).

User spec: canonical 1K column (E-deep/I-superficial gradient, dense L2/3)
HDP tuned so H is steady at ~1.0, weights bounded by w_floor/w_ceiling,
population firing rate stable at 5.0+-2.5 Hz, using
jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp's H_boost_gain
mechanism (strong drive boost when H_i < 1.0).

This script no longer hand-maintains its own build_model/
apply_drive_correction copies -- those live once in jaxfne.hdp_network and
are shared with scripts/hdp_100_stability_sweep.py. N_NEURONS is just a
config field (HDPColumnConfig.n_neurons), not part of any function name --
per the "one network maker for any TFNE-Izhikevich network" mandate.

HDP_KWARGS below uses jaxfne.hdp_network.DEFAULT_HDP (K_HDP=0.01,
tau_0_ms=200.0, K_ctrl=5.0, barrier_c=barrier_d=0.01) -- the 5-seed,
20s-validated freeze candidate from scripts/hdp_100_stability_sweep.py --
not the older K_HDP=0.05/K_ctrl=0.5 pair this script previously used,
which was only verified at a 1000ms tuning window and known to drift out
of band by 2000ms.

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

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.hdp_network import (
    HDPColumnConfig, build_model, apply_drive_correction,
    layer_size_scale_override, run, BASE_HDP_KWARGS_DEFAULT, DEFAULT_HDP,
    BASE_DRIVE_BY_CELL_TYPE_DEFAULT,
)

OUTPUT_DIR = Path("outputs/hdp_1000_laminar_column_boosted")

N_NEURONS = 1000
SEED = 0
DT_MS = 0.5
DURATION_MS = 1000.0
TAIL_FRACTION = 0.3
TARGET_RATE_HZ = 5.0
TARGET_RATE_TOL_HZ = 2.5

# F-023: explicit base_drive_by_cell_type preserves this script's prior
# always-4.0-for-all behavior now that HDPColumnConfig's dataclass default
# is None (emitter-preset passthrough) -- see skills/FRICTIONS_STACK.md F-023.
# hdp_suite2_visualizations.py reuses this exact CFG, so it inherits the fix.
CFG = HDPColumnConfig(n_neurons=N_NEURONS, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED,
                       probe_name="hdp_1000_probe",
                       base_drive_by_cell_type=dict(BASE_DRIVE_BY_CELL_TYPE_DEFAULT))

HDP_KWARGS = dict(BASE_HDP_KWARGS_DEFAULT)
HDP_KWARGS.update(DEFAULT_HDP)


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
    print(f"=== Building {N_NEURONS}-neuron laminar column (via jaxfne.hdp_network) ===")
    model = build_model(CFG)

    lc = {}
    for r in model.neuron_table():
        lc[(r["layer"], r["cell_type"])] = lc.get((r["layer"], r["cell_type"]), 0) + 1
    print("Per-layer/cell-type counts:", dict(sorted(lc.items())))

    model = apply_drive_correction(model, CFG)
    print(f"Applied per-cell-type drive correction: {dict(CFG.drive_correction_by_cell_type)}")

    size_override = layer_size_scale_override(model, CFG)
    print(f"Applied per-layer size scale: {dict(CFG.layer_size_scale)}")

    print(f"\n=== HDP run, duration={DURATION_MS}ms, kwargs={HDP_KWARGS} ===")
    run_out = run(model, CFG, DURATION_MS, SEED, hdp_kwargs=HDP_KWARGS, size_scale_override=size_override)
    summary = summarize(model, run_out)
    print(json.dumps(summary, indent=2))

    print("\n=== Spectrolaminar run (same HDP-validated parameters) ===")
    spectro_duration_ms = DURATION_MS
    spectro_run = run(model, CFG, spectro_duration_ms, SEED + 1, hdp_kwargs=HDP_KWARGS, size_scale_override=size_override)
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
        "layer_cell_type_frac": dict(CFG.layer_cell_type_frac),
        "hdp_kwargs": HDP_KWARGS,
        "layer_size_scale": dict(CFG.layer_size_scale),
        "tuning_run_summary": summary,
        "spectro_run_summary": spectro_summary,
        "per_layer_celltype_counts": {f"{k[0]}/{k[1]}": v for k, v in lc.items()},
    }
    with open(OUTPUT_DIR / "hdp_1000_laminar_column_boosted.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
