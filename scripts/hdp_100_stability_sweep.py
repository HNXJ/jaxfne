"""Fast small-N HDP stability sweep for long-duration simulations.

Goal: find HDP parameter settings (chiefly K_HDP) that stay stable over a
LONG duration (10-20s), not just the short tuning window used for the
1000-neuron build in hdp_1000_laminar_column_boosted.py. That script's
K_HDP=0.05 was only verified at 1000ms; the per-cell-type drive correction
there already documented a caveat that rates drift out of band by 2000ms.

Two corrections made during this sweep, in order:

1. N=100 was tried first and rejected: the canonical 6-layer x 4-cell-type
   fractions leave VIP with only 5 neurons total (0 in L4/L5/L6) and SST
   with 7 (0 in L4) -- rates read as spurious 0Hz from pure spike-count
   granularity at that population size, independent of any real dynamics.
   N=250 (current N_NEURONS) gives per-cell-type totals of E=189/PV=25/
   SST=18/VIP=18, enough for stable rate estimates.

2. At N=250, even K_HDP=0 (weights frozen, plasticity fully off) was NOT
   stationary with the old N=1000-tuned drive correction
   ({'PV':0.45,'SST':0.09,'VIP':5.6}) -- PV/VIP decayed toward 0Hz over a
   long window. This proved the instability was in the HDP-OFF baseline
   itself, not in K_HDP. DRIVE_CORRECTION_BY_CELL_TYPE below was
   re-derived via per-cell-type bisection against a 10s window at K_HDP=0,
   and verified STATIONARY over 20s (flat rates, no drift) before any HDP
   plasticity was re-introduced.

With that stationary baseline, K_HDP was re-swept jointly with K_ctrl/
barrier_c/barrier_d (the H-restoring-force terms): the original
K_ctrl=0.5/barrier=0.01 pair never holds past ~5-10s at any K_HDP>0;
K_ctrl=5.0 does. RECOMMENDED_K_HDP=0.01 with K_ctrl=5.0 is verified stable
over 20s: rates flat in-band, H pinned at ~1.0000-1.0006, weight
saturation ~10.5% (not pinned to floor/ceiling).

This sweep uses N=250 (same canonical layer/cell-type structure, scaled
down from N=1000) so each run is fast, to cheaply screen K_HDP x tau_0_ms
before committing to an expensive 1000-neuron run with these parameters.

Stability metric: compare early-window vs late-window rate-by-type and H
stats within a single run. "Stable" = late values stay within the target
band and don't drift far from the early values (no runaway).

Usage: PYTHONPATH=. python3 scripts/hdp_100_stability_sweep.py
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

OUTPUT_DIR = Path("outputs/hdp_100_stability_sweep")

LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6"]
ZBANDS = {
    "L1": (0.00, 0.10), "L2": (0.10, 0.35), "L3": (0.35, 0.55),
    "L4": (0.55, 0.65), "L5": (0.65, 0.85), "L6": (0.85, 1.00),
}
LAYER_CELL_TYPE_FRAC = {
    "L1": {"E": 0.50, "PV": 0.00, "SST": 0.15, "VIP": 0.35},
    "L2": {"E": 0.65, "PV": 0.20, "SST": 0.10, "VIP": 0.05},
    "L3": {"E": 0.80, "PV": 0.08, "SST": 0.08, "VIP": 0.04},
    "L4": {"E": 0.75, "PV": 0.18, "SST": 0.04, "VIP": 0.03},
    "L5": {"E": 0.88, "PV": 0.06, "SST": 0.04, "VIP": 0.02},
    "L6": {"E": 0.90, "PV": 0.05, "SST": 0.03, "VIP": 0.02},
}
BASE_DRIVE_BY_CELL_TYPE = {"E": 4.0, "PV": 4.0, "SST": 4.0, "VIP": 4.0}

# Re-derived at N=250 (the old N=1000-tuned {'PV':0.45,'SST':0.09,'VIP':5.6}
# does NOT generalize -- see module docstring). Found by per-cell-type
# bisection search against a 10s window with K_HDP=0 (HDP off, plasticity
# frozen): verified STATIONARY (no drift) over a 20s window at K_HDP=0 --
# this is the prerequisite fixed point HDP is layered on top of.
DRIVE_CORRECTION_BY_CELL_TYPE = {
    "E": 1.0, "PV": 0.8757734374999999, "SST": 0.06492187500000002, "VIP": 5.7509375,
}

N_NEURONS = 250
SEED = 0
DT_MS = 0.5
DURATION_MS = 20000.0
EARLY_WINDOW_MS = 1000.0   # steps [0, EARLY_WINDOW_MS)
LATE_WINDOW_MS = 1000.0    # steps [DURATION_MS - LATE_WINDOW_MS, DURATION_MS)
TARGET_RATE_HZ = 5.0
TARGET_RATE_TOL_HZ = 2.5

# K_ctrl/barrier_c/barrier_d re-derived alongside K_HDP below: the original
# K_ctrl=0.5, barrier=0.01 pair (from the 1000-neuron 1000ms-window build)
# is NOT long-term stable at ANY K_HDP>0 tested here -- it always drifts
# into runaway by 5-10s. K_ctrl=5.0 (10x stronger H-restoring force) is
# what makes K_HDP=0.01 hold flat over 20s (see module docstring).
BASE_HDP_KWARGS = dict(
    H_min=0.1, H_max=10.0,
    alpha=0.01, beta=0.0, gamma=0.0, delta=0.0, C_spike=0.0,
    K_ctrl=5.0, barrier_c=0.01, barrier_d=0.01, barrier_eps=1.0e-3,
    w_floor=0.01, w_ceiling=10.0, H_boost_gain=4.0,
)

# Verified long-term stable (20s, flat rates in-band, H pinned at ~1.0000-
# 1.0006, weight_saturation_fraction~0.105 not pinned to floor/ceiling):
# K_HDP=0.01 with K_ctrl=5.0, barrier_c=barrier_d=0.01 above.
RECOMMENDED_K_HDP = 0.01

SWEEP_K_HDP = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]
SWEEP_TAU0_MS = [200.0]


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
        .probe(name="hdp_100_probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(builder)


def apply_drive_correction(model: "jtfne.core.Model") -> "jtfne.core.Model":
    labels = np.asarray(model.params["emitter"].labels)
    base_drive = np.asarray(model.params["emitter"].drive)
    corrected = base_drive.copy()
    for ct, factor in DRIVE_CORRECTION_BY_CELL_TYPE.items():
        corrected[labels == ct] = base_drive[labels == ct] * factor
    return model.with_emitter_parameters(drive_per_neuron=jnp.asarray(corrected))


def window_summary(labels: np.ndarray, spikes: np.ndarray, H_trace: np.ndarray,
                    w_trace: np.ndarray, dt_ms: float, w_floor: float, w_ceiling: float) -> dict[str, Any]:
    duration_s = spikes.shape[0] * dt_ms / 1000.0
    rate_by_type = {
        ct: float(spikes[:, labels == ct].sum() / ((labels == ct).sum() * duration_s))
        for ct in sorted(set(labels.tolist())) if (labels == ct).any()
    }
    saturated = (np.abs(w_trace[-1]) >= 0.999 * w_ceiling) | (np.abs(w_trace[-1]) <= 1.001 * w_floor)
    return {
        "rate_by_type_hz": rate_by_type,
        "rate_in_target_band": all(
            abs(r - TARGET_RATE_HZ) <= TARGET_RATE_TOL_HZ for r in rate_by_type.values()
        ),
        "H_mean": float(H_trace.mean()),
        "H_var": float(H_trace.var()),
        "weight_saturation_fraction": float(saturated.mean()),
    }


def run_one(model: "jtfne.core.Model", K_HDP: float, tau_0_ms: float) -> dict[str, Any]:
    hdp_kwargs = dict(BASE_HDP_KWARGS)
    hdp_kwargs["K_HDP"] = K_HDP
    hdp_kwargs["tau_0_ms"] = tau_0_ms
    n_steps = int(round(DURATION_MS / DT_MS))
    key = jax.random.PRNGKey(SEED)
    _, spikes, _, diagnostics = simulate_edge_recurrent_izhikevich_hdp(
        model.params["emitter"], model.params["edge_list"], n_steps, DT_MS, key, **hdp_kwargs,
    )
    labels = np.asarray(model.params["emitter"].labels)
    spikes_np = np.asarray(spikes)
    H_trace = np.asarray(diagnostics["H_trace"])
    w_trace = np.asarray(diagnostics["w_trace"])

    early_n = int(round(EARLY_WINDOW_MS / DT_MS))
    late_n = int(round(LATE_WINDOW_MS / DT_MS))
    early = window_summary(labels, spikes_np[:early_n], H_trace[:early_n], w_trace[:early_n],
                            DT_MS, hdp_kwargs["w_floor"], hdp_kwargs["w_ceiling"])
    late = window_summary(labels, spikes_np[-late_n:], H_trace[-late_n:], w_trace[-late_n:],
                           DT_MS, hdp_kwargs["w_floor"], hdp_kwargs["w_ceiling"])

    max_rate_drift_hz = max(
        abs(late["rate_by_type_hz"][ct] - early["rate_by_type_hz"][ct])
        for ct in early["rate_by_type_hz"]
    )
    blew_up = any(r > 100.0 for r in late["rate_by_type_hz"].values()) or late["H_mean"] > 3.0 or not np.isfinite(late["H_mean"])

    return {
        "K_HDP": K_HDP, "tau_0_ms": tau_0_ms,
        "early": early, "late": late,
        "max_rate_drift_hz": max_rate_drift_hz,
        "stable_long_term": (not blew_up) and late["rate_in_target_band"] and max_rate_drift_hz <= TARGET_RATE_TOL_HZ,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Building {N_NEURONS}-neuron laminar column ===")
    model = build_model()
    model = apply_drive_correction(model)

    results = []
    print(f"\n=== HDP stability sweep, duration={DURATION_MS}ms ===")
    for tau_0_ms in SWEEP_TAU0_MS:
        for K_HDP in SWEEP_K_HDP:
            r = run_one(model, K_HDP, tau_0_ms)
            results.append(r)
            print(f"K_HDP={K_HDP:<5} tau_0_ms={tau_0_ms:<6} "
                  f"early_rate={r['early']['rate_by_type_hz']} "
                  f"late_rate={r['late']['rate_by_type_hz']} "
                  f"drift={r['max_rate_drift_hz']:.2f}Hz "
                  f"H_late={r['late']['H_mean']:.4f} "
                  f"stable={r['stable_long_term']}")

    stable = [r for r in results if r["stable_long_term"]]
    print(f"\n{len(stable)}/{len(results)} configs stable over {DURATION_MS}ms.")
    if stable:
        best = min(stable, key=lambda r: r["max_rate_drift_hz"])
        print(f"Best stable config: K_HDP={best['K_HDP']}, tau_0_ms={best['tau_0_ms']}, "
              f"drift={best['max_rate_drift_hz']:.3f}Hz")
    else:
        print("No swept config held the target band over the full duration without drift/runaway.")

    with open(OUTPUT_DIR / "hdp_100_stability_sweep.json", "w") as f:
        json.dump({"duration_ms": DURATION_MS, "results": results}, f, indent=2, allow_nan=False)
    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
