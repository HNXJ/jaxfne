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
over 20s (and over a 5-seed multi-seed gate): rates flat in-band, H pinned
at ~1.0000-1.0029, weight saturation ~10.5% (not pinned to floor/ceiling).
These are exactly jaxfne.hdp_network.DEFAULT_HDP -- the proposed 0.4.6
freeze point.

N=250 (same canonical layer/cell-type structure, scaled down from N=1000
via the shared jaxfne.hdp_network builder -- not a separate hand-copied
build function) so each run is fast, to cheaply screen K_HDP x tau_0_ms
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

import numpy as np

from jaxfne.hdp_network import (
    HDPColumnConfig, build_model, apply_drive_correction, run,
    BASE_HDP_KWARGS_DEFAULT, DEFAULT_HDP,
)

OUTPUT_DIR = Path("outputs/hdp_100_stability_sweep")

N_NEURONS = 250
SEED = 0
DT_MS = 0.5
DURATION_MS = 20000.0
EARLY_WINDOW_MS = 1000.0   # steps [0, EARLY_WINDOW_MS)
LATE_WINDOW_MS = 1000.0    # steps [DURATION_MS - LATE_WINDOW_MS, DURATION_MS)
TARGET_RATE_HZ = 5.0
TARGET_RATE_TOL_HZ = 2.5

CFG = HDPColumnConfig(n_neurons=N_NEURONS, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED,
                       probe_name="hdp_100_probe")

# Re-exported for any callers/notebooks that still import these names
# directly from this module (kept identical to jaxfne.hdp_network's
# defaults -- this script no longer hand-maintains its own copy).
DRIVE_CORRECTION_BY_CELL_TYPE = dict(CFG.drive_correction_by_cell_type)
RECOMMENDED_K_HDP = 0.01

BASE_HDP_KWARGS = dict(BASE_HDP_KWARGS_DEFAULT)
BASE_HDP_KWARGS.update({k: v for k, v in DEFAULT_HDP.items() if k != "K_HDP" and k != "tau_0_ms"})

SWEEP_K_HDP = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]
SWEEP_TAU0_MS = [200.0]


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
    out = run(model, CFG, duration_ms=DURATION_MS, seed=SEED, hdp_kwargs=hdp_kwargs)
    labels = np.asarray(model.params["emitter"].labels)
    spikes_np = np.asarray(out["spikes"])
    H_trace = np.asarray(out["diagnostics"]["H_trace"])
    w_trace = np.asarray(out["diagnostics"]["w_trace"])

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
    print(f"=== Building {N_NEURONS}-neuron laminar column (via jaxfne.hdp_network) ===")
    model = build_model(CFG)
    model = apply_drive_correction(model, CFG)

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
