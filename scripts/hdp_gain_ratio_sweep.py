"""Gain-ratio (K_ctrl vs alpha+gamma+delta) sweep for the HDP linear controller.

Diagnostic only -- not a tuned operating point. Motivated by a discrepancy
observed in spectrolaminar_tfne_izhikevich_pipeline.py: a short diagnostic
window showed H pinned near 1.0 (controller dominating the resource state),
while the full-duration motif run at the same hdp_kwargs drifted to
H_final_mean=4.88 with pathological synchrony (kappa=0.85, 177-294 Hz). This
sweeps R = K_ctrl / (alpha + gamma + delta) across both window lengths to see
where the controller stops erasing the resource-estimator signal without
reintroducing the tau_0_ms=20 instability.

Usage: PYTHONPATH=. python3 scripts/hdp_gain_ratio_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import jaxfne as jtfne
from spectrolaminar_tfne_izhikevich_pipeline import (
    build_config,
    build_model,
    compute_hdp_state_diagnostics,
    classify_hdp_state,
    run_hdp,
    summarize_run,
)

OUTPUT_DIR = Path("outputs/hdp_spectrolaminar_pipeline")

K_CTRL_GRID = [1.0, 0.1, 0.05, 0.02, 0.01, 0.0]
SHORT_DURATION_MS = 500.0
FULL_TAU_0_MS = 20.0  # the previously-identified unstable point; deliberately held fixed


def point_summary(
    model: "jtfne.core.Model", cfg: dict[str, Any], *, K_ctrl: float, duration_ms: float
) -> dict[str, Any]:
    overrides = {
        "K_HDP": 1.0,
        "tau_0_ms": FULL_TAU_0_MS,
        "K_ctrl": K_ctrl,
        "barrier_c": 0.01,
        "barrier_d": 1.0,
    }
    run = run_hdp(model, cfg, seed=cfg["seed"], duration_ms=duration_ms, overrides=overrides)
    summary = summarize_run(model, cfg, run)
    per_type = compute_hdp_state_diagnostics(model, cfg, run)
    classification = classify_hdp_state(per_type, cfg["target_rate_hz"])

    H_trace = np.asarray(run["diagnostics"]["H_trace"])
    mean_pressure = float((H_trace - 1.0).mean())
    var_H = float(H_trace.var())

    return {
        "K_ctrl": K_ctrl,
        "duration_ms": duration_ms,
        "mean_H_minus_1": mean_pressure,
        "var_H": var_H,
        "kappa_synchrony": summary["kappa_synchrony"],
        "overall_rate_hz": summary["overall_rate_hz"],
        "rate_by_type_hz": summary["rate_by_type_hz"],
        "weight_saturation_fraction": summary["weight_saturation_fraction"],
        "classification": {ct: c["classification"] for ct, c in classification.items()},
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config()

    print("=== Building laminar column Model (one-time construct) ===")
    model = build_model(cfg)

    full_duration_ms = cfg["duration_ms"]
    results = {"short": [], "full": []}

    for K_ctrl in K_CTRL_GRID:
        R = K_ctrl / (cfg["hdp_defaults"]["alpha"] + cfg["hdp_defaults"]["gamma"] + cfg["hdp_defaults"]["delta"])
        print(f"\n--- K_ctrl={K_ctrl:.3f}  R={R:.2f} ---")

        short = point_summary(model, cfg, K_ctrl=K_ctrl, duration_ms=SHORT_DURATION_MS)
        results["short"].append(short)
        print(
            f"  short ({SHORT_DURATION_MS:.0f}ms): mean(H-1)={short['mean_H_minus_1']:+.4f}  "
            f"var(H)={short['var_H']:.4f}  SST={short['rate_by_type_hz'].get('SST', float('nan')):.2f}Hz  "
            f"PV={short['rate_by_type_hz'].get('PV', float('nan')):.2f}Hz  kappa={short['kappa_synchrony']:.3f}  "
            f"class={short['classification']}"
        )

        full = point_summary(model, cfg, K_ctrl=K_ctrl, duration_ms=full_duration_ms)
        results["full"].append(full)
        print(
            f"  full ({full_duration_ms:.0f}ms):  mean(H-1)={full['mean_H_minus_1']:+.4f}  "
            f"var(H)={full['var_H']:.4f}  SST={full['rate_by_type_hz'].get('SST', float('nan')):.2f}Hz  "
            f"PV={full['rate_by_type_hz'].get('PV', float('nan')):.2f}Hz  kappa={full['kappa_synchrony']:.3f}  "
            f"class={full['classification']}"
        )

    with open(OUTPUT_DIR / "hdp_gain_ratio_sweep.json", "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)

    _plot(results, OUTPUT_DIR / "hdp_gain_ratio_sweep.png")
    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")


def _plot(results: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    K_ctrl_by_window = {w: [p["K_ctrl"] for p in pts] for w, pts in results.items()}
    mean_H_minus_1_by_window = {w: [p["mean_H_minus_1"] for p in pts] for w, pts in results.items()}
    SST_rate_by_window = {w: [p["rate_by_type_hz"].get("SST", float("nan")) for p in pts] for w, pts in results.items()}
    kappa_synchrony_by_window = {w: [p["kappa_synchrony"] for p in pts] for w, pts in results.items()}

    fig = jtfne.vis.gain_ratio_sweep_panels(
        K_ctrl_by_window, mean_H_minus_1_by_window, SST_rate_by_window, kappa_synchrony_by_window,
    )
    fig.savefig(out_path, dpi=150)
    jtfne.vis.close_all()


if __name__ == "__main__":
    main()
