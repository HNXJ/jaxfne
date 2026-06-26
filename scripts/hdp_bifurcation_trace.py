"""Time-resolved bifurcation trace for the K_ctrl=0 / tau_0_ms=20 HDP instability.

The gain-ratio sweep (hdp_gain_ratio_sweep.py) showed the *uncontrolled*
kernel (K_ctrl=0 -- i.e. unmodified pre-existing HDP) is stable over a 500ms
window but runs away into pathological synchrony (kappa~0.94) by 1000ms at
tau_0_ms=20. This script bins the same run into 50ms windows and logs, per
window: rate_E/PV/SST/VIP, kappa_synchrony, mean_weight, weight_variance,
H_mean, H_variance -- to locate the bifurcation time and determine which
variable departs from baseline first (weights, synchrony, or H).

Usage: PYTHONPATH=. python3 scripts/hdp_bifurcation_trace.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import jaxfne as jtfne
from spectrolaminar_tfne_izhikevich_pipeline import build_config, build_model, run_hdp

OUTPUT_DIR = Path("outputs/hdp_spectrolaminar_pipeline")
BIN_MS = 50.0
TRACE_DURATION_MS = 1500.0  # extend past the 1000ms motif duration to see the full runaway
TAU_0_MS = 20.0


def trace(model: "jtfne.core.Model", cfg: dict[str, Any]) -> list[dict[str, Any]]:
    overrides = {"K_HDP": 1.0, "tau_0_ms": TAU_0_MS, "K_ctrl": 0.0, "barrier_c": 0.0, "barrier_d": 0.0}
    run = run_hdp(model, cfg, seed=cfg["seed"], duration_ms=TRACE_DURATION_MS, overrides=overrides)

    labels = np.asarray(model.params["emitter"].labels)
    spikes = np.asarray(run["spikes"])  # (n_steps, n_neurons)
    H_trace = np.asarray(run["diagnostics"]["H_trace"])  # (n_steps, n_neurons)
    w_trace = np.asarray(run["diagnostics"]["w_trace"])  # (n_steps, n_edges)
    dt_ms = cfg["dt_ms"]
    n_steps = spikes.shape[0]
    bin_steps = max(1, int(round(BIN_MS / dt_ms)))
    n_bins = n_steps // bin_steps

    points = []
    for b in range(n_bins):
        s0, s1 = b * bin_steps, (b + 1) * bin_steps
        t_ms = (s0 + s1) / 2.0 * dt_ms
        spikes_bin = spikes[s0:s1]
        dur_s = (s1 - s0) * dt_ms / 1000.0

        rate_by_type = {}
        for cell_type in sorted(set(labels.tolist())):
            mask = labels == cell_type
            if not mask.any():
                continue
            rate_by_type[cell_type] = float(spikes_bin[:, mask].sum() / (mask.sum() * dur_s))

        kappa = float(jtfne.kappa_synchrony(spikes_bin, dt_ms=dt_ms))
        H_bin = H_trace[s0:s1]
        w_bin = w_trace[s0:s1]

        points.append({
            "t_ms": t_ms,
            "rate_by_type_hz": rate_by_type,
            "kappa_synchrony": kappa,
            "mean_weight": float(np.abs(w_bin).mean()),
            "weight_variance": float(w_bin.var()),
            "H_mean": float(H_bin.mean()),
            "H_variance": float(H_bin.var()),
        })
        print(
            f"t={t_ms:>6.1f}ms  rates={ {k: round(v, 1) for k, v in rate_by_type.items()} }  "
            f"kappa={kappa:.3f}  mean|w|={points[-1]['mean_weight']:.3f}  "
            f"var(w)={points[-1]['weight_variance']:.4f}  H_mean={points[-1]['H_mean']:.4f}  "
            f"var(H)={points[-1]['H_variance']:.4f}"
        )
    return points


def _first_departure_bin(points: list[dict[str, Any]], key_fn, baseline_n: int = 4, n_sigma: float = 3.0) -> int | None:
    """First bin index where key_fn(point) departs > n_sigma*std from the mean of the first baseline_n bins."""
    baseline_vals = np.array([key_fn(p) for p in points[:baseline_n]])
    mu, sigma = baseline_vals.mean(), baseline_vals.std()
    if sigma == 0:
        sigma = 1e-9
    for i, p in enumerate(points):
        if abs(key_fn(p) - mu) > n_sigma * sigma:
            return i
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config()

    print("=== Building laminar column Model (one-time construct) ===")
    model = build_model(cfg)

    print(f"\n=== Time-resolved trace: K_ctrl=0, tau_0_ms={TAU_0_MS}, duration={TRACE_DURATION_MS}ms, bin={BIN_MS}ms ===")
    points = trace(model, cfg)

    departures = {
        "kappa_synchrony": _first_departure_bin(points, lambda p: p["kappa_synchrony"]),
        "weight_variance": _first_departure_bin(points, lambda p: p["weight_variance"]),
        "H_variance": _first_departure_bin(points, lambda p: p["H_variance"]),
        "SST_rate": _first_departure_bin(points, lambda p: p["rate_by_type_hz"].get("SST", float("nan"))),
    }
    print("\n=== First-departure bin index (>3 sigma from first 4 bins' baseline) ===")
    for key, idx in departures.items():
        t = points[idx]["t_ms"] if idx is not None else None
        print(f"  {key}: bin {idx} (t={t}ms)" if idx is not None else f"  {key}: no departure detected")

    with open(OUTPUT_DIR / "hdp_bifurcation_trace.json", "w") as f:
        json.dump({"points": points, "first_departure_bin": departures}, f, indent=2, allow_nan=False)

    _plot(points, departures, OUTPUT_DIR / "hdp_bifurcation_trace.png")
    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")


def _plot(points: list[dict[str, Any]], departures: dict[str, int | None], out_path: Path) -> None:
    t = [p["t_ms"] for p in points]
    rate_by_type_hz = {
        cell_type: [p["rate_by_type_hz"].get(cell_type, float("nan")) for p in points]
        for cell_type in sorted(points[0]["rate_by_type_hz"].keys())
    }
    fig = jtfne.vis.bifurcation_state_trace_panels(
        t,
        rate_by_type_hz,
        [p["kappa_synchrony"] for p in points],
        [p["mean_weight"] for p in points],
        [p["weight_variance"] for p in points],
        [p["H_mean"] for p in points],
        [p["H_variance"] for p in points],
        departures,
        title="Bifurcation trace: K_ctrl=0, tau_0_ms=20",
    )
    fig.savefig(out_path, dpi=150)
    jtfne.vis.close_all()


if __name__ == "__main__":
    main()
