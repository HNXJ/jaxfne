"""High-resolution dH-component trace across the K_ctrl=0/tau_0_ms=20 bifurcation.

hdp_bifurcation_trace.py located the bifurcation at ~350-375ms using 50ms
bins, but couldn't separate "H moves first" from "H and w move together"
at that resolution, nor identify which additive term of dH_i/dt (income,
rate-spending, weight-spending, K_ctrl, barrier) is responsible. This script
uses the new record_dH_components=True kernel output (emitters.py) to log
each term separately at 5ms resolution across the critical 250-500ms window.

cfg["hdp_defaults"] has delta=0.0, so dH_weight (-delta*W_burden) is
identically zero by construction here -- it cannot be the destabilizing
term in this configuration regardless of what the trace shows. The real
candidates are dH_income (alpha*I_syn) and dH_rate (-gamma*r).

Usage: PYTHONPATH=. python3 scripts/hdp_dH_component_trace.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import simulate_edge_recurrent_izhikevich_hdp
from spectrolaminar_tfne_izhikevich_pipeline import build_config, build_model

OUTPUT_DIR = Path("outputs/hdp_spectrolaminar_pipeline")
BIN_MS = 5.0
WINDOW_START_MS = 250.0
WINDOW_END_MS = 500.0
TAU_0_MS = 20.0


def main() -> None:
    import jax

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config()

    print("=== Building laminar column Model (one-time construct) ===")
    model = build_model(cfg)

    hdp_kwargs = dict(cfg["hdp_defaults"])
    hdp_kwargs.update({"K_HDP": 1.0, "tau_0_ms": TAU_0_MS, "K_ctrl": 0.0, "barrier_c": 0.0, "barrier_d": 0.0})
    print(f"hdp_kwargs (delta={hdp_kwargs['delta']} -> dH_weight is identically zero): {hdp_kwargs}")

    dt_ms = cfg["dt_ms"]
    n_steps = int(round(WINDOW_END_MS / dt_ms))
    key = jax.random.PRNGKey(cfg["seed"])

    _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        model.params["emitter"], model.params["edge_list"], n_steps, dt_ms, key,
        record_dH_components=True, **hdp_kwargs,
    )

    H_trace = np.asarray(diag["H_trace"])
    w_trace = np.asarray(diag["w_trace"])
    dH_income = np.asarray(diag["dH_income_trace"])
    dH_rate = np.asarray(diag["dH_rate_trace"])
    dH_weight = np.asarray(diag["dH_weight_trace"])
    dH_ctrl = np.asarray(diag["dH_ctrl_trace"])
    dH_barrier = np.asarray(diag["dH_barrier_trace"])

    bin_steps = max(1, int(round(BIN_MS / dt_ms)))
    start_step = int(round(WINDOW_START_MS / dt_ms))
    n_bins = (n_steps - start_step) // bin_steps

    points = []
    for b in range(n_bins):
        s0 = start_step + b * bin_steps
        s1 = s0 + bin_steps
        t_ms = (s0 + s1) / 2.0 * dt_ms
        points.append({
            "t_ms": t_ms,
            "H_mean": float(H_trace[s0:s1].mean()),
            "var_H": float(H_trace[s0:s1].var()),
            "mean_abs_w": float(np.abs(w_trace[s0:s1]).mean()),
            "var_w": float(w_trace[s0:s1].var()),
            "dH_income_mean": float(dH_income[s0:s1].mean()),
            "dH_rate_mean": float(dH_rate[s0:s1].mean()),
            "dH_weight_mean": float(dH_weight[s0:s1].mean()),
            "dH_ctrl_mean": float(dH_ctrl[s0:s1].mean()),
            "dH_barrier_mean": float(dH_barrier[s0:s1].mean()),
        })
        p = points[-1]
        print(
            f"t={p['t_ms']:>6.1f}ms  H_mean={p['H_mean']:.5f}  var(H)={p['var_H']:.6f}  "
            f"mean|w|={p['mean_abs_w']:.4f}  var(w)={p['var_w']:.5f}  "
            f"dH_income={p['dH_income_mean']:+.5f}  dH_rate={p['dH_rate_mean']:+.5f}  "
            f"dH_weight={p['dH_weight_mean']:+.5f}"
        )

    # First-departure detection for H itself and for each dH component's magnitude.
    baseline_n = 6
    def first_departure(key_fn, n_sigma=3.0):
        baseline = np.array([key_fn(p) for p in points[:baseline_n]])
        mu, sigma = baseline.mean(), baseline.std()
        sigma = sigma if sigma > 0 else 1e-12
        for i, p in enumerate(points):
            if abs(key_fn(p) - mu) > n_sigma * sigma:
                return i
        return None

    departures = {
        "H_mean": first_departure(lambda p: p["H_mean"]),
        "var_H": first_departure(lambda p: p["var_H"]),
        "var_w": first_departure(lambda p: p["var_w"]),
        "dH_income": first_departure(lambda p: p["dH_income_mean"]),
        "dH_rate": first_departure(lambda p: p["dH_rate_mean"]),
    }
    print("\n=== First-departure bin (>3 sigma vs first 6 bins' baseline) ===")
    for k, idx in departures.items():
        t = points[idx]["t_ms"] if idx is not None else None
        print(f"  {k}: {'t=' + str(t) + 'ms' if idx is not None else 'no departure'}")

    with open(OUTPUT_DIR / "hdp_dH_component_trace.json", "w") as f:
        json.dump({"points": points, "first_departure": departures}, f, indent=2, allow_nan=False)

    _plot(points, departures, OUTPUT_DIR / "hdp_dH_component_trace.png")
    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")


def _plot(points: list[dict[str, Any]], departures: dict[str, int | None], out_path: Path) -> None:
    t = [p["t_ms"] for p in points]
    fig = jtfne.vis.dH_component_trace_panels(
        t,
        [p["H_mean"] for p in points],
        [p["var_H"] for p in points],
        [p["mean_abs_w"] for p in points],
        [p["var_w"] for p in points],
        [p["dH_income_mean"] for p in points],
        [p["dH_rate_mean"] for p in points],
        [p["dH_weight_mean"] for p in points],
        departures,
        window_start_ms=WINDOW_START_MS, window_end_ms=WINDOW_END_MS, bin_ms=BIN_MS,
    )
    fig.savefig(out_path, dpi=150)
    jtfne.vis.close_all()


if __name__ == "__main__":
    main()
