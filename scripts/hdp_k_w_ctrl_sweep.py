"""Multi-seed check for DEFAULT_HDP's K_w_ctrl candidate values
(plans.json:hdp-k-w-ctrl-default-runaway-gap; jaxfne/hdp_network.py's
K_w_ctrl caveat comment).

Real finding (2026-07-14, ad-hoc, single-seed, 80s/40-chained-trials, custom
20-neuron all-to-all topology): DEFAULT_HDP's K_w_ctrl=0.0 lets |w|_mean grow
unboundedly; K_w_ctrl=0.001 (DEFAULT_HDP_V1_PFC_AAAB's value) over-corrects on
that topology, collapsing weight differentiation ~150x vs no-restoring.

Scope of THIS check (2026-07-15): a genuine multi-seed (5 seeds), single
continuous-run (not chained-trials) check at N=20 on the production
HDPColumnConfig/build_model path, testing K_w_ctrl in {0.0, 0.0001, 0.001} for
10s duration. This is deliberately NOT the full rigorous gate the original
finding called for (that needs the same chained-trial protocol, duration, and
topology as the original finding, matching the rigor of the existing
K_ctrl/rho_passive sweeps) -- it is a real, additional, but reduced-scope data
point, honestly labeled as such below.

Usage: PYTHONPATH=. python3 scripts/hdp_k_w_ctrl_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import jax

import jaxfne as jtfne
from jaxfne.hdp_network import (
    BASE_DRIVE_BY_CELL_TYPE_DEFAULT,
    BASE_HDP_KWARGS_DEFAULT,
    DEFAULT_HDP,
    HDPColumnConfig,
    build_model,
)

N_NEURONS = 20
DURATION_MS = 10_000.0
DT_MS = 0.5
SEEDS = [0, 1, 2, 3, 4]
K_W_CTRL_CANDIDATES = [0.0, 0.0001, 0.001]

OUT_DIR = Path("artifacts/hdp_k_w_ctrl_sweep")


def _run_one(k_w_ctrl: float, seed: int) -> dict:
    cfg = HDPColumnConfig(
        n_neurons=N_NEURONS,
        duration_ms=DURATION_MS,
        dt_ms=DT_MS,
        seed=seed,
        base_drive_by_cell_type=dict(BASE_DRIVE_BY_CELL_TYPE_DEFAULT),
    )
    model = build_model(cfg)
    emitter = model.params["emitter"]
    edges = model.params["edge_list"]

    hdp_kw = {**DEFAULT_HDP, "K_w_ctrl": k_w_ctrl}
    combined_kw = {**hdp_kw, **BASE_HDP_KWARGS_DEFAULT}

    n_steps = int(DURATION_MS / DT_MS)
    _, sig, _, diagnostics = jtfne.emitters.simulate_edge_recurrent_izhikevich_hdp(
        params=emitter, edges=edges, n_steps=n_steps, dt_ms=DT_MS,
        key=jax.random.PRNGKey(seed), **combined_kw,
    )

    w_trace = np.asarray(diagnostics["w_trace"])  # (n_steps, n_edges)
    H_trace = np.asarray(diagnostics["H_trace"])
    abs_w_mean_trace = np.mean(np.abs(w_trace), axis=1)  # (n_steps,)

    T = abs_w_mean_trace.shape[0]
    first_quarter = abs_w_mean_trace[: T // 4]
    last_quarter = abs_w_mean_trace[3 * T // 4 :]

    growth_ratio = float(np.mean(last_quarter) / np.mean(first_quarter))
    final_w_std = float(np.std(w_trace[-1]))
    initial_w_std = float(np.std(w_trace[0]))
    differentiation_ratio = float(final_w_std / initial_w_std) if initial_w_std > 0 else float("nan")

    return dict(
        k_w_ctrl=k_w_ctrl, seed=seed,
        finite=bool(np.all(np.isfinite(w_trace)) and np.all(np.isfinite(H_trace))),
        abs_w_mean_first_quarter=float(np.mean(first_quarter)),
        abs_w_mean_last_quarter=float(np.mean(last_quarter)),
        growth_ratio=growth_ratio,
        differentiation_ratio=differentiation_ratio,
        H_mean_final=float(H_trace[-1].mean()),
        mean_rate_hz=float(np.mean(sig) / (DT_MS * 1e-3)),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for k_w_ctrl in K_W_CTRL_CANDIDATES:
        for seed in SEEDS:
            r = _run_one(k_w_ctrl, seed)
            results.append(r)
            print(f"K_w_ctrl={k_w_ctrl:<8} seed={seed}  growth_ratio={r['growth_ratio']:.3f}  "
                  f"diff_ratio={r['differentiation_ratio']:.3f}  finite={r['finite']}  "
                  f"rate={r['mean_rate_hz']:.2f}Hz")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print("\n=== Per-candidate summary (mean +/- std across 5 seeds) ===")
    summary = {}
    for k_w_ctrl in K_W_CTRL_CANDIDATES:
        rows = [r for r in results if r["k_w_ctrl"] == k_w_ctrl]
        growth = [r["growth_ratio"] for r in rows]
        diff = [r["differentiation_ratio"] for r in rows]
        all_finite = all(r["finite"] for r in rows)
        summary[k_w_ctrl] = {
            "growth_ratio_mean": float(np.mean(growth)),
            "growth_ratio_std": float(np.std(growth)),
            "differentiation_ratio_mean": float(np.mean(diff)),
            "differentiation_ratio_std": float(np.std(diff)),
            "all_finite": all_finite,
        }
        print(f"K_w_ctrl={k_w_ctrl:<8} growth_ratio={np.mean(growth):.3f}+/-{np.std(growth):.3f}  "
              f"diff_ratio={np.mean(diff):.3f}+/-{np.std(diff):.3f}  all_finite={all_finite}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
