"""Small-scale H-dynamics balance sweep: 10 random all-to-all networks each
at N in {10, 30, 70, 100} (40 networks total), pure H dynamics in isolation.

Goal: with weights frozen (K_HDP=0, no plasticity) and rate/weight-spending
disabled (gamma=delta=0), the H_i ODE reduces to its three remaining named
terms -- income (alpha*I_syn, "a"), constant offset (beta, "b"), and the
linear equilibrium controller (K_ctrl*(1-H), "K") plus the symmetric
double-barrier safety potential (barrier_c=barrier_d, "c") -- so any
observed H steady-state mean/variance is attributable only to a, b, K, c.
This isolates "just H dynamics" per the user's request, independent of the
plasticity/rate-feedback loop characterized in hdp_synaptic_channel_trace.py.

For each network size N, sweep K_ctrl over K_GRID (a, b, c held fixed at
hdp_defaults-style values) across the 10 random all-to-all networks, average
H-tail mean/variance over networks, and pick the K* that best balances H
(closest to its 1.0 equilibrium with lowest variance). Comparing K* across
N estimates how the controller gain must scale with network size.

Usage: PYTHONPATH=. python3 scripts/hdp_small_scale_balance_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import jaxfne as jtfne

OUTPUT_DIR = Path("outputs/hdp_small_scale_balance_sweep")

N_LIST = [10, 30, 70, 100]
N_NETWORKS_PER_SIZE = 10
K_GRID = [0.02, 0.05, 0.1, 0.3, 1.0]

# Fixed terms (not swept this round -- see report caveat).
ALPHA_A = 0.01      # "a": income gain on I_syn
BETA_B = 0.0        # "b": constant H offset
BARRIER_C = 0.01    # "c": symmetric double-barrier strength (barrier_d = barrier_c)
TAU_0_MS = 100.0    # known-stable HDP time constant (NOT the pathological 20.0)
DT_MS = 0.5
WEIGHT_RANGE = (0.01, 0.05)  # uniform random all-to-all excitatory weight

# H's relaxation timescale is tau_0_ms/K_ctrl -- a fixed 500ms duration looked
# "flat across K" in the first pass because low-K runs (e.g. tau_0/K=5000ms)
# never left their transient within 500ms. Scale duration per K so every run
# actually reaches steady state, with a fixed safety floor.
RELAXATION_MULTIPLE = 8.0
MIN_DURATION_MS = 1000.0
TAIL_FRACTION = 0.3  # last 30% of each run's (K-dependent) duration


def duration_for_k(k_ctrl: float) -> float:
    return max(MIN_DURATION_MS, RELAXATION_MULTIPLE * TAU_0_MS / max(k_ctrl, 1e-6))


def build_all_to_all(n: int, seed: int):
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    W = rng.uniform(WEIGHT_RANGE[0], WEIGHT_RANGE[1], size=(n, n)).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    return jnp.asarray(W)


def run_one(n: int, seed: int, k_ctrl: float) -> dict[str, float]:
    import jax

    from jaxfne.emitters import (
        izhikevich_params_from_labels,
        make_edge_list_from_dense,
        simulate_edge_recurrent_izhikevich_hdp,
    )

    params = izhikevich_params_from_labels(["E"] * n)
    edges = make_edge_list_from_dense(build_all_to_all(n, seed))
    key = jax.random.PRNGKey(seed)
    duration_ms = duration_for_k(k_ctrl)
    n_steps = int(round(duration_ms / DT_MS))
    tail_steps = int(round(n_steps * TAIL_FRACTION))

    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        params, edges, n_steps, DT_MS, key,
        alpha=ALPHA_A, beta=BETA_B, gamma=0.0, delta=0.0,
        K_HDP=0.0, K_ctrl=k_ctrl, barrier_c=BARRIER_C, barrier_d=BARRIER_C,
        tau_0_ms=TAU_0_MS,
    )
    H_tail = np.asarray(diag["H_trace"])[-tail_steps:]
    return {"H_mean": float(H_tail.mean()), "H_var": float(H_tail.var())}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[float, list[dict[str, float]]]] = {}

    print(f"=== 40-network H-dynamics balance sweep (a={ALPHA_A}, b={BETA_B}, c={BARRIER_C}, K_HDP=0 frozen) ===")
    for n in N_LIST:
        results[n] = {k: [] for k in K_GRID}
        for k_ctrl in K_GRID:
            for seed in range(N_NETWORKS_PER_SIZE):
                r = run_one(n, seed, k_ctrl)
                results[n][k_ctrl].append(r)
            means = [r["H_mean"] for r in results[n][k_ctrl]]
            varz = [r["H_var"] for r in results[n][k_ctrl]]
            print(
                f"N={n:>4}  K={k_ctrl:<5}  H_mean(avg over 10 nets)={np.mean(means):.5f}  "
                f"H_var(avg over 10 nets)={np.mean(varz):.6f}  H_mean(std across nets)={np.std(means):.5f}"
            )

    # Pick K* per N: minimize |H_mean-1| + H_var (network-averaged).
    summary = []
    for n in N_LIST:
        scored = []
        for k_ctrl in K_GRID:
            means = [r["H_mean"] for r in results[n][k_ctrl]]
            varz = [r["H_var"] for r in results[n][k_ctrl]]
            h_mean_avg = float(np.mean(means))
            h_var_avg = float(np.mean(varz))
            score = abs(h_mean_avg - 1.0) + h_var_avg
            scored.append((score, k_ctrl, h_mean_avg, h_var_avg))
        scored.sort(key=lambda x: x[0])
        best_score, best_k, best_mean, best_var = scored[0]
        summary.append({
            "N": n, "K_star": best_k, "H_mean_at_K_star": best_mean,
            "H_var_at_K_star": best_var, "balance_score": best_score,
        })

    print("\n=== Per-N balance summary (K* minimizes |H_mean-1| + H_var) ===")
    for s in summary:
        print(f"  N={s['N']:>4}  K*={s['K_star']:<5}  H_mean={s['H_mean_at_K_star']:.5f}  H_var={s['H_var_at_K_star']:.6f}")

    # Scale-effect-on-K estimate: log-log slope of K* vs N (only meaningful if K* varies).
    Ns = np.array([s["N"] for s in summary], dtype=float)
    Ks = np.array([s["K_star"] for s in summary], dtype=float)
    if len(set(Ks.tolist())) > 1:
        slope, intercept = np.polyfit(np.log(Ns), np.log(Ks), 1)
        print(f"\nEstimated K* ~ N^{slope:.3f} (log-log fit, intercept={intercept:.3f})")
    else:
        slope = 0.0
        print(f"\nK* is constant ({Ks[0]}) across all N in this grid -- no resolvable scale effect at this grid resolution.")

    payload = {
        "fixed_params": {"a_alpha": ALPHA_A, "b_beta": BETA_B, "c_barrier": BARRIER_C, "K_HDP": 0.0, "tau_0_ms": TAU_0_MS},
        "K_grid": K_GRID,
        "n_networks_per_size": N_NETWORKS_PER_SIZE,
        "raw": {str(n): {str(k): results[n][k] for k in K_GRID} for n in N_LIST},
        "summary": summary,
        "K_star_vs_N_loglog_slope": float(slope),
    }
    with open(OUTPUT_DIR / "hdp_small_scale_balance_sweep.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    _plot(results, summary, OUTPUT_DIR / "hdp_small_scale_balance_sweep.png")
    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")
    print(
        "\nCAVEAT: this sweep only swept K (alpha=a, beta=b, barrier_c=c held fixed at "
        f"a={ALPHA_A}, b={BETA_B}, c={BARRIER_C}). A full a/b/c grid was out of scope for "
        "this pass -- flag explicitly if a/b/c also need to be swept per N."
    )


def _plot(results: dict[int, dict[float, list[dict[str, float]]]], summary: list[dict[str, Any]], out_path: Path) -> None:
    H_mean_by_n = {n: [np.mean([r["H_mean"] for r in results[n][k]]) for k in K_GRID] for n in N_LIST}
    H_var_by_n = {n: [np.mean([r["H_var"] for r in results[n][k]]) for k in K_GRID] for n in N_LIST}
    fig = jtfne.vis.balance_sweep_by_network_size(K_GRID, H_mean_by_n, H_var_by_n)
    fig.savefig(out_path, dpi=150)
    jtfne.vis.close_all()


if __name__ == "__main__":
    main()
