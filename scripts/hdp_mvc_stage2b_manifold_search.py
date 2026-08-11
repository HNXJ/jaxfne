#!/usr/bin/env python3
"""Stage 2b: read-only operating-manifold search for HDP-MVC.

Phases A–F: map reachable (r_E, r_I), trajectory nondegeneracy, J_W at promising
points, optional Izh heterogeneity, scientific AGSDR if justified, horizon qual.

No package / HDP-rule mutations. Stops before sustained-perturbation B/C.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

import jaxfne as jtfne
from jaxfne._model_tune import _model_with_parameters
from jaxfne.hdp_network import DEFAULT_HDP
from jaxfne.io import json_safe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "msvc_hdp_diagnostic"

DT_MS = 0.1
BURN_IN_MS = 20.0
SIM_SEED = 17
OPT_SEED = 42
MAP_DURATION_MS = 2000.0
REFERENCE_MS = 10_000.0
HORIZONS_MS = (500.0, 1000.0, 2000.0, 5000.0, 10_000.0)
TARGET_R_E_HZ = 15.0
TARGET_R_I_HZ = 10.0
PARAM_NAMES = ("m_EE", "m_EI", "m_IE", "m_II")
BOUNDS = (0.1, 5.0)
SOBOL_N = 64
FD_EPS = 0.05
EPS_PRIMARY = 0.15
EPS_CONFIRM = 0.20
IZH_HET_AMP = 0.05
J_MIN_FRO = 0.1
SIGMA2_MIN = 0.1
SIGMA_RATIO_MIN = 0.05
H_DOT_MAX = 1e-4
W_DOT_MAX = 1e-4
RATE_SAT_MAX = 100.0
HAMMING_MIN = 1e-4
NONDEG_SD_MIN_HZ = 0.5
MANIFOLD_2D_RATIO = 0.05
TARGET_L_OP_FEASIBLE = 0.08
SCIENTIFIC_GENERATIONS = 12
SCIENTIFIC_POPULATION = 16
STAT_TOL = {"rate_hz": 0.5, "H_mean": 0.0005}


def mcc3_config():
    return (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(
            name="V1",
            kind="cortical_column",
            n=10,
            cell_types={"E": 0.5, "PV": 0.5},
        )
        .cell_type_drives({"E": 8.0, "PV": 8.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="probe", modes=["spikes", "V_m"])
    )


def mcc3_specs():
    return {
        name: jtfne.edge_parameter(
            pre={"cell_type": pre},
            post={"cell_type": post},
            bounds=BOUNDS,
        )
        for name, pre, post in (
            ("m_EE", "E", "E"),
            ("m_EI", "E", "PV"),
            ("m_IE", "PV", "E"),
            ("m_II", "PV", "PV"),
        )
    }


def mcc3_runtime(enabled: bool) -> jtfne.RuntimeConfig:
    params = dict(DEFAULT_HDP)
    params.update({"H_min": 0.1, "H_max": 10.0, "w_min": -10.0, "w_max": 10.0})
    return jtfne.RuntimeConfig(
        enable_hdp=enabled,
        recurrent_backend="edge_list",
        jit=False,
        hdp_params=params if enabled else {},
    )


def fixed_pattern(n: int = 10) -> np.ndarray:
    raw = np.array(
        [0.12, -0.08, 0.05, -0.10, 0.06, -0.07, 0.09, -0.04, 0.03, -0.06],
        dtype=float,
    )
    return raw[:n] - raw[:n].mean()


def apply_drive_heterogeneity(model: jtfne.Model, eps: float) -> jtfne.Model:
    pattern = fixed_pattern()
    base = np.asarray(model.params["emitter"].drive, dtype=np.float32)
    scaled = base * (1.0 + eps * pattern).astype(np.float32)
    return jtfne.with_emitter_parameters(model, drive_per_neuron=jnp.asarray(scaled))


def apply_izh_heterogeneity(model: jtfne.Model, amp: float) -> jtfne.Model:
    em = model.params["emitter"]
    pat = fixed_pattern()
    pat_b = np.roll(pat, 3)
    a = np.asarray(em.a, dtype=np.float32) * (1.0 + amp * pat).astype(np.float32)
    b = np.asarray(em.b, dtype=np.float32) * (1.0 + amp * pat_b).astype(np.float32)
    return jtfne.with_emitter_parameters(
        model, a_per_neuron=jnp.asarray(a), b_per_neuron=jnp.asarray(b)
    )


def sobol_theta_samples(n: int, seed: int = 0) -> list[dict[str, float]]:
    engine = qmc.Sobol(d=4, scramble=True, seed=seed)
    unit = engine.random(n)
    lo, hi = BOUNDS
    scaled = qmc.scale(unit, lo, hi)
    out = []
    for row in scaled:
        out.append({name: float(v) for name, v in zip(PARAM_NAMES, row)})
    return out


def L_op(r_e: float, r_i: float) -> float:
    return float(
        ((r_e - TARGET_R_E_HZ) / TARGET_R_E_HZ) ** 2
        + ((r_i - TARGET_R_I_HZ) / TARGET_R_I_HZ) ** 2
    )


def ei_rates(spikes: np.ndarray, burn_in_ms: float = BURN_IN_MS) -> dict[str, float]:
    start = int(math.ceil(burn_in_ms / DT_MS))
    window = spikes[start:]
    r_e = float(window[:, :5].mean() * (1000.0 / DT_MS)) if window.size else 0.0
    r_i = float(window[:, 5:].mean() * (1000.0 / DT_MS)) if window.size else 0.0
    pop = float(window.mean() * (1000.0 / DT_MS)) if window.size else 0.0
    return {"population_hz": pop, "r_E_hz": r_e, "r_I_hz": r_i}


def spike_population_trajectory_stats(spikes: np.ndarray, cols: slice) -> dict[str, Any]:
    start = int(math.ceil(BURN_IN_MS / DT_MS))
    trains = (spikes[start:, cols] > 0.5).astype(np.uint8)
    n = trains.shape[1]
    if n < 2:
        return {
            "mean_pairwise_hamming": 0.0,
            "max_pairwise_correlation": 1.0,
            "identical_spike_trains": True,
        }
    hamming, corrs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            hamming.append(float(np.mean(trains[:, i] != trains[:, j])))
            ai = trains[:, i].astype(float)
            aj = trains[:, j].astype(float)
            if ai.std() == 0 and aj.std() == 0:
                corrs.append(1.0)
            elif ai.std() == 0 or aj.std() == 0:
                corrs.append(0.0)
            else:
                corrs.append(float(np.corrcoef(ai, aj)[0, 1]))
    return {
        "mean_pairwise_hamming": float(np.mean(hamming)),
        "max_pairwise_correlation": float(np.max(corrs)),
        "identical_spike_trains": bool(np.max(hamming) < 1e-12),
    }


def nondegeneracy_metrics(spikes: np.ndarray) -> dict[str, Any]:
    start = int(math.ceil(BURN_IN_MS / DT_MS))
    per_neuron = spikes[start:].mean(axis=0) * (1000.0 / DT_MS)
    e_traj = spike_population_trajectory_stats(spikes, slice(0, 5))
    i_traj = spike_population_trajectory_stats(spikes, slice(5, 10))
    traj_degenerate = e_traj["identical_spike_trains"] and i_traj["identical_spike_trains"]
    traj_pass = (
        e_traj["mean_pairwise_hamming"] >= HAMMING_MIN
        or i_traj["mean_pairwise_hamming"] >= HAMMING_MIN
    ) and not traj_degenerate
    sd_pass = float(np.std(per_neuron)) >= NONDEG_SD_MIN_HZ
    return {
        "per_neuron_hz": per_neuron.tolist(),
        "sd_across_neurons_hz": float(np.std(per_neuron)),
        "E_trajectory": e_traj,
        "I_trajectory": i_traj,
        "trajectory_nondegenerate": traj_pass,
        "sd_nondegenerate": sd_pass,
        "nondegenerate_hard": traj_pass or sd_pass,
        "permutation_degenerate": traj_degenerate,
    }


def baseline_hdp_drift(diag: dict[str, Any] | None) -> dict[str, Any]:
    if diag is None:
        return {"H_dot_mean_per_ms": None, "W_dot_mean_per_ms": None}
    start = int(math.ceil(BURN_IN_MS / DT_MS))
    out: dict[str, Any] = {}
    h_trace = diag.get("H_trace")
    if h_trace is not None:
        h = np.asarray(h_trace)
        h_series = h[start:, :].mean(axis=1) if h.ndim == 2 else h[start:]
        dh = np.diff(h_series) / DT_MS if len(h_series) > 1 else np.array([0.0])
        out["H_dot_mean_per_ms"] = float(np.mean(np.abs(dh)))
    else:
        out["H_dot_mean_per_ms"] = None
    w_trace = diag.get("w_trace")
    if w_trace is not None:
        w = np.asarray(w_trace)
        if w.ndim == 2 and w.shape[0] > start + 1:
            wg = np.mean(np.abs(w[start:, :]), axis=1)
            dw = np.diff(wg) / DT_MS
            out["W_dot_mean_per_ms"] = float(np.mean(np.abs(dw)))
        else:
            out["W_dot_mean_per_ms"] = 0.0
    else:
        out["W_dot_mean_per_ms"] = None
    return out


def evaluate_theta(
    hetero_model: jtfne.Model,
    specs: dict,
    theta: dict[str, float],
    duration_ms: float,
) -> dict[str, Any]:
    model = _model_with_parameters(hetero_model, theta, specs)
    sim = jtfne.simulation(
        duration_ms=duration_ms,
        dt_ms=DT_MS,
        seed=SIM_SEED,
        runtime=mcc3_runtime(True),
    )
    sig = model.simulate(sim)
    spikes = np.asarray(sig.spikes)
    rates = ei_rates(spikes)
    per = spikes[int(math.ceil(BURN_IN_MS / DT_MS)) :].mean(axis=0) * (1000.0 / DT_MS)
    nd = nondegeneracy_metrics(spikes)
    diag = model.last_hdp_diagnostics()
    drift = baseline_hdp_drift(diag)
    silent_e = bool(np.any(per[:5] < 0.01))
    silent_i = bool(np.any(per[5:] < 0.01))
    saturated = bool(np.max(per) > RATE_SAT_MAX)
    finite = bool(np.all(np.isfinite(per)))
    active = finite and not silent_e and not silent_i and not saturated
    return {
        "theta": theta,
        "duration_ms": duration_ms,
        **rates,
        "L_op": L_op(rates["r_E_hz"], rates["r_I_hz"]),
        "finite": finite,
        "silent_E": silent_e,
        "silent_I": silent_i,
        "saturated": saturated,
        "active": active,
        "nondegeneracy": nd,
        "baseline_hdp_drift": drift,
    }


def frobenius_jw(
    hetero_model: jtfne.Model,
    theta: dict[str, float],
    specs: dict,
    duration_ms: float,
) -> dict[str, Any]:
    j = np.zeros((2, 4), dtype=float)
    for col, name in enumerate(PARAM_NAMES):
        tp, tm = dict(theta), dict(theta)
        delta = max(FD_EPS * abs(theta[name]), 0.01)
        tp[name] = float(np.clip(theta[name] + delta, *BOUNDS))
        tm[name] = float(np.clip(theta[name] - delta, *BOUNDS))
        yp = evaluate_theta(hetero_model, specs, tp, duration_ms)
        ym = evaluate_theta(hetero_model, specs, tm, duration_ms)
        j[:, col] = (
            np.array([yp["r_E_hz"], yp["r_I_hz"]])
            - np.array([ym["r_E_hz"], ym["r_I_hz"]])
        ) / (tp[name] - tm[name] + 1e-12)
    u, s, vt = np.linalg.svd(j, full_matrices=False)
    rank2 = bool(s[1] >= SIGMA2_MIN and (s[1] / (s[0] + 1e-12)) >= SIGMA_RATIO_MIN)
    cols = {
        name: {
            "partial_r_E": float(j[0, k]),
            "partial_r_I": float(j[1, k]),
            "column_norm": float(np.linalg.norm(j[:, k])),
        }
        for k, name in enumerate(PARAM_NAMES)
    }
    return {
        "J_W": j.tolist(),
        "frobenius_norm": float(np.linalg.norm(j)),
        "singular_values": s.tolist(),
        "sigma2_over_sigma1": float(s[1] / (s[0] + 1e-12)),
        "numerical_rank_2_ok": rank2,
        "U": u.tolist(),
        "Vt": vt.tolist(),
        "columns": cols,
    }


def manifold_analysis(samples: list[dict[str, Any]]) -> dict[str, Any]:
    active = [s for s in samples if s["active"]]
    if len(active) < 3:
        return {"n_active": len(active), "dimension": "insufficient_data"}
    pts = np.array([[s["r_E_hz"], s["r_I_hz"]] for s in active], dtype=float)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ratio = float(evals[1] / (evals[0] + 1e-12)) if evals[0] > 0 else 0.0
    l_ops = np.array([s["L_op"] for s in active])
    best_idx = int(np.argmin(l_ops))
    best = active[best_idx]
    dist_eucl = np.linalg.norm(pts - np.array([TARGET_R_E_HZ, TARGET_R_I_HZ]), axis=1)
    return {
        "n_active": len(active),
        "n_total": len(samples),
        "covariance_eigenvalues": evals.tolist(),
        "lambda2_over_lambda1": ratio,
        "manifold_dimension": "2D" if ratio >= MANIFOLD_2D_RATIO else "1D",
        "r_E_range_hz": [float(pts[:, 0].min()), float(pts[:, 0].max())],
        "r_I_range_hz": [float(pts[:, 1].min()), float(pts[:, 1].max())],
        "min_L_op": float(l_ops.min()),
        "best_active_theta": best["theta"],
        "best_active_r_E_hz": best["r_E_hz"],
        "best_active_r_I_hz": best["r_I_hz"],
        "min_euclidean_distance_to_target_hz": float(dist_eucl.min()),
        "target_feasible_L_op": bool(l_ops.min() <= TARGET_L_OP_FEASIBLE),
        "target_feasible_interior": bool(dist_eucl.min() < 4.0),
    }


def select_promising(samples: list[dict[str, Any]], max_n: int = 8) -> list[dict[str, Any]]:
    active = [s for s in samples if s["active"]]
    if not active:
        return []
    chosen: list[dict[str, Any]] = []
    used_theta: list[tuple] = []

    def key_theta(s: dict) -> tuple:
        t = s["theta"]
        return tuple(round(t[n], 4) for n in PARAM_NAMES)

    def add(s: dict) -> None:
        k = key_theta(s)
        if k not in used_theta:
            used_theta.append(k)
            chosen.append(s)

    by_lop = sorted(active, key=lambda s: s["L_op"])
    for s in by_lop[:3]:
        add(s)

    pts = np.array([[s["r_E_hz"], s["r_I_hz"]] for s in active])
    if len(active) >= 2:
        dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmax(dist), dist.shape)
        add(active[i])
        add(active[j])

    by_nd = sorted(
        active,
        key=lambda s: (
            s["nondegeneracy"]["sd_across_neurons_hz"]
            + s["nondegeneracy"]["E_trajectory"]["mean_pairwise_hamming"]
            + s["nondegeneracy"]["I_trajectory"]["mean_pairwise_hamming"]
        ),
        reverse=True,
    )
    for s in by_nd:
        if len(chosen) >= max_n:
            break
        add(s)

    for s in by_lop:
        if len(chosen) >= max_n:
            break
        add(s)
    return chosen[:max_n]


def qualify_jw_point(eval_10s: dict[str, Any], jw: dict[str, Any]) -> dict[str, Any]:
    drift = eval_10s["baseline_hdp_drift"]
    nd = eval_10s["nondegeneracy"]
    checks = {
        "active_finite_nonsaturated": eval_10s["active"],
        "trajectory_or_sd_nondegenerate": nd["nondegenerate_hard"],
        "not_permutation_degenerate": not nd["permutation_degenerate"],
        "J_W_frobenius": jw["frobenius_norm"] > J_MIN_FRO,
        "rank_2": jw["numerical_rank_2_ok"],
        "H_drift_small": drift.get("H_dot_mean_per_ms") is None
        or drift["H_dot_mean_per_ms"] <= H_DOT_MAX,
        "W_drift_small": drift.get("W_dot_mean_per_ms") is None
        or drift["W_dot_mean_per_ms"] <= W_DOT_MAX,
    }
    return {
        "checks": checks,
        "passes_all": bool(all(checks.values())),
        "failed": [k for k, v in checks.items() if not v],
    }


def horizon_convergence(
    hetero_model: jtfne.Model, specs: dict, theta: dict[str, float]
) -> dict[str, Any]:
    ref = evaluate_theta(hetero_model, specs, theta, REFERENCE_MS)
    rows, t_op = [], None
    for dur in HORIZONS_MS:
        row = evaluate_theta(hetero_model, specs, theta, dur)
        ok = (
            abs(row["r_E_hz"] - ref["r_E_hz"]) <= STAT_TOL["rate_hz"]
            and abs(row["r_I_hz"] - ref["r_I_hz"]) <= STAT_TOL["rate_hz"]
        )
        row["converged_to_10s"] = ok
        rows.append(row)
        if ok and t_op is None:
            t_op = dur
    return {"reference_10s": ref, "horizons": rows, "T_op_ms": t_op}


def run_manifold_phase(
    label: str,
    hetero_model: jtfne.Model,
    specs: dict,
    thetas: list[dict[str, float]],
) -> dict[str, Any]:
    print(f"  Phase A [{label}]: {len(thetas)} Sobol points @ {MAP_DURATION_MS} ms...", flush=True)
    samples = [evaluate_theta(hetero_model, specs, th, MAP_DURATION_MS) for th in thetas]
    manifold = manifold_analysis(samples)
    promising = select_promising(samples)
    print(f"    active={manifold.get('n_active')} dim={manifold.get('manifold_dimension')} "
          f"min_L_op={manifold.get('min_L_op', float('nan')):.4f}", flush=True)

    jw_results = []
    for s in promising:
        print(f"    Phase C: J_W @ theta≈{[round(s['theta'][n],3) for n in PARAM_NAMES]}...", flush=True)
        jw = frobenius_jw(hetero_model, s["theta"], specs, MAP_DURATION_MS)
        ev10 = evaluate_theta(hetero_model, specs, s["theta"], REFERENCE_MS)
        qual = qualify_jw_point(ev10, jw)
        jw_results.append(
            {
                "theta": s["theta"],
                "map_eval": s,
                "eval_10s": ev10,
                "jacobian": jw,
                "qualification": qual,
            }
        )

    qualified = [r for r in jw_results if r["qualification"]["passes_all"]]
    return {
        "label": label,
        "samples": samples,
        "manifold": manifold,
        "promising_jacobian": jw_results,
        "n_qualified_rank2": len(qualified),
        "qualified_thetas": [r["theta"] for r in qualified],
    }


def scientific_agsdr(
    hetero_model: jtfne.Model,
    specs: dict,
    seed_theta: dict[str, float],
    duration_ms: float,
) -> dict[str, Any]:
    model = _model_with_parameters(hetero_model, seed_theta, specs)
    objective = jtfne.rate_targets(
        groups={"E": list(range(5)), "I": list(range(5, 10))},
        targets_hz={"E": TARGET_R_E_HZ, "I": TARGET_R_I_HZ},
        burn_in_ms=BURN_IN_MS,
    )
    sim = jtfne.simulation(
        duration_ms=duration_ms,
        dt_ms=DT_MS,
        seed=SIM_SEED,
        runtime=mcc3_runtime(True),
    )
    optimizer = jtfne.agsdr(
        parameters=specs,
        generations=SCIENTIFIC_GENERATIONS,
        population_size=SCIENTIFIC_POPULATION,
        seed=OPT_SEED,
    )
    print(
        f"  Phase E: scientific AGSDR {SCIENTIFIC_GENERATIONS}x{SCIENTIFIC_POPULATION} "
        f"@ {duration_ms} ms from seed θ...",
        flush=True,
    )
    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim)
    theta = dict(result.best_parameters)
    ev_map = evaluate_theta(hetero_model, specs, theta, duration_ms)
    ev10 = evaluate_theta(hetero_model, specs, theta, REFERENCE_MS)
    jw = frobenius_jw(hetero_model, theta, specs, duration_ms)
    qual = qualify_jw_point(ev10, jw)
    horizon = horizon_convergence(hetero_model, specs, theta)
    return {
        "seed_theta": seed_theta,
        "duration_ms": duration_ms,
        "theta_star": theta,
        "best_score": float(result.best_score),
        "eval_at_opt_horizon": ev_map,
        "eval_10s": ev10,
        "jacobian": jw,
        "qualification": qual,
        "horizon_convergence": horizon,
        "summary": dict(result.summary) if result.summary else {},
    }


def make_figure(report: dict[str, Any], path: Path) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    colors = {"drive_0.15": "tab:blue", "drive_0.20": "tab:orange", "drive_izh": "tab:green"}
    for phase_key, color in colors.items():
        block = report.get("phases", {}).get(phase_key)
        if not block:
            continue
        active = [s for s in block["samples"] if s["active"]]
        if not active:
            continue
        re = [s["r_E_hz"] for s in active]
        ri = [s["r_I_hz"] for s in active]
        ax0.scatter(re, ri, s=18, alpha=0.55, c=color, label=phase_key)

    ax0.scatter(
        [TARGET_R_E_HZ], [TARGET_R_I_HZ], marker="*", s=220, c="red", zorder=5, label="target (15,10)"
    )

    for pt in report.get("highlight_points", []):
        mk = "o" if pt.get("qualified") else "x"
        ec = "lime" if pt.get("qualified") else "crimson"
        ax0.scatter(
            [pt["r_E_hz"]],
            [pt["r_I_hz"]],
            s=90,
            facecolors="none",
            edgecolors=ec,
            linewidths=2,
            marker=mk,
        )

    ax0.set_xlabel("r_E (Hz)")
    ax0.set_ylabel("r_I (Hz)")
    ax0.set_title("Reachable (r_E, r_I) cloud")
    ax0.legend(fontsize=7, loc="best")
    ax0.grid(True, alpha=0.25)

    ax1 = fig.add_subplot(gs[0, 1])
    labels, s1, s2 = [], [], []
    for block in report.get("phases", {}).values():
        for row in block.get("promising_jacobian", []):
            jw = row["jacobian"]
            labels.append(
                f"{block['label'][:8]}\nσ={jw['singular_values'][0]:.2f}"
            )
            s1.append(jw["singular_values"][0])
            s2.append(jw["singular_values"][1] if len(jw["singular_values"]) > 1 else 0.0)
    if labels:
        x = np.arange(len(labels))
        ax1.bar(x - 0.15, s1, width=0.3, label="σ₁")
        ax1.bar(x + 0.15, s2, width=0.3, label="σ₂")
        ax1.set_xticks(x, labels, fontsize=6, rotation=45, ha="right")
        ax1.set_ylabel("singular value")
        ax1.set_title("J_W singular values (promising points)")
        ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    lines = [
        f"Target feasible (L_op): {report.get('target_feasible')}",
        f"Manifold 2D (ε=0.15): {report.get('manifold_015')}",
        f"Qualified rank-2 (drive only): {report.get('qualified_drive_only')}",
        f"Qualified rank-2 (after Izh): {report.get('qualified_with_izh')}",
        f"Scientific AGSDR run: {report.get('scientific_agsdr_run')}",
        f"Final qualified candidate: {report.get('final_qualified')}",
        f"Conclusion: {report.get('conclusion')}",
    ]
    if report.get("scientific_result"):
        sr = report["scientific_result"]
        lines.append(
            f"θ* r_E/r_I @10s: {sr['eval_10s']['r_E_hz']:.2f}/{sr['eval_10s']['r_I_hz']:.2f} "
            f"L_op={sr['eval_10s']['L_op']:.4f}"
        )
    ax2.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)

    fig.suptitle("HDP-MVC Stage 2b operating-manifold search", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = jtfne.construct(mcc3_config())
    specs = mcc3_specs()
    thetas = sobol_theta_samples(SOBOL_N, seed=7)

    phases: dict[str, Any] = {}
    highlight: list[dict[str, Any]] = []

    # Phase A/C @ ε=0.15
    m15 = apply_drive_heterogeneity(base, EPS_PRIMARY)
    phases["drive_0.15"] = run_manifold_phase("drive_eps_0.15", m15, specs, thetas)

    # Confirm @ ε=0.20 (same Sobol design)
    m20 = apply_drive_heterogeneity(base, EPS_CONFIRM)
    phases["drive_0.20"] = run_manifold_phase("drive_eps_0.20", m20, specs, thetas)

    qualified_drive = []
    for key in ("drive_0.15", "drive_0.20"):
        for row in phases[key]["promising_jacobian"]:
            if row["qualification"]["passes_all"]:
                qualified_drive.append((key, row))
            highlight.append(
                {
                    "phase": key,
                    "r_E_hz": row["eval_10s"]["r_E_hz"],
                    "r_I_hz": row["eval_10s"]["r_I_hz"],
                    "qualified": row["qualification"]["passes_all"],
                    "theta": row["theta"],
                }
            )

    # Phase D if needed
    qualified_with_izh: list[tuple[str, dict]] = []
    izh_block = None
    if not qualified_drive:
        print("  Phase D: adding deterministic Izh heterogeneity (5%)...", flush=True)
        m_izh = apply_izh_heterogeneity(apply_drive_heterogeneity(base, EPS_PRIMARY), IZH_HET_AMP)
        izh_block = run_manifold_phase("drive_0.15_izh_0.05", m_izh, specs, thetas)
        phases["drive_izh"] = izh_block
        for row in izh_block["promising_jacobian"]:
            if row["qualification"]["passes_all"]:
                qualified_with_izh.append(("drive_izh", row))
            highlight.append(
                {
                    "phase": "drive_izh",
                    "r_E_hz": row["eval_10s"]["r_E_hz"],
                    "r_I_hz": row["eval_10s"]["r_I_hz"],
                    "qualified": row["qualification"]["passes_all"],
                    "theta": row["theta"],
                }
            )

    all_qualified = qualified_drive + qualified_with_izh
    target_feasible = bool(
        phases["drive_0.15"]["manifold"].get("target_feasible_L_op")
        or phases["drive_0.20"]["manifold"].get("target_feasible_L_op")
        or (izh_block and izh_block["manifold"].get("target_feasible_L_op"))
    )

    scientific_result = None
    scientific_run = False
    final_qualified = False
    conclusion = ""

    if all_qualified and target_feasible:
        # pick best qualified by L_op @ 10s
        best_key, best_row = min(
            all_qualified,
            key=lambda kv: kv[1]["eval_10s"]["L_op"],
        )
        hetero = {
            "drive_0.15": m15,
            "drive_0.20": m20,
            "drive_izh": apply_izh_heterogeneity(
                apply_drive_heterogeneity(base, EPS_PRIMARY), IZH_HET_AMP
            ),
        }[best_key if best_key != "drive_izh" else "drive_izh"]
        seed_theta = best_row["theta"]
        t_op = MAP_DURATION_MS
        scientific_run = True
        scientific_result = scientific_agsdr(hetero, specs, seed_theta, t_op)
        # Phase F: horizon
        t_op_new = scientific_result["horizon_convergence"]["T_op_ms"] or t_op
        if t_op_new != t_op:
            print(f"  Phase F: re-finalizing at T_op={t_op_new} ms...", flush=True)
            scientific_result = scientific_agsdr(hetero, specs, seed_theta, float(t_op_new))
        final_qualified = scientific_result["qualification"]["passes_all"]
        conclusion = (
            "qualified_candidate_for_stage3_review"
            if final_qualified
            else "scientific_agsdr_failed_qualification"
        )
    elif not target_feasible:
        conclusion = "STOP_target_15_10_outside_reachable_interior"
        scientific_run = False
    elif not all_qualified:
        conclusion = "STOP_no_rank2_region_circuit_geometry_limit"
        scientific_run = False

    report = json_safe(
        {
            "schema": "hdp_mvc_stage2b_manifold.v0.1",
            "forbidden": ["K_HDP", "tau_H", "H_to_W", "package_mutations"],
            "targets_hz": {"r_E": TARGET_R_E_HZ, "r_I": TARGET_R_I_HZ},
            "sobol_n": SOBOL_N,
            "map_duration_ms": MAP_DURATION_MS,
            "predeclared_gates": {
                "J_W_fro_min": J_MIN_FRO,
                "sigma2_min": SIGMA2_MIN,
                "sigma_ratio_min": SIGMA_RATIO_MIN,
                "hamming_min": HAMMING_MIN,
                "nondeg_sd_min_hz": NONDEG_SD_MIN_HZ,
                "target_L_op_feasible": TARGET_L_OP_FEASIBLE,
                "manifold_2d_ratio": MANIFOLD_2D_RATIO,
            },
            "phases": phases,
            "target_feasible": target_feasible,
            "manifold_015": phases["drive_0.15"]["manifold"].get("manifold_dimension"),
            "manifold_020": phases["drive_0.20"]["manifold"].get("manifold_dimension"),
            "qualified_drive_only": len(qualified_drive),
            "qualified_with_izh": len(qualified_with_izh),
            "scientific_agsdr_run": scientific_run,
            "scientific_agsdr_budget": {
                "generations": SCIENTIFIC_GENERATIONS,
                "population_size": SCIENTIFIC_POPULATION,
            },
            "scientific_result": scientific_result,
            "final_qualified": final_qualified,
            "conclusion": conclusion,
            "highlight_points": highlight,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "stop_before": "sustained_perturbation_BC",
        }
    )

    json_path = OUT / "hdp_mvc_stage2b_manifold.json"
    fig_path = OUT / "hdp_mvc_stage2b_manifold.png"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_figure(report, fig_path)

    summary = {
        "conclusion": conclusion,
        "target_feasible": target_feasible,
        "manifold_eps_015": phases["drive_0.15"]["manifold"],
        "qualified_rank2_drive": len(qualified_drive),
        "scientific_agsdr_run": scientific_run,
        "final_qualified": final_qualified,
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
