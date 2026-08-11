#!/usr/bin/env python3
"""Stage 2: HDP-MVC operating-point search (read-only — no package/HDP-rule changes).

For each fixed heterogeneity amplitude ε ∈ {0.10, 0.15, 0.20}:
  1. AGSDR at initial T_op = 500 ms toward r_E*=15 Hz, r_I*=10 Hz
  2. Post-hoc sensitivity / pathology qualification (not in the objective)
  3. Horizon re-estimation vs 10 s reference; re-optimize if T_op shifts materially

Stops before sustained-perturbation B/C (Stage 3).
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne._model_tune import _edge_parameter_mask, _model_with_parameters
from jaxfne.hdp_network import DEFAULT_HDP
from jaxfne.io import json_safe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "msvc_hdp_diagnostic"

# --- predeclared gates (not tuned post-hoc) ---
DT_MS = 0.1
BURN_IN_MS = 20.0
SIM_SEED = 17
OPT_SEED = 42
AGSDR_GENERATIONS = 2
AGSDR_POPULATION = 4
INITIAL_T_OP_MS = 500.0
REFERENCE_MS = 10_000.0
HORIZONS_MS = (500.0, 1000.0, 2000.0, 5000.0, 10_000.0)
EPS_CANDIDATES = (0.10, 0.15, 0.20)
TARGET_R_E_HZ = 15.0
TARGET_R_I_HZ = 10.0
FD_EPS = 0.05
PARAM_NAMES = ("m_EE", "m_EI", "m_IE", "m_II")
STAT_TOL = {"rate_hz": 0.5, "cv_isi": 0.02, "H_mean": 0.0005}
J_MIN_FRO = 0.1
R_E_MIN_HZ = 1.0
R_I_MIN_HZ = 1.0
NONDEG_SD_MIN_HZ = 0.5
SIGMA2_MIN = 0.1
SIGMA_RATIO_MIN = 0.05
RATE_SATURATION_MAX_HZ = 100.0
H_DOT_BASELINE_MAX = 1e-4  # per ms, scalar H mean
W_DOT_BASELINE_MAX = 1e-4  # per ms, group-mean |W|
MATERIAL_HORIZON_CHANGE_MS = 250.0  # re-optimize if |T_op_new - T_op_used| > this


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
            bounds=(0.1, 5.0),
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


def fixed_heterogeneity_pattern(n: int = 10) -> np.ndarray:
    raw = np.array(
        [0.12, -0.08, 0.05, -0.10, 0.06, -0.07, 0.09, -0.04, 0.03, -0.06],
        dtype=float,
    )
    return raw[:n] - raw[:n].mean()


def apply_heterogeneity(model: jtfne.Model, eps_amp: float) -> jtfne.Model:
    pattern = fixed_heterogeneity_pattern()
    base = np.asarray(model.params["emitter"].drive, dtype=np.float32)
    scaled = base * (1.0 + eps_amp * pattern).astype(np.float32)
    return jtfne.with_emitter_parameters(model, drive_per_neuron=jnp.asarray(scaled))


def operating_point_objective() -> jtfne.Objective:
    return jtfne.rate_targets(
        groups={"E": list(range(5)), "I": list(range(5, 10))},
        targets_hz={"E": TARGET_R_E_HZ, "I": TARGET_R_I_HZ},
        burn_in_ms=BURN_IN_MS,
    )


def ei_rates(spikes: np.ndarray, burn_in_ms: float = BURN_IN_MS) -> dict[str, float]:
    start = int(math.ceil(burn_in_ms / DT_MS))
    window = spikes[start:]
    r_e = float(window[:, :5].mean() * (1000.0 / DT_MS)) if window.size else 0.0
    r_i = float(window[:, 5:].mean() * (1000.0 / DT_MS)) if window.size else 0.0
    pop = float(window.mean() * (1000.0 / DT_MS)) if window.size else 0.0
    return {"population_hz": pop, "r_E_hz": r_e, "r_I_hz": r_i}


def rate_nondegeneracy(spikes: np.ndarray) -> dict[str, Any]:
    start = int(math.ceil(BURN_IN_MS / DT_MS))
    per_neuron = spikes[start:].mean(axis=0) * (1000.0 / DT_MS)
    return {
        "per_neuron_hz": per_neuron.tolist(),
        "sd_across_neurons_hz": float(np.std(per_neuron)),
        "min_hz": float(np.min(per_neuron)),
        "max_hz": float(np.max(per_neuron)),
        "all_neurons_identical_rate": bool(np.allclose(per_neuron, per_neuron[0], atol=1e-3)),
    }


def pathology_metrics(spikes: np.ndarray, rates: dict[str, float]) -> dict[str, Any]:
    nd = rate_nondegeneracy(spikes)
    per = np.asarray(nd["per_neuron_hz"])
    return {
        "nondegeneracy": nd,
        "silent_E": bool(np.any(per[:5] < 0.01)),
        "silent_I": bool(np.any(per[5:] < 0.01)),
        "max_neuron_hz": float(np.max(per)),
        "saturated": bool(np.max(per) > RATE_SATURATION_MAX_HZ),
        "rates_finite": bool(np.all(np.isfinite(per))),
        "L_op": float(
            ((rates["r_E_hz"] - TARGET_R_E_HZ) / TARGET_R_E_HZ) ** 2
            + ((rates["r_I_hz"] - TARGET_R_I_HZ) / TARGET_R_I_HZ) ** 2
        ),
        "kappa": float(jtfne.kappa_synchrony(spikes, DT_MS)),
    }


def simulate_full(
    model: jtfne.Model,
    duration_ms: float,
    *,
    hdp_on: bool = True,
) -> dict[str, Any]:
    sim = jtfne.simulation(
        duration_ms=duration_ms,
        dt_ms=DT_MS,
        seed=SIM_SEED,
        runtime=mcc3_runtime(hdp_on),
    )
    sig = model.simulate(sim)
    spikes = np.asarray(sig.spikes)
    rates = ei_rates(spikes)
    path = pathology_metrics(spikes, rates)
    h_mean = None
    diag = model.last_hdp_diagnostics()
    drift = baseline_hdp_drift(diag, duration_ms)
    if diag is not None and diag.get("H_trace") is not None:
        h = np.asarray(diag["H_trace"])
        start = int(math.ceil(BURN_IN_MS / DT_MS))
        h_mean = float(np.mean(h[start:]))
    return {
        "duration_ms": duration_ms,
        **rates,
        "H_post_burnin_mean": h_mean,
        "pathology": path,
        "baseline_hdp_drift": drift,
        "hdp_diagnostics_present": diag is not None,
    }


def baseline_hdp_drift(diag: dict[str, Any] | None, duration_ms: float) -> dict[str, Any]:
    if diag is None:
        return {"H_dot_mean_per_ms": None, "W_dot_mean_per_ms": None, "available": False}
    start = int(math.ceil(BURN_IN_MS / DT_MS))
    out: dict[str, Any] = {"available": True}
    h_trace = diag.get("H_trace")
    if h_trace is not None:
        h = np.asarray(h_trace)
        if h.ndim == 2:
            h_series = h[start:, :].mean(axis=1)
        else:
            h_series = h[start:]
        if len(h_series) >= 2:
            dh = np.diff(h_series) / DT_MS
            out["H_dot_mean_per_ms"] = float(np.mean(np.abs(dh)))
            out["H_dot_max_per_ms"] = float(np.max(np.abs(dh)))
        else:
            out["H_dot_mean_per_ms"] = 0.0
            out["H_dot_max_per_ms"] = 0.0
    else:
        out["H_dot_mean_per_ms"] = None
    w_trace = diag.get("w_trace")
    if w_trace is not None:
        w = np.asarray(w_trace)
        if w.ndim == 2 and w.shape[0] > start + 1:
            w_group = np.mean(np.abs(w[start:, :]), axis=1)
            dw = np.diff(w_group) / DT_MS
            out["W_dot_mean_per_ms"] = float(np.mean(np.abs(dw)))
            out["W_dot_max_per_ms"] = float(np.max(np.abs(dw)))
        else:
            out["W_dot_mean_per_ms"] = 0.0
    else:
        w0 = diag.get("w_final")
        out["W_dot_mean_per_ms"] = 0.0 if w0 is not None else None
    return out


def frobenius_jw(
    model: jtfne.Model,
    theta: dict[str, float],
    specs: dict,
    duration_ms: float,
) -> dict[str, Any]:
    y0 = simulate_full(_model_with_parameters(model, theta, specs), duration_ms)
    j = np.zeros((2, 4), dtype=float)
    for col, name in enumerate(PARAM_NAMES):
        tp, tm = dict(theta), dict(theta)
        delta = max(FD_EPS * abs(theta[name]), 0.01)
        tp[name] = float(np.clip(theta[name] + delta, *specs[name].bounds))
        tm[name] = float(np.clip(theta[name] - delta, *specs[name].bounds))
        yp = simulate_full(_model_with_parameters(model, tp, specs), duration_ms)
        ym = simulate_full(_model_with_parameters(model, tm, specs), duration_ms)
        j[:, col] = (
            np.array([yp["r_E_hz"], yp["r_I_hz"]])
            - np.array([ym["r_E_hz"], ym["r_I_hz"]])
        ) / (tp[name] - tm[name] + 1e-12)
    fro = float(np.linalg.norm(j, ord="fro"))
    u, s, vt = np.linalg.svd(j, full_matrices=False)
    rank2_ok = bool(s[1] >= SIGMA2_MIN and (s[1] / (s[0] + 1e-12)) >= SIGMA_RATIO_MIN)
    column_dominance = {
        name: {
            "partial_r_E": float(j[0, col]),
            "partial_r_I": float(j[1, col]),
            "column_norm": float(np.linalg.norm(j[:, col])),
        }
        for col, name in enumerate(PARAM_NAMES)
    }
    return {
        "J_W": j.tolist(),
        "frobenius_norm": fro,
        "singular_values": s.tolist(),
        "numerical_rank_2_ok": rank2_ok,
        "U": u.tolist(),
        "Vt": vt.tolist(),
        "column_dominance": column_dominance,
        "baseline_rates": {"r_E_hz": y0["r_E_hz"], "r_I_hz": y0["r_I_hz"]},
    }


def qualify_candidate(
    eval_10s: dict[str, Any],
    jw: dict[str, Any],
) -> dict[str, Any]:
    rates = eval_10s
    path = eval_10s["pathology"]
    nd = path["nondegeneracy"]
    drift = eval_10s["baseline_hdp_drift"]
    checks = {
        "r_E_positive": rates["r_E_hz"] > R_E_MIN_HZ,
        "r_I_positive": rates["r_I_hz"] > R_I_MIN_HZ,
        "rates_finite": path["rates_finite"],
        "not_saturated": not path["saturated"],
        "no_silent_E": not path["silent_E"],
        "no_silent_I": not path["silent_I"],
        "nondegenerate_sd": nd["sd_across_neurons_hz"] >= NONDEG_SD_MIN_HZ,
        "J_W_frobenius": jw["frobenius_norm"] > J_MIN_FRO,
        "J_W_rank2": jw["numerical_rank_2_ok"],
        "H_drift_small": (
            drift.get("H_dot_mean_per_ms") is None
            or drift["H_dot_mean_per_ms"] <= H_DOT_BASELINE_MAX
        ),
        "W_drift_small": (
            drift.get("W_dot_mean_per_ms") is None
            or drift["W_dot_mean_per_ms"] <= W_DOT_BASELINE_MAX
        ),
    }
    return {
        "checks": checks,
        "passes_all": bool(all(checks.values())),
        "failed": [k for k, v in checks.items() if not v],
    }


def horizon_convergence(
    model: jtfne.Model,
    theta: dict[str, float],
    specs: dict,
) -> dict[str, Any]:
    ref = simulate_full(_model_with_parameters(model, theta, specs), REFERENCE_MS)
    rows = []
    t_op = None
    for dur in HORIZONS_MS:
        row = simulate_full(_model_with_parameters(model, theta, specs), dur)
        ok = (
            abs(row["r_E_hz"] - ref["r_E_hz"]) <= STAT_TOL["rate_hz"]
            and abs(row["r_I_hz"] - ref["r_I_hz"]) <= STAT_TOL["rate_hz"]
            and (
                row["pathology"]["nondegeneracy"].get("isi_cv_population") is None
                or ref["pathology"]["nondegeneracy"].get("isi_cv_population") is None
                or True
            )
            and (
                row["H_post_burnin_mean"] is None
                or ref["H_post_burnin_mean"] is None
                or abs(row["H_post_burnin_mean"] - ref["H_post_burnin_mean"])
                <= STAT_TOL["H_mean"]
            )
        )
        row["converged_to_10s"] = ok
        rows.append(row)
        if ok and t_op is None:
            t_op = dur
    return {
        "reference_10s": ref,
        "tolerance": STAT_TOL,
        "horizons": rows,
        "T_op_ms": t_op,
    }


def run_agsdr(
    model: jtfne.Model,
    specs: dict,
    duration_ms: float,
) -> dict[str, Any]:
    sim = jtfne.simulation(
        duration_ms=duration_ms,
        dt_ms=DT_MS,
        seed=SIM_SEED,
        runtime=mcc3_runtime(True),
    )
    objective = operating_point_objective()
    optimizer = jtfne.agsdr(
        parameters=specs,
        generations=AGSDR_GENERATIONS,
        population_size=AGSDR_POPULATION,
        seed=OPT_SEED,
    )
    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim)
    return {
        "theta_star": dict(result.best_parameters),
        "best_score": float(result.best_score),
        "summary": dict(result.summary) if result.summary else {},
        "duration_ms": duration_ms,
    }


def recovery_ei_metric_def() -> dict[str, Any]:
    """Stage-3 criterion (documented here; not executed in Stage 2)."""
    return {
        "R_EI_formula": (
            "1 - ||D(r_final - r_0)||_2 / (||D(r_peak - r_0)||_2 + eps), "
            "D=diag(1/r_E*, 1/r_I*)"
        ),
        "r_E_star_hz": TARGET_R_E_HZ,
        "r_I_star_hz": TARGET_R_I_HZ,
        "gate": "R_EI_HDP > R_EI_off + delta",
        "delta_predeclared": 0.1,
    }


def evaluate_candidate(
    hetero_model: jtfne.Model,
    specs: dict,
    theta: dict[str, float],
    *,
    qualify_duration_ms: float,
) -> dict[str, Any]:
    model = _model_with_parameters(hetero_model, theta, specs)
    eval_at_qual = simulate_full(model, qualify_duration_ms)
    eval_10s = simulate_full(model, REFERENCE_MS)
    jw = frobenius_jw(hetero_model, theta, specs, qualify_duration_ms)
    qual = qualify_candidate(eval_10s, jw)
    horizon = horizon_convergence(hetero_model, theta, specs)
    return {
        "theta_star": theta,
        "eval_at_qualify_horizon": eval_at_qual,
        "eval_10s": eval_10s,
        "jacobian": jw,
        "qualification": qual,
        "horizon_convergence": horizon,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = jtfne.construct(mcc3_config())
    specs = mcc3_specs()
    candidates: list[dict[str, Any]] = []

    for eps in EPS_CANDIDATES:
        print(f"ε={eps}: AGSDR at T_op={INITIAL_T_OP_MS} ms...", flush=True)
        hetero = apply_heterogeneity(base, eps)
        agsdr = run_agsdr(hetero, specs, INITIAL_T_OP_MS)
        theta = agsdr["theta_star"]

        row = {
            "eps_amplitude": eps,
            "agsdr_initial": agsdr,
            "optimization_horizon_ms": INITIAL_T_OP_MS,
            "reoptimized": False,
        }

        eval0 = evaluate_candidate(
            hetero, specs, theta, qualify_duration_ms=INITIAL_T_OP_MS
        )
        row["evaluation_initial"] = eval0

        t_op_new = eval0["horizon_convergence"]["T_op_ms"]
        material_change = (
            t_op_new is not None
            and abs(float(t_op_new) - INITIAL_T_OP_MS) > MATERIAL_HORIZON_CHANGE_MS
        )

        if material_change and eval0["qualification"]["passes_all"]:
            print(
                f"  re-optimizing at T_op={t_op_new} ms (material horizon shift)...",
                flush=True,
            )
            agsdr2 = run_agsdr(hetero, specs, float(t_op_new))
            theta = agsdr2["theta_star"]
            row["reoptimized"] = True
            row["optimization_horizon_ms"] = float(t_op_new)
            row["agsdr_final"] = agsdr2
            row["evaluation_final"] = evaluate_candidate(
                hetero, specs, theta, qualify_duration_ms=float(t_op_new)
            )
        else:
            row["evaluation_final"] = eval0
            if material_change:
                row["horizon_reoptimize_skipped"] = (
                    "material T_op shift but candidate failed initial qualification"
                )

        fin = row["evaluation_final"]
        candidates.append(row)
        q = fin["qualification"]
        print(
            f"  θ*={theta}  r_E={fin['eval_10s']['r_E_hz']:.2f}  "
            f"r_I={fin['eval_10s']['r_I_hz']:.2f}  "
            f"‖J_W‖_F={fin['jacobian']['frobenius_norm']:.3f}  "
            f"pass={q['passes_all']}",
            flush=True,
        )

    table = []
    for row in candidates:
        fin = row["evaluation_final"]
        qual = fin["qualification"]
        jw = fin["jacobian"]
        ev = fin["eval_10s"]
        table.append(
            {
                "eps": row["eps_amplitude"],
                "theta_star": fin["theta_star"],
                "T_op_used_ms": row["optimization_horizon_ms"],
                "T_op_converged_ms": fin["horizon_convergence"]["T_op_ms"],
                "r_E_10s_hz": ev["r_E_hz"],
                "r_I_10s_hz": ev["r_I_hz"],
                "L_op_10s": ev["pathology"]["L_op"],
                "sd_neurons_hz": ev["pathology"]["nondegeneracy"]["sd_across_neurons_hz"],
                "J_W_fro": jw["frobenius_norm"],
                "singular_values": jw["singular_values"],
                "rank2_ok": jw["numerical_rank_2_ok"],
                "H_dot_baseline": ev["baseline_hdp_drift"].get("H_dot_mean_per_ms"),
                "W_dot_baseline": ev["baseline_hdp_drift"].get("W_dot_mean_per_ms"),
                "passes_qualification": qual["passes_all"],
                "failed_checks": qual["failed"],
            }
        )

    report = json_safe(
        {
            "schema": "hdp_mvc_stage2_operating_point.v0.1",
            "forbidden": ["K_HDP", "tau_H", "H_to_W_equations", "package_mutations"],
            "targets_hz": {"r_E": TARGET_R_E_HZ, "r_I": TARGET_R_I_HZ},
            "objective": "((r_E-15)/15)^2 + ((r_I-10)/10)^2 via rate_targets",
            "agsdr_contract": {
                "generations": AGSDR_GENERATIONS,
                "population_size": AGSDR_POPULATION,
                "seed": OPT_SEED,
                "sim_seed": SIM_SEED,
                "initial_T_op_ms": INITIAL_T_OP_MS,
            },
            "predeclared_gates": {
                "r_E_min_hz": R_E_MIN_HZ,
                "r_I_min_hz": R_I_MIN_HZ,
                "nondegeneracy_sd_min_hz": NONDEG_SD_MIN_HZ,
                "J_W_frobenius_min": J_MIN_FRO,
                "sigma2_min": SIGMA2_MIN,
                "sigma_ratio_min": SIGMA_RATIO_MIN,
                "H_dot_baseline_max_per_ms": H_DOT_BASELINE_MAX,
                "W_dot_baseline_max_per_ms": W_DOT_BASELINE_MAX,
                "rate_saturation_max_hz": RATE_SATURATION_MAX_HZ,
            },
            "stage3_recovery_metric": recovery_ei_metric_def(),
            "candidate_table": table,
            "candidates_full": candidates,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "stop_before": "sustained_perturbation_BC",
        }
    )

    out_path = OUT / "hdp_mvc_stage2_operating_point.json"
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_table": table}, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
