#!/usr/bin/env python3
"""Read-only HDP-MVC experimental geometry search.

Stages (no package / HDP-rule mutations):
  0. Horizon convergence — find T_op where stats match 10 s reference
  1. Heterogeneity sweep — break permutation symmetry; measure J_W Frobenius norm
  2. (placeholder) Operating-point optimization with post-hoc |J_W|_F gate
  3. (placeholder) Sustained-perturbation B vs C with recovery index R

Forbidden during this phase: changing K_HDP, tau_H, or H→W equations.
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
from jaxfne._model_tune import _model_with_parameters
from jaxfne.hdp_network import DEFAULT_HDP
from jaxfne.io import json_safe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "msvc_hdp_diagnostic"

DT_MS = 0.1
BURN_IN_MS = 20.0
SIM_SEED = 17
REFERENCE_MS = 10_000.0
FD_EPS = 0.05
PARAM_NAMES = ("m_EE", "m_EI", "m_IE", "m_II")
HORIZONS_MS = (500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10_000.0)
EPS_AMPLITUDES = (0.0, 0.05, 0.10, 0.15, 0.20)
STAT_TOL = {"rate_hz": 0.5, "cv_isi": 0.02, "H_mean": 0.0005}
J_MIN_FRO = 0.1  # predeclared sensitivity gate (not tuned post-hoc)


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
    """Deterministic zero-mean multipliers for drive heterogeneity."""
    raw = np.array([0.12, -0.08, 0.05, -0.10, 0.06, -0.07, 0.09, -0.04, 0.03, -0.06], dtype=float)
    return raw[:n] - raw[:n].mean()


def apply_heterogeneity(model: jtfne.Model, eps_amp: float) -> jtfne.Model:
    if eps_amp == 0.0:
        return model
    pattern = fixed_heterogeneity_pattern()
    base = np.asarray(model.params["emitter"].drive, dtype=np.float32)
    scaled = base * (1.0 + eps_amp * pattern).astype(np.float32)
    return jtfne.with_emitter_parameters(model, drive_per_neuron=jnp.asarray(scaled))


def ei_rates(spikes: np.ndarray, dt_ms: float, burn_in_ms: float = BURN_IN_MS) -> dict[str, float]:
    start = int(math.ceil(burn_in_ms / dt_ms))
    window = spikes[start:]
    e_idx, i_idx = list(range(5)), list(range(5, 10))
    r_e = float(window[:, e_idx].mean() * (1000.0 / dt_ms)) if window.size else 0.0
    r_i = float(window[:, i_idx].mean() * (1000.0 / dt_ms)) if window.size else 0.0
    pop = float(window.mean() * (1000.0 / dt_ms)) if window.size else 0.0
    return {"population_hz": pop, "r_E_hz": r_e, "r_I_hz": r_i}


def population_isi_cv(spikes: np.ndarray, dt_ms: float) -> float | None:
    isis = []
    for i in range(spikes.shape[1]):
        idx = np.flatnonzero(spikes[:, i] > 0.5)
        if len(idx) >= 2:
            isis.extend((np.diff(idx) * dt_ms).tolist())
    if len(isis) < 2:
        return None
    isis = np.asarray(isis)
    return float(np.std(isis) / (np.mean(isis) + 1e-9))


def rate_nondegeneracy(spikes: np.ndarray, dt_ms: float) -> dict[str, Any]:
    """Non-degeneracy diagnostics (not biological CV target)."""
    start = int(math.ceil(BURN_IN_MS / dt_ms))
    per_neuron = spikes[start:].mean(axis=0) * (1000.0 / dt_ms)
    return {
        "per_neuron_hz": per_neuron.tolist(),
        "sd_across_neurons_hz": float(np.std(per_neuron)),
        "min_hz": float(np.min(per_neuron)),
        "max_hz": float(np.max(per_neuron)),
        "isi_cv_population": population_isi_cv(spikes, dt_ms),
        "all_neurons_identical_rate": bool(np.allclose(per_neuron, per_neuron[0], atol=1e-3)),
    }


def simulate_stats(
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
    rates = ei_rates(spikes, DT_MS)
    nd = rate_nondegeneracy(spikes, DT_MS)
    h_mean = None
    diag = model.last_hdp_diagnostics()
    if diag is not None and diag.get("H_trace") is not None:
        h = np.asarray(diag["H_trace"])
        start = int(math.ceil(BURN_IN_MS / DT_MS))
        h_mean = float(h[start:].mean())
    return {
        "duration_ms": duration_ms,
        **rates,
        "H_post_burnin_mean": h_mean,
        "nondegeneracy": nd,
        "kappa": float(jtfne.kappa_synchrony(spikes, DT_MS)),
    }


def frobenius_jw(
    model: jtfne.Model,
    theta: dict[str, float],
    specs: dict,
    duration_ms: float,
) -> dict[str, Any]:
    y0 = simulate_stats(_model_with_parameters(model, theta, specs), duration_ms)
    j = np.zeros((2, 4), dtype=float)
    for col, name in enumerate(PARAM_NAMES):
        tp, tm = dict(theta), dict(theta)
        delta = max(FD_EPS * abs(theta[name]), 0.01)
        tp[name] = float(np.clip(theta[name] + delta, *specs[name].bounds))
        tm[name] = float(np.clip(theta[name] - delta, *specs[name].bounds))
        yp = simulate_stats(_model_with_parameters(model, tp, specs), duration_ms)
        ym = simulate_stats(_model_with_parameters(model, tm, specs), duration_ms)
        j[:, col] = (
            np.array([yp["r_E_hz"], yp["r_I_hz"]])
            - np.array([ym["r_E_hz"], ym["r_I_hz"]])
        ) / (tp[name] - tm[name] + 1e-12)
    fro = float(np.linalg.norm(j, ord="fro"))
    return {
        "J_W": j.tolist(),
        "frobenius_norm": fro,
        "passes_gate": fro > J_MIN_FRO,
        "baseline_rates": y0,
    }


def horizon_convergence(base: jtfne.Model, theta: dict[str, float], specs: dict) -> dict[str, Any]:
    ref = simulate_stats(_model_with_parameters(base, theta, specs), REFERENCE_MS)
    rows = []
    t_op = None
    for dur in HORIZONS_MS:
        row = simulate_stats(_model_with_parameters(base, theta, specs), dur)
        ok = (
            abs(row["r_E_hz"] - ref["r_E_hz"]) <= STAT_TOL["rate_hz"]
            and abs(row["r_I_hz"] - ref["r_I_hz"]) <= STAT_TOL["rate_hz"]
            and (
                row["nondegeneracy"]["isi_cv_population"] is None
                or ref["nondegeneracy"]["isi_cv_population"] is None
                or abs(
                    row["nondegeneracy"]["isi_cv_population"]
                    - ref["nondegeneracy"]["isi_cv_population"]
                )
                <= STAT_TOL["cv_isi"]
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


def heterogeneity_sweep(
    base: jtfne.Model,
    theta: dict[str, float],
    specs: dict,
    duration_ms: float,
) -> list[dict[str, Any]]:
    rows = []
    for eps in EPS_AMPLITUDES:
        model = apply_heterogeneity(_model_with_parameters(base, theta, specs), eps)
        stats = simulate_stats(model, duration_ms)
        jw = frobenius_jw(model, theta, specs, duration_ms)
        rows.append(
            {
                "eps_amplitude": eps,
                "stats": stats,
                "J_W": jw,
            }
        )
    return rows


def recovery_index(r0: float, r_peak: float, r_final: float, eps: float = 1e-6) -> float:
    """R=0 no recovery, R=1 complete, negative = overshoot away from baseline."""
    numer = abs(r_peak - r0) - abs(r_final - r0)
    return float(numer / (abs(r_peak - r0) + eps))


def ei_recovery_indices(
    spikes_post: np.ndarray, r0_e: float, r0_i: float, bin_steps: int
) -> dict[str, float]:
    def pop_series(col_slice: slice) -> list[float]:
        return [
            float(spikes_post[b0 : b0 + bin_steps, col_slice].mean() * (1000.0 / DT_MS))
            for b0 in range(0, spikes_post.shape[0], bin_steps)
            if b0 + bin_steps <= spikes_post.shape[0]
        ]

    out: dict[str, float] = {}
    for key, r0, sl in (("R_E", r0_e, slice(0, 5)), ("R_I", r0_i, slice(5, 10))):
        series = pop_series(sl)
        if not series:
            out[key] = float("nan")
            continue
        peak = max(series, key=lambda r: abs(r - r0))
        final = series[-1]
        out[key] = recovery_index(r0, peak, final)
    return out


def sustained_perturbation_recovery(
    model: jtfne.Model,
    *,
    hdp_on: bool,
    baseline_scale: float = 1.0,
    perturbed_scale: float = 1.5,
    baseline_ms: float = 3000.0,
    perturb_ms: float = 7000.0,
) -> dict[str, Any]:
    """Sustained perturbation: U0 baseline then U1 held (no restore)."""
    from jaxfne._model_simulate import _simulate_continuation_arrays

    phases = [
        ("baseline", baseline_ms, baseline_scale),
        ("sustained_perturb", perturb_ms, perturbed_scale),
    ]
    continuation = None
    phase_rates = []
    all_spikes = []
    for label, dur, scale in phases:
        base_drive = np.asarray(model.params["emitter"].drive, dtype=np.float32)
        scaled = jtfne.with_emitter_parameters(
            model,
            drive_per_neuron=jnp.asarray(base_drive * scale, dtype=base_drive.dtype),
        )
        sim = jtfne.simulation(
            duration_ms=dur,
            dt_ms=DT_MS,
            seed=SIM_SEED,
            runtime=mcc3_runtime(hdp_on),
        )
        schedule = jnp.zeros((sim.n_steps, base_drive.shape[0]), dtype=base_drive.dtype)
        _, spikes, _, continuation = _simulate_continuation_arrays(
            scaled,
            sim,
            sim.resolved_runtime,
            schedule,
            continuation,
        )
        spikes = np.asarray(spikes)
        all_spikes.append(spikes)
        # 1 s bin rates post-burn-in within phase
        bins = []
        bin_ms = 1000.0
        bin_steps = int(round(bin_ms / DT_MS))
        for b0 in range(0, spikes.shape[0], bin_steps):
            b1 = min(b0 + bin_steps, spikes.shape[0])
            if b1 <= b0:
                break
            bins.append(float(spikes[b0:b1].mean() * (1000.0 / DT_MS)))
        phase_rates.append(
            {
                "phase": label,
                "drive_scale": scale,
                "window_mean_hz": bins,
                "full_phase": ei_rates(spikes, DT_MS, burn_in_ms=0.0),
            }
        )

    spikes_all = np.concatenate(all_spikes, axis=0)
    # Recovery after perturbation onset: compare final 2s vs peak in perturb phase
    total_ms = baseline_ms + perturb_ms
    onset_idx = int(round(baseline_ms / DT_MS))
    post = spikes_all[onset_idx:]
    bin_steps = int(round(1000.0 / DT_MS))
    post_bins = [
        float(post[b0 : b0 + bin_steps].mean() * (1000.0 / DT_MS))
        for b0 in range(0, post.shape[0], bin_steps)
        if b0 + bin_steps <= post.shape[0]
    ]
    r0 = phase_rates[0]["full_phase"]["population_hz"]
    r_peak = max(post_bins) if post_bins else phase_rates[1]["full_phase"]["population_hz"]
    r_final = post_bins[-1] if post_bins else r_peak
    r_index = recovery_index(r0, r_peak, r_final)
    ei_r = ei_recovery_indices(
        post,
        phase_rates[0]["full_phase"]["r_E_hz"],
        phase_rates[0]["full_phase"]["r_I_hz"],
        bin_steps,
    )

    return {
        "hdp_on": hdp_on,
        "baseline_scale": baseline_scale,
        "perturbed_scale": perturbed_scale,
        "phases": phase_rates,
        "r0_pop_hz": r0,
        "r_peak_pop_hz": r_peak,
        "r_final_pop_hz": r_final,
        "R_population": float(r_index),
        "R_E": ei_r["R_E"],
        "R_I": ei_r["R_I"],
        "post_perturb_bins_hz": post_bins,
    }


def load_theta_hat() -> dict[str, float]:
    path = ROOT / "artifacts" / "mcc3_10s_checkpoint" / "mcc3_10s_metrics.json"
    if path.exists():
        return dict(json.loads(path.read_text())["agsdr"]["theta_hat"])
    raise FileNotFoundError("Run mcc3_10s_scientific_checkpoint.py first for theta_hat")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = jtfne.construct(mcc3_config())
    specs = mcc3_specs()
    theta_hat = load_theta_hat()

    print("Stage 0: horizon convergence...", flush=True)
    stage0 = horizon_convergence(base, theta_hat, specs)
    t_op = stage0["T_op_ms"] or 3000.0
    print(f"  T_op_ms = {t_op}", flush=True)

    print("Stage 1: heterogeneity sweep...", flush=True)
    stage1 = heterogeneity_sweep(base, theta_hat, specs, duration_ms=t_op)

    print("Stage 3 preview: sustained perturbation at theta_hat (symmetric)...", flush=True)
    model = _model_with_parameters(base, theta_hat, specs)
    sustained_on = sustained_perturbation_recovery(model, hdp_on=True)
    sustained_off = sustained_perturbation_recovery(model, hdp_on=False)

    report = json_safe(
        {
            "schema": "hdp_mvc_readonly_search.v0.1",
            "constraints": {
                "forbidden": ["K_HDP", "tau_H", "H_to_W_equations", "package_mutations"],
                "decision_tree": [
                    "nondegenerate E/I operating point",
                    "J_W != 0",
                    "sustained perturbation with H,W response",
                    "R_HDP > R_off + delta",
                ],
            },
            "acceptance_criteria_predeclared": {
                "r_E_min_hz": 1.0,
                "r_I_min_hz": 1.0,
                "nondegeneracy_sd_min_hz": 0.5,
                "J_W_frobenius_min": J_MIN_FRO,
                "delta_H_min": 0.001,
                "delta_W_min": 0.01,
                "R_HDP_minus_R_off_min": 0.1,
                "note": "CV not a hard biological gate; non-degeneracy replaces symmetry trap",
            },
            "theta_hat": theta_hat,
            "stage0_horizon_convergence": stage0,
            "stage1_heterogeneity_sweep": stage1,
            "stage3_sustained_perturbation_preview": {
                "hdp_on": sustained_on,
                "hdp_off": sustained_off,
                "note": "Preview at symmetric theta_hat; repeat after sensitivity-qualified theta*",
            },
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
    )

    out_path = OUT / "hdp_mvc_readonly_search.json"
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    best_eps = max(stage1, key=lambda r: r["J_W"]["frobenius_norm"])
    print(
        json.dumps(
            {
                "T_op_ms": t_op,
                "best_eps_amplitude": best_eps["eps_amplitude"],
                "best_J_W_fro": best_eps["J_W"]["frobenius_norm"],
                "symmetric_J_W_fro": stage1[0]["J_W"]["frobenius_norm"],
                "sustained_R_hdp_on": sustained_on["R_population"],
                "sustained_R_hdp_off": sustained_off["R_population"],
                "sustained_R_E_on": sustained_on["R_E"],
                "sustained_R_E_off": sustained_off["R_E"],
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
