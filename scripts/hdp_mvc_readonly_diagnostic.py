#!/usr/bin/env python3
"""Read-only HDP-MVC diagnostic on the optimized MCC-3 10-neuron circuit.

Tests whether B≈C arises from J_W·Δθ_HDP≈0 (insensitive subspace) versus
HDP rule failure. Does not modify package code.
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

import jaxfne as jtfne
from jaxfne._model_tune import _edge_parameter_mask, _model_with_parameters
from jaxfne.hdp_network import DEFAULT_HDP
from jaxfne.io import json_safe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "msvc_hdp_diagnostic"
METRICS_10S = ROOT / "artifacts" / "mcc3_10s_checkpoint" / "mcc3_10s_metrics.json"

DT_MS = 0.1
BURN_IN_MS = 20.0
SIM_SEED = 17
OPT_SEED = 42
FD_EPS = 0.05  # relative magnitude perturbation for Jacobian
PARAM_NAMES = ("m_EE", "m_EI", "m_IE", "m_II")
DRIVE_SCALES = (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)


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


def ei_indices(model: jtfne.Model) -> tuple[list[int], list[int]]:
    table = model.neuron_table()
    e_idx = [int(r["neuron_id"]) for r in table if r["cell_type"] == "E"]
    i_idx = [int(r["neuron_id"]) for r in table if r["cell_type"] == "PV"]
    return e_idx, i_idx


def rates_from_spikes(
    spikes: np.ndarray, dt_ms: float, burn_in_ms: float = BURN_IN_MS
) -> tuple[float, float, float]:
    start = int(math.ceil(burn_in_ms / dt_ms))
    window = spikes[start:]
    pop = float(window.mean() * (1000.0 / dt_ms))
    e_idx, i_idx = list(range(5)), list(range(5, 10))
    r_e = float(window[:, e_idx].mean() * (1000.0 / dt_ms)) if e_idx else 0.0
    r_i = float(window[:, i_idx].mean() * (1000.0 / dt_ms)) if i_idx else 0.0
    return pop, r_e, r_i


def continuation_segment(
    model: jtfne.Model,
    duration_ms: float,
    *,
    hdp_on: bool,
    drive_scale: float,
    continuation=None,
    seed: int = SIM_SEED,
) -> tuple[np.ndarray, Any, Any]:
    """Continuation segment with explicit drive schedule (emitter.drive * scale)."""
    from jaxfne._model_simulate import _simulate_continuation_arrays

    runtime = mcc3_runtime(hdp_on)
    sim = jtfne.simulation(
        duration_ms=duration_ms,
        dt_ms=DT_MS,
        seed=seed,
        runtime=runtime,
    )
    base_drive = np.asarray(model.params["emitter"].drive, dtype=np.float32)
    scaled_model = jtfne.with_emitter_parameters(
        model,
        drive_per_neuron=jnp.asarray(base_drive * drive_scale, dtype=base_drive.dtype),
    )
    # Continuation path adds schedule to emitter.drive — use zero schedule so
    # drive_scale is carried only through the scaled emitter vector.
    schedule = jnp.zeros((sim.n_steps, int(base_drive.shape[0])), dtype=base_drive.dtype)
    _, spikes, _, state = _simulate_continuation_arrays(
        scaled_model,
        sim,
        sim.resolved_runtime,
        schedule,
        continuation,
    )
    return np.asarray(spikes), state, scaled_model.last_hdp_diagnostics()


def evaluate_at_theta(
    base: jtfne.Model,
    theta: dict[str, float],
    specs: dict,
    duration_ms: float,
    hdp_on: bool,
    drive_scale: float = 1.0,
) -> dict[str, float]:
    model = _model_with_parameters(base, theta, specs)
    if drive_scale != 1.0:
        d = np.asarray(model.params["emitter"].drive)
        model = jtfne.with_emitter_parameters(
            model, drive_per_neuron=jnp.asarray(d * drive_scale)
        )
    sim = jtfne.simulation(
        duration_ms=duration_ms,
        dt_ms=DT_MS,
        seed=SIM_SEED,
        runtime=mcc3_runtime(hdp_on),
    )
    sig = model.simulate(sim)
    spikes = np.asarray(sig.spikes)
    pop, r_e, r_i = rates_from_spikes(spikes, DT_MS)
    h_mean = None
    diag = model.last_hdp_diagnostics()
    if diag is not None and diag.get("H_trace") is not None:
        h = np.asarray(diag["H_trace"])
        start = int(math.ceil(BURN_IN_MS / DT_MS))
        h_mean = float(h[start:].mean())
    return {
        "population_rate_hz": pop,
        "r_E_hz": r_e,
        "r_I_hz": r_i,
        "H_post_burnin_mean": h_mean,
        "kappa": float(jtfne.kappa_synchrony(spikes, DT_MS)),
    }


def finite_difference_jacobian(
    base: jtfne.Model,
    theta: dict[str, float],
    specs: dict,
    duration_ms: float,
) -> dict[str, Any]:
    y0 = evaluate_at_theta(base, theta, specs, duration_ms, hdp_on=True)
    r0 = np.array([y0["r_E_hz"], y0["r_I_hz"]], dtype=float)
    h0 = y0["H_post_burnin_mean"]
    j_r = np.zeros((2, 4), dtype=float)
    j_h = np.zeros(4, dtype=float)
    for j, name in enumerate(PARAM_NAMES):
        tp = dict(theta)
        tm = dict(theta)
        delta = max(FD_EPS * abs(theta[name]), FD_EPS * 0.1)
        tp[name] = float(np.clip(theta[name] + delta, *specs[name].bounds))
        tm[name] = float(np.clip(theta[name] - delta, *specs[name].bounds))
        yp = evaluate_at_theta(base, tp, specs, duration_ms, hdp_on=True)
        ym = evaluate_at_theta(base, tm, specs, duration_ms, hdp_on=True)
        rp = np.array([yp["r_E_hz"], yp["r_I_hz"]])
        rm = np.array([ym["r_E_hz"], ym["r_I_hz"]])
        j_r[:, j] = (rp - rm) / (tp[name] - tm[name] + 1e-12)
        if h0 is not None and yp["H_post_burnin_mean"] is not None:
            j_h[j] = (yp["H_post_burnin_mean"] - ym["H_post_burnin_mean"]) / (
                tp[name] - tm[name] + 1e-12
            )
    return {
        "baseline": y0,
        "J_W": j_r.tolist(),
        "J_H": j_h.tolist(),
        "param_order": list(PARAM_NAMES),
        "observable_order": ["r_E", "r_I"],
        "fd_eps_relative": FD_EPS,
        "duration_ms": duration_ms,
    }


def hdp_displacement_vector(weight_groups: dict[str, dict]) -> np.ndarray:
  """Effective Δθ_HDP from group mean |W| change (HDP-on segment)."""
  return np.array(
      [
          weight_groups[name]["abs_W_final"] - weight_groups[name]["abs_W0"]
          for name in PARAM_NAMES
      ],
      dtype=float,
  )


def estimate_h_timescale(h_trace: np.ndarray, dt_ms: float) -> float | None:
    """Crude 1/e decay time of |H - H_terminal| from end backward."""
    if h_trace.ndim != 2:
        return None
    h_mean = h_trace.mean(axis=1)
    terminal = h_mean[-1]
    dev = np.abs(h_mean - terminal)
    if dev.max() < 1e-9:
        return 0.0
    target = dev[0] / math.e
    idx = np.where(dev <= target)[0]
    if len(idx) == 0:
        return float(len(h_mean) * dt_ms)
    return float(idx[0] * dt_ms)


def phased_recovery_protocol(
    model: jtfne.Model,
    *,
    hdp_on: bool,
    perturb_scale: float = 1.5,
) -> dict[str, Any]:
    """baseline 3s → perturb 5s → restore 5s (13s total)."""
    phases = [
        ("baseline", 3000.0, 1.0),
        ("perturb", 5000.0, perturb_scale),
        ("restore", 5000.0, 1.0),
    ]
    continuation = None
    phase_metrics = []
    total_spikes = 0
    for label, dur, scale in phases:
        spikes, continuation, diag = continuation_segment(
            model, dur, hdp_on=hdp_on, drive_scale=scale, continuation=continuation
        )
        total_spikes += int(spikes.sum())
        pop, r_e, r_i = rates_from_spikes(spikes, DT_MS, burn_in_ms=0.0)
        phase_metrics.append(
            {
                "phase": label,
                "duration_ms": dur,
                "drive_scale": scale,
                "population_rate_hz_full_window": pop,
                "r_E_hz": r_e,
                "r_I_hz": r_i,
            }
        )
    # Recovery metric on restore phase vs baseline target
    base_pop = phase_metrics[0]["population_rate_hz_full_window"]
    pert_pop = phase_metrics[1]["population_rate_hz_full_window"]
    rest_pop = phase_metrics[2]["population_rate_hz_full_window"]
    eps = 1e-6
    r_hdp = 1.0 - abs(rest_pop - base_pop) / (abs(pert_pop - base_pop) + eps)
    return {
        "hdp_on": hdp_on,
        "phases": phase_metrics,
        "recovery_metric_R": float(r_hdp),
        "baseline_rate": base_pop,
        "post_perturb_rate": pert_pop,
        "restore_rate": rest_pop,
        "total_spikes": total_spikes,
    }


def load_theta_hat() -> dict[str, float]:
    if METRICS_10S.exists():
        data = json.loads(METRICS_10S.read_text())
        return dict(data["agsdr"]["theta_hat"])
    # Fallback: rerun canonical AGSDR
    base = jtfne.construct(mcc3_config())
    specs = mcc3_specs()
    objective = jtfne.rate_targets(
        groups={"all": list(range(10))},
        targets_hz={"all": 20.0},
        burn_in_ms=BURN_IN_MS,
    )
    tune_sim = jtfne.simulation(
        duration_ms=100.0, dt_ms=DT_MS, seed=SIM_SEED, runtime=mcc3_runtime(True)
    )
    result = base.tune(
        objectives=objective,
        optimizer=jtfne.agsdr(
            parameters=specs,
            generations=2,
            population_size=4,
            seed=OPT_SEED,
        ),
        simulation=tune_sim,
    )
    return dict(result.best_parameters)


def make_diagnostic_figure(report: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    j = np.asarray(report["jacobian_10s"]["J_W"])
    names = PARAM_NAMES
    x = np.arange(4)
    axes[0, 0].bar(x - 0.15, j[0], width=0.3, label="∂r_E/∂m")
    axes[0, 0].bar(x + 0.15, j[1], width=0.3, label="∂r_I/∂m")
    axes[0, 0].set_xticks(x, names)
    axes[0, 0].set_title("J_W (10 s, HDP on)")
    axes[0, 0].legend(fontsize=8)

    dtheta = np.asarray(report["hdp_displacement"]["delta_m_abs"])
    pred = j @ dtheta
    axes[0, 1].bar(["Δr_E pred", "Δr_I pred"], pred, color=["tab:red", "tab:blue"])
    axes[0, 1].set_title("J_W · Δ|W|_HDP")
    axes[0, 1].axhline(0, color="k", lw=0.5)

    scales = [row["drive_scale"] for row in report["drive_sweep_hdp_on"]]
    r_pop_on = [row["population_rate_hz"] for row in report["drive_sweep_hdp_on"]]
    r_pop_off = [row["population_rate_hz"] for row in report["drive_sweep_hdp_off"]]
    axes[0, 2].plot(scales, r_pop_on, "-o", label="HDP on")
    axes[0, 2].plot(scales, r_pop_off, "-s", label="HDP off")
    axes[0, 2].set_xlabel("drive scale")
    axes[0, 2].set_ylabel("population rate (Hz)")
    axes[0, 2].set_title("Drive sensitivity (10 s)")
    axes[0, 2].legend()

    rec_on = report["phased_recovery"]["hdp_on"]
    rec_off = report["phased_recovery"]["hdp_off"]
    axes[1, 0].bar(
        ["R_HDP on", "R_HDP off"],
        [rec_on["recovery_metric_R"], rec_off["recovery_metric_R"]],
    )
    axes[1, 0].set_title("Recovery metric R")
    axes[1, 0].set_ylim(0, 1.05)

    for row, lab in ((rec_on, "HDP on"), (rec_off, "HDP off")):
        phases = [p["phase"] for p in row["phases"]]
        rates = [p["population_rate_hz_full_window"] for p in row["phases"]]
        axes[1, 1].plot(phases, rates, "-o", label=lab)
    axes[1, 1].set_title("Phased protocol rates")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].axis("off")
    txt = (
        f"J_W·Δm ≈ {report['projection']['J_W_dot_delta_m']}\n"
        f"|projection|_2 = {report['projection']['projection_norm']:.4f}\n"
        f"τ_H est ≈ {report['timescales']['tau_H_ms']} ms\n"
        f"Verdict: {report['interpretation']['verdict']}"
    )
    axes[1, 2].text(0.05, 0.95, txt, va="top", family="monospace", fontsize=9)

    fig.suptitle("HDP-MVC read-only diagnostic (optimized MCC-3 circuit)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = mcc3_specs()
    base = jtfne.construct(mcc3_config())
    theta_hat = load_theta_hat()
    model = _model_with_parameters(base, theta_hat, specs)

    print("Computing J_W and J_H at 10 s...", flush=True)
    jacobian_10s = finite_difference_jacobian(base, theta_hat, specs, 10_000.0)
    jacobian_3s = finite_difference_jacobian(base, theta_hat, specs, 3_000.0)

    # HDP displacement from 10s HDP-on/off at theta_hat
    spikes_b, _, diag_b = continuation_segment(
        model, 10_000.0, hdp_on=True, drive_scale=1.0, continuation=None
    )
    spikes_c, _, diag_c = continuation_segment(
        model, 10_000.0, hdp_on=False, drive_scale=1.0, continuation=None
    )
    masks = {n: _edge_parameter_mask(model, n, specs[n]) for n in PARAM_NAMES}
    w0 = np.asarray(model.params["edge_list"].weight)
    w_final = np.asarray(diag_b["w_final"]) if diag_b else w0
    weight_groups = {}
    for name, mask in masks.items():
        w_tr = np.asarray(diag_b["w_trace"]) if diag_b and diag_b.get("w_trace") is not None else None
        series = w_tr[:, mask].mean(axis=1) if w_tr is not None else w0[mask]
        mag = np.abs(series)
        weight_groups[name] = {
            "abs_W0": float(np.abs(w0[mask]).mean()),
            "abs_W_final": float(np.abs(w_final[mask]).mean()),
            "delta_abs_W": float(np.abs(w_final[mask]).mean() - np.abs(w0[mask]).mean()),
        }
    delta_m = hdp_displacement_vector(weight_groups)
    j_w = np.asarray(jacobian_10s["J_W"])
    projection = j_w @ delta_m

    h_trace = np.asarray(diag_b["H_trace"]) if diag_b and diag_b.get("H_trace") is not None else None
    tau_h = estimate_h_timescale(h_trace, DT_MS) if h_trace is not None else None

    print("Drive sweep...", flush=True)
    drive_on, drive_off = [], []
    for scale in DRIVE_SCALES:
        drive_on.append(
            {"drive_scale": scale, **evaluate_at_theta(base, theta_hat, specs, 10_000.0, True, scale)}
        )
        drive_off.append(
            {"drive_scale": scale, **evaluate_at_theta(base, theta_hat, specs, 10_000.0, False, scale)}
        )

    print("Phased recovery protocol...", flush=True)
    rec_on = phased_recovery_protocol(model, hdp_on=True, perturb_scale=1.5)
    rec_off = phased_recovery_protocol(model, hdp_on=False, perturb_scale=1.5)

    pop_b, r_e_b, r_i_b = rates_from_spikes(spikes_b, DT_MS)
    pop_c, r_e_c, r_i_c = rates_from_spikes(spikes_c, DT_MS)
    b_eq_c = bool(
        np.allclose(spikes_b, spikes_c)
        and math.isclose(pop_b, pop_c, rel_tol=0, abs_tol=1e-6)
    )

    proj_norm = float(np.linalg.norm(projection))
    verdict = (
        "insensitive_subspace"
        if proj_norm < 0.5 and b_eq_c
        else "inspect_hdp_rule"
        if proj_norm >= 0.5 and rec_on["recovery_metric_R"] <= rec_off["recovery_metric_R"]
        else "mixed"
    )

    report = json_safe(
        {
            "identity": {
                "branch": subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "theta_hat_source": str(METRICS_10S) if METRICS_10S.exists() else "agsdr_rerun",
                "theta_hat": theta_hat,
            },
            "jacobian_10s": jacobian_10s,
            "jacobian_3s": jacobian_3s,
            "hdp_displacement": {
                "weight_groups": weight_groups,
                "delta_m_abs": delta_m.tolist(),
            },
            "projection": {
                "J_W_dot_delta_m": projection.tolist(),
                "projection_norm": proj_norm,
                "observed_rate_delta_B_minus_C": {
                    "population": pop_b - pop_c,
                    "r_E": r_e_b - r_e_c,
                    "r_I": r_i_b - r_i_c,
                },
                "spike_trains_identical_B_C": b_eq_c,
            },
            "drive_sweep_hdp_on": drive_on,
            "drive_sweep_hdp_off": drive_off,
            "phased_recovery": {"hdp_on": rec_on, "hdp_off": rec_off},
            "timescales": {
                "tau_H_ms": tau_h,
                "DEFAULT_HDP_tau_0_ms": DEFAULT_HDP.get("tau_0_ms"),
            },
            "acceptance_criteria_draft": {
                "note": "Predeclared targets for future HDP-MVC; not tuned post-hoc.",
                "R_HDP_min_delta_over_off": 0.1,
                "both_populations_active_baseline": True,
                "delta_H_min": 0.001,
                "delta_W_min": 0.01,
            },
            "interpretation": {
                "verdict": verdict,
                "mcc3_role": "software integration MCC — keep fast",
                "msvc_hdp_role": "scientific adaptation validation — separate experiment",
                "bad_validation_design": True,
                "hdp_rule_undetermined": verdict == "insensitive_subspace",
            },
        }
    )

    report_path = OUT / "hdp_mvc_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig_path = OUT / "hdp_mvc_diagnostic.png"
    make_diagnostic_figure(report, fig_path)
    print(json.dumps(report["projection"], indent=2))
    print(json.dumps(report["phased_recovery"], indent=2))
    print(f"wrote {report_path} and {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
