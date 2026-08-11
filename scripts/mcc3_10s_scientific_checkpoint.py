#!/usr/bin/env python3
"""MCC-3 10-second scientific checkpoint (analysis only — no package mutation)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxfne as jtfne
from jaxfne._model_tune import (
    _candidate_state_evidence,
    _edge_parameter_mask,
    _model_with_parameters,
)
from jaxfne.hdp_network import DEFAULT_HDP
from jaxfne.io import json_safe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "mcc3_10s_checkpoint"
EPS = 1e-9

DURATION_MS = 10_000.0
DT_MS = 0.1
BURN_IN_MS = 20.0
TARGET_HZ = 20.0
SIM_SEED = 17
OPT_SEED = 42
AGSDR_GENERATIONS = 2
AGSDR_POPULATION = 4
TUNE_DURATION_MS = 100.0  # canonical MCC-3 training horizon


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


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_branch() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ei_label(cell_type: str) -> str:
    return "E" if cell_type == "E" else "I"


def build_network_table(model: jtfne.Model) -> list[dict[str, Any]]:
    table = model.neuron_table()
    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre, dtype=int)
    post = np.asarray(edges.post, dtype=int)
    in_count = np.bincount(post, minlength=10)
    out_count = np.bincount(pre, minlength=10)
    rows = []
    positions = model.params.get("positions")
    for row in table:
        idx = int(row["neuron_id"])
        pos = None
        if positions is not None:
            pos = np.asarray(positions[idx]).tolist()
        rows.append(
            {
                "neuron_index": idx,
                "cell_type": row["cell_type"],
                "ei_identity": ei_label(str(row["cell_type"])),
                "layer": row.get("layer", "unspecified"),
                "area": row.get("area", "network"),
                "position": pos,
                "incoming_edge_count": int(in_count[idx]),
                "outgoing_edge_count": int(out_count[idx]),
            }
        )
    return rows


def edge_group_counts(model: jtfne.Model, specs: dict) -> dict[str, int]:
    return {
        name: int(_edge_parameter_mask(model, name, spec).sum())
        for name, spec in specs.items()
    }


def group_masks(model: jtfne.Model, specs: dict) -> dict[str, np.ndarray]:
    return {name: _edge_parameter_mask(model, name, spec) for name, spec in specs.items()}


def rate_hz(spikes: np.ndarray, dt_ms: float, start_idx: int, end_idx: int) -> float:
    window = spikes[start_idx:end_idx]
    return float(window.mean() * (1000.0 / dt_ms))


def per_neuron_rates(
    spikes: np.ndarray, dt_ms: float, start_idx: int, end_idx: int
) -> np.ndarray:
    window = spikes[start_idx:end_idx]
    return window.mean(axis=0) * (1000.0 / dt_ms)


def windowed_population_rates(
    spikes: np.ndarray, dt_ms: float, burn_in_ms: float, bin_ms: float = 1000.0
) -> dict[str, Any]:
    n_steps = spikes.shape[0]
    start_idx = int(math.ceil(burn_in_ms / dt_ms))
    bin_steps = int(round(bin_ms / dt_ms))
    rates = []
    edges = []
    t = []
    for b0 in range(start_idx, n_steps, bin_steps):
        b1 = min(b0 + bin_steps, n_steps)
        if b1 <= b0:
            break
        rates.append(rate_hz(spikes, dt_ms, b0, b1))
        edges.append((b0 * dt_ms, b1 * dt_ms))
        t.append(0.5 * (edges[-1][0] + edges[-1][1]))
    return {
        "bin_ms": bin_ms,
        "time_ms": t,
        "rates_hz": rates,
        "first_hz": rates[0] if rates else None,
        "last_hz": rates[-1] if rates else None,
        "mean_hz": float(np.mean(rates)) if rates else None,
        "sd_hz": float(np.std(rates)) if rates else None,
    }


def isi_stats(spikes: np.ndarray, dt_ms: float) -> dict[str, Any]:
    """ISI summary from binary spike matrix."""
    isis: list[float] = []
    per_neuron = []
    for i in range(spikes.shape[1]):
        idx = np.flatnonzero(spikes[:, i] > 0.5)
        if len(idx) < 2:
            per_neuron.append({"neuron": i, "n_spikes": int(len(idx)), "isi_cv": None})
            continue
        isi = np.diff(idx) * dt_ms
        isis.extend(isi.tolist())
        cv = float(np.std(isi) / (np.mean(isi) + EPS)) if len(isi) > 1 else None
        per_neuron.append(
            {"neuron": i, "n_spikes": int(len(idx)), "isi_cv": cv}
        )
    pop_cv = float(np.std(isis) / (np.mean(isis) + EPS)) if len(isis) > 1 else None
    return {
        "population_isi_cv": pop_cv,
        "n_isi_samples": len(isis),
        "per_neuron": per_neuron,
    }


def weight_group_trajectory(
    w_trace: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        series = w_trace[:, mask].mean(axis=1)
        mag = np.abs(series)
        out[name] = {
            "mean_W_t": series.tolist(),
            "mean_abs_W_t": mag.tolist(),
            "W0": float(series[0]),
            "W_final": float(series[-1]),
            "abs_W0": float(mag[0]),
            "abs_W_final": float(mag[-1]),
            "delta_mean_W": float(series[-1] - series[0]),
            "delta_abs_W": float(mag[-1] - mag[0]),
            "rel_delta_abs_W": float((mag[-1] - mag[0]) / (mag[0] + EPS)),
        }
    return out


def source_field_summary(signals: jtfne.Signals) -> dict[str, Any]:
    src = np.asarray(signals.sources) if signals.sources is not None else None
    meta = signals.metadata or {}
    rep = meta.get("representation", "relative")
    out: dict[str, Any] = {
        "representation": rep,
        "physical_amplitude_calibrated": meta.get("physical_amplitude_calibrated", False),
        "field_claim_level": meta.get("field_claim_level", "proxy_readout"),
        "field_solver_status": meta.get("field_solver_status", "linear_solver"),
        "source_calibration_status": meta.get("source_calibration_status"),
    }
    if src is not None:
        out.update(
            {
                "source_mean": float(np.mean(src)),
                "source_abs_max": float(np.max(np.abs(src))),
                "source_rms": float(np.sqrt(np.mean(src**2))),
            }
        )
    field = signals.field
    if field is not None:
        lfp = getattr(field, "lfp_proxy", None)
        if lfp is not None:
            lfp_a = np.asarray(lfp)
            out["lfp_proxy_rms"] = float(np.sqrt(np.mean(lfp_a**2)))
        csd = getattr(field, "csd_proxy", None)
        if csd is not None:
            csd_a = np.asarray(csd)
            out["csd_proxy_rms"] = float(np.sqrt(np.mean(csd_a**2)))
    return out


def psd_summary(signals: jtfne.Signals) -> dict[str, Any]:
    if signals.sources is None:
        return {"status": "no_sources"}
    pop = np.asarray(signals.sources).mean(axis=1, keepdims=True)  # (T,1)
    sig = jnp.asarray(pop[None, :, :])  # (1,T,1)
    fs = 1000.0 / float(signals.metadata.get("dt_ms", DT_MS))
    psd = np.asarray(jtfne.spectrolaminar_psd_jax(sig, fs=fs))
    freqs = np.linspace(1.0, 150.0, psd.shape[0])
    peak_idx = int(np.argmax(psd[:, 0]))
    return {
        "fs_hz": fs,
        "dominant_freq_hz": float(freqs[peak_idx]),
        "dominant_power": float(psd[peak_idx, 0]),
        "freqs_hz": freqs.tolist(),
        "psd": psd[:, 0].tolist(),
    }


def run_condition(
    base_model: jtfne.Model,
    params: dict[str, float] | None,
    specs: dict,
    sim: jtfne.Simulation,
    objective: jtfne.Objective,
    label: str,
) -> dict[str, Any]:
    model = (
        _model_with_parameters(base_model, params, specs)
        if params is not None
        else base_model
    )
    w0 = np.asarray(model.params["edge_list"].weight).copy()
    signals = model.simulate(sim)
    diag = model.last_hdp_diagnostics()
    report = model.evaluate(
        signals,
        objective,
        state_diagnostics=diag,
    )
    spikes = np.asarray(signals.spikes)
    dt_ms = float(signals.metadata.get("dt_ms", DT_MS))
    start_idx = int(math.ceil(BURN_IN_MS / dt_ms))
    end_idx = spikes.shape[0]
    table = model.neuron_table()
    e_idx = [int(r["neuron_id"]) for r in table if r["cell_type"] == "E"]
    i_idx = [int(r["neuron_id"]) for r in table if r["cell_type"] == "PV"]
    neuron_rates = per_neuron_rates(spikes, dt_ms, start_idx, end_idx)
    pop_rate = rate_hz(spikes, dt_ms, start_idx, end_idx)
    masks = group_masks(model, specs)
    w_traj = None
    h_traj = None
    if diag is not None:
        if diag.get("w_trace") is not None:
            w_traj = weight_group_trajectory(np.asarray(diag["w_trace"]), masks)
        if diag.get("H_trace") is not None:
            h_arr = np.asarray(diag["H_trace"])
            if h_arr.ndim == 2:
                h_traj = {
                    "mean_H_t": h_arr.mean(axis=1).tolist(),
                    "per_neuron_terminal_H": h_arr[-1].tolist(),
                    "H0_mean": float(h_arr[0].mean()),
                    "H_final_mean": float(h_arr[-1].mean()),
                    "H_range": [float(h_arr.min()), float(h_arr.max())],
                    "H_drift": float(h_arr[-1].mean() - h_arr[0].mean()),
                }
    w_final = (
        np.asarray(diag["w_final"])
        if diag is not None and diag.get("w_final") is not None
        else np.asarray(model.params["edge_list"].weight)
    )
    return {
        "label": label,
        "model": model,
        "signals": signals,
        "report": report,
        "diagnostics": diag,
        "W0": w0,
        "W_final": w_final,
        "population_rate_hz": pop_rate,
        "E_rate_hz": rate_hz(spikes[:, e_idx], dt_ms, start_idx, end_idx) if e_idx else None,
        "I_rate_hz": rate_hz(spikes[:, i_idx], dt_ms, start_idx, end_idx) if i_idx else None,
        "neuron_rates_hz": neuron_rates.tolist(),
        "rate_target_error_hz": abs(pop_rate - TARGET_HZ),
        "windowed_rates": windowed_population_rates(spikes, dt_ms, BURN_IN_MS),
        "spike_total": int(spikes.sum()),
        "spikes_per_neuron": spikes.sum(axis=0).astype(int).tolist(),
        "isi": isi_stats(spikes, dt_ms),
        "kappa_synchrony": float(jtfne.kappa_synchrony(spikes, dt_ms)),
        "weight_groups": w_traj,
        "H": h_traj,
        "source_field": source_field_summary(signals),
        "psd": psd_summary(signals),
        "state_evidence": _candidate_state_evidence(model, diag),
        "Vm_mean": np.asarray(signals.V_m).mean(axis=0).tolist(),
        "Vm_range": [float(np.min(signals.V_m)), float(np.max(signals.V_m))],
        "Vm_finite": bool(np.isfinite(signals.V_m).all()),
    }


def make_figure(
    network_rows: list[dict],
    theta0: dict,
    thetah: dict,
    tune_summary: dict,
    A: dict,
    B: dict,
    C: dict,
    out_png: Path,
) -> None:
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(6, 4, hspace=0.35, wspace=0.3)

    # A network schematic
    ax_net = fig.add_subplot(gs[0, :2])
    pre = np.asarray(A["model"].params["edge_list"].pre)
    post = np.asarray(A["model"].params["edge_list"].post)
    w = np.asarray(A["model"].params["edge_list"].weight)
    xy = {}
    for r in network_rows:
        idx = r["neuron_index"]
        xy[idx] = (idx % 5, idx // 5)
        color = "tab:red" if r["ei_identity"] == "E" else "tab:blue"
        ax_net.scatter(*xy[idx], s=120, c=color, edgecolors="k", zorder=3)
        ax_net.text(xy[idx][0], xy[idx][1] + 0.15, f"{idx}", ha="center", fontsize=8)
    for p, q, wt in zip(pre, post, w):
        ax_net.annotate(
            "",
            xy=xy[int(q)],
            xytext=xy[int(p)],
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.3 + 0.4 * min(abs(wt), 3)),
        )
    ax_net.set_title("A. 10-neuron MCC-3 network (E=red, I/PV=blue)")
    ax_net.set_aspect("equal")
    ax_net.axis("off")

    # B parameter change
    ax_p = fig.add_subplot(gs[0, 2:])
    names = list(theta0.keys())
    x = np.arange(len(names))
    b0 = [theta0[n] for n in names]
    b1 = [thetah[n] for n in names]
    ax_p.bar(x - 0.15, b0, width=0.3, label=r"$\theta_0$")
    ax_p.bar(x + 0.15, b1, width=0.3, label=r"$\hat\theta$")
    ax_p.set_xticks(x, names)
    ax_p.set_ylabel("magnitude")
    ax_p.set_title("B. AGSDR grouped magnitudes")
    ax_p.legend()

    t = np.asarray(A["signals"].time_ms)
    dt = float(A["signals"].metadata.get("dt_ms", DT_MS))

    def raster(ax, cond, title):
        sp = np.asarray(cond["signals"].spikes)
        ids, ts = np.where(sp > 0.5)
        ax.scatter(ts * dt, ids, s=1, c="k", marker="|")
        ax.set_title(title)
        ax.set_xlim(0, DURATION_MS)
        ax.set_ylim(-0.5, 9.5)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("neuron")

    ax_c = fig.add_subplot(gs[1, :2])
    raster(ax_c, A, "C. Raster — Pre AGSDR (A)")
    ax_d = fig.add_subplot(gs[1, 2:])
    raster(ax_d, B, "D. Raster — Post AGSDR (B)")

    ax_e = fig.add_subplot(gs[2, :2])
    for cond, lab, col in ((A, "A pre", "tab:gray"), (B, "B post", "tab:green"), (C, "C HDP-off", "tab:orange")):
        wr = cond["windowed_rates"]
        ax_e.plot(wr["time_ms"], wr["rates_hz"], "-o", ms=3, label=lab, color=col)
    ax_e.axhline(TARGET_HZ, color="k", ls="--", lw=1, label="target")
    ax_e.set_title("E. Population rate (1 s bins, post burn-in)")
    ax_e.set_xlabel("time (ms)")
    ax_e.set_ylabel("Hz")
    ax_e.legend(fontsize=8)

    ax_f = fig.add_subplot(gs[2, 2:])
    rep_n = [0, 4, 9]
    for cond, lab in ((A, "A"), (B, "B")):
        vm = np.asarray(cond["signals"].V_m)
        for n in rep_n:
            ax_f.plot(t, vm[:, n], lw=0.8, alpha=0.8, label=f"{lab} n{n}")
    ax_f.set_title("F. Representative Vm (native units)")
    ax_f.set_xlabel("time (ms)")

    ax_g = fig.add_subplot(gs[3, :2])
    if A.get("H") and B.get("H"):
        ax_g.plot(t, A["H"]["mean_H_t"], label="A mean H")
        ax_g.plot(t, B["H"]["mean_H_t"], label="B mean H")
        ax_g.set_title("G. Mean H-state trajectory")
        ax_g.legend(fontsize=8)

    ax_h = fig.add_subplot(gs[3, 2:])
    if A.get("weight_groups") and B.get("weight_groups"):
        for g in ("m_EE", "m_EI", "m_IE", "m_II"):
            ax_h.plot(
                t,
                A["weight_groups"][g]["mean_abs_W_t"],
                ls="--",
                lw=0.8,
                label=f"A |{g}|",
            )
            ax_h.plot(t, B["weight_groups"][g]["mean_abs_W_t"], lw=1.0, label=f"B |{g}|")
        ax_h.set_title("H. Group |W| trajectories")
        ax_h.legend(fontsize=6, ncol=2)

    ax_i = fig.add_subplot(gs[4, :2])
    if A["signals"].sources is not None:
        src = np.asarray(A["signals"].sources).mean(axis=1)
        src_b = np.asarray(B["signals"].sources).mean(axis=1)
        ax_i.plot(t, src, lw=0.6, label="A pop source")
        ax_i.plot(t, src_b, lw=0.6, label="B pop source")
        ax_i.set_title("I. Population source mean")
        ax_i.legend(fontsize=8)

    ax_j = fig.add_subplot(gs[4, 2:])
    if A["psd"].get("psd"):
        freqs = np.asarray(A["psd"]["freqs_hz"])
        ax_j.semilogy(freqs, A["psd"]["psd"], label="A")
        ax_j.semilogy(freqs, B["psd"]["psd"], label="B")
        ax_j.set_title("J. Population-source PSD")
        ax_j.set_xlabel("Hz")
        ax_j.legend()

    ax_k = fig.add_subplot(gs[5, :2])
    ax_k.axis("off")
    ax_k.set_title("K. Objective summary (100 ms tune window)")
    lines = [
        f"initial score: {tune_summary.get('initial_score')}",
        f"best score: {tune_summary.get('best_score')}",
        f"A rate err @10s: {A['rate_target_error_hz']:.3f} Hz",
        f"B rate err @10s: {B['rate_target_error_hz']:.3f} Hz",
        f"candidates: {tune_summary.get('n_candidates')}",
        f"rejected: {tune_summary.get('n_rejected')}",
    ]
    ax_k.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)

    ax_l = fig.add_subplot(gs[5, 2:])
    ax_l.axis("off")
    ax_l.set_title("L. 10 s summary")
    summary_lines = [
        f"pop rate A/B/C: {A['population_rate_hz']:.2f} / {B['population_rate_hz']:.2f} / {C['population_rate_hz']:.2f}",
        f"E rate A/B: {A['E_rate_hz']:.2f} / {B['E_rate_hz']:.2f}",
        f"I rate A/B: {A['I_rate_hz']:.2f} / {B['I_rate_hz']:.2f}",
        f"kappa A/B: {A['kappa_synchrony']:.3f} / {B['kappa_synchrony']:.3f}",
        f"spikes A/B/C: {A['spike_total']} / {B['spike_total']} / {C['spike_total']}",
    ]
    ax_l.text(0.02, 0.95, "\n".join(summary_lines), va="top", family="monospace", fontsize=9)

    fig.suptitle("MCC-3 10 s scientific checkpoint — Pre vs Post AGSDR", fontsize=14, y=0.995)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = mcc3_specs()
    objective = jtfne.rate_targets(
        groups={"all": list(range(10))},
        targets_hz={"all": TARGET_HZ},
        burn_in_ms=BURN_IN_MS,
    )
    base = jtfne.construct(mcc3_config())
    network_table = build_network_table(base)
    group_ct = edge_group_counts(base, specs)
    n_edges = int(base.params["edge_list"].n_edges)

    # Canonical AGSDR at 100 ms training horizon (same as accepted MCC-3 test).
    tune_sim = jtfne.simulation(
        duration_ms=TUNE_DURATION_MS,
        dt_ms=DT_MS,
        seed=SIM_SEED,
        runtime=mcc3_runtime(True),
    )
    optimizer = jtfne.agsdr(
        parameters=specs,
        generations=AGSDR_GENERATIONS,
        population_size=AGSDR_POPULATION,
        seed=OPT_SEED,
    )
    print("Running canonical AGSDR (100 ms training sim)...", flush=True)
    tune_result = base.tune(
        objectives=objective,
        optimizer=optimizer,
        simulation=tune_sim,
    )
    theta0 = dict(tune_result.summary["theta_0"])
    thetah = dict(tune_result.best_parameters)

    sim_10s_on = jtfne.simulation(
        duration_ms=DURATION_MS,
        dt_ms=DT_MS,
        seed=SIM_SEED,
        runtime=mcc3_runtime(True),
    )
    sim_10s_off = jtfne.simulation(
        duration_ms=DURATION_MS,
        dt_ms=DT_MS,
        seed=SIM_SEED,
        runtime=mcc3_runtime(False),
    )

    print("Running A (theta0, HDP on) 10 s...", flush=True)
    A = run_condition(base, None, specs, sim_10s_on, objective, "A")
    print("Running B (theta_hat, HDP on) 10 s...", flush=True)
    B = run_condition(base, thetah, specs, sim_10s_on, objective, "B")
    print("Running C (theta_hat, HDP off) 10 s...", flush=True)
    C = run_condition(base, thetah, specs, sim_10s_off, objective, "C")

    # Topology identity checks
    assert np.array_equal(
        np.asarray(A["model"].params["edge_list"].pre),
        np.asarray(B["model"].params["edge_list"].pre),
    )
    assert np.array_equal(
        np.asarray(A["model"].params["edge_list"].post),
        np.asarray(B["model"].params["edge_list"].post),
    )

    identity = {
        "branch": git_branch(),
        "sha": git_head(),
        "package_version": jtfne.__version__,
        "working_tree_status": git_status(),
        "simulation_seed": SIM_SEED,
        "optimizer_seed": OPT_SEED,
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "burn_in_ms": BURN_IN_MS,
        "target_firing_rate_hz": TARGET_HZ,
        "agsdr_training_duration_ms": TUNE_DURATION_MS,
        "agsdr_generations": AGSDR_GENERATIONS,
        "agsdr_population_size": AGSDR_POPULATION,
        "agsdr_regenerated": True,
        "agsdr_note": (
            "AGSDR rerun with exact accepted MCC-3 optimizer configuration; "
            "training sim remains 100 ms, evaluation sim is 10 s."
        ),
    }

    agsdr_report = {
        "theta_0": theta0,
        "theta_hat": thetah,
        "delta_m": {k: float(thetah[k] - theta0[k]) for k in theta0},
        "pct_delta_m": {
            k: float(100.0 * (thetah[k] - theta0[k]) / (abs(theta0[k]) + EPS))
            for k in theta0
        },
        "bounds": {k: list(specs[k].bounds) for k in specs},
        "signs_unchanged": bool(
            np.all(
                np.sign(np.asarray(B["W0"]))
                == np.sign(np.asarray(base.params["edge_list"].weight))
            )
        ),
        "initial_objective_100ms": (
            tune_result.summary["candidate_evaluations"][0]["objective"].get("total_score")
            if tune_result.summary.get("candidate_evaluations")
            else None
        ),
        "best_objective_100ms": tune_result.best_score,
        "initial_rate_error_100ms": abs(
            tune_result.summary["candidate_evaluations"][0]["objective"].get("rate", 0)
            - TARGET_HZ
        )
        if tune_result.summary["candidate_evaluations"]
        else None,
        "optimized_rate_error_100ms": abs(
            tune_result.summary["best_evaluation"]["objective"].get("rate", 0) - TARGET_HZ
        )
        if tune_result.summary.get("best_evaluation")
        else None,
        "n_candidate_evaluations": len(tune_result.summary.get("candidate_evaluations", [])),
        "n_rejected": tune_result.summary.get("candidate_rejection_count", 0),
    }

    abc_table = {
        "population_rate_hz": {
            "A": A["population_rate_hz"],
            "B": B["population_rate_hz"],
            "C": C["population_rate_hz"],
        },
        "target_error_hz": {
            "A": A["rate_target_error_hz"],
            "B": B["rate_target_error_hz"],
            "C": C["rate_target_error_hz"],
        },
        "E_rate_hz": {"A": A["E_rate_hz"], "B": B["E_rate_hz"], "C": C["E_rate_hz"]},
        "I_rate_hz": {"A": A["I_rate_hz"], "B": B["I_rate_hz"], "C": C["I_rate_hz"]},
        "total_spikes": {
            "A": A["spike_total"],
            "B": B["spike_total"],
            "C": C["spike_total"],
        },
        "source_rms": {
            "A": A["source_field"].get("source_rms"),
            "B": B["source_field"].get("source_rms"),
            "C": C["source_field"].get("source_rms"),
        },
        "readout_rms": {
            "A": A["source_field"].get("lfp_proxy_rms"),
            "B": B["source_field"].get("lfp_proxy_rms"),
            "C": C["source_field"].get("lfp_proxy_rms"),
        },
        "objective_total_score_10s": {
            "A": A["report"].get("total_score"),
            "B": B["report"].get("total_score"),
            "C": C["report"].get("total_score"),
        },
    }

    pathology = {
        "finite_Vm": {"A": A["Vm_finite"], "B": B["Vm_finite"], "C": C["Vm_finite"]},
        "weight_bounds": B["report"].get("state_validity", {}).get("bounds"),
        "H_finite": {
            "A": A["report"].get("state_validity", {}).get("finite", {}).get("H_trace"),
            "B": B["report"].get("state_validity", {}).get("finite", {}).get("H_trace"),
        },
        "silence": {k: v["spike_total"] == 0 for k, v in (("A", A), ("B", B), ("C", C))},
        "extreme_sync_kappa": {
            "A": A["kappa_synchrony"],
            "B": B["kappa_synchrony"],
            "C": C["kappa_synchrony"],
        },
    }

    metrics = json_safe(
        {
            "identity": identity,
            "network": {
                "N": 10,
                "E_edges": n_edges,
                "group_edge_counts": group_ct,
                "neurons": network_table,
            },
            "agsdr": agsdr_report,
            "abc_table": abc_table,
            "firing": {
                "A": {
                    k: A[k]
                    for k in (
                        "population_rate_hz",
                        "E_rate_hz",
                        "I_rate_hz",
                        "neuron_rates_hz",
                        "rate_target_error_hz",
                        "windowed_rates",
                        "spike_total",
                        "spikes_per_neuron",
                        "isi",
                        "kappa_synchrony",
                    )
                },
                "B": {
                    k: B[k]
                    for k in (
                        "population_rate_hz",
                        "E_rate_hz",
                        "I_rate_hz",
                        "neuron_rates_hz",
                        "rate_target_error_hz",
                        "windowed_rates",
                        "spike_total",
                        "spikes_per_neuron",
                        "isi",
                        "kappa_synchrony",
                    )
                },
                "C": {
                    k: C[k]
                    for k in (
                        "population_rate_hz",
                        "E_rate_hz",
                        "I_rate_hz",
                        "neuron_rates_hz",
                        "rate_target_error_hz",
                        "windowed_rates",
                        "spike_total",
                        "spikes_per_neuron",
                        "isi",
                        "kappa_synchrony",
                    )
                },
            },
            "H_state": {"A": A.get("H"), "B": B.get("H")},
            "weights": {"A": A.get("weight_groups"), "B": B.get("weight_groups")},
            "source_field": {"A": A["source_field"], "B": B["source_field"], "C": C["source_field"]},
            "spectral": {"A": A["psd"], "B": B["psd"]},
            "pathology": pathology,
        }
    )

    # CSV exports
    with (OUT / "mcc3_10s_rates.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "neuron_index", "rate_hz"])
        for cond in (A, B, C):
            for i, r in enumerate(cond["neuron_rates_hz"]):
                w.writerow([cond["label"], i, r])

    with (OUT / "mcc3_10s_weights.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "group", "time_ms", "mean_abs_W"])
        t = np.asarray(A["signals"].time_ms)
        for cond in (A, B):
            if not cond.get("weight_groups"):
                continue
            for g, meta in cond["weight_groups"].items():
                for ti, val in zip(t, meta["mean_abs_W_t"]):
                    w.writerow([cond["label"], g, float(ti), val])

    with (OUT / "mcc3_10s_H.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "time_ms", "mean_H"])
        t = np.asarray(A["signals"].time_ms)
        for cond in (A, B):
            if not cond.get("H"):
                continue
            for ti, val in zip(t, cond["H"]["mean_H_t"]):
                w.writerow([cond["label"], float(ti), val])

    metrics_path = OUT / "mcc3_10s_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    png_path = OUT / "mcc3_10s_pre_post.png"
    make_figure(network_table, theta0, thetah, agsdr_report, A, B, C, png_path)

    manifest = json_safe(
        {
            "checkpoint": "mcc3_10s_scientific",
            "identity": identity,
            "artifacts": {
                "figure_png": str(png_path.relative_to(ROOT)),
                "metrics_json": str(metrics_path.relative_to(ROOT)),
                "rates_csv": "artifacts/mcc3_10s_checkpoint/mcc3_10s_rates.csv",
                "weights_csv": "artifacts/mcc3_10s_checkpoint/mcc3_10s_weights.csv",
                "H_csv": "artifacts/mcc3_10s_checkpoint/mcc3_10s_H.csv",
            },
            "hashes": {
                "mcc3_10s_pre_post.png": sha256_file(png_path),
                "mcc3_10s_metrics.json": sha256_file(metrics_path),
            },
            "abc_table": abc_table,
        }
    )
    manifest_path = OUT / "mcc3_10s_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
