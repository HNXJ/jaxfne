"""Spectrolaminar TFNE Izhikevich + HDP tuning pipeline.

Practical test/debugging guide for Homeostasis-Dependent Plasticity (HDP),
i.e. ``jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp``. HDP is not
yet wired into ``core.py``/``RuntimeConfig`` (open follow-up task), so this
script drives the raw kernel directly against a ``jtfne.construct()``-built
laminar-column ``Model``, reusing its already-built ``emitter``/``edge_list``/
``positions`` params rather than calling ``model.simulate()``.

Cells (run top-to-bottom, or import and call the functions individually):
  1. CONFIG -- every tunable in one dict: network size/shape (N, areas,
     layers, cell-type fractions, duration_ms, dt_ms) and every HDP gain
     (H_min/H_max/tau_0_ms/alpha/beta/gamma/delta/C_spike/K_HDP/w_floor/
     w_ceiling), plus the two sweep grids (K_HDP_GRID, TAU0_GRID).
  2. build_model(cfg) -- construct the laminar column once. construct() is
     the slow step (~tens of ms-s depending on N); every later cell reuses
     this same built Model (see project memory
     `reuse-built-model-not-rebuild`); only structural cfg changes
     (counts/layers/cell-types) require a rebuild.
  3. run_hdp(...) + summarize_run(...) -- one HDP-kernel run plus a
     population report: per-cell-type firing rate (Hz), kappa synchrony,
     H_i drift from its 1.0 equilibrium, and weight-saturation fraction.
  4. K_HDP sweep -- scans the global plasticity gain at fixed tau_0_ms,
     looking for a value that holds the user's target firing-rate band
     (5-15 Hz for most units) without uniform-suppression collapse. A
     prior smoke test found gamma/delta's "spending" terms can dominate
     and globally brake the whole population rather than selectively
     correcting only the overactive subset -- this sweep is how that
     failure mode is supposed to surface and get tuned out.
  5. tau_0_ms sweep (short vs. long HDP time constant) at the chosen K_HDP.
  6. Spectrolaminar motif -- project_laminar_sources + spectrolaminar_psd_jax
     give a depth x frequency power map at the tuned operating point.
  7. E-vs-I spike-waveform/size diagnostic. Caveat: Izhikevich's hard
     v-reset has no real action-potential shape, so "waveform width" here
     is a spike-triggered V_m average (a scaffold proxy), not a calibrated
     extracellular AP waveform; cell-type *size* (tau_i scaling) is real
     kernel state (DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE), not a proxy.
  8. Save a JSON-safe receipt + PNG figures under
     outputs/hdp_spectrolaminar_pipeline/.

Usage:
    PYTHONPATH=. python scripts/spectrolaminar_tfne_izhikevich_pipeline.py

Claim status: computational scaffold / proxy diagnostics only (see
jaxfne/AGENTS.md SS13 claim-language rules). No biological-validation
claim; HDP's calibration is an open question this script is meant to
inform, not a validated result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import simulate_edge_recurrent_izhikevich_hdp

OUTPUT_DIR = Path("outputs/hdp_spectrolaminar_pipeline")

# %% [1] CONFIG -- every parameter used anywhere in this script.


def build_config() -> dict[str, Any]:
    """Single source of truth for network shape, HDP gains, and sweep grids."""

    return {
        # --- Network shape ---
        "areas": ["V1"],
        "layers": ["L1", "L2/3", "L4", "L5", "L6"],
        "n_neurons": 1000,
        "cell_type_fractions": {"E": 0.78, "PV": 0.12, "SST": 0.06, "VIP": 0.04},
        # Canonical column: count proportional to thickness (see project
        # memory `canonical-cortical-column-template`).
        "layer_fractions": {
            "L1": (0.00, 0.10),
            "L2/3": (0.10, 0.45),
            "L4": (0.45, 0.65),
            "L5": (0.65, 0.85),
            "L6": (0.85, 1.00),
        },
        "seed": 0,
        # --- Timing ---
        "duration_ms": 1000.0,
        "dt_ms": 0.5,
        # Longer duration used only for the final spectrolaminar-motif cell,
        # where frequency resolution benefits from more samples.
        "spectro_duration_ms": 2000.0,
        # Shorter duration for the K_HDP x tau_0_ms stability map (24 runs
        # at the grid sizes below) -- diagnostic only, not the tuned point.
        "stability_duration_ms": 500.0,
        # --- Field/probe (objective-grammar Configuration requires both) ---
        "field_kwargs": dict(
            domain="laminar_column", conductivity="proxy",
            boundary="mean_zero_neumann", gauge="mean_zero",
        ),
        "probe_kwargs": dict(name="hdp_probe", modes=["spikes", "V_m"]),
        "n_contacts": 16,
        "contact_width": 0.10,
        "projection_mode": "density_preserving",
        # --- HDP defaults (the H_i + global K_HDP kernel; see emitters.py) ---
        "hdp_defaults": {
            "H_min": 0.1,
            "H_max": 10.0,
            "tau_0_ms": 200.0,
            "alpha": 0.01,
            "beta": 0.0,
            "gamma": 0.02,
            "delta": 0.0,
            "C_spike": 0.0,
            "K_HDP": 0.2,
            "w_floor": 1.0e-3,
            "w_ceiling": 50.0,
        },
        # --- Sweep grids ---
        "K_HDP_GRID": [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
        "TAU0_GRID_MS": [20.0, 50.0, 200.0, 800.0],
        # --- Target band (user-specified tuning goal) ---
        "target_rate_hz": (5.0, 15.0),
        "kappa_stable_max": 0.3,  # see project memory `low-synchrony-kappa-for-spectrolaminar`
    }


# %% [2] Build the laminar column Model once; reuse it for every run below.


def build_model(cfg: dict[str, Any]) -> "jtfne.core.Model":
    """Construct the laminar-column Model. Call once; reuse across all sweeps."""

    builder = jtfne.laminar_cortex_config(
        areas=cfg["areas"],
        layers=cfg["layers"],
        n=cfg["n_neurons"],
        duration_ms=cfg["duration_ms"],
        dt_ms=cfg["dt_ms"],
        seed=cfg["seed"],
        emitter="izhikevich",
    ).layer_fractions(layer_fractions=cfg["layer_fractions"])

    for area in cfg["areas"]:
        builder = builder.area_layer_cell_types(
            area, {layer: dict(cfg["cell_type_fractions"]) for layer in cfg["layers"]}
        )

    builder = builder.field(**cfg["field_kwargs"]).probe(**cfg["probe_kwargs"])
    return jtfne.construct(builder)


# %% [3] One HDP run + a population-rate/stability report.


def run_hdp(
    model: "jtfne.core.Model",
    cfg: dict[str, Any],
    *,
    seed: int,
    duration_ms: float | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the raw HDP kernel against the model's existing emitter/edge_list params."""

    hdp_kwargs = dict(cfg["hdp_defaults"])
    if overrides:
        hdp_kwargs.update(overrides)

    duration_ms = cfg["duration_ms"] if duration_ms is None else duration_ms
    n_steps = int(round(duration_ms / cfg["dt_ms"]))
    key = jax.random.PRNGKey(seed)

    voltages, spikes, sources, diagnostics = simulate_edge_recurrent_izhikevich_hdp(
        model.params["emitter"],
        model.params["edge_list"],
        n_steps,
        cfg["dt_ms"],
        key,
        **hdp_kwargs,
    )
    return {
        "voltages": voltages,
        "spikes": spikes,
        "sources": sources,
        "diagnostics": diagnostics,
        "hdp_kwargs": hdp_kwargs,
        "n_steps": n_steps,
    }


def summarize_run(model: "jtfne.core.Model", cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Per-cell-type rate (Hz), synchrony, HDP drift, and weight-saturation stats."""

    labels = np.asarray(model.params["emitter"].labels)
    spikes_np = np.asarray(run["spikes"])  # (n_steps, n_neurons)
    duration_s = (run["n_steps"] * cfg["dt_ms"]) / 1000.0

    rate_by_type: dict[str, float] = {}
    for cell_type in sorted(set(labels.tolist())):
        mask = labels == cell_type
        if not mask.any():
            continue
        rate_by_type[cell_type] = float(spikes_np[:, mask].sum() / (mask.sum() * duration_s))

    overall_rate = float(spikes_np.sum() / (spikes_np.shape[1] * duration_s))
    kappa = float(jtfne.kappa_synchrony(spikes_np, dt_ms=cfg["dt_ms"]))

    diagnostics = run["diagnostics"]
    H_final = np.asarray(diagnostics["H_final"])
    w_final = np.asarray(diagnostics["w_final"])
    w_ceiling = run["hdp_kwargs"]["w_ceiling"]
    w_floor = run["hdp_kwargs"]["w_floor"]
    saturated = (np.abs(w_final) >= 0.999 * w_ceiling) | (np.abs(w_final) <= 1.001 * w_floor)

    return {
        "rate_by_type_hz": rate_by_type,
        "overall_rate_hz": overall_rate,
        "kappa_synchrony": kappa,
        "H_final_mean": float(H_final.mean()),
        "H_final_max_abs_drift": float(np.abs(H_final - 1.0).max()),
        "weight_saturation_fraction": float(saturated.mean()),
        "hdp_kwargs": run["hdp_kwargs"],
    }


def _in_target_band(rate_by_type_hz: dict[str, float], band: tuple[float, float]) -> bool:
    lo, hi = band
    return all(lo <= r <= hi for r in rate_by_type_hz.values())


# %% [3b] HDP state-trajectory diagnostics (H_trace/w_trace are already
# returned by the kernel -- no kernel change needed). This directly answers
# the unresolved question raised in review: when an overactive cell type's
# rate sits outside the target band, is H not tracking it (estimator-blind,
# mean(H-1) ~= 0) or is H tracking it but the weight update isn't correcting
# it (controller-ineffective, |H-1| >> 0)? A single final-state snapshot
# cannot distinguish "H never moves" / "H moves but weights saturate" / "H
# oscillates" -- only the full time series can.


def compute_hdp_state_diagnostics(
    model: "jtfne.core.Model", cfg: dict[str, Any], run: dict[str, Any], *, bin_ms: float = 20.0
) -> dict[str, dict[str, np.ndarray]]:
    """Per-cell-type time series: mean_H, var_H, mean_weight (full res) + binned rate/pressure."""

    labels = np.asarray(model.params["emitter"].labels)
    diagnostics = run["diagnostics"]
    H_trace = np.asarray(diagnostics["H_trace"])  # (n_steps, n_neurons)
    w_trace = np.asarray(diagnostics["w_trace"])  # (n_steps, n_edges)
    spikes = np.asarray(run["spikes"])  # (n_steps, n_neurons)
    pre = np.asarray(model.params["edge_list"].pre)
    dt_ms = cfg["dt_ms"]
    n_steps = H_trace.shape[0]
    t_ms = np.arange(n_steps) * dt_ms

    bin_steps = max(1, int(round(bin_ms / dt_ms)))
    n_bins = n_steps // bin_steps

    per_type: dict[str, dict[str, np.ndarray]] = {}
    for cell_type in sorted(set(labels.tolist())):
        mask = labels == cell_type
        if not mask.any():
            continue
        H_ct = H_trace[:, mask]  # (n_steps, n_ct)
        mean_H_t = H_ct.mean(axis=1)
        var_H_t = H_ct.var(axis=1)

        edge_mask = mask[pre]
        w_ct = w_trace[:, edge_mask] if edge_mask.any() else np.full((n_steps, 0), np.nan)
        mean_weight_t = np.abs(w_ct).mean(axis=1) if w_ct.shape[1] else np.full(n_steps, np.nan)

        spikes_ct = spikes[:, mask][: n_bins * bin_steps]
        rate_binned_hz = (
            spikes_ct.reshape(n_bins, bin_steps, -1).sum(axis=(1, 2))
            / (mask.sum() * (bin_steps * dt_ms / 1000.0))
        )
        H_binned = mean_H_t[: n_bins * bin_steps].reshape(n_bins, bin_steps).mean(axis=1)
        t_bin_ms = (np.arange(n_bins) * bin_steps + bin_steps / 2.0) * dt_ms

        per_type[cell_type] = {
            "t_ms": t_ms,
            "mean_H_t": mean_H_t,
            "var_H_t": var_H_t,
            "mean_weight_t": mean_weight_t,
            "t_bin_ms": t_bin_ms,
            "rate_binned_hz": rate_binned_hz,
            "pressure_binned": H_binned - 1.0,
        }
    return per_type


def classify_hdp_state(per_type: dict[str, dict[str, np.ndarray]], band: tuple[float, float], *, pressure_eps: float = 0.05) -> dict[str, dict[str, Any]]:
    """Steady-state (second-half) rate/pressure classification per cell type.

    Operationalizes the estimator-blind vs. controller-ineffective question:
    if a cell type's rate is out of band, low |H-1| means H is not encoding
    the overactivity (estimator-blind); large |H-1| means H sees it but the
    weight update isn't correcting it (controller-ineffective).
    """

    classification: dict[str, dict[str, Any]] = {}
    for cell_type, d in per_type.items():
        n_bins = d["rate_binned_hz"].shape[0]
        half = n_bins // 2
        steady_rate = float(np.mean(d["rate_binned_hz"][half:])) if n_bins else float("nan")
        steady_pressure = float(np.mean(d["pressure_binned"][half:])) if n_bins else float("nan")
        in_band = band[0] <= steady_rate <= band[1]
        if in_band:
            label = "in_band"
        elif abs(steady_pressure) < pressure_eps:
            label = "estimator_blind"  # rate out of band, H not tracking it
        else:
            label = "controller_ineffective"  # H tracking it, weight update not correcting it
        classification[cell_type] = {
            "steady_rate_hz": steady_rate,
            "steady_pressure_H_minus_1": steady_pressure,
            "in_band": in_band,
            "classification": label,
        }
    return classification


# %% [4] K_HDP sweep -- at fixed tau_0_ms, scan the global plasticity gain.


def sweep_K_HDP(model: "jtfne.core.Model", cfg: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for k in cfg["K_HDP_GRID"]:
        run = run_hdp(model, cfg, seed=cfg["seed"], overrides={"K_HDP": k})
        summary = summarize_run(model, cfg, run)
        summary["K_HDP"] = k
        summary["in_target_band"] = _in_target_band(summary["rate_by_type_hz"], cfg["target_rate_hz"])
        results.append(summary)
        print(
            f"K_HDP={k:>5.2f}  rates={summary['rate_by_type_hz']}  "
            f"kappa={summary['kappa_synchrony']:.3f}  "
            f"sat={summary['weight_saturation_fraction']:.3f}  "
            f"in_band={summary['in_target_band']}"
        )
    return results


# %% [5] tau_0_ms sweep -- short vs. long HDP time constant at the chosen K_HDP.


def sweep_tau_0_ms(model: "jtfne.core.Model", cfg: dict[str, Any], K_HDP: float) -> list[dict[str, Any]]:
    results = []
    for tau in cfg["TAU0_GRID_MS"]:
        run = run_hdp(model, cfg, seed=cfg["seed"], overrides={"K_HDP": K_HDP, "tau_0_ms": tau})
        summary = summarize_run(model, cfg, run)
        summary["tau_0_ms"] = tau
        summary["in_target_band"] = _in_target_band(summary["rate_by_type_hz"], cfg["target_rate_hz"])
        results.append(summary)
        print(
            f"tau_0_ms={tau:>6.1f}  rates={summary['rate_by_type_hz']}  "
            f"kappa={summary['kappa_synchrony']:.3f}  "
            f"sat={summary['weight_saturation_fraction']:.3f}  "
            f"in_band={summary['in_target_band']}"
        )
    return results


# %% [5b] K_HDP x tau_0_ms stability map -- distinguishes a genuine
# tuning problem from a feedback-control instability. max_var_H / max_var_w
# are population-variance proxies for a runaway/oscillatory controller: if
# instability (high max_var_H, high weight_saturation) appears below a
# critical tau_0_ms regardless of K_HDP, that's a controller-bandwidth
# limit, not evidence that more gain (or per-cell-type gain) is needed.


def sweep_stability_map(model: "jtfne.core.Model", cfg: dict[str, Any]) -> list[dict[str, Any]]:
    duration_ms = cfg.get("stability_duration_ms", cfg["duration_ms"])
    results = []
    for k in cfg["K_HDP_GRID"]:
        for tau in cfg["TAU0_GRID_MS"]:
            run = run_hdp(model, cfg, seed=cfg["seed"], duration_ms=duration_ms, overrides={"K_HDP": k, "tau_0_ms": tau})
            summary = summarize_run(model, cfg, run)
            H_trace = np.asarray(run["diagnostics"]["H_trace"])
            w_trace = np.asarray(run["diagnostics"]["w_trace"])
            max_var_H = float(H_trace.var(axis=1).max())
            max_var_w = float(w_trace.var(axis=1).max())
            point = {
                "K_HDP": k,
                "tau_0_ms": tau,
                "mean_rate_hz": summary["overall_rate_hz"],
                "weight_saturation_fraction": summary["weight_saturation_fraction"],
                "kappa_synchrony": summary["kappa_synchrony"],
                "max_var_H": max_var_H,
                "max_var_w": max_var_w,
            }
            results.append(point)
            print(
                f"K_HDP={k:>5.2f} tau_0_ms={tau:>6.1f}  rate={point['mean_rate_hz']:.2f}Hz  "
                f"sat={point['weight_saturation_fraction']:.3f}  max_var_H={max_var_H:.4f}  "
                f"max_var_w={max_var_w:.4f}"
            )
    return results


def pick_best(results: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Prefer an in-band, low-synchrony, low-saturation point; else lowest |rate-10Hz|."""

    candidates = [r for r in results if r["in_target_band"] and r["kappa_synchrony"] <= cfg["kappa_stable_max"]]
    pool = candidates or results

    def _score(r: dict[str, Any]) -> float:
        mean_rate = float(np.mean(list(r["rate_by_type_hz"].values()))) if r["rate_by_type_hz"] else 0.0
        return abs(mean_rate - 10.0) + r["weight_saturation_fraction"]

    return min(pool, key=_score)


# %% [6] Spectrolaminar motif at the tuned operating point.


def compute_spectrolaminar_motif(
    model: "jtfne.core.Model", cfg: dict[str, Any], hdp_overrides: dict[str, Any]
) -> dict[str, Any]:
    run = run_hdp(model, cfg, seed=cfg["seed"] + 1, duration_ms=cfg["spectro_duration_ms"], overrides=hdp_overrides)
    field = jtfne.project_laminar_sources(
        run["sources"],
        model.params["positions"],
        n_contacts=cfg["n_contacts"],
        width=cfg["contact_width"],
        mode=cfg["projection_mode"],
    )
    fs = 1000.0 / cfg["dt_ms"]
    lfp = np.asarray(field.lfp_proxy)[None, :, :]  # (1, n_steps, n_contacts)
    power = np.asarray(jtfne.spectrolaminar_psd_jax(jnp.asarray(lfp), fs=fs))  # (n_freqs, n_contacts) or similar
    summary = summarize_run(model, cfg, run)
    return {
        "field": field,
        "power": power,
        "fs": fs,
        "summary": summary,
        "contact_depths": np.asarray(field.contact_depths),
    }


# %% [7] E-vs-I spike-waveform/size diagnostic (spike-triggered V_m proxy).


def spike_triggered_waveform(
    model: "jtfne.core.Model", run: dict[str, Any], cfg: dict[str, Any], window_ms: float = 20.0
) -> dict[str, np.ndarray]:
    """Per-cell-type average V_m around each spike (scaffold proxy, not a real AP shape)."""

    labels = np.asarray(model.params["emitter"].labels)
    voltages = np.asarray(run["voltages"])  # (n_steps, n_neurons)
    spikes = np.asarray(run["spikes"])
    half = int(round(window_ms / cfg["dt_ms"] / 2))

    waveforms: dict[str, np.ndarray] = {}
    for cell_type in sorted(set(labels.tolist())):
        idx = np.where(labels == cell_type)[0]
        if idx.size == 0:
            continue
        snippets = []
        for n_idx in idx:
            spike_steps = np.where(spikes[:, n_idx] > 0.5)[0]
            for s in spike_steps[:50]:  # cap per-neuron for cost
                lo, hi = s - half, s + half
                if lo < 0 or hi >= voltages.shape[0]:
                    continue
                snippets.append(voltages[lo:hi, n_idx])
        if snippets:
            waveforms[cell_type] = np.mean(np.stack(snippets, axis=0), axis=0)
    return waveforms


def waveform_width_ms(waveform: np.ndarray, dt_ms: float, half_max_ref: float = -65.0) -> float:
    """Full-width at half-max above a resting reference, in ms (a proxy, not a calibrated AP width)."""

    peak = waveform.max()
    threshold = half_max_ref + 0.5 * (peak - half_max_ref)
    above = waveform > threshold
    if not above.any():
        return 0.0
    idx = np.where(above)[0]
    return float((idx[-1] - idx[0] + 1) * dt_ms)


# %% [8] Plotting + JSON-safe receipt.


def _plot_rate_histogram(summary: dict[str, Any], band: tuple[float, float], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    types = list(summary["rate_by_type_hz"].keys())
    rates = [summary["rate_by_type_hz"][t] for t in types]
    ax.bar(types, rates, color="steelblue")
    ax.axhspan(band[0], band[1], color="green", alpha=0.15, label="target band")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("Per-cell-type firing rate (HDP-tuned operating point)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sweep(results: list[dict[str, Any]], x_key: str, band: tuple[float, float], out_path: Path, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    xs = [r[x_key] for r in results]
    cell_types = sorted({t for r in results for t in r["rate_by_type_hz"]})
    for cell_type in cell_types:
        ys = [r["rate_by_type_hz"].get(cell_type, np.nan) for r in results]
        ax.plot(xs, ys, marker="o", label=cell_type)
    ax.axhspan(band[0], band[1], color="green", alpha=0.1)
    if x_key == "K_HDP" or x_key == "tau_0_ms":
        ax.set_xscale("symlog" if 0 in xs else "log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("firing rate (Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_spectrolaminar_motif(motif: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    power = motif["power"]
    depths = motif["contact_depths"]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        power, aspect="auto", origin="lower",
        extent=[0, power.shape[1], depths.min(), depths.max()] if power.ndim == 2 else None,
        cmap="magma",
    )
    ax.set_xlabel("frequency bin")
    ax.set_ylabel("relative laminar depth")
    ax.set_title("Spectrolaminar motif (depth x frequency power, proxy)")
    fig.colorbar(im, ax=ax, label="power (a.u.)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_waveforms(waveforms: dict[str, np.ndarray], dt_ms: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for cell_type, wf in waveforms.items():
        t_ms = np.arange(wf.shape[0]) * dt_ms
        width = waveform_width_ms(wf, dt_ms)
        ax.plot(t_ms, wf, label=f"{cell_type} (FWHM~{width:.1f} ms)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("V_m (mV, spike-triggered avg)")
    ax.set_title("E vs. I spike-triggered waveform proxy (not a calibrated AP shape)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_H_trajectories(per_type: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cell_type, d in per_type.items():
        ax.plot(d["t_ms"], d["mean_H_t"], label=cell_type)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="H=1 (equilibrium)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mean H")
    ax.set_title("HDP master state H_i over time (per cell type)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pressure_trajectories(per_type: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cell_type, d in per_type.items():
        ax.plot(d["t_ms"], d["mean_H_t"] - 1.0, label=cell_type)
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mean(H-1)  [controller error]")
    ax.set_title("HDP pressure (error signal) over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_rate_vs_pressure(per_type: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for cell_type, d in per_type.items():
        ax.scatter(d["pressure_binned"], d["rate_binned_hz"], s=10, alpha=0.5, label=cell_type)
    ax.axvline(0.0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("mean(H-1)  [controller error]")
    ax.set_ylabel("binned firing rate (Hz)")
    ax.set_title("Rate vs. controller error (healthy controller -> monotonic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_stability_map(results: list[dict[str, Any]], cfg: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ks = cfg["K_HDP_GRID"]
    taus = cfg["TAU0_GRID_MS"]
    by_point = {(r["K_HDP"], r["tau_0_ms"]): r for r in results}

    fields = [
        ("mean_rate_hz", "mean rate (Hz)"),
        ("weight_saturation_fraction", "weight saturation frac."),
        ("max_var_H", "max var(H)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, (key, title) in zip(axes, fields):
        grid = np.array([[by_point[(k, t)][key] for t in taus] for k in ks])
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f"{t:g}" for t in taus])
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([f"{k:g}" for k in ks])
        ax.set_xlabel("tau_0_ms")
        ax.set_ylabel("K_HDP")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("K_HDP x tau_0_ms stability map")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config()

    print("=== [2] Building laminar column Model (one-time construct) ===")
    model = build_model(cfg)

    print("\n=== [4] K_HDP sweep ===")
    k_results = sweep_K_HDP(model, cfg)
    best_k_point = pick_best(k_results, cfg)
    best_K_HDP = best_k_point["K_HDP"]
    print(f"--> chosen K_HDP = {best_K_HDP}")

    print("\n=== [5] tau_0_ms sweep (at chosen K_HDP) ===")
    tau_results = sweep_tau_0_ms(model, cfg, best_K_HDP)
    best_tau_point = pick_best(tau_results, cfg)
    best_tau_0_ms = best_tau_point["tau_0_ms"]
    print(f"--> chosen tau_0_ms = {best_tau_0_ms}")

    tuned_overrides = {"K_HDP": best_K_HDP, "tau_0_ms": best_tau_0_ms}

    print("\n=== [3b] HDP state-trajectory diagnostics (at tuned point) ===")
    diag_run = run_hdp(model, cfg, seed=cfg["seed"], overrides=tuned_overrides)
    per_type_diag = compute_hdp_state_diagnostics(model, cfg, diag_run)
    state_classification = classify_hdp_state(per_type_diag, cfg["target_rate_hz"])
    for cell_type, c in state_classification.items():
        print(
            f"{cell_type:>5s}: steady_rate={c['steady_rate_hz']:.2f}Hz  "
            f"H-1={c['steady_pressure_H_minus_1']:+.4f}  -> {c['classification']}"
        )

    print("\n=== [5b] K_HDP x tau_0_ms stability map ===")
    stability_results = sweep_stability_map(model, cfg)

    print("\n=== [6] Spectrolaminar motif at tuned operating point ===")
    motif = compute_spectrolaminar_motif(model, cfg, tuned_overrides)
    print(f"tuned-point summary: {motif['summary']}")

    print("\n=== [7] E-vs-I waveform/size diagnostic ===")
    final_run = run_hdp(model, cfg, seed=cfg["seed"], overrides=tuned_overrides)
    waveforms = spike_triggered_waveform(model, final_run, cfg)
    widths = {ct: waveform_width_ms(wf, cfg["dt_ms"]) for ct, wf in waveforms.items()}
    print(f"spike-triggered FWHM by cell type (ms, proxy): {widths}")

    print("\n=== [8] Saving figures + receipt ===")
    final_summary = summarize_run(model, cfg, final_run)
    _plot_rate_histogram(final_summary, cfg["target_rate_hz"], OUTPUT_DIR / "rate_histogram.png")
    _plot_sweep(k_results, "K_HDP", cfg["target_rate_hz"], OUTPUT_DIR / "k_hdp_sweep.png", "K_HDP")
    _plot_sweep(tau_results, "tau_0_ms", cfg["target_rate_hz"], OUTPUT_DIR / "tau_0_ms_sweep.png", "tau_0_ms (ms)")
    _plot_spectrolaminar_motif(motif, OUTPUT_DIR / "spectrolaminar_motif.png")
    _plot_waveforms(waveforms, cfg["dt_ms"], OUTPUT_DIR / "ei_waveform_proxy.png")
    _plot_H_trajectories(per_type_diag, OUTPUT_DIR / "hdp_H_trajectories.png")
    _plot_pressure_trajectories(per_type_diag, OUTPUT_DIR / "hdp_pressure_trajectories.png")
    _plot_rate_vs_pressure(per_type_diag, OUTPUT_DIR / "hdp_rate_vs_pressure.png")
    _plot_stability_map(stability_results, cfg, OUTPUT_DIR / "hdp_stability_map.png")

    receipt = {
        "claim_level": "computational_scaffold",
        "config": {k: v for k, v in cfg.items() if k not in ("field_kwargs", "probe_kwargs")},
        "k_hdp_sweep": k_results,
        "tau_0_ms_sweep": tau_results,
        "chosen": tuned_overrides,
        "tuned_operating_point_summary": final_summary,
        "spike_triggered_fwhm_ms_proxy": widths,
        "spectrolaminar_summary": motif["summary"],
        "hdp_state_classification": state_classification,
        "hdp_stability_map": stability_results,
    }
    with open(OUTPUT_DIR / "receipt.json", "w") as f:
        json.dump(receipt, f, indent=2, allow_nan=False)

    print(f"\nDone. Figures + receipt.json written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
