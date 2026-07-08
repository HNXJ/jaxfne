#!/usr/bin/env python3
"""Generate PNG showcase figures for docs/guides/showcases.md.

Runs real jaxfne simulations with proxy-safe titles. Sizes are reduced from
the prose in showcases.md where a full-scale run would be impractical for
doc regeneration (e.g. 10k-neuron columns use 200–500 neurons).

Usage:
  python scripts/generate_showcase_figures.py [--output-dir docs/assets/showcases]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import jaxfne as jtfne

OUTPUT_NAMES = [
    "homeostasis_rate_change_10s.png",
    "homeostasis_full_raster_10s.png",
    "plasticity_random_stim_stability.png",
    "plasticity_weight_distribution.png",
    "spectrolaminar_slow_homeostasis_suite.png",
    "spectrolaminar_depth_distribution_crossings.png",
    "spectrolaminar_suite_corrected.png",
    "spectrolaminar_absolute_power_1f_check.png",
]


def _save(fig, path: Path, title: str) -> None:
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    jtfne.vis.close_all()
    print(f"  wrote {path.name}")


def _column_cfg(n: int, *, homeostasis: bool = False, k_gain: float = 1.0, **homeo_kw):
    cfg = (
        jtfne.build_laminar_column("V1", n=n, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=min(32, max(8, n // 4)))
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    if homeostasis:
        cfg = cfg.homeostasis(relative_baseline=1.0, r_star=10.0, k_gain=k_gain, **homeo_kw)
    return cfg


def gen_homeostasis_figures(out_dir: Path) -> None:
    """200-neuron column, 10 s, homeostasis off vs on."""
    import matplotlib.pyplot as plt

    duration_ms = 10_000.0
    dt_ms = 0.5
    n = 200
    window_ms = 500.0
    n_windows = int(duration_ms / window_ms)

    cfg_off = _column_cfg(n)
    cfg_on = _column_cfg(n, homeostasis=True, k_gain=1.0)
    model_off = jtfne.construct(cfg_off)
    model_on = jtfne.construct(cfg_on)

    sig_off = jtfne.simulate(model_off, duration_ms=duration_ms, dt_ms=dt_ms, seed=0)
    sig_on = jtfne.simulate(model_on, duration_ms=duration_ms, dt_ms=dt_ms, seed=0)

    def windowed_rates(spikes: np.ndarray) -> np.ndarray:
        steps_per_win = int(window_ms / dt_ms)
        rates = []
        for w in range(n_windows):
            chunk = spikes[w * steps_per_win : (w + 1) * steps_per_win]
            rates.append(float(chunk.sum() / (n * (window_ms / 1000.0))))
        return np.asarray(rates)

    sp_off = np.asarray(sig_off.spikes)
    sp_on = np.asarray(sig_on.spikes)
    t_win = np.arange(n_windows) * (window_ms / 1000.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_win, windowed_rates(sp_off), label="homeostasis off", lw=1.5)
    ax.plot(t_win, windowed_rates(sp_on), label="homeostasis on (k_gain=1)", lw=1.5)
    ax.axhline(10.0, color="gray", ls="--", alpha=0.5, label="r*=10 Hz")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population rate (Hz)")
    ax.set_title("Simulated population rate — homeostasis comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / OUTPUT_NAMES[0], "Firing-rate change with homeostasis (proxy scaffold)")

    fig2 = jtfne.vis.raster(sig_on, figsize=(10, 4))
    fig2.axes[0].set_title("Spike raster proxy — homeostasis on, 10 s")
    _save(fig2, out_dir / OUTPUT_NAMES[1], "Full 10 s raster, homeostasis on (simulated)")


def gen_stdp_figures(out_dir: Path) -> None:
    """100-neuron E/I cloud, closed-loop STDP under random noise."""
    import matplotlib.pyplot as plt

    from jaxfne.plasticity import STDPPlasticityConfig, STDPState
    from jaxfne.solvers import SolverConfig

    n = 100
    duration_ms = 10_000.0
    dt_ms = 0.5
    n_steps = int(duration_ms / dt_ms)
    chunk_ms = 1000.0

    _, exc_mask, inh_mask, W0 = jtfne.make_ei_cloud_network(n, seed=42)
    exc_mask = np.asarray(exc_mask)
    inh_mask = np.asarray(inh_mask)
    W0 = np.asarray(W0)

    v0 = jnp.full(n, -65.0)
    u0 = jnp.zeros(n)
    s0 = jnp.zeros(n)
    a = jnp.where(exc_mask, 0.02, 0.1)
    b = jnp.where(exc_mask, -14.0, 0.0)
    c = jnp.full(n, -65.0)
    d = jnp.where(exc_mask, 8.0, 2.0)

    key = jr.PRNGKey(0)
    noise = jr.normal(key, (n_steps, n)) * 0.8

    stdp_state = STDPState(W=jnp.asarray(W0), trace_pre=jnp.zeros(n), trace_post=jnp.zeros(n))
    plasticity_config = STDPPlasticityConfig(A_plus=0.01, A_minus=0.012)
    solver_config = SolverConfig(method="euler", dt=dt_ms)

    (v_final, u_final, s_final, final_state), traj = jtfne.run_stdp_stream(
        v_init=v0,
        u_init=u0,
        s_init=s0,
        stdp_state=stdp_state,
        stim_drive=jnp.zeros((n_steps, n)),
        noise=noise,
        solver_config=solver_config,
        plasticity_config=plasticity_config,
        plasticity_scale=0.1,
        exc_mask=jnp.asarray(exc_mask),
        inh_mask=jnp.asarray(inh_mask),
        a=a,
        b=b,
        c=c,
        d=d,
        chunk_size_ms=chunk_ms,
        downsample_factor=1,
    )

    summaries = traj["chunk_summaries"]
    chunk_rates = [float(s["mean_firing_rate_hz"]) for s in summaries]
    chunk_wmean = [float(s["mean_weight"]) for s in summaries]
    W_final = np.asarray(final_state.W)
    n_chunks = len(chunk_rates)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    xs = np.arange(n_chunks) + 1
    ax1.plot(xs, chunk_rates, "o-", lw=1.5)
    ax1.set_ylabel("Rate (Hz)")
    ax1.set_title("Activity stability under random stimulation (STDP on)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(xs, chunk_wmean, "s-", color="C1", lw=1.5)
    ax2.set_xlabel("1 s chunk")
    ax2.set_ylabel("Mean exc. weight")
    ax2.set_title("Synaptic weight drift (closed-loop STDP)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / OUTPUT_NAMES[2], "STDP stability under random drive (computational scaffold)")

    w_exc_before = W0[exc_mask][:, exc_mask].ravel()
    w_exc_after = W_final[exc_mask][:, exc_mask].ravel()
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.hist(w_exc_before, bins=30, alpha=0.6, label="before", density=True)
    ax.hist(w_exc_after, bins=30, alpha=0.6, label="after 10 s", density=True)
    ax.set_xlabel("Excitatory weight")
    ax.set_ylabel("Density")
    ax.set_title("Excitatory weight distribution — STDP under random noise")
    ax.legend(fontsize=8)
    _save(fig2, out_dir / OUTPUT_NAMES[3], "Excitatory weight distribution (simulated STDP)")


def _graded_homeostasis_cfg(n: int):
    """Depth-graded homeostasis on a reduced column (doc regen size)."""
    cfg = _column_cfg(n)
    model = jtfne.construct(cfg)
    nt = model.neuron_table()
    layers = [row["layer"] for row in nt]
    tau = np.where(np.isin(layers, ["L5", "L6"]), 1500.0, 300.0).astype(np.float32)
    g_min = np.where(np.isin(layers, ["L5", "L6"]), -20.0, -12.0).astype(np.float32)
    g_max = np.where(np.isin(layers, ["L5", "L6"]), 14.0, 8.0).astype(np.float32)
    cfg = cfg.homeostasis(
        relative_baseline=1.0,
        k_gain=1.0,
        r_star=10.0,
        tau_r_ms=jnp.asarray(tau),
        g_min=jnp.asarray(g_min),
        g_max=jnp.asarray(g_max),
    )
    return cfg


def gen_spectrolaminar_figures(out_dir: Path) -> None:
    """Spectrolaminar proxy readouts at doc-regeneration scale."""
    import matplotlib.pyplot as plt

    # Slow-deep homeostasis suite (reduced n for regen; same API path)
    cfg_slow = _graded_homeostasis_cfg(300)
    model_slow = jtfne.construct(cfg_slow)
    sig_slow = jtfne.simulate(model_slow, duration_ms=1000.0, dt_ms=0.5, seed=0)
    fig = jtfne.vis.spectrolaminar_suite(
        sig_slow,
        title="Simulated laminar proxy readout — depth-graded homeostasis",
        max_freq_hz=80.0,
    )
    _save(fig, out_dir / OUTPUT_NAMES[4], "Spectrolaminar suite, slow-deep homeostasis (proxy)")

    # 3-panel corrected crossing methodology (100 n, few trials)
    cfg = _column_cfg(100, homeostasis=True, k_gain=0.1)
    model = jtfne.construct(cfg)
    figs = jtfne.vis.spectrolaminar_suite_3panel(
        model,
        n_trials=5,
        duration_ms=2000.0,
        dt_ms=0.5,
        seed=0,
        signal="lfp",
        enable_homeostasis=True,
        homeostasis_params={"k_gain": 0.1, "r_star": 10.0},
    )
    area = next(iter(figs))
    fig3 = figs[area]
    _save(fig3, out_dir / OUTPUT_NAMES[6], "Spectrolaminar 3-panel suite (proxy readout)")

    # Depth-distribution crossing panel: compare 300 vs 100 neuron depth profiles
    cfg_big = _column_cfg(300, homeostasis=True, k_gain=0.1)
    cfg_small = _column_cfg(100, homeostasis=True, k_gain=0.1)
    figs_big = jtfne.vis.spectrolaminar_suite_3panel(
        jtfne.construct(cfg_big), n_trials=3, duration_ms=1000.0, dt_ms=0.5, seed=1, signal="lfp"
    )
    figs_small = jtfne.vis.spectrolaminar_suite_3panel(
        jtfne.construct(cfg_small), n_trials=5, duration_ms=2000.0, dt_ms=0.5, seed=0, signal="lfp"
    )
    # Extract crossing subplot (panel 3) from each — last axes in 3-panel figure
    fig_cross, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, figs, label in [
        (axes[0], figs_big, "300 neurons"),
        (axes[1], figs_small, "100 neurons"),
    ]:
        src_fig = figs[next(iter(figs))]
        src_ax = src_fig.axes[-1]
        for line in src_ax.get_lines():
            ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), lw=line.get_linewidth())
        ax.set_title(f"Depth-distribution crossing — {label}")
        ax.set_xlabel(src_ax.get_xlabel())
        ax.set_ylabel(src_ax.get_ylabel())
        ax.grid(True, alpha=0.3)
        if src_ax.get_legend():
            ax.legend(fontsize=7)
        jtfne.vis.close_all()
    fig_cross.tight_layout()
    _save(fig_cross, out_dir / OUTPUT_NAMES[5], "Depth-distribution crossings (proxy PSD)")

    # Absolute power 1/f background check (superficial vs deep contact groups)
    sig = jtfne.simulate(model, duration_ms=2000.0, dt_ms=0.5, seed=0)
    lfp = np.asarray(sig.field.lfp_proxy)
    n_contacts = lfp.shape[1]
    mid = n_contacts // 2
    superficial = lfp[:, :mid].mean(axis=1)
    deep = lfp[:, mid:].mean(axis=1)
    fs = 1000.0 / 0.5
    from scipy import signal as scipy_signal

    f_sup, p_sup = scipy_signal.welch(superficial, fs=fs, nperseg=min(512, len(superficial)))
    f_deep, p_deep = scipy_signal.welch(deep, fs=fs, nperseg=min(512, len(deep)))
    fig_pow, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(f_sup[1:], p_sup[1:], label="superficial mean contact", lw=1.5)
    ax.loglog(f_deep[1:], p_deep[1:], label="deep mean contact", lw=1.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (proxy units²/Hz)")
    ax.set_title("Absolute LFP-proxy power — superficial vs deep")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig_pow, out_dir / OUTPUT_NAMES[7], "Absolute power spectra — 1/f background check (proxy)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate showcase PNG figures for docs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/showcases"),
        help="Directory for PNG output",
    )
    parser.add_argument("--skip-stdp", action="store_true", help="Skip slow STDP 10 s run")
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating homeostasis figures...")
    gen_homeostasis_figures(out_dir)

    if not args.skip_stdp:
        print("Generating STDP figures (10 s run)...")
        gen_stdp_figures(out_dir)
    else:
        print("Skipping STDP figures")

    print("Generating spectrolaminar figures...")
    gen_spectrolaminar_figures(out_dir)

    missing = [name for name in OUTPUT_NAMES if not (out_dir / name).exists()]
    if missing:
        print(f"WARNING: missing outputs: {missing}", file=sys.stderr)
        return 1
    print(f"Done — {len(OUTPUT_NAMES)} figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
