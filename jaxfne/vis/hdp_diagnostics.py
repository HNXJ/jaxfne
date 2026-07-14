"""Rendering helpers for the HDP (homeostatic-dynamic-plasticity) diagnostic
scripts under ``scripts/hdp_*.py`` and the standalone 5 Hz sweep/verify
scripts.

These plain functions take pre-extracted NumPy/JAX-host arrays (or simple
dict/list records the calling script already built) and return a matplotlib
``Figure`` -- they never write files or call ``plt.show``. Scripts stay free
of direct ``matplotlib`` imports, per the package-wide vis-isolation rule
(all rendering lives under ``jaxfne.vis``).

Evaluated as an uncalibrated computational scaffold; all readouts here are
simulated proxy diagnostics of the HDP homeostatic-plasticity controller,
not physical measurements or biological mechanism claims.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .core import prepare_static_plot_matrix


# ---------------------------------------------------------------------------
# scripts/hdp_synaptic_channel_trace.py
# ---------------------------------------------------------------------------

def synaptic_channel_decomposition_trace(
    t_ms: Any,
    series_by_class: dict[str, Any],
    departures: dict[str, int | None],
    *,
    window_start_ms: float,
    window_end_ms: float,
    bin_ms: float,
    title: str | None = None,
) -> Any:
    """Per-connection-class synaptic current contribution over time.

    ``series_by_class`` maps a connection-class label (e.g. ``"E->PV"``) to
    a 1D array of mean per-step channel current contribution per time bin.
    ``departures`` maps the same labels to the first-departure bin index (or
    ``None``); a vertical dotted marker is drawn at that bin's time for any
    class with a detected departure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    t = np.asarray(prepare_static_plot_matrix(t_ms))
    classes = list(series_by_class.keys())

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = cm.tab20(np.linspace(0, 1, max(len(classes), 1)))
    for cls, color in zip(classes, colors):
        y = np.asarray(prepare_static_plot_matrix(series_by_class[cls]))
        ax.plot(t, y, label=cls, color=color)
        idx = departures.get(cls)
        if idx is not None:
            ax.axvline(t[idx], color=color, ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0.0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mean per-step channel current contribution")
    ax.set_title(
        title
        or f"Synaptic-channel decomposition, {window_start_ms:.0f}-{window_end_ms:.0f}ms, bin={bin_ms:.0f}ms"
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# scripts/hdp_dH_component_trace.py
# ---------------------------------------------------------------------------

def dH_component_trace_panels(
    t_ms: Any,
    H_mean: Any,
    var_H: Any,
    mean_abs_w: Any,
    var_w: Any,
    dH_income_mean: Any,
    dH_rate_mean: Any,
    dH_weight_mean: Any,
    departures: dict[str, int | None],
    *,
    window_start_ms: float,
    window_end_ms: float,
    bin_ms: float,
) -> Any:
    """Three-panel HDP resource-state (H) decomposition trace.

    Panel 1: H_mean (left axis) + var(H) (right axis), with the H=1
    equilibrium reference line. Panel 2: mean|w| (left) + var(w) (right).
    Panel 3: the three additive dH/dt terms (income, rate-spending,
    weight-spending). ``departures`` maps trace-name -> first-departure bin
    index (or None); a dotted vertical marker is drawn on the owning panel.
    """
    import matplotlib.pyplot as plt

    t = np.asarray(prepare_static_plot_matrix(t_ms))
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(t, np.asarray(prepare_static_plot_matrix(H_mean)), label="H_mean")
    ax0b = axes[0].twinx()
    ax0b.plot(t, np.asarray(prepare_static_plot_matrix(var_H)), color="tab:green", label="var(H)")
    axes[0].axhline(1.0, color="k", ls="--", lw=0.5)
    axes[0].set_ylabel("H_mean")
    ax0b.set_ylabel("var(H)", color="tab:green")
    axes[0].set_title(f"dH-component trace, {window_start_ms:.0f}-{window_end_ms:.0f}ms, bin={bin_ms:.0f}ms")

    axes[1].plot(t, np.asarray(prepare_static_plot_matrix(mean_abs_w)), label="mean|w|")
    ax1b = axes[1].twinx()
    ax1b.plot(t, np.asarray(prepare_static_plot_matrix(var_w)), color="tab:orange", label="var(w)")
    axes[1].set_ylabel("mean|w|")
    ax1b.set_ylabel("var(w)", color="tab:orange")

    axes[2].plot(t, np.asarray(prepare_static_plot_matrix(dH_income_mean)), label="dH_income (alpha*I_syn)")
    axes[2].plot(t, np.asarray(prepare_static_plot_matrix(dH_rate_mean)), label="dH_rate (-gamma*r)")
    axes[2].plot(t, np.asarray(prepare_static_plot_matrix(dH_weight_mean)), label="dH_weight (-delta*W, =0)")
    axes[2].axhline(0.0, color="k", ls="--", lw=0.5)
    axes[2].set_ylabel("mean dH term")
    axes[2].set_xlabel("time (ms)")
    axes[2].legend(fontsize=8)

    colors = {"H_mean": "tab:blue", "var_H": "tab:green", "var_w": "tab:orange", "dH_income": "tab:purple", "dH_rate": "tab:red"}
    ax_for = {"H_mean": axes[0], "var_H": axes[0], "var_w": axes[1], "dH_income": axes[2], "dH_rate": axes[2]}
    for key, idx in departures.items():
        if idx is not None and key in ax_for:
            ax_for[key].axvline(t[idx], color=colors[key], ls=":", lw=1.0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# scripts/hdp_bifurcation_trace.py
# ---------------------------------------------------------------------------

def bifurcation_state_trace_panels(
    t_ms: Any,
    rate_by_type_hz: dict[str, Any],
    kappa_synchrony: Any,
    mean_weight: Any,
    weight_variance: Any,
    H_mean: Any,
    H_variance: Any,
    departures: dict[str, int | None],
    *,
    title: str = "Bifurcation trace: K_ctrl=0, tau_0_ms=20",
) -> Any:
    """Four-panel time-resolved bifurcation trace: per-type rates, synchrony
    (kappa), weight mean/variance, and HDP resource-state H mean/variance.

    ``rate_by_type_hz`` maps cell-type label -> 1D rate array (one entry per
    bin); ``departures`` maps {"kappa_synchrony","weight_variance",
    "H_variance","SST_rate"} -> first-departure bin index or None.
    """
    import matplotlib.pyplot as plt

    t = np.asarray(prepare_static_plot_matrix(t_ms))
    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)

    for cell_type in sorted(rate_by_type_hz.keys()):
        axes[0].plot(t, np.asarray(prepare_static_plot_matrix(rate_by_type_hz[cell_type])), label=cell_type)
    axes[0].set_ylabel("rate (Hz)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)

    axes[1].plot(t, np.asarray(prepare_static_plot_matrix(kappa_synchrony)), color="tab:red")
    axes[1].set_ylabel("kappa_synchrony")

    axes[2].plot(t, np.asarray(prepare_static_plot_matrix(mean_weight)), label="mean|w|")
    ax2b = axes[2].twinx()
    ax2b.plot(t, np.asarray(prepare_static_plot_matrix(weight_variance)), color="tab:orange", label="var(w)")
    axes[2].set_ylabel("mean|w|")
    ax2b.set_ylabel("var(w)", color="tab:orange")

    axes[3].plot(t, np.asarray(prepare_static_plot_matrix(H_mean)), label="H_mean")
    ax3b = axes[3].twinx()
    ax3b.plot(t, np.asarray(prepare_static_plot_matrix(H_variance)), color="tab:green", label="var(H)")
    axes[3].axhline(1.0, color="k", ls="--", lw=0.5)
    axes[3].set_ylabel("H_mean")
    ax3b.set_ylabel("var(H)", color="tab:green")
    axes[3].set_xlabel("time (ms)")

    colors = {"kappa_synchrony": "tab:red", "weight_variance": "tab:orange", "H_variance": "tab:green", "SST_rate": "tab:blue"}
    ax_for = {"kappa_synchrony": axes[1], "weight_variance": axes[2], "H_variance": axes[3], "SST_rate": axes[0]}
    for key, idx in departures.items():
        if idx is not None and key in ax_for:
            ax_for[key].axvline(t[idx], color=colors[key], ls=":", lw=1.2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# scripts/hdp_gain_ratio_sweep.py
# ---------------------------------------------------------------------------

def gain_ratio_sweep_panels(
    K_ctrl_by_window: dict[str, Any],
    mean_H_minus_1_by_window: dict[str, Any],
    SST_rate_by_window: dict[str, Any],
    kappa_synchrony_by_window: dict[str, Any],
    *,
    suptitle: str = "HDP gain-ratio sweep (R = K_ctrl / (alpha+gamma+delta)), short vs. full window",
) -> Any:
    """Three-panel K_ctrl gain-ratio sweep: controller signal mean(H-1),
    SST rate, and synchrony kappa, each vs. K_ctrl, one line per window
    length (e.g. "short" / "full"). All ``*_by_window`` dicts share the same
    keys (window labels) and use ``markers`` o/s in insertion order.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    markers = ["o", "s", "^", "d"]
    for i, window in enumerate(K_ctrl_by_window.keys()):
        marker = markers[i % len(markers)]
        K_ctrl_vals = np.asarray(prepare_static_plot_matrix(K_ctrl_by_window[window]))
        axes[0].plot(K_ctrl_vals, np.asarray(prepare_static_plot_matrix(mean_H_minus_1_by_window[window])), marker=marker, label=window)
        axes[1].plot(K_ctrl_vals, np.asarray(prepare_static_plot_matrix(SST_rate_by_window[window])), marker=marker, label=window)
        axes[2].plot(K_ctrl_vals, np.asarray(prepare_static_plot_matrix(kappa_synchrony_by_window[window])), marker=marker, label=window)

    axes[0].axhline(0.0, color="k", ls="--", lw=0.5)
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xlabel("K_ctrl")
    axes[0].set_ylabel("mean(H-1)")
    axes[0].set_title("controller signal vs. K_ctrl")
    axes[0].legend()

    axes[1].axhspan(5.0, 15.0, color="g", alpha=0.1, label="target band")
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_xlabel("K_ctrl")
    axes[1].set_ylabel("SST rate (Hz)")
    axes[1].set_title("SST rate vs. K_ctrl")
    axes[1].legend()

    axes[2].axhline(0.3, color="r", ls="--", lw=0.5, label="kappa_stable_max")
    axes[2].set_xscale("symlog", linthresh=0.01)
    axes[2].set_xlabel("K_ctrl")
    axes[2].set_ylabel("kappa_synchrony")
    axes[2].set_title("synchrony vs. K_ctrl")
    axes[2].legend()

    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# scripts/hdp_small_scale_balance_sweep.py
# ---------------------------------------------------------------------------

def balance_sweep_by_network_size(
    K_grid: Any,
    H_mean_by_n: dict[int, Any],
    H_var_by_n: dict[int, Any],
    *,
    suptitle: str = "H-dynamics balance vs K_ctrl, by network size (10 all-to-all networks/size)",
) -> Any:
    """Two-panel H_mean / H_var vs K_ctrl, one line per network size N.

    ``H_mean_by_n`` / ``H_var_by_n`` map network size N -> 1D array aligned
    with ``K_grid`` (network-averaged tail-window H statistics).
    """
    import matplotlib.pyplot as plt

    K_grid_np = np.asarray(prepare_static_plot_matrix(K_grid))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for n in H_mean_by_n.keys():
        axes[0].plot(K_grid_np, np.asarray(prepare_static_plot_matrix(H_mean_by_n[n])), marker="o", label=f"N={n}")
        axes[1].plot(K_grid_np, np.asarray(prepare_static_plot_matrix(H_var_by_n[n])), marker="o", label=f"N={n}")
    axes[0].axhline(1.0, color="k", ls="--", lw=0.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("K_ctrl")
    axes[0].set_ylabel("H_mean (avg over 10 networks, tail window)")
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("K_ctrl")
    axes[1].set_ylabel("H_var (avg over 10 networks, tail window)")
    axes[1].legend(fontsize=8)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# scripts/hdp_suite2_visualizations.py
# ---------------------------------------------------------------------------

def lfp_traces_stacked_by_depth(
    time_ms: Any,
    lfp: Any,
    contact_depths: Any,
    *,
    title: str = "LFP-proxy traces, Z-sorted by depth (offset stacked)",
) -> Any:
    """Offset-stacked LFP-proxy traces ordered by contact depth.

    ``lfp`` has shape ``(n_steps, n_contacts)``; ``contact_depths`` has
    shape ``(n_contacts,)``. Each trace is normalized by the global std and
    offset by its depth rank so all channels are visible on one axis.
    """
    import matplotlib.pyplot as plt

    lfp_np = np.asarray(prepare_static_plot_matrix(lfp))
    depths_np = np.asarray(prepare_static_plot_matrix(contact_depths))
    time_np = np.asarray(prepare_static_plot_matrix(time_ms))
    order = np.argsort(depths_np)

    fig, ax = plt.subplots(figsize=(10, 12))
    scale = np.nanstd(lfp_np) or 1.0
    for rank, ci in enumerate(order):
        ax.plot(time_np, lfp_np[:, ci] / scale + rank, lw=0.6, color="C0")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{depths_np[ci]:.3f}" for ci in order], fontsize=6)
    ax.set_ylabel("Contact relative depth (z, superficial->deep)")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"{title} ({len(order)} channels)")
    fig.tight_layout()
    return fig


def etude2_extended_panel_grid(
    *,
    layer_order: list[str],
    cell_types_present: list[str],
    ct_colors: dict[str, str],
    layer_density_fracs: dict[str, Any],
    freq_hz: Any,
    relative_power: Any,
    pos_from_l4_mm: Any,
    alpha_beta_profile: Any,
    gamma_profile: Any,
    time_win_ms: Any,
    lfp_win: Any,
    contact_depths: Any,
    spike_times_ms: Any,
    spike_neuron_rank: Any,
    n_neurons_zsorted: int,
    rate_table_hz: Any,
    k_hdp_used: float,
    last_window_ms: float,
) -> Any:
    """Etude No. 2 extended A-F static panel grid (dark theme).

    A: cell-type relative density per layer (stacked).
    B: spectrolaminar relative-power depth x frequency heatmap (pre-smoothed
       by the caller -- this function only renders ``relative_power``).
    C: alpha-beta / gamma band-power vs depth crossing line (pre-smoothed).
    D: LFP-proxy traces, last `last_window_ms`, offset-stacked by depth.
    E: spike raster, last `last_window_ms`, z-sorted by depth (caller passes
       already-filtered/sorted scatter coordinates).
    F: E/I mean firing-rate table per layer (2 rows x len(layer_order)).

    G (interactive 3D column scatter) is intentionally not part of this
    static grid -- it is produced separately via
    ``jaxfne.vis.visualize_network_3d`` as its own Plotly HTML.
    """
    import matplotlib.pyplot as plt

    BG, LINE = "#1a1a1a", "#f5f5f5"

    fig = plt.figure(figsize=(20, 11), dpi=150, facecolor=BG)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[0.8, 1.3, 0.8])
    axA = fig.add_subplot(grid[0, 0])
    axB = fig.add_subplot(grid[0, 1])
    axC = fig.add_subplot(grid[0, 2], sharey=axB)
    axD = fig.add_subplot(grid[1, 0])
    axE = fig.add_subplot(grid[1, 1])
    axF = fig.add_subplot(grid[1, 2])
    for ax in (axA, axB, axC, axD, axE, axF):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.tick_params(colors=LINE, labelsize=8)
        ax.xaxis.label.set_color(LINE)
        ax.yaxis.label.set_color(LINE)
        ax.title.set_color(LINE)

    # A: cell-type relative density per layer (stacked, L1 top -> L6 bottom)
    bottoms = np.zeros(len(layer_order))
    for ct in cell_types_present:
        fracs = np.asarray(prepare_static_plot_matrix(layer_density_fracs[ct]))
        axA.barh(range(len(layer_order)), fracs, left=bottoms, height=0.7,
                 color=ct_colors.get(ct, "#888888"), label=ct)
        bottoms += fracs
    axA.set_yticks(range(len(layer_order)))
    axA.set_yticklabels(layer_order)
    axA.invert_yaxis()
    axA.set_xlabel("Relative density (fraction of layer)")
    axA.set_title("A: Cell-type relative density per layer", fontsize=10)
    axA.legend(fontsize=7, loc="lower right", facecolor="#2a2a2a", edgecolor="#555555", labelcolor=LINE)

    # B: depth x frequency relative-power heatmap (caller pre-smooths)
    freq_np = np.asarray(prepare_static_plot_matrix(freq_hz))
    power_np = np.asarray(prepare_static_plot_matrix(relative_power))
    pos_np = np.asarray(prepare_static_plot_matrix(pos_from_l4_mm))
    im = axB.imshow(
        power_np.T, aspect="auto", cmap="magma", origin="upper",
        extent=[freq_np[0], freq_np[-1], pos_np[-1], pos_np[0]],
    )
    axB.set_xlabel("Frequency (Hz)")
    axB.set_ylabel("Depth from L4 (mm, superficial top)")
    axB.set_title("B: Spectrolaminar relative power (smoothed)", fontsize=10)
    cb = fig.colorbar(im, ax=axB, fraction=0.04)
    cb.ax.tick_params(colors=LINE, labelsize=7)

    # C: alpha-beta / gamma crossing line vs depth (caller pre-smooths)
    ab_np = np.asarray(prepare_static_plot_matrix(alpha_beta_profile))
    gm_np = np.asarray(prepare_static_plot_matrix(gamma_profile))
    axC.plot(ab_np, pos_np, color="#3b82f6", lw=2.5, label="Alpha-beta")
    axC.plot(gm_np, pos_np, color="#ef4444", lw=2.5, label="Gamma")
    axC.axhline(0, color=LINE, lw=1.0, alpha=0.6)
    axC.set_xlabel("Relative power")
    axC.set_title("C: Alpha-beta / gamma crossing", fontsize=10)
    axC.legend(fontsize=7, loc="lower left", facecolor="#2a2a2a", edgecolor="#555555", labelcolor=LINE)

    # D: LFP traces, last window, offset-stacked by depth
    time_win_np = np.asarray(prepare_static_plot_matrix(time_win_ms))
    lfp_win_np = np.asarray(prepare_static_plot_matrix(lfp_win))
    depths_np = np.asarray(prepare_static_plot_matrix(contact_depths))
    order = np.argsort(depths_np)
    scale = np.nanstd(lfp_win_np) or 1.0
    for rank, ci in enumerate(order):
        axD.plot(time_win_np, lfp_win_np[:, ci] / scale + rank, lw=0.5, color="#4fc3f7")
    axD.set_yticks(range(len(order)))
    axD.set_yticklabels([f"{depths_np[ci]:.2f}" for ci in order], fontsize=6)
    axD.set_xlabel("Time (ms)")
    axD.set_ylabel("Contact relative depth (superficial->deep)")
    axD.set_title(f"D: LFP-proxy, last {last_window_ms:.0f} ms ({len(order)} ch)", fontsize=10)

    # E: spike raster, last window, z-sorted by depth
    spike_t_np = np.asarray(prepare_static_plot_matrix(spike_times_ms))
    spike_n_np = np.asarray(prepare_static_plot_matrix(spike_neuron_rank))
    axE.scatter(spike_t_np, spike_n_np, s=1.0, c="#f5f5f5", alpha=0.7, rasterized=True)
    axE.set_xlabel("Time (ms)")
    axE.set_ylabel("Neuron rank (z-sorted, superficial->deep)")
    axE.set_title(f"E: Raster, last {last_window_ms:.0f} ms (z-sorted)", fontsize=10)
    if time_win_np.size:
        axE.set_xlim(time_win_np[0], time_win_np[-1])

    # F: E/I mean firing-rate table per layer (2 x len(layer_order))
    rate_table_np = np.asarray(prepare_static_plot_matrix(rate_table_hz))
    axF.axis("off")
    cell_text = [[f"{v:.2f}" if np.isfinite(v) else "-" for v in row] for row in rate_table_np]
    tbl = axF.table(
        cellText=cell_text, rowLabels=["E (Hz)", "I (Hz)"], colLabels=layer_order,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#2a2a2a" if row > 0 else "#3a3a3a")
        cell.set_text_props(color=LINE)
        cell.set_edgecolor("#555555")
    axF.set_title("F: E/I mean firing rate per layer", fontsize=10, color=LINE)

    fig.suptitle(
        f"Etude No. 2 extended panels A-F (HDP, K_HDP={k_hdp_used}) -- "
        f"G (3D column scatter) is the separate network_3d.html",
        fontsize=11, color=LINE,
    )
    return fig


# ---------------------------------------------------------------------------
# scripts/verify_5hz_traces.py
# ---------------------------------------------------------------------------

def voltage_trace_with_spikes_by_celltype(
    time_ms: Any,
    voltages_by_celltype: dict[str, Any],
    spikes_by_celltype: dict[str, Any],
    params_by_celltype: dict[str, dict[str, float]],
    *,
    colors: dict[str, str] | None = None,
    suptitle: str = (
        "Verification of 5.0 Hz (5 spikes / 1000ms) Firing Rates across Cell Types\n"
        "Truth Mode:  | Status: computational_scaffold"
    ),
) -> Any:
    """Stacked voltage-trace panels (one per cell type) with spike ticks.

    ``voltages_by_celltype``/``spikes_by_celltype`` map cell-type label ->
    1D arrays over ``time_ms``; ``params_by_celltype`` maps cell-type label
    -> ``{"drive":..., "noise":..., "spike_count":...}`` used for the panel
    subtitle.
    """
    import matplotlib.pyplot as plt

    default_colors = {"E": "#1f77b4", "PV": "#ff7f0e", "SST": "#2ca02c", "VIP": "#9467bd"}
    colors = colors or default_colors

    time_np = np.asarray(prepare_static_plot_matrix(time_ms))
    cell_types = list(voltages_by_celltype.keys())
    fig, axes = plt.subplots(len(cell_types), 1, figsize=(12, 10), sharex=True, dpi=150)
    if len(cell_types) == 1:
        axes = [axes]

    for idx, cell_type in enumerate(cell_types):
        v_np = np.asarray(prepare_static_plot_matrix(voltages_by_celltype[cell_type]))
        s_np = np.asarray(prepare_static_plot_matrix(spikes_by_celltype[cell_type]))
        p = params_by_celltype[cell_type]
        ax = axes[idx]
        ax.plot(time_np, v_np, color=colors.get(cell_type, "C0"), linewidth=1.5, label=f"{cell_type} Voltage")
        spike_times = time_np[s_np > 0.5]
        ax.plot(spike_times, np.ones_like(spike_times) * 35.0, "|", color="red", markersize=12, markeredgewidth=2, label="Spikes")
        ax.set_ylabel("Voltage (mV)", fontsize=11)
        ax.set_title(
            f"{cell_type} Neuron (Drive = {p['drive']:.3f}, Noise = {p['noise']:.3f}) | "
            f"Spikes: {int(p['spike_count'])} (5.0 Hz)",
            fontsize=12, fontweight="bold", pad=5,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim([-90, 45])

    axes[-1].set_xlabel("Time (ms)", fontsize=12)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
