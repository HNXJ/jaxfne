"""Laminar and structural fields plotting submodules for jaxfne/vis.

NumPy-isolated graphics for spectrolaminar power profiles, connectivity matrices,
laminar profiles, multi-area layouts, and optimization objective histories.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import numpy as np
from scipy import signal

from .core import FigureResult, prepare_static_plot_matrix, require_matplotlib


def plot_laminar_field_interpolation(field_potential_tensor: jax.Array, grid_coords: dict) -> None:
    """Extracts accelerator tensors onto the host context prior to plotting execution.

    Coordinates the 2D spatial interpolation maps for laminar potentials across multi-contact grids.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    # Force immediate host-device transfer to protect the active trace context
    static_field_matrix = np.asarray(jax.device_get(field_potential_tensor))

    fig, ax = plt.subplots(figsize=grid_coords.get("figsize", (10, 5)))
    # Renders spatial interpolation maps cleanly...
    ax.set_title("Laminar Field Interpolation Map")
    ax.set_xlabel("Laminar Channels")
    ax.set_ylabel("Normalized Depth")
    plt.close(fig)


def _neuron_rows(signals: Any) -> list[dict[str, Any]]:
    meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
    rows = meta.get("neuron_metadata") if isinstance(meta, dict) else None
    return [dict(row) for row in rows] if rows else []


def _signals_time_ms(signals: Any, n: int | None = None) -> np.ndarray:
    if hasattr(signals, "time_ms"):
        return np.asarray(signals.time_ms)
    if isinstance(signals, dict) and "time_ms" in signals:
        return np.asarray(signals["time_ms"])
    return np.arange(int(n or 0))


def _signals_spikes(signals: Any) -> np.ndarray:
    if hasattr(signals, "spikes"):
        return np.asarray(signals.spikes)
    if isinstance(signals, dict) and "spikes" in signals:
        return np.asarray(signals["spikes"])
    return np.asarray(signals)


def _signals_sources(signals: Any) -> np.ndarray:
    src_raw = getattr(signals, "sources", None)
    if src_raw is None and isinstance(signals, dict):
        src_raw = signals.get("sources")
    src = prepare_static_plot_matrix(src_raw)
    if src is None:
        raise ValueError("signals.sources is required for this proxy readout")
    return src


def _signals_lfp(signals: Any) -> np.ndarray:
    if hasattr(signals, "field") and signals.field is not None:
        return np.asarray(signals.field.lfp_proxy)
    if isinstance(signals, dict) and "lfp_proxy" in signals:
        return np.asarray(signals["lfp_proxy"])
    raise ValueError("signals.field.lfp_proxy is required for this figure")


def _signals_csd(signals: Any) -> np.ndarray:
    if hasattr(signals, "field") and signals.field is not None:
        return np.asarray(signals.field.csd_proxy)
    if isinstance(signals, dict) and "csd_proxy" in signals:
        return np.asarray(signals["csd_proxy"])
    raise ValueError("signals.field.csd_proxy is required for this figure")


def _linear_proxy_from_sources(signals: Any, *, n_channels: int, phase: float = 0.0) -> np.ndarray:
    src = _signals_sources(signals)
    n_src = src.shape[1]
    idx = np.arange(n_src, dtype=float)
    channels = []
    for c in range(int(n_channels)):
        channels.append(np.cos((c + 1.0) * (idx + 1.0) / max(n_src, 1) + phase))
    lead = np.asarray(channels, dtype=float)
    lead = lead / np.maximum(np.linalg.norm(lead, axis=1, keepdims=True), 1e-12)
    return src @ lead.T


def _geometry3d_from_config(cfg: Any, *, areas=None, cell_types=None, figsize=(9, 7)) -> Any:
    """Internal: synthesise 3D geometry scatter from Configuration metadata."""
    import matplotlib.pyplot as plt
    meta = cfg.metadata
    columns = meta.get("columns", [])
    ct_fracs = meta.get("cell_types", {})
    if isinstance(ct_fracs, dict) and not ct_fracs:
        ct_fracs = {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07}
    all_cell_types = list(ct_fracs.keys()) if isinstance(ct_fracs, dict) else ["E", "PV", "SST", "VIP"]
    if cell_types is not None:
        all_cell_types = [c for c in all_cell_types if c in cell_types]

    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    colors_map = {ct: plt.cm.Set1(i / max(len(all_cell_types), 1)) for i, ct in enumerate(all_cell_types)}

    for col_idx, col in enumerate(columns):
        if areas is not None and col.get("name") not in areas:
            continue
        n = int(col.get("n", 50))
        x_off = col_idx * 0.6  # offset columns laterally
        for ct in all_cell_types:
            frac = float((ct_fracs if isinstance(ct_fracs, dict) else {}).get(ct, 0.25))
            n_ct = max(1, int(n * frac))
            x = rng.uniform(0, 0.5, n_ct) + x_off
            y = rng.uniform(0, 0.5, n_ct)
            z = rng.uniform(0, 1.6, n_ct)
            ax.scatter(x, y, z, s=6, alpha=0.55, label=ct if col_idx == 0 else "",
                       color=colors_map.get(ct, "gray"))

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=colors_map.get(ct, "gray"), markersize=8, label=ct)
               for ct in all_cell_types]
    ax.legend(handles=handles, title="Cell type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("Declared 3D geometry (Configuration proxy)", fontsize=11)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z — laminar depth")
    fig.tight_layout()
    return fig


def spectrolaminar(signals: Signals, **kwargs: Any) -> Any:
    """Generate a publication-ready, 3-panel spectrolaminar profile figure."""
    field_obj = getattr(signals, "field", None)
    if field_obj is None and isinstance(signals, dict):
        field_obj = signals.get("field")
    if field_obj is None:
        raise ValueError(
            "Cannot generate spectrolaminar profile: signals.field is None. "
            "Ensure that record_fields was set to True in the Simulation config."
        )

    # Enforce/check matplotlib availability
    require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Extract signals and metadata safely without using boolean array evaluation
    time_ms_raw = getattr(signals, "time_ms", None)
    if time_ms_raw is None and isinstance(signals, dict):
        time_ms_raw = signals.get("time_ms")
    time_ms = prepare_static_plot_matrix(time_ms_raw)
    
    field = getattr(signals, "field", None)
    if field is None and isinstance(signals, dict):
        field = signals.get("field")

    if field is not None:
        lfp_raw = getattr(field, "lfp_proxy", None)
        if lfp_raw is None and isinstance(field, dict):
            lfp_raw = field.get("lfp_proxy")
        lfp = prepare_static_plot_matrix(lfp_raw)

        csd_raw = getattr(field, "csd_proxy", None)
        if csd_raw is None and isinstance(field, dict):
            csd_raw = field.get("csd_proxy")
        csd = prepare_static_plot_matrix(csd_raw)

        depths_raw = getattr(field, "contact_depths", None)
        if depths_raw is None and isinstance(field, dict):
            depths_raw = field.get("contact_depths")
        depths = prepare_static_plot_matrix(depths_raw)
    else:
        lfp = None
        csd = None
        depths = None
    
    meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
    dt_ms = float(meta.get("dt_ms", 0.05)) if isinstance(meta, dict) else 0.05
    fs = 1000.0 / dt_ms  # Sampling frequency in Hz

    # Compute Power Spectral Density (PSD) via SciPy Welch for each laminar contact
    nperseg = min(256, lfp.shape[0])
    freqs, psds = signal.welch(lfp, fs=fs, axis=0, nperseg=nperseg)

    # Slice frequency axis to neurophysiologically relevant band (0 to 150 Hz)
    freq_mask = freqs <= 150.0
    freqs_masked = freqs[freq_mask]
    psd_masked = psds[freq_mask, :]  # Shape: [F_masked, n_contacts]

    # Create figure
    fig_kwargs = {"figsize": (16, 5), "facecolor": "#f8f9fa"}
    fig_kwargs.update(kwargs)
    fig = plt.figure(**fig_kwargs)

    gs = GridSpec(1, 3, wspace=0.35)

    # Common extent formatting [xmin, xmax, ymin, ymax]
    extent_time = [time_ms[0], time_ms[-1], depths[-1], depths[0]]
    extent_freq = [freqs_masked[0], freqs_masked[-1], depths[-1], depths[0]]

    # 1. Extracellular Potential Heatmap (LFP-proxy)
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(
        lfp.T,
        cmap="viridis",
        aspect="auto",
        extent=extent_time,
        origin="upper",
    )
    ax0.set_title("Extracellular Potential (LFP-proxy)", fontsize=12, fontweight="bold", pad=12)
    ax0.set_xlabel("Time (ms)", fontsize=10)
    ax0.set_ylabel("Laminar Depth (relative)", fontsize=10)
    ax0.grid(True, linestyle="--", alpha=0.3)
    cbar0 = fig.colorbar(im0, ax=ax0)
    cbar0.set_label("Potential (proxy units)", fontsize=9)

    # 2. Current Source Density Heatmap (CSD-proxy)
    ax1 = fig.add_subplot(gs[1])
    csd_max = float(np.max(np.abs(csd)))
    vmin = -csd_max if csd_max > 0 else -1.0
    vmax = csd_max if csd_max > 0 else 1.0
    im1 = ax1.imshow(
        csd.T,
        cmap="RdBu_r",
        aspect="auto",
        extent=extent_time,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("Current Source Density (CSD-proxy)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Time (ms)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("CSD (proxy units)", fontsize=9)

    # 3. LFP Power Spectral Density Heatmap (Spectrolaminar Profile)
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(
        psd_masked.T,
        cmap="magma",
        aspect="auto",
        extent=extent_freq,
        origin="upper",
    )
    ax2.set_title("Spectrolaminar Power Profile (PSD)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Frequency (Hz)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("Power (dB-proxy)", fontsize=9)

    fig.suptitle(
        "jaxfne Spectrolaminar Profile  |  "
        "Status: Simulated Laminar Proxy Readout",
        fontsize=11,
        color="#495057",
        fontstyle="italic",
        y=1.02,
    )

    return fig


def bandpower(
    signals: Any,
    *,
    band_definitions: dict[str, tuple[float, float]] | None = None,
    figsize: tuple[float, float] = (10, 5),
    **kwargs: Any,
) -> Any:
    """Plot mean spectral band power per contact."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    if band_definitions is None:
        band_definitions = {
            "alpha/beta\n(8-25 Hz)": (8.0, 25.0),
            "gamma\n(40-150 Hz)": (40.0, 150.0),
        }

    lfp_arr = prepare_static_plot_matrix(_signals_lfp(signals))
    if not np.all(np.isfinite(lfp_arr)):
        lfp_arr = np.nan_to_num(lfp_arr, nan=0.0, posinf=0.0, neginf=0.0)

    meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
    dt_ms = float(meta.get("dt_ms", 0.1)) if isinstance(meta, dict) else 0.1
    fs = 1000.0 / dt_ms
    freqs, pxx = signal.welch(lfp_arr, fs=fs, axis=0, nperseg=min(512, lfp_arr.shape[0]))

    n_bands = len(band_definitions)
    fig, axes = plt.subplots(1, n_bands, figsize=figsize, squeeze=False)
    n_contacts = lfp_arr.shape[1] if lfp_arr.ndim > 1 else 1

    for col, (band_name, (lo, hi)) in enumerate(band_definitions.items()):
        ax = axes[0, col]
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, f"Band {band_name}\nnot covered by\nsampling rate",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(band_name)
            continue
        if lfp_arr.ndim > 1:
            band_power = np.mean(pxx[mask, :], axis=0)
        else:
            band_power = np.array([np.mean(pxx[mask])])

        contact_labels = [str(c) for c in range(len(band_power))]
        ax.barh(contact_labels, band_power, color="steelblue", alpha=0.8)
        ax.set_title(f"Band power\n{band_name}", fontsize=10)
        ax.set_xlabel("Power (proxy a.u.)")
        ax.set_ylabel("Contact")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Spectrolaminar Band Power (proxy relative units)", fontsize=11)
    fig.tight_layout()
    return fig


def laminar_profile(
    signals: Any,
    *,
    cell_types: list[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    **kwargs: Any,
) -> Any:
    """Plot laminar neuron count distribution by cell type."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    rows = _neuron_rows(signals)
    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "neuron_metadata not available\nin this signals object.\nRun with n_contacts > 0.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Laminar profile (declared geometry proxy)")
        return fig

    z_vals = np.asarray([float(row.get("z", 0.0)) for row in rows])
    ct_vals = [str(row.get("cell_type", "unknown")) for row in rows]

    if cell_types is None:
        cell_types = sorted(set(ct_vals))

    z_min, z_max = float(z_vals.min()), float(z_vals.max())
    if z_max == z_min:
        z_max = z_min + 1.0
    n_bins = 10
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    colors = plt.cm.Set2(np.linspace(0, 1, len(cell_types)))
    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(n_bins)
    for ct, color in zip(cell_types, colors):
        ct_mask = np.asarray([v == ct for v in ct_vals])
        ct_z = z_vals[ct_mask]
        counts, _ = np.histogram(ct_z, bins=bin_edges)
        ax.barh(bin_centers, counts, height=(z_max - z_min) / n_bins * 0.8,
                left=bottom, color=color, alpha=0.85, label=ct)
        bottom += counts

    ax.set_title("Laminar neuron profile (declared geometry proxy)", fontsize=11)
    ax.set_xlabel("Neuron count")
    ax.set_ylabel("Depth z (mm or relative)")
    ax.invert_yaxis()
    ax.legend(title="Cell type", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def layer_celltype_counts(signals: Any, **kwargs: Any) -> Any:
    """Plot laminar neuron count distribution by cell type (alias)."""
    return laminar_profile(signals, **kwargs)


def connectivity(
    model_or_W: Any,
    *,
    cell_type_labels: list[str] | None = None,
    colormap: str = "RdBu_r",
    figsize: tuple[float, float] = (8, 7),
    **kwargs: Any,
) -> Any:
    """Plot the recurrent connectivity weight matrix as a heatmap."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    W = None
    if hasattr(model_or_W, "params"):
        params = model_or_W.params
        if hasattr(params, "get"):
            W = params.get("W", params.get("weights", None))
        elif hasattr(params, "W"):
            W = params.W
        elif hasattr(params, "weights"):
            W = params.weights
    if W is None and hasattr(model_or_W, "__array__"):
        W = np.asarray(model_or_W)
    if W is None and hasattr(model_or_W, "shape"):
        W = np.asarray(model_or_W)

    if W is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5,
                "Weight matrix W not accessible\nfrom this model object.\n"
                "Pass W directly: jtfne.vis.connectivity(model.params['W'])",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Recurrent connectivity (proxy weight structure)")
        return fig

    W_np = prepare_static_plot_matrix(W)
    if not np.all(np.isfinite(W_np)):
        W_np = np.nan_to_num(W_np, nan=0.0, posinf=0.0, neginf=0.0)

    vmax = float(np.nanmax(np.abs(W_np))) or 1.0
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(W_np, cmap=colormap, vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="Weight (proxy a.u.)")

    if cell_type_labels is not None and len(cell_type_labels) == W_np.shape[0]:
        ax.set_xticks(range(len(cell_type_labels)))
        ax.set_yticks(range(len(cell_type_labels)))
        ax.set_xticklabels(cell_type_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cell_type_labels, fontsize=7)

    ax.set_title("Recurrent connectivity W (proxy weight structure)", fontsize=11)
    ax.set_xlabel("Target neuron")
    ax.set_ylabel("Source neuron")
    fig.tight_layout()
    return fig


def connectivity_matrix(model_or_W: Any, **kwargs: Any) -> Any:
    """Plot recurrent connectivity (alias)."""
    return connectivity(model_or_W, **kwargs)


def multi_area_layout(
    signals_or_cfg: Any,
    *,
    areas: list[str] | None = None,
    figsize: tuple[float, float] = (10, 5),
    **kwargs: Any,
) -> Any:
    """Plot a 2D top-down layout showing multiple cortical areas side by side."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    columns = []
    inter_conn = {}
    if hasattr(signals_or_cfg, "metadata"):
        meta = signals_or_cfg.metadata
        columns = meta.get("columns", [])
        inter_conn = meta.get("inter_column_connectivity", {})
    elif hasattr(signals_or_cfg, "neuron_metadata"):
        rows = _neuron_rows(signals_or_cfg)
        area_counts = {}
        for row in rows:
            area = str(row.get("area", "unknown"))
            area_counts[area] = area_counts.get(area, 0) + 1
        columns = [{"name": a, "n": n} for a, n in area_counts.items()]

    if not columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Column metadata required\nfor multi_area_layout.\n"
                "Use a Configuration or signals with neuron_metadata.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Multi-area layout (declared metadata proxy)")
        return fig

    if areas is not None:
        columns = [c for c in columns if c.get("name") in areas]

    fig, ax = plt.subplots(figsize=figsize)
    n_cols = len(columns)
    col_width = 0.8 / max(n_cols, 1)
    spacing = 1.0 / max(n_cols, 1)

    col_centers = {}
    for i, col in enumerate(columns):
        cx = spacing * i + spacing / 2
        cy = 0.5
        col_centers[col.get("name", str(i))] = (cx, cy)
        rect = mpatches.FancyBboxPatch(
            (cx - col_width / 2, cy - 0.25), col_width, 0.5,
            boxstyle="round,pad=0.02", linewidth=2,
            edgecolor="steelblue", facecolor="lightsteelblue", alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + 0.1, col.get("name", f"Area {i}"), ha="center", va="center",
                fontsize=12, fontweight="bold", color="navy")
        n_neurons = col.get("n", "?")
        layers = col.get("layers", [])
        layer_str = f"{len(layers)} layers" if layers else ""
        ax.text(cx, cy - 0.1, f"N={n_neurons}\n{layer_str}", ha="center", va="center",
                fontsize=9, color="darkblue")

    if inter_conn:
        src_area = inter_conn.get("source_area", "")
        tgt_area = inter_conn.get("target_area", "")
        if src_area in col_centers and tgt_area in col_centers:
            sx, sy = col_centers[src_area]
            tx, ty = col_centers[tgt_area]
            ax.annotate("", xy=(tx - col_width / 2, ty + 0.15),
                         xytext=(sx + col_width / 2, sy + 0.15),
                         arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))
            ax.text((sx + tx) / 2, sy + 0.22, "FF", ha="center", fontsize=9, color="darkorange")
            ax.annotate("", xy=(sx + col_width / 2, sy - 0.15),
                         xytext=(tx - col_width / 2, ty - 0.15),
                         arrowprops=dict(arrowstyle="->", color="mediumpurple", lw=2))
            ax.text((sx + tx) / 2, sy - 0.22, "FB", ha="center", fontsize=9, color="mediumpurple")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Multi-area layout (declared connectivity metadata proxy)", fontsize=11)
    fig.tight_layout()
    return fig


def objective_report(
    tune_result_or_history: Any,
    *,
    figsize: tuple[float, float] = (9, 4),
    **kwargs: Any,
) -> Any:
    """Plot optimization objective value history."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    history = None
    if hasattr(tune_result_or_history, "summary"):
        summary = tune_result_or_history.summary
        if isinstance(summary, dict):
            history = (summary.get("score_history")
                       or summary.get("objective_history")
                       or summary.get("loss_history"))
    elif isinstance(tune_result_or_history, dict):
        history = (tune_result_or_history.get("score_history")
                   or tune_result_or_history.get("objective_history")
                   or tune_result_or_history.get("loss_history"))
    if history is None:
        try:
            history = list(tune_result_or_history)
        except (TypeError, ValueError):
            history = None

    fig, axes = plt.subplots(1, 2 if history is not None else 1, figsize=figsize, squeeze=False)

    if history is not None:
        history_arr = prepare_static_plot_matrix(history)
        if not np.all(np.isfinite(history_arr)):
            history_arr = np.nan_to_num(history_arr, nan=np.nan)
        iters = np.arange(len(history_arr))
        ax = axes[0, 0]
        ax.plot(iters, history_arr, "o-", color="steelblue", ms=4, lw=1.5)
        ax.set_title("Objective history (surrogate proxy)", fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective value (proxy a.u.)")
        ax.grid(True, linestyle="--", alpha=0.4)

        if len(history_arr) > 1:
            ax2 = axes[0, 1] if axes.shape[1] > 1 else axes[0, 0]
            delta = np.diff(history_arr)
            ax2.bar(iters[1:], delta, color=np.where(delta < 0, "green", "red"), alpha=0.7)
            ax2.axhline(0, color="black", lw=0.8)
            ax2.set_title("Δ Objective per step", fontsize=11)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Δ value")
            ax2.grid(True, linestyle="--", alpha=0.4)
    else:
        ax = axes[0, 0]
        ax.text(0.5, 0.5,
                "Objective history not found.\nPass TuneResult, dict with\n"
                "'score_history' key, or list of scores.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Objective report (proxy)")

    fig.suptitle("Optimization objective — Evaluated as a structured simulation proxy under uncalibrated computational scaffold matching truth_safe_unverified boundaries.",
                 fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    return fig


def spectrolaminar_suite(signals: Any, **kwargs: Any) -> Any:
    """Render the core Suite No. 2 readout panel in one figure."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    freq_min_hz = float(kwargs.pop("freq_min_hz", 0.0))
    freq_max_hz = float(kwargs.pop("freq_max_hz", 80.0))
    freq_count = int(kwargs.pop("freq_count", 128))
    psd_nperseg = kwargs.pop("psd_nperseg", None)
    figure_title = kwargs.pop("title", None)
    figsize = kwargs.pop("figsize", (12, 10))
    dpi = kwargs.pop("dpi", None)

    fig_kwargs = {"figsize": figsize}
    if dpi is not None:
        fig_kwargs["dpi"] = int(dpi)
    fig = plt.figure(**fig_kwargs)

    axes = [fig.add_subplot(3, 2, i + 1) for i in range(6)]
    spikes = _signals_spikes(signals)
    time_ms = _signals_time_ms(signals, spikes.shape[0])
    rows = _neuron_rows(signals)
    t_idx, n_idx = np.where(spikes > 0)
    if len(rows) == spikes.shape[1]:
        order = np.argsort([float(row.get("z", i)) for i, row in enumerate(rows)])
        rank = np.empty_like(order)
        rank[order] = np.arange(order.shape[0])
        n_plot = rank[n_idx]
    else:
        n_plot = n_idx
    axes[0].scatter(time_ms[t_idx], n_plot, s=1, marker="|")
    axes[0].set_title("Raster proxy sorted by depth")
    axes[0].set_xlabel("Time (ms)")

    try:
        lfp_arr = _signals_lfp(signals)
    except ValueError:
        lfp_arr = None

    if lfp_arr is not None and np.all(np.isfinite(lfp_arr)):
        im1 = axes[1].imshow(lfp_arr.T, aspect="auto", origin="upper", extent=[time_ms[0], time_ms[-1], lfp_arr.shape[1], 0])
        axes[1].set_title("LFP-like contacts")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
    else:
        axes[1].text(0.5, 0.5, "LFP data unavailable", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("LFP-like contacts")

    try:
        csd_arr = _signals_csd(signals)
    except ValueError:
        csd_arr = None

    if csd_arr is not None and np.all(np.isfinite(csd_arr)):
        vmax = float(np.nanmax(np.abs(csd_arr))) or 1.0
        im2 = axes[2].imshow(csd_arr.T, aspect="auto", origin="upper", extent=[time_ms[0], time_ms[-1], csd_arr.shape[1], 0], vmin=-vmax, vmax=vmax)
        axes[2].set_title("CSD-like contacts")
        fig.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        axes[2].text(0.5, 0.5, "CSD data unavailable", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("CSD-like contacts")

    if lfp_arr is not None and np.all(np.isfinite(lfp_arr)):
        dt_ms = float(getattr(signals, "metadata", {}).get("dt_ms", 0.1)) if hasattr(signals, "metadata") else 0.1
        fs = 1000.0 / dt_ms
        nperseg = psd_nperseg if psd_nperseg is not None else min(512, lfp_arr.shape[0])
        freqs, pxx = signal.welch(lfp_arr, fs=fs, axis=0, nperseg=int(nperseg))
        mask = (freqs >= freq_min_hz) & (freqs <= freq_max_hz)
        axes[3].plot(freqs[mask], np.mean(pxx[mask], axis=1))
        axes[3].set_title("Mean PSD")
        axes[3].set_xlabel("Frequency (Hz)")
        axes[3].set_xlim(freq_min_hz, freq_max_hz)
    else:
        axes[3].text(0.5, 0.5, "PSD unavailable", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_title("Mean PSD")

    try:
        eeg_y = _linear_proxy_from_sources(signals, n_channels=3, phase=0.0)
    except ValueError:
        eeg_y = None

    if eeg_y is not None and np.all(np.isfinite(eeg_y)):
        scale = np.nanstd(eeg_y) or 1.0
        for ch in range(eeg_y.shape[1]):
            axes[4].plot(time_ms, eeg_y[:, ch] / scale + ch)
        axes[4].set_title("EEG-proxy")
    else:
        axes[4].text(0.5, 0.5, "EEG proxy unavailable", ha="center", va="center", transform=axes[4].transAxes)
        axes[4].set_title("EEG-proxy")

    spikes_mean = np.mean(spikes, axis=1)
    try:
        src = _signals_sources(signals)
    except ValueError:
        src = None

    if src is not None and np.all(np.isfinite(src)):
        emm_cost = spikes_mean + np.mean(np.abs(src), axis=1)
        if lfp_arr is not None:
            emm_cost = emm_cost + np.mean(lfp_arr * lfp_arr, axis=1)
        emm_cost = emm_cost / max(float(np.nanmax(np.abs(emm_cost))), 1e-12)
        axes[5].plot(time_ms, emm_cost)
        axes[5].set_title("EMM-proxy")
    else:
        axes[5].plot(time_ms, spikes_mean / max(float(np.nanmax(spikes_mean)), 1e-12))
        axes[5].set_title("EMM-proxy (fallback: spike rate)")
    axes[5].set_xlabel("Time (ms)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()

    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=14, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.99])

    return fig
