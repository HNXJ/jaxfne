"""One-off report/diagnostic figures for jaxfne's scripts/ tooling.

These back narrowly-scoped plotting calls in scripts/*.py report generators
(COOP/plasticity gate reports, delta-test notebooks, AGSDR sweeps) that
previously called matplotlib/plotly directly. Each function takes plain
arrays (numpy/jax) and returns a matplotlib Figure -- no rendering happens
outside jaxfne.vis.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .core import require_matplotlib, prepare_static_plot_matrix


def dual_raster_comparison(
    spikes_a: Any,
    spikes_b: Any,
    *,
    duration_s: float,
    title_a: str = "Raster A",
    title_b: str = "Raster B",
    figsize: tuple = (10, 6),
) -> Any:
    """Two stacked binary-spike rasters (imshow), sharing the time axis.

    ``spikes_a``/``spikes_b`` are (n_steps, n_neurons) binary arrays.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    a = prepare_static_plot_matrix(spikes_a)
    b = prepare_static_plot_matrix(spikes_b)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].imshow(a.T, aspect="auto", cmap="binary", origin="lower", extent=[0, duration_s, 0, a.shape[1]])
    axes[0].set_title(title_a)
    axes[0].set_ylabel("Neuron ID")
    axes[1].imshow(b.T, aspect="auto", cmap="binary", origin="lower", extent=[0, duration_s, 0, b.shape[1]])
    axes[1].set_title(title_b)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Neuron ID")
    fig.tight_layout()
    return fig


def optimization_progress_line(
    steps: Sequence[float],
    values: Sequence[float],
    *,
    xlabel: str = "Step",
    ylabel: str = "Score",
    title: str = "Optimization Progress",
    color: str = "purple",
    figsize: tuple = (8, 5),
) -> Any:
    """Single-line step/score progress plot (PID/sweep/tuning-loop diagnostics)."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(list(steps), list(values), "o-", color=color, label=ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    return fig


def spike_grid_heatmap(
    spikes: Any,
    *,
    duration_s: float,
    title: str = "Spike Grid",
    figsize: tuple = (11, 6),
) -> Any:
    """Binary spike grid (n_steps, n_neurons) as a black/white imshow."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    s = prepare_static_plot_matrix(spikes)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(s.T, aspect="auto", cmap="binary", extent=[0, duration_s, 0, s.shape[1]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron ID")
    ax.set_title(title)
    return fig


def signed_heatmap_with_colorbar(
    data: Any,
    *,
    extent: Optional[Sequence[float]] = None,
    transpose: bool = True,
    title: str = "Heatmap",
    xlabel: str = "Time (s)",
    ylabel: str = "Channel",
    cbar_label: str = "Amplitude",
    cmap: str = "RdBu_r",
    figsize: tuple = (11, 6),
) -> Any:
    """Generic signed 2D heatmap with a colorbar (CSD-proxy style, or any
    depth/channel x time/frequency array). ``data`` is (rows, cols); by
    default transposed before imshow so a (time, channel) array renders as
    channel-on-y / time-on-x (set ``transpose=False`` if already laid out
    as (channel/row, time/col))."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    d = prepare_static_plot_matrix(data)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(d.T if transpose else d, aspect="auto", cmap=cmap, extent=extent)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def grouped_bar_comparison(
    categories: Sequence[str],
    group_values: dict,
    *,
    xlabel: str = "Category",
    ylabel: str = "Count",
    title: str = "Grouped Bar Comparison",
    colors: Optional[Sequence[str]] = None,
    bar_width: float = 0.4,
    figsize: tuple = (11, 6),
) -> Any:
    """Grouped bar chart: ``group_values`` maps group label -> per-category values."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(categories))
    n_groups = len(group_values)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_width
    for i, (label, values) in enumerate(group_values.items()):
        color = colors[i] if colors is not None else None
        ax.bar(x + offsets[i], values, width=bar_width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(list(categories))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def agsdr_rate_tuning_panel_grid(
    *,
    gains: Sequence[float],
    scores: Sequence[float],
    mean_rates: Sequence[float],
    best_score: float,
    best_gain: float,
    target_rate_hz: float,
    rate_tolerance_hz: float,
    baseline_vals: Sequence[float],
    tuned_vals: Sequence[float],
    min_neuron_rate_gate_hz: float,
    tuned_rates_histogram: Any,
    categories: Sequence[str] = ("Mean Rate", "Min Rate"),
    suptitle: str = "AGSDR Connectivity-Gain Tuning toward Spike-Rate Target",
    figsize: tuple = (14, 10),
) -> Any:
    """2x2 AGSDR connectivity-gain tuning diagnostic: score curve, rate curve,
    baseline-vs-tuned bar comparison, tuned per-neuron rate histogram."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(suptitle, fontsize=14)

    # Panel A: candidate gain vs objective score
    axes[0, 0].plot(list(gains), list(scores), "o-", linewidth=2, markersize=8)
    axes[0, 0].axhline(y=best_score, color="r", linestyle="--", label="best")
    axes[0, 0].set_xlabel("Connectivity Gain")
    axes[0, 0].set_ylabel("Objective Score")
    axes[0, 0].set_title("A. Candidate Gain vs Objective Score")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Panel B: candidate gain vs mean firing rate
    axes[0, 1].plot(list(gains), list(mean_rates), "o-", linewidth=2, markersize=8, color="green")
    axes[0, 1].axhline(y=target_rate_hz, color="orange", linestyle="--", label="target")
    axes[0, 1].axhline(y=target_rate_hz + rate_tolerance_hz, color="orange", linestyle=":", alpha=0.5)
    axes[0, 1].axhline(y=target_rate_hz - rate_tolerance_hz, color="orange", linestyle=":", alpha=0.5)
    axes[0, 1].axvline(x=best_gain, color="r", linestyle="--", label="best gain")
    axes[0, 1].set_xlabel("Connectivity Gain")
    axes[0, 1].set_ylabel("Mean Firing Rate (Hz)")
    axes[0, 1].set_title("B. Candidate Gain vs Mean Firing Rate")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Panel C: baseline vs tuned mean/min firing rate
    x_pos = np.arange(len(categories))
    width = 0.35
    axes[1, 0].bar(x_pos - width / 2, list(baseline_vals), width, label="baseline")
    axes[1, 0].bar(x_pos + width / 2, list(tuned_vals), width, label="tuned")
    axes[1, 0].axhline(y=min_neuron_rate_gate_hz, color="r", linestyle="--", label="min gate (1.0 Hz)")
    axes[1, 0].set_ylabel("Firing Rate (Hz)")
    axes[1, 0].set_title("C. Baseline vs Tuned Rates")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(list(categories))
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Panel D: tuned per-neuron rate histogram with min gate
    hist_data = prepare_static_plot_matrix(tuned_rates_histogram)
    axes[1, 1].hist(hist_data, bins=20, edgecolor="black", alpha=0.7)
    axes[1, 1].axvline(x=min_neuron_rate_gate_hz, color="r", linestyle="--", linewidth=2, label="min gate (1.0 Hz)")
    axes[1, 1].set_xlabel("Firing Rate (Hz)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("D. Tuned Per-Neuron Firing Rate")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def open_pdf_pages(path: Any) -> Any:
    """Open a multi-page PDF writer (``matplotlib.backends.backend_pdf.PdfPages``).

    Thin wrapper so report-assembler scripts never import matplotlib
    directly just to write a PDF; use as ``with jaxfne.vis.open_pdf_pages(path) as pdf:``.
    """
    require_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    return PdfPages(path)


def pdf_text_page(pdf: Any, lines: Sequence[str], *, title: Optional[str] = None) -> None:
    """Render one text-only PDF page (monospace body, optional bold title,
    ``## `` -prefixed lines as section headers) and add it to ``pdf``
    (a handle from :func:`open_pdf_pages`)."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.05)
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.98
    if title:
        ax.text(0.0, y, title, fontsize=18, fontweight="bold", va="top", transform=ax.transAxes)
        y -= 0.045
    for ln in lines:
        if ln.startswith("## "):
            y -= 0.012
            ax.text(0.0, y, ln[3:], fontsize=13, fontweight="bold", va="top", transform=ax.transAxes)
            y -= 0.028
        else:
            ax.text(0.0, y, ln, fontsize=10, va="top", family="monospace", transform=ax.transAxes)
            y -= 0.022
        if y < 0.05:  # spill protection
            break
    pdf.savefig(fig)
    plt.close(fig)


def pdf_figure_page(pdf: Any, fig_path: Any, caption: str) -> None:
    """Embed an existing image file as one full-page PDF page with a caption
    and add it to ``pdf`` (a handle from :func:`open_pdf_pages`)."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(str(fig_path))
    fig = plt.figure(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.06)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.03,
              "computational_scaffold / proxy_readout / no physical amplitude claim",
              ha="center", fontsize=8, style="italic", color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def celltype_sweep_heatmap_grid(
    grids: dict,
    *,
    title: str,
    ylabel: str,
    xlabel: str,
    xticklabels: Sequence[float],
    yticklabels: Sequence[float],
    cell_types: Sequence[str] = ("E", "PV", "SST", "VIP"),
    colormaps: Optional[dict] = None,
    value_label: str = "Spike Count",
    figsize: tuple = (14, 12),
) -> Any:
    """2x2 grid of per-cell-type annotated parameter-sweep heatmaps.

    ``grids`` maps cell type -> 2D array (n_noise, n_drive) of integer/scalar
    counts; each panel gets its own colormap (default per-cell-type palette)
    and per-cell text annotation.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    if colormaps is None:
        colormaps = {"E": "Blues", "PV": "Oranges", "SST": "Greens", "VIP": "Purples"}

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=150)
    axes = axes.flatten()

    for idx, cell_type in enumerate(cell_types):
        ax = axes[idx]
        grid = prepare_static_plot_matrix(grids[cell_type])
        cmap = colormaps.get(cell_type, "viridis")
        n_rows, n_cols = grid.shape

        im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(f"{cell_type} Neuron Spikes (1000 ms)", fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels([f"{x:.1f}" for x in xticklabels], fontsize=10)
        ax.set_yticklabels([f"{y:.1f}" for y in yticklabels], fontsize=10)

        grid_max = np.max(grid) if grid.size else 1.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = grid[i, j]
                text_color = "white" if val > grid_max * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                         color=text_color, fontsize=9, fontweight="bold")

        fig.colorbar(im, ax=ax, shrink=0.8, label=value_label)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def gain_matrix_heatmap(
    gain_matrix: Any,
    area_labels: Sequence[str],
    *,
    title: str = "Optimized Structured Gain Matrix",
    cbar_label: str = "Synaptic Gain Scale",
    xlabel: str = "Presynaptic Source Area",
    ylabel: str = "Postsynaptic Destination Area",
    vmin: float = 0.0,
    vmax: float = 3.0,
    annotate_threshold: float = 1.5,
    skip_diagonal: bool = True,
    figsize: tuple = (11, 8.5),
) -> Any:
    """Annotated NxN inter-area gain matrix heatmap (off-diagonal values
    text-labeled, white-on-dark / black-on-light by value vs ``annotate_threshold``)."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    g = prepare_static_plot_matrix(gain_matrix)
    n = g.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(g, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(list(area_labels))
    ax.set_yticklabels(list(area_labels))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if skip_diagonal and i == j:
                continue
            val = float(g[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val < annotate_threshold else "black")
    fig.tight_layout()
    return fig
