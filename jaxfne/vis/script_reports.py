"""Rendering helpers for ``scripts/*.py`` (figure/data-build CLIs).

These plain functions take pre-extracted NumPy/JAX-host arrays and return a
matplotlib ``Figure`` or a Plotly ``Figure`` object — they never write files
or call ``plt.show``. Scripts stay free of direct ``matplotlib``/``plotly``
imports, per the package-wide vis-isolation rule (all rendering lives under
``jaxfne.vis``).

Evaluated as an uncalibrated computational scaffold; all readouts here are
simulated proxy diagnostics, not physical measurements.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .core import prepare_static_plot_matrix


# ---------------------------------------------------------------------------
# scripts/build_v037_source_column_3d.py
# ---------------------------------------------------------------------------

def column_network_3d_scatter(
    x: Any,
    y: Any,
    z: Any,
    rates_hz: Any,
    hover_text: list[str],
    *,
    title: str = "Interactive 3D Source/Field/Probe Cortical Column",
    subtitle: str = "",
    equation_note: str = "",
    width: int = 1200,
    height: int = 700,
) -> Any:
    """3D Plotly scatter of neuron positions colored by firing rate.

    Returns a Plotly ``Figure`` with one ``Scatter3d`` trace plus an
    equation-annotation footer. Caller supplies coordinates, per-neuron
    rate (used for marker color), and pre-built hover text.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=list(x),
        y=list(y),
        z=list(z),
        mode="markers",
        marker=dict(
            size=5,
            color=list(rates_hz),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Rate (Hz)", thickness=15, len=0.7),
            line=dict(width=0.5, color="rgba(200, 200, 200, 0.5)"),
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        name="Neurons",
    ))

    title_html = f"<b>{title}</b>"
    if subtitle:
        title_html += f"<br><sub>{subtitle}</sub>"

    annotations = []
    if equation_note:
        annotations.append(dict(
            text=equation_note,
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            align="center",
            font=dict(size=11, color="gray"),
        ))

    fig.update_layout(
        title=dict(text=title_html, font=dict(size=16)),
        scene=dict(
            xaxis=dict(title="X (µm)", backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
            yaxis=dict(title="Y (µm)", backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
            zaxis=dict(title="Depth (µm)", backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        width=width,
        height=height,
        hovermode="closest",
        showlegend=False,
        font=dict(size=11),
        annotations=annotations,
    )
    return fig


def column_readout_panels(
    t_ms: Any,
    source_mean: Any | None,
    bin_times: Any,
    pop_rates_hz: Any,
    lfp_proxy: Any | None,
    layer_names: list[str],
    layer_voltage_means: dict[str, Any],
    layer_colors: dict[str, str],
    *,
    title: str = "Readout Panels: Source / Field / Probe",
    height: int = 700,
) -> Any:
    """2x2 Plotly subplot grid: source summary, population rate, LFP proxy, per-layer V_m.

    Any of ``source_mean``/``lfp_proxy`` may be ``None`` (omits that trace,
    matching the original script's ``if size > 0`` guards).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Source Summary",
            "Population Firing Rate",
            "LFP-Proxy (sample contact)",
            "Mean Voltage by Layer",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    if source_mean is not None:
        fig.add_trace(
            go.Scatter(x=list(t_ms), y=list(source_mean), mode="lines", name="Source mean",
                       line=dict(color="steelblue", width=1)),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(x=list(bin_times), y=list(pop_rates_hz), mode="lines", name="Population rate",
                   line=dict(color="darkgreen", width=2)),
        row=1, col=2,
    )

    if lfp_proxy is not None:
        fig.add_trace(
            go.Scatter(x=list(t_ms), y=list(lfp_proxy), mode="lines", name="LFP-proxy",
                       line=dict(color="purple", width=1)),
            row=2, col=1,
        )

    for layer in layer_names:
        fig.add_trace(
            go.Scatter(x=list(t_ms), y=list(layer_voltage_means[layer]), mode="lines", name=layer,
                       line=dict(color=layer_colors[layer])),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Mean source", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Rate (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="LFP proxy", row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=2)
    fig.update_yaxes(title_text="Mean voltage", row=2, col=2)

    fig.update_layout(title_text=title, height=height, showlegend=True)
    return fig


# ---------------------------------------------------------------------------
# scripts/spectrolaminar_tfne_izhikevich_pipeline.py
# ---------------------------------------------------------------------------

def rate_histogram_by_celltype(
    rate_by_type_hz: dict[str, float],
    band: tuple[float, float],
    *,
    title: str = "Per-cell-type firing rate (HDP-tuned operating point)",
) -> Any:
    """Bar chart of firing rate per cell type with a target-band shaded span."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    types = list(rate_by_type_hz.keys())
    rates = [rate_by_type_hz[t] for t in types]
    ax.bar(types, rates, color="steelblue")
    ax.axhspan(band[0], band[1], color="green", alpha=0.15, label="target band")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def sweep_rate_lines(
    xs: list[float],
    rate_by_type_per_point: list[dict[str, float]],
    band: tuple[float, float],
    xlabel: str,
    *,
    x_key_is_log: bool = False,
) -> Any:
    """Per-cell-type firing-rate line plot across a 1D sweep grid (e.g. K_HDP or tau_0_ms)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    cell_types = sorted({t for r in rate_by_type_per_point for t in r})
    for cell_type in cell_types:
        ys = [r.get(cell_type, np.nan) for r in rate_by_type_per_point]
        ax.plot(xs, ys, marker="o", label=cell_type)
    ax.axhspan(band[0], band[1], color="green", alpha=0.1)
    if x_key_is_log:
        ax.set_xscale("symlog" if 0 in xs else "log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("firing rate (Hz)")
    ax.legend()
    fig.tight_layout()
    return fig


def spectrolaminar_motif_heatmap(
    power: Any,
    depths: Any,
    *,
    title: str = "Spectrolaminar motif (depth x frequency power, proxy)",
) -> Any:
    """Depth x frequency power heatmap (proxy spectrolaminar motif)."""
    import matplotlib.pyplot as plt

    power = prepare_static_plot_matrix(power)
    depths = prepare_static_plot_matrix(depths)

    fig, ax = plt.subplots(figsize=(6, 4))
    extent = [0, power.shape[1], float(depths.min()), float(depths.max())] if power.ndim == 2 else None
    im = ax.imshow(power, aspect="auto", origin="lower", extent=extent, cmap="magma")
    ax.set_xlabel("frequency bin")
    ax.set_ylabel("relative laminar depth")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="power (a.u.)")
    fig.tight_layout()
    return fig


def spike_triggered_waveforms_plot(
    waveforms: dict[str, Any],
    dt_ms: float,
    widths_ms: dict[str, float],
    *,
    title: str = "E vs. I spike-triggered waveform proxy (not a calibrated AP shape)",
) -> Any:
    """Per-cell-type spike-triggered average V_m waveform, with FWHM in the legend."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for cell_type, wf in waveforms.items():
        wf = prepare_static_plot_matrix(wf)
        t_ms = np.arange(wf.shape[0]) * dt_ms
        ax.plot(t_ms, wf, label=f"{cell_type} (FWHM~{widths_ms[cell_type]:.1f} ms)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("V_m (mV, spike-triggered avg)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def hdp_state_trajectories(
    per_type: dict[str, dict[str, Any]],
    *,
    mode: str = "H",
    title: str | None = None,
) -> Any:
    """Per-cell-type HDP master-state trajectory over time.

    ``mode="H"`` plots ``mean_H_t`` with an H=1 equilibrium line; ``mode="pressure"``
    plots ``mean_H_t - 1`` (the controller error) with a zero line.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cell_type, d in per_type.items():
        t_ms = prepare_static_plot_matrix(d["t_ms"])
        mean_H_t = prepare_static_plot_matrix(d["mean_H_t"])
        y = mean_H_t if mode == "H" else (mean_H_t - 1.0)
        ax.plot(t_ms, y, label=cell_type)

    if mode == "H":
        ax.axhline(1.0, color="k", ls="--", lw=0.8, label="H=1 (equilibrium)")
        ax.set_ylabel("mean H")
        default_title = "HDP master state H_i over time (per cell type)"
    else:
        ax.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set_ylabel("mean(H-1)  [controller error]")
        default_title = "HDP pressure (error signal) over time"

    ax.set_xlabel("time (ms)")
    ax.set_title(title or default_title)
    ax.legend()
    fig.tight_layout()
    return fig


def rate_vs_pressure_scatter(
    per_type: dict[str, dict[str, Any]],
    *,
    title: str = "Rate vs. controller error (healthy controller -> monotonic)",
) -> Any:
    """Per-cell-type scatter of binned firing rate vs. HDP controller error."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for cell_type, d in per_type.items():
        pressure = prepare_static_plot_matrix(d["pressure_binned"])
        rate = prepare_static_plot_matrix(d["rate_binned_hz"])
        ax.scatter(pressure, rate, s=10, alpha=0.5, label=cell_type)
    ax.axvline(0.0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("mean(H-1)  [controller error]")
    ax.set_ylabel("binned firing rate (Hz)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def hdp_H_occupancy_histogram(
    H_by_celltype: dict[str, Any],
    *,
    bins: int = 60,
    title: str = "HDP master-state occupancy (per cell type)",
) -> Any:
    """Histogram of H_i values across all timesteps/neurons, one outline per cell type."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cell_type, values in H_by_celltype.items():
        values = prepare_static_plot_matrix(values)
        ax.hist(values.ravel(), bins=bins, histtype="step", density=True, label=cell_type)
    ax.axvline(1.0, color="k", ls="--", lw=0.8, label="H=1 (equilibrium)")
    ax.set_xlabel("H_i")
    ax.set_ylabel("density (over time x neurons)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def hdp_stability_map(
    grids: dict[str, Any],
    k_hdp_grid: list[float],
    tau0_grid_ms: list[float],
    *,
    title: str = "K_HDP x tau_0_ms stability map",
) -> Any:
    """3-panel imshow grid (mean rate / weight saturation / max var(H)) over a K_HDP x tau_0_ms sweep.

    ``grids`` maps field name -> 2D array shaped (len(k_hdp_grid), len(tau0_grid_ms)),
    already assembled by the caller.
    """
    import matplotlib.pyplot as plt

    fields = [
        ("mean_rate_hz", "mean rate (Hz)"),
        ("weight_saturation_fraction", "weight saturation frac."),
        ("max_var_H", "max var(H)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, (key, subtitle) in zip(axes, fields):
        grid = prepare_static_plot_matrix(grids[key])
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(tau0_grid_ms)))
        ax.set_xticklabels([f"{t:g}" for t in tau0_grid_ms])
        ax.set_yticks(range(len(k_hdp_grid)))
        ax.set_yticklabels([f"{k:g}" for k in k_hdp_grid])
        ax.set_xlabel("tau_0_ms")
        ax.set_ylabel("K_HDP")
        ax.set_title(subtitle)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# scripts/generate_tutorial_figures.py
# ---------------------------------------------------------------------------

def tutorial_spike_raster(spikes: Any, *, title: str = "Spike Raster (Izhikevich Simulation)") -> Any:
    """Vline spike raster over (time_steps, n_units) binary spike matrix."""
    import matplotlib.pyplot as plt

    spikes = prepare_static_plot_matrix(spikes)
    time_steps, n_units = spikes.shape
    fig, ax = plt.subplots(figsize=(12, 4))
    for unit_idx in range(n_units):
        spike_times = np.where(spikes[:, unit_idx] > 0.5)[0]
        ax.vlines(spike_times, unit_idx - 0.4, unit_idx + 0.4, colors="black", linewidth=0.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Unit index")
    ax.set_title(title)
    ax.set_ylim(-1, n_units)
    ax.set_xlim(0, time_steps)
    fig.tight_layout()
    return fig


def tutorial_voltage_traces(
    v_m: Any,
    *,
    n_display: int = 6,
    title: str = "Membrane Voltage Traces (Izhikevich Native)",
) -> Any:
    """Stacked subplot of up to ``n_display`` evenly-spaced unit voltage traces."""
    import matplotlib.pyplot as plt

    v_m = prepare_static_plot_matrix(v_m)
    time_steps, n_units = v_m.shape
    n_display = min(n_display, n_units)

    fig, axes = plt.subplots(n_display, 1, figsize=(12, 2 * n_display), sharex=True)
    if n_display == 1:
        axes = [axes]

    unit_indices = np.linspace(0, n_units - 1, n_display, dtype=int)
    for i, ax in enumerate(axes):
        unit_idx = unit_indices[i]
        ax.plot(v_m[:, unit_idx], linewidth=0.5, color="steelblue")
        ax.set_ylabel(f"Unit {unit_idx}\n(mV)")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time step")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def tutorial_matrix_heatmap(
    matrix: Any,
    *,
    transpose: bool = True,
    cmap: str = "RdBu_r",
    xlabel: str = "Time step",
    ylabel: str = "Unit index",
    title: str = "",
    colorbar_label: str = "",
    figsize: tuple[float, float] = (12, 5),
) -> Any:
    """Generic (time x channel) heatmap used for source/CSD/phi_e/source-spatial proxies."""
    import matplotlib.pyplot as plt

    matrix = prepare_static_plot_matrix(matrix)
    plotted = matrix.T if transpose else matrix

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plotted, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig


def tutorial_lfp_proxy_trace(lfp_mean: Any, *, title: str = "LFP Proxy (Averaged Across Contacts)") -> Any:
    """Single-line trace of LFP proxy averaged across contacts."""
    import matplotlib.pyplot as plt

    lfp_mean = prepare_static_plot_matrix(lfp_mean)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(lfp_mean, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Proxy amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def tutorial_conservation_bar(
    metrics: dict[str, float],
    *,
    title: str = "Conservation Proxy Diagnostics (Laminar Field)",
) -> Any:
    """Log-scale bar chart of conservation-proxy diagnostic magnitudes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax.bar(names, values, color=colors[: len(names)], alpha=0.7, edgecolor="black", linewidth=1)
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def tutorial_contact_depths_barh(
    contact_depths: Any,
    *,
    title: str = "Laminar Profile (Contact Depths)",
) -> Any:
    """Horizontal bar chart of per-contact depth (laminar profile axis)."""
    import matplotlib.pyplot as plt

    contact_depths = prepare_static_plot_matrix(contact_depths)
    n_contacts = len(contact_depths)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.arange(n_contacts), contact_depths, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_ylabel("Contact index")
    ax.set_xlabel("Depth (μm, proxy)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def tutorial_status_text_panel(
    text_content: str,
    *,
    suptitle: str = "Statement Gates Summary",
) -> Any:
    """Axis-off figure rendering a monospace status/gate text block.

    Note: the original ``generate_tutorial_figures.py::gen_status_summary``
    called ``ax.text(..., verticalcomparison="top", ...)`` -- an invalid
    matplotlib kwarg (typo for ``verticalalignment``) that always raised
    ``AttributeError`` and was silently swallowed by that script's
    ``try/except`` in ``main()``, so figure 11 was never actually produced.
    This helper uses the correct ``verticalalignment="top"`` so the figure
    now renders; this is a pre-existing-bug fix, not a behavior change this
    migration set out to make, but a correctly-named function cannot also
    reproduce a misspelled kwarg.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(
        0.1, 0.9, text_content, transform=ax.transAxes, fontfamily="monospace",
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def tutorial_spectral_summary(
    freqs_pos: Any,
    power_pos: Any,
    *,
    title: str = "Spectral Summary (Network Activity FFT)",
) -> Any:
    """Log-scale power-spectrum line plot (positive frequencies only)."""
    import matplotlib.pyplot as plt

    freqs_pos = prepare_static_plot_matrix(freqs_pos)
    power_pos = prepare_static_plot_matrix(power_pos)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freqs_pos, power_pos + 1e-10, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Frequency (normalized)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# scripts/run_all_tutorials.py
# ---------------------------------------------------------------------------

def spike_raster_plotly(
    time_ms: list[float],
    unit_id: list[float],
    *,
    title: str = "Spike Raster",
    height: int = 500,
    width: int = 1200,
) -> Any:
    """Plotly scatter spike raster (time vs. unit id) from flat event lists."""
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Scatter(
        x=time_ms,
        y=unit_id,
        mode="markers",
        marker=dict(size=3, color="black", opacity=0.6),
        name="spikes",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (timestep)",
        yaxis_title="Unit ID",
        height=height,
        width=width,
        hovermode="closest",
    )
    return fig


def spectrolaminar_profile_bars_plotly(
    layers: list[Any],
    alpha_beta_profile: list[float],
    gamma_profile: list[float],
    *,
    title: str = "Spectrolaminar Profile",
    height: int = 500,
    width: int = 1200,
) -> Any:
    """Plotly grouped bar chart of alpha/beta vs. gamma relative power per layer/window."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layers, y=alpha_beta_profile, name="Alpha/Beta power",
        marker_color="steelblue", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=layers, y=gamma_profile, name="Gamma power",
        marker_color="coral", opacity=0.7,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Window",
        yaxis_title="Power (relative units)",
        barmode="group",
        height=height,
        width=width,
        hovermode="x unified",
    )
    return fig


def tutorial_spike_raster_vlines_plotly(
    spike_times: list,
    *,
    title: str = "Spike Raster",
) -> Any:
    """Plotly figure with one dashed vline per spike event, annotated by neuron id.

    ``spike_times`` is a list of per-neuron spike-time lists (ms).
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for neuron_id, spike_times_i in enumerate(spike_times):
        for spike_time in spike_times_i:
            fig.add_vline(
                x=spike_time, line_dash="dash", line_color="gray",
                annotation_text=f"Neuron {neuron_id}", annotation_position="top",
            )
    fig.update_layout(
        title=title, xaxis_title="Time (ms)", yaxis_title="Neuron",
        hovermode="x unified", height=400 + len(spike_times) * 5,
    )
    return fig


def tutorial_voltage_trace_plotly(
    time_ms: list,
    voltage_traces: dict,
    *,
    title: str = "Membrane Voltage",
) -> Any:
    """Plotly multi-neuron voltage trace figure. ``voltage_traces`` maps neuron_id -> voltage array (mV)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for neuron_id, voltages in voltage_traces.items():
        fig.add_trace(go.Scatter(
            x=time_ms, y=voltages, mode="lines", name=f"Neuron {neuron_id}",
            hovertemplate="<b>Neuron %{fullData.name}</b><br>Time: %{x:.1f} ms<br>V_m: %{y:.1f} mV<extra></extra>",
        ))
    fig.update_layout(
        title=title, xaxis_title="Time (ms)", yaxis_title="Membrane Potential (mV)",
        hovermode="x unified", height=500,
    )
    return fig


def tutorial_firing_rate_plotly(
    time_ms: list,
    firing_rate_hz: list,
    *,
    title: str = "Population Firing Rate",
) -> Any:
    """Plotly filled-area population firing-rate figure."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_ms, y=firing_rate_hz, mode="lines", name="Population firing rate",
        fill="tozeroy", hovertemplate="Time: %{x:.1f} ms<br>Rate: %{y:.1f} Hz<extra></extra>",
    ))
    fig.update_layout(
        title=title, xaxis_title="Time (ms)", yaxis_title="Firing Rate (Hz)",
        hovermode="x unified", height=400,
    )
    return fig
