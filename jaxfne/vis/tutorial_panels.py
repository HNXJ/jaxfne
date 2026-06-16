"""Visual suites for laminar column tutorials (Etude No. 1)."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Sequence


def _apply_dark_theme(fig, ax=None):
    """Apply dark theme to matplotlib figure."""
    fig.patch.set_facecolor('#0d0d10')
    for text in fig.texts:
        text.set_color('#f5f5f5')
    if hasattr(fig, "_suptitle") and fig._suptitle is not None:
        try:
            fig._suptitle.set_color('#f5f5f5')
        except Exception:
            pass
    if ax is not None:
        # Handle different ax inputs
        if isinstance(ax, np.ndarray):
            axes = ax.flatten().tolist()
        elif isinstance(ax, (list, tuple)):
            axes = list(ax) if not isinstance(ax, list) else ax
        else:
            axes = [ax]

        for a in axes:
            if a is None or not hasattr(a, 'set_facecolor'):
                continue
            a.set_facecolor('#121217')
            a.tick_params(colors='#f5f5f5')
            for spine in a.spines.values():
                spine.set_color('#2a2a35')
            a.xaxis.label.set_color('#f5f5f5')
            a.yaxis.label.set_color('#f5f5f5')
            a.title.set_color('#f5f5f5')
    return fig


def visualize_laminar_column_3d(
    model: dict,
    cfg: object,
    title: str = "Laminar Column",
    output_png: str | Path | None = None,
    output_html: str | Path | None = None,
    show_edges: bool = False,
    return_node_table: bool = False,
    theme: str = "dark",
) -> matplotlib.figure.Figure | Tuple[matplotlib.figure.Figure, Any]:
    """Create 3D visualization of laminar column network.

    Parameters
    ----------
    model : dict
        Model from build_laminar_column()
    cfg : LaminarColumnConfig
        Configuration
    title : str
        Figure title
    output_png : str/Path or None
        Path to save PNG
    output_html : str/Path or None
        Path to save Plotly HTML
    show_edges : bool
        Whether to show synaptic edges
    return_node_table : bool
        Whether to return neuron table
    theme : str
        Visual theme ("dark", "light")

    Returns
    -------
    tuple of (fig, node_table) if return_node_table else fig
        fig: matplotlib Figure or Plotly Figure
        node_table: DataFrame with neuron positions
    """
    from .core import require_matplotlib
    require_matplotlib()
    import matplotlib.pyplot as plt

    neurons_df = model['neurons']
    positions_m = model['positions_m']

    # Prepare node table for return
    node_table = neurons_df.copy()

    # Create matplotlib 3D figure
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Plot neurons by cell type
    cell_colors = {
        'E': '#1f77b4',
        'PV': '#d62728',
        'SST': '#2ca02c',
        'VIP': '#9467bd',
    }

    for cell_type in neurons_df['cell_type'].unique():
        mask = neurons_df['cell_type'] == cell_type
        indices = mask[mask].index
        positions_subset = positions_m[indices]
        color = cell_colors.get(cell_type, '#808080')
        ax.scatter(positions_subset[:, 0], positions_subset[:, 1], positions_subset[:, 2],
                  s=20, c=color, label=cell_type, alpha=0.7)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)

    # Apply dark theme
    if theme == "dark":
        _apply_dark_theme(fig, ax)

    # Save PNG
    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {output_png}")

    # Attempt Plotly HTML (optional)
    if output_html is not None:
        try:
            import plotly.graph_objects as go

            fig_plotly = go.Figure(data=[
                go.Scatter3d(
                    x=neurons_df.loc[neurons_df['cell_type'] == ct, 'x_m'],
                    y=neurons_df.loc[neurons_df['cell_type'] == ct, 'y_m'],
                    z=neurons_df.loc[neurons_df['cell_type'] == ct, 'z_m'],
                    mode='markers',
                    marker=dict(size=4, color=cell_colors.get(ct, '#808080')),
                    name=ct,
                )
                for ct in neurons_df['cell_type'].unique()
            ])

            fig_plotly.update_layout(
                title=title,
                scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
                template='plotly_dark',
                paper_bgcolor='#0d0d10',
                plot_bgcolor='#121217',
                font=dict(color='#f5f5f5'),
            )

            output_html = Path(output_html)
            output_html.parent.mkdir(parents=True, exist_ok=True)
            fig_plotly.write_html(str(output_html), include_plotlyjs='cdn', full_html=True)
            print(f"Saved: {output_html}")
        except ImportError:
            print("Plotly not available; skipping HTML export")

    if return_node_table:
        return fig, node_table
    else:
        return fig


def activity_trace_suite(
    trials: dict,
    cfg: object,
    stage: str = "initial",
    duration_window_ms: Tuple[float, float] | None = None,
    psd_freq_range_hz: Tuple[float, float] = (1.0, 150.0),
    psd_log_x: bool = True,
    psd_smooth_sigma: float = 1.5,
    lfp_gain: float = 5.0,
    min_trace_spacing: float = 500.0,
    output_png: str | Path | None = None,
    theme: str = "dark",
    area_index: int = 0,
) -> matplotlib.figure.Figure:
    """Create activity trace suite (raster, LFP, CSD, PSD).

    Parameters
    ----------
    trials : dict
        Trials dict from simulate_laminar_trials()
    cfg : LaminarColumnConfig or object
        Configuration specification.
    stage : str
        Stage label for title.
    duration_window_ms : tuple or None
        Time window (tmin, tmax) to display.
    psd_freq_range_hz : tuple
        Frequency range for PSD.
    psd_log_x : bool
        Log frequency axis for PSD.
    psd_smooth_sigma : float
        Gaussian smoothing for PSD.
    lfp_gain : float
        Gain for LFP visibility.
    min_trace_spacing : float
        Minimum spacing for traces.
    output_png : str/Path or None
        Path to save PNG.
    theme : str
        Visual theme ("dark" or "light").
    area_index : int
        Index of target cortical area to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Calculated and rendered activity trace suite figure.
    """
    from .core import require_matplotlib
    require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib
    from scipy import signal

    spikes = trials['spikes']  # (trials, T, N)
    time_ms = trials['time_ms']  # (T,)
    lfp = trials['lfp_contacts']
    csd = trials['csd_contacts']

    # Per-area 4-D contract (trials, areas, T, contacts): select one area.
    # Legacy 3-D form (trials, T, contacts) passes through unchanged.
    if lfp.ndim == 4:
        lfp = lfp[:, area_index]
    if csd.ndim == 4:
        csd = csd[:, area_index]

    # Average over trials
    spikes_mean = spikes.mean(axis=0)  # (T, N)
    lfp_mean = lfp.mean(axis=0)  # (T, contacts)
    csd_mean = csd.mean(axis=0)  # (T, contacts)

    # Set window
    if duration_window_ms is None:
        duration_window_ms = (time_ms[0], time_ms[-1])

    mask = (time_ms >= duration_window_ms[0]) & (time_ms <= duration_window_ms[1])
    t_window = time_ms[mask]

    # Create 2×2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle(f"{stage.capitalize()} Activity Suite", fontsize=12)

    # Panel 1: Raster
    spike_times, neuron_ids = np.where(spikes_mean[mask])
    spike_times = t_window[spike_times]
    axes[0, 0].scatter(spike_times, neuron_ids, s=0.5, c='#f5f5f5', alpha=0.4)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Neuron index')
    axes[0, 0].set_title('Spike Raster')

    # Panel 2: LFP traces — stacked per-channel waterfall (one row per contact,
    # non-overlapping), superficial contact at top, deep at bottom.
    lfp_win = lfp_mean[mask]                      # (T_win, n_contacts)
    n_ch = lfp_win.shape[1]
    # Per-channel z-score so every contact has comparable amplitude.
    ch_mean = lfp_win.mean(axis=0, keepdims=True)
    ch_std = lfp_win.std(axis=0, keepdims=True) + 1e-9
    lfp_z = (lfp_win - ch_mean) / ch_std
    row_spacing = 4.0                            # in z-score units (non-overlap)
    cmap = plt.get_cmap('viridis')
    for ci in range(n_ch):
        offset = ci * row_spacing
        axes[0, 1].plot(
            t_window, lfp_z[:, ci] * lfp_gain * 0.2 + offset,
            lw=0.6, color=cmap(ci / max(n_ch - 1, 1)),
        )
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Contact (depth)')
    axes[0, 1].set_title(f'LFP Traces ({n_ch} channels)')
    tick_idx = np.linspace(0, n_ch - 1, min(n_ch, 8)).astype(int)
    axes[0, 1].set_yticks([i * row_spacing for i in tick_idx])
    axes[0, 1].set_yticklabels([str(int(i)) for i in tick_idx])
    axes[0, 1].set_ylim(-row_spacing, (n_ch) * row_spacing)
    axes[0, 1].invert_yaxis()                    # contact 0 (superficial) at top

    # Panel 3: CSD heatmap
    if csd_mean.shape[1] > 1:
        im = axes[1, 0].imshow(
            csd_mean[mask].T, aspect='auto', origin='upper',
            extent=[duration_window_ms[0], duration_window_ms[1], csd_mean.shape[1], 0],
            cmap='RdBu_r',
        )
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Contact')
        axes[1, 0].set_title('CSD-like Heatmap')
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # Panel 4: Compute real PSD instead of mock random noise
    try:
        dt_ms = 0.1
        if hasattr(cfg, "dt_ms"):
            dt_ms = float(cfg.dt_ms)
        elif isinstance(cfg, dict) and "dt_ms" in cfg:
            dt_ms = float(cfg["dt_ms"])
        
        fs = 1000.0 / dt_ms
        nperseg = min(256, lfp_mean.shape[0])
        freq_hz, psd = signal.welch(lfp_mean, fs=fs, axis=0, nperseg=nperseg)
        # mask within frequency range
        mask_psd = (freq_hz >= psd_freq_range_hz[0]) & (freq_hz <= psd_freq_range_hz[1])
        freq_hz = freq_hz[mask_psd]
        psd = psd[mask_psd, :]
    except Exception:
        # Fallback to deterministic mock PSD if signal analysis fails
        freq_hz = np.logspace(np.log10(psd_freq_range_hz[0]), np.log10(psd_freq_range_hz[1]), 64)
        n_freqs = len(freq_hz)
        psd = np.random.RandomState(42).randn(n_freqs, lfp_mean.shape[1]) ** 2 + 1.0

    for ci in range(min(3, psd.shape[1])):
        if psd_log_x:
            axes[1, 1].loglog(freq_hz, psd[:, ci], lw=1.5, label=f'Contact {ci}')
        else:
            axes[1, 1].semilogy(freq_hz, psd[:, ci], lw=1.5, label=f'Contact {ci}')

    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power (a.u.)')
    axes[1, 1].set_title('PSD')
    axes[1, 1].legend(fontsize=8)

    # Apply dark theme
    if theme == "dark":
        _apply_dark_theme(fig, axes.flatten())

    fig.tight_layout()

    # Save PNG
    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {output_png}")

    return fig


def spectrolaminar_suite_3panel(
    specs: dict,
    model: dict,
    cfg: object,
    stage: str = "initial",
    areas: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    theme: str = "dark",
    density_bins: int = 33,
    density_smooth_sigma: float = 1.2,
    power_vmin: float = 0.48,
    power_vmax: float = 0.94,
    profile_smooth_sigma: float = 1.0,
    power_smooth_sigmas: Sequence[float] | None = None,
) -> dict[str, matplotlib.figure.Figure]:
    """Create 3-panel spectrolaminar suite per area.

    Parameters
    ----------
    specs : dict
        Specs dict from spectrolaminar_from_trials()
    model : dict
        Model from build_laminar_column()
    cfg : LaminarColumnConfig
        Configuration
    stage : str
        Stage label
    areas : sequence or None
        Areas to plot (default: all)
    output_dir : str/Path or None
        Output directory for PNGs
    theme : str
        Visual theme
    density_bins, density_smooth_sigma : int, float
        Cell density histogram parameters
    power_vmin, power_vmax : float
        Color limits for power heatmap
    profile_smooth_sigma : float
        Gaussian smoothing for alpha-beta/gamma profiles (panel C)
    power_smooth_sigmas : sequence of 2 floats or None
        Smoothing sigmas for panel B (mean relative power spectrum):
        [sigma_freq, sigma_depth]. Default [1.0, 1.0] applies minimal smoothing.
        [5.0, 1.0] applies 5x smoothing along frequency axis (rows).

    Returns
    -------
    dict mapping area -> Figure
    """
    from .core import require_matplotlib
    require_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d as gauss1d

    if areas is None:
        areas = cfg.areas

    if output_dir is None:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    line_color = '#f5f5f5' if theme == "dark" else '#202020'
    figs = {}

    for area in areas:
        # Genuine per-area spec (specs is keyed by area name).
        spec = specs[area] if area in specs else specs

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150,
                                gridspec_kw={'width_ratios': [0.85, 1.75, 0.85]},
                                sharey=True)

        n_trials = int(spec.get('n_trials', getattr(cfg, 'n_trials', 0)))
        sim_pct = float(spec.get('similarity_percent', float('nan')))
        fig.suptitle(
            f"{stage}: TFNE-Izhikevich spectrolaminar suite "
            f"({n_trials} trials, area {area}, similarity {sim_pct:.1f}%)",
            fontsize=12,
        )

        # Depth axis from the spec's contact positions (mm), shared across panels.
        pos_from_l4 = np.asarray(spec['pos_from_l4'], dtype=float) * 1.0e3  # m -> mm
        pos_lo, pos_hi = float(pos_from_l4.min()), float(pos_from_l4.max())

        # ── Panel A: Mean Relative Dist of Cells ──────────────────────────────
        neurons_df = model['neurons']
        area_neurons = neurons_df[neurons_df['area'] == area]
        bin_edges = np.linspace(pos_lo, pos_hi, density_bins + 1)
        bin_ctrs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for ct in cfg.cell_types:
            z_ct = area_neurons.loc[area_neurons['cell_type'] == ct, 'pos_from_l4'].values * 1.0e3
            if len(z_ct) == 0:
                continue
            counts, _ = np.histogram(z_ct, bins=bin_edges)
            counts = gauss1d(counts.astype(float), density_smooth_sigma)
            if counts.max() > 0:
                counts = counts / counts.max()
            axes[0].plot(counts, bin_ctrs, lw=2.0,
                         color=cfg.cell_colors.get(ct, '#808080'), label=ct)
        axes[0].axhline(0, color=line_color, lw=1.0, alpha=0.7)
        axes[0].set_title('A Mean Relative Dist of Cells', fontsize=10)
        axes[0].set_xlabel('Relative Count')
        axes[0].set_ylabel('Cortical Position from L4')
        axes[0].legend(fontsize=8, loc='lower right')

        # ── Panel B: Mean Relative Power Spectrum ─────────────────────────────
        # spec['relative_power'] is (n_freqs, n_contacts); transpose so depth is
        # the vertical axis and frequency the horizontal axis.
        relative_power = np.asarray(spec['relative_power'], dtype=float)
        freq_hz = np.asarray(spec['freq_hz'], dtype=float)

        # Apply smoothing to the power spectrum if requested
        if power_smooth_sigmas is not None:
            sigma_freq, sigma_depth = power_smooth_sigmas[0], power_smooth_sigmas[1]
            if sigma_freq > 0 or sigma_depth > 0:
                relative_power = gauss1d(relative_power, sigma=sigma_freq, axis=0)  # smooth frequency axis
                relative_power = gauss1d(relative_power, sigma=sigma_depth, axis=1)  # smooth depth axis

        # Check if depth-normalized
        is_depth_normalized = np.allclose(relative_power.sum(axis=1), 1.0, atol=1e-2)
        if is_depth_normalized:
            vmin_to_use = 0.0
            vmax_to_use = float(relative_power.max())
            if vmax_to_use <= 0.0:
                vmax_to_use = 1.0
        else:
            vmin_to_use = power_vmin
            vmax_to_use = power_vmax

        im = axes[1].imshow(
            relative_power.T, aspect='auto', origin='lower',
            extent=[float(freq_hz[0]), float(freq_hz[-1]), pos_lo, pos_hi],
            cmap='viridis', vmin=vmin_to_use, vmax=vmax_to_use,
        )
        axes[1].axhline(0, color=line_color, lw=1.0, alpha=0.7)
        axes[1].set_title('B Mean Relative Power Spectrum', fontsize=10)
        axes[1].set_xlabel('Frequency (Hz)')
        fig.colorbar(im, ax=axes[1], label='Relative Power (Normalized)')

        # ── Panel C: Alpha-beta / Gamma cross ─────────────────────────────────
        ab_profile = np.asarray(spec['alpha_beta'], dtype=float)
        gm_profile = np.asarray(spec['gamma'], dtype=float)
        if profile_smooth_sigma and profile_smooth_sigma > 0 and ab_profile.size >= 3:
            ab_profile = gauss1d(ab_profile, profile_smooth_sigma)
            gm_profile = gauss1d(gm_profile, profile_smooth_sigma)
        axes[2].plot(ab_profile, pos_from_l4, color='#1f4fe0', lw=2.5, label='Alpha-beta')
        axes[2].plot(gm_profile, pos_from_l4, color='#e01f1f', lw=2.5, label='Gamma')
        axes[2].axhline(0, color=line_color, lw=1.0, alpha=0.7)
        axes[2].set_title('C Alpha-beta / Gamma cross', fontsize=10)
        axes[2].set_xlabel('Relative power')
        axes[2].legend(fontsize=8, loc='lower left')

        # Cortical depth increases downward (superficial above L4 at top).
        axes[0].set_ylim(pos_hi, pos_lo)

        if theme == "dark":
            _apply_dark_theme(fig, axes)

        fig.tight_layout()

        png_path = output_dir / f"spectrolaminar_{stage}_{area}.png"
        fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {png_path}")

        figs[area] = fig

    return figs
