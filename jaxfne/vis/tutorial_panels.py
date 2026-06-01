"""Visual suites for laminar column tutorials (Etude No. 1)."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Sequence


def _apply_dark_theme(fig, ax=None):
    """Apply dark theme to matplotlib figure."""
    fig.patch.set_facecolor('#0d0d10')
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
) -> Tuple:
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
) -> object:
    """Create activity trace suite (raster, LFP, CSD, PSD).

    Parameters
    ----------
    trials : dict
        Trials dict from simulate_laminar_trials()
    cfg : LaminarColumnConfig
        Configuration
    stage : str
        Stage label for title
    duration_window_ms : tuple or None
        Time window (tmin, tmax) to display
    psd_freq_range_hz : tuple
        Frequency range for PSD
    psd_log_x : bool
        Log frequency axis for PSD
    psd_smooth_sigma : float
        Gaussian smoothing for PSD
    lfp_gain : float
        Gain for LFP visibility
    min_trace_spacing : float
        Minimum spacing for traces
    output_png : str/Path or None
        Path to save PNG
    theme : str
        Visual theme

    Returns
    -------
    matplotlib Figure
    """
    from .core import require_matplotlib
    require_matplotlib()
    import matplotlib.pyplot as plt

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

    # Panel 2: LFP traces
    for ci in range(min(4, lfp_mean.shape[1])):
        axes[0, 1].plot(t_window, lfp_mean[mask, ci] * lfp_gain, lw=1, label=f'Contact {ci}')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('LFP-proxy (V)')
    axes[0, 1].set_title('LFP Traces')
    axes[0, 1].legend(fontsize=8)

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

    # Panel 4: PSD (mock)
    freq_hz = np.logspace(np.log10(psd_freq_range_hz[0]), np.log10(psd_freq_range_hz[1]), 64)
    n_freqs = len(freq_hz)
    psd = np.random.RandomState(42).randn(n_freqs, lfp_mean.shape[1]) ** 2 + 1.0

    for ci in range(min(3, psd.shape[1])):
        axes[1, 1].loglog(freq_hz, psd[:, ci], lw=1.5, label=f'Contact {ci}')

    axes[1, 1].set_xlabel('Frequency (Hz)' if psd_log_x else 'Frequency (Hz)')
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
) -> dict:
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

    figs = {}

    for area in areas:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150,
                                gridspec_kw={'width_ratios': [0.85, 1.75, 0.85]},
                                sharey=True)

        fig.suptitle(f"{area} Spectrolaminar ({stage})", fontsize=11)

        # Get area neurons
        neurons_df = model['neurons']
        area_mask = neurons_df['area'] == area
        area_neurons = neurons_df[area_mask]

        # Panel A: Cell density
        pos_from_l4 = specs.get('pos_from_l4', np.linspace(-0.5, 0.5, 33))
        bin_edges = np.linspace(pos_from_l4.min() - 0.05, pos_from_l4.max() + 0.05, density_bins + 1)
        bin_ctrs = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for ct in cfg.cell_types:
            ct_mask = area_neurons['cell_type'] == ct
            z_ct = area_neurons.loc[ct_mask, 'pos_from_l4'].values
            if len(z_ct) == 0:
                continue
            counts, _ = np.histogram(z_ct, bins=bin_edges)
            counts = gauss1d(counts.astype(float), density_smooth_sigma)
            if counts.max() > 0:
                counts = counts / counts.max()
            axes[0].plot(counts, bin_ctrs, lw=2.0, color=cfg.cell_colors.get(ct, '#808080'),
                        label=ct)

        axes[0].axhline(0, color='#f5f5f5', lw=1.0, ls='--', alpha=0.5)
        axes[0].set_title('A Cell Density', fontsize=10)
        axes[0].set_xlabel('Relative Count')
        axes[0].set_ylabel('Cortical Position from L4 (mm)')
        axes[0].legend(fontsize=8)

        # Panel B: Power heatmap
        relative_power = specs.get('relative_power', np.random.rand(64, 32))
        freq_hz = specs.get('freq_hz', np.logspace(0, 2.2, 64))

        im = axes[1].imshow(relative_power, aspect='auto', origin='lower',
                           extent=[freq_hz[0], freq_hz[-1], pos_from_l4.min(), pos_from_l4.max()],
                           cmap='viridis', vmin=power_vmin, vmax=power_vmax)
        axes[1].axhline(0, color='#f5f5f5', lw=1.0, ls='--', alpha=0.5)
        axes[1].set_title('B Power Spectrum', fontsize=10)
        axes[1].set_xlabel('Frequency (Hz)')
        fig.colorbar(im, ax=axes[1], label='Rel Power')

        # Panel C: Band profiles
        ab_profile = specs.get('alpha_beta', np.ones(len(pos_from_l4)))
        gm_profile = specs.get('gamma', np.ones(len(pos_from_l4)))

        axes[2].plot(ab_profile, pos_from_l4, color='#4169e1', lw=2.5, label='Alpha-beta')
        axes[2].plot(gm_profile, pos_from_l4, color='#dc143c', lw=2.5, label='Gamma')
        axes[2].axhline(0, color='#f5f5f5', lw=1.0, ls='--', alpha=0.5)
        axes[2].set_title('C Band Profiles', fontsize=10)
        axes[2].set_xlabel('Relative Power')
        axes[2].legend(fontsize=8)

        # Apply dark theme
        if theme == "dark":
            _apply_dark_theme(fig, axes)

        fig.tight_layout()

        # Save PNG
        png_path = output_dir / f"spectrolaminar_{stage}_{area}.png"
        fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {png_path}")

        figs[area] = fig

    return figs
