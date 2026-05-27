#!/usr/bin/env python3
"""
Build interactive 3D cortical column visualization (Plotly HTML).

Demonstrates source bookkeeping, field handoff, and probe readout via:
- 3D neuron scatter (colored by layer/cell type)
- Hover metadata (neuron id, layer, cell type, depth, source index, mean rate)
- Readout panels: source summary, LFP-proxy, CSD-proxy, population rate
- Equation annotations
- Interactive camera control

Output:
  docs/assets/interactive/v037_source_column_3d.html

Usage:
  python scripts/build_v037_source_column_3d.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    import jaxfne as jtfne
except ImportError:
    raise ImportError("jaxfne not found. Install: pip install jaxfne")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("plotly not found. Install: pip install plotly")


# ============================================================================
# Configuration
# ============================================================================

SEED = 42
DTYPE = "float32"
DURATION_MS = 1000.0  # Shorter for interactive viz
DT_MS = 0.1

# Layer metadata
LAYERS = ["L2/3", "L4", "L5", "L6"]
LAYER_DEPTHS_UM = {
    "L2/3": 250.0,
    "L4": 500.0,
    "L5": 750.0,
    "L6": 950.0,
}

CELL_TYPES = {"E": 0.70, "PV": 0.15, "SST": 0.10, "VIP": 0.05}

# Color scheme
LAYER_COLORS = {
    "L2/3": "rgb(220, 100, 100)",  # Reddish
    "L4": "rgb(100, 150, 220)",    # Blueish
    "L5": "rgb(100, 220, 100)",    # Greenish
    "L6": "rgb(220, 200, 100)",    # Yellowish
}

CELL_TYPE_SYMBOLS = {
    "E": "circle",
    "PV": "diamond",
    "SST": "square",
    "VIP": "cross",
}


# ============================================================================
# Cortical Column Configuration & Simulation
# ============================================================================

def build_column_config(n_per_layer: int = 12) -> "jtfne.Configuration":
    """
    Build laminar column configuration.

    Args:
        n_per_layer: neurons per layer

    Returns:
        jtfne.Configuration object
    """
    cfg = (jtfne.Configuration()
        .runtime(seed=SEED, dtype=DTYPE, duration_ms=DURATION_MS, dt_ms=DT_MS)
        .column(name="interactive_column", layers=LAYERS, n=n_per_layer * len(LAYERS))
        .cell_types(CELL_TYPES)
        .connectivity(kind="laminar_signed_metadata", recurrent=True)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP-proxy", "CSD-proxy"], n_contacts=16))

    return cfg


def simulate_column(cfg: "jtfne.Configuration") -> Tuple[Any, Any]:
    """
    Construct and simulate the column.

    Args:
        cfg: Configuration object

    Returns:
        (model, signals) tuple
    """
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(
        model,
        duration_ms=DURATION_MS,
        dt_ms=DT_MS,
        seed=SEED
    )
    return model, signals


def extract_neuron_metadata(
    signals: Any,
    n_per_layer: int = 12
) -> Dict[str, List[Any]]:
    """
    Extract per-neuron metadata for hover display.

    Args:
        signals: Signals object from simulate()
        n_per_layer: neurons per layer

    Returns:
        Dict with keys: neuron_ids, layers, cell_types, depths, rates, sources
    """
    n_neurons = len(LAYERS) * n_per_layer
    spikes = np.asarray(signals.spikes)
    dt_ms = float(DURATION_MS / spikes.shape[0])

    neuron_ids = []
    layers = []
    cell_types = []
    depths = []
    rates_hz = []
    source_indices = []

    # Assign neurons to layers and cell types
    n_per_type = {
        "E": int(n_per_layer * CELL_TYPES["E"]),
        "PV": int(n_per_layer * CELL_TYPES["PV"]),
        "SST": int(n_per_layer * CELL_TYPES["SST"]),
        "VIP": int(n_per_layer * CELL_TYPES["VIP"]),
    }

    # Adjust for rounding
    n_per_type["E"] += n_per_layer - sum(n_per_type.values())

    neuron_idx = 0
    for layer_idx, layer in enumerate(LAYERS):
        layer_depth = LAYER_DEPTHS_UM.get(layer, 0.0)

        for cell_type in ["E", "PV", "SST", "VIP"]:
            for _ in range(n_per_type[cell_type]):
                if neuron_idx >= n_neurons:
                    break

                neuron_ids.append(neuron_idx)
                layers.append(layer)
                cell_types.append(cell_type)
                depths.append(layer_depth)

                # Compute firing rate
                spike_count = spikes[:, neuron_idx].sum()
                rate = float(spike_count * (1000.0 / DURATION_MS))
                rates_hz.append(rate)

                source_indices.append(neuron_idx)

                neuron_idx += 1

    return {
        "neuron_ids": neuron_ids,
        "layers": layers,
        "cell_types": cell_types,
        "depths": depths,
        "rates_hz": rates_hz,
        "source_indices": source_indices,
    }


# ============================================================================
# 3D Visualization
# ============================================================================

def create_3d_column_plot(
    signals: Any,
    neuron_metadata: Dict[str, List[Any]],
    n_per_layer: int = 12
) -> go.Figure:
    """
    Create interactive 3D column visualization.

    Args:
        signals: Signals object
        neuron_metadata: Output from extract_neuron_metadata()
        n_per_layer: neurons per layer

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # 3D neuron scatter
    # Generate spatial coordinates (simple grid)
    x_coords = []
    y_coords = []
    z_coords = []

    for layer_idx, layer in enumerate(LAYERS):
        for neuron_local_idx in range(n_per_layer):
            # Arrange neurons in a simple square grid within each layer
            grid_size = int(np.sqrt(n_per_layer)) + 1
            x = (neuron_local_idx % grid_size) * 10.0  # 10 µm spacing
            y = (neuron_local_idx // grid_size) * 10.0
            z = LAYER_DEPTHS_UM[layer]

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

    # Create scatter trace
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=5,
            color=neuron_metadata["rates_hz"],  # Color by firing rate
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Rate (Hz)",
                thickness=15,
                len=0.7,
            ),
            line=dict(width=0.5, color='rgba(200, 200, 200, 0.5)'),
        ),
        text=[
            f"<b>Neuron {nid}</b><br>"
            f"Layer: {layer}<br>"
            f"Cell type: {ct}<br>"
            f"Depth: {d:.0f} µm<br>"
            f"Rate: {r:.2f} Hz<br>"
            f"Source index: {si}"
            for nid, layer, ct, d, r, si in zip(
                neuron_metadata["neuron_ids"],
                neuron_metadata["layers"],
                neuron_metadata["cell_types"],
                neuron_metadata["depths"],
                neuron_metadata["rates_hz"],
                neuron_metadata["source_indices"],
            )
        ],
        hovertemplate='%{text}<extra></extra>',
        name='Neurons',
    ))

    # Update layout with equation annotations
    fig.update_layout(
        title=dict(
            text=(
                "<b>Interactive 3D Source/Field/Probe Cortical Column</b><br>"
                "<sub>v0.3.7: Source Bookkeeping, Field Handoff, Probe Readout</sub>"
            ),
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(
                title='X (µm)',
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
            ),
            yaxis=dict(
                title='Y (µm)',
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
            ),
            zaxis=dict(
                title='Depth (µm)',
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
            ),
        ),
        width=1200,
        height=700,
        hovermode='closest',
        showlegend=False,
        font=dict(size=11),
        annotations=[
            # Equation annotations
            dict(
                text=(
                    "<b>Source Bookkeeping:</b> S(t) ∈ ℝ^{T×N} | "
                    "<b>Field Handoff:</b> Y(t) = P·S(t) | "
                    "<b>Probe Readout:</b> R_k(t) = Q_k·Y(t)"
                ),
                xref='paper', yref='paper',
                x=0.5, y=-0.05,
                showarrow=False,
                align='center',
                font=dict(size=11, color='gray'),
            ),
        ],
    )

    return fig


def create_readout_panels(signals: Any) -> go.Figure:
    """
    Create side-panel readout plots (source, LFP, CSD, population rate).

    Args:
        signals: Signals object

    Returns:
        Plotly Figure with subplots
    """
    spikes = np.asarray(signals.spikes)
    V_m = np.asarray(signals.V_m)
    sources = np.asarray(signals.sources) if hasattr(signals, 'sources') else np.zeros_like(spikes)
    t = np.asarray(signals.time_ms)

    # Create 2x2 subplots
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

    # 1. Source summary (mean source over time)
    if sources.size > 0:
        source_mean = sources.mean(axis=1) if len(sources.shape) > 1 else sources
        fig.add_trace(
            go.Scatter(x=t, y=source_mean, mode='lines', name='Source mean',
                      line=dict(color='steelblue', width=1)),
            row=1, col=1
        )

    # 2. Population firing rate (binned)
    bin_size_ms = 50.0
    bin_edges = np.arange(0, t.max() + bin_size_ms, bin_size_ms)
    rates = []
    bin_times = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (t >= lo) & (t < hi)
        if mask.sum() > 0:
            rate = spikes[mask].mean() * (1000.0 / DT_MS)
            rates.append(rate)
            bin_times.append(0.5 * (lo + hi))

    fig.add_trace(
        go.Scatter(x=bin_times, y=rates, mode='lines', name='Population rate',
                  line=dict(color='darkgreen', width=2)),
        row=1, col=2
    )

    # 3. LFP-proxy (if available)
    # Use voltage as proxy (simplified)
    if V_m.size > 0:
        lfp_proxy = V_m.mean(axis=1) if len(V_m.shape) > 1 else V_m
        fig.add_trace(
            go.Scatter(x=t, y=lfp_proxy, mode='lines', name='LFP-proxy',
                      line=dict(color='purple', width=1)),
            row=2, col=1
        )

    # 4. Mean voltage by layer (sample neurons from each layer)
    n_per_layer = spikes.shape[1] // 4
    for layer_idx, layer in enumerate(LAYERS):
        start_idx = layer_idx * n_per_layer
        end_idx = start_idx + n_per_layer
        layer_v_mean = V_m[:, start_idx:end_idx].mean(axis=1)
        fig.add_trace(
            go.Scatter(x=t, y=layer_v_mean, mode='lines', name=layer,
                      line=dict(color=LAYER_COLORS[layer])),
            row=2, col=2
        )

    # Update axes labels
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Mean source", row=1, col=1)

    fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Rate (Hz)", row=1, col=2)

    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="LFP proxy", row=2, col=1)

    fig.update_xaxes(title_text="Time (ms)", row=2, col=2)
    fig.update_yaxes(title_text="Mean voltage", row=2, col=2)

    fig.update_layout(
        title_text="Readout Panels: Source / Field / Probe",
        height=700,
        showlegend=True,
    )

    return fig


# ============================================================================
# Manifest & Output
# ============================================================================

def create_manifest(
    cfg: "jtfne.Configuration",
    signals: Any,
    neuron_metadata: Dict[str, List[Any]],
) -> Dict[str, Any]:
    """
    Create scope/metadata manifest.

    Args:
        cfg: Configuration object
        signals: Signals object
        neuron_metadata: Neuron metadata dict

    Returns:
        Manifest dict
    """
    spikes = np.asarray(signals.spikes)
    V_m = np.asarray(signals.V_m)

    mean_rate_hz = float(
        spikes.mean() * (1000.0 / DURATION_MS)
    )

    manifest = {
        "scope_status": "computational_scaffold",
        "readout_status": "simulated_proxy",
        "field_mode": "proxy_convolution_no_pde",
        "physical_amplitude_claim_allowed": False,
        "visualizer": "plotly_standalone_html",
        "duration_ms": DURATION_MS,
        "dt_ms": DT_MS,
        "dtype": DTYPE,
        "seed": SEED,
        "n_neurons": len(neuron_metadata["neuron_ids"]),
        "layers": LAYERS,
        "cell_types": CELL_TYPES,
        "mean_population_rate_hz": round(mean_rate_hz, 4),
        "voltage_range_mv_like": [
            round(float(V_m.min()), 2),
            round(float(V_m.max()), 2),
        ],
        "all_outputs_finite": bool(
            np.isfinite(spikes).all() and np.isfinite(V_m).all()
        ),
        "equations": {
            "source_bookkeeping": "S(t) ∈ ℝ^{T×N}",
            "field_handoff": "Y(t) = P·S(t)",
            "probe_readout": "R_k(t) = Q_k·Y(t)",
        },
        "html_output": "docs/assets/interactive/v037_source_column_3d.html",
    }

    return manifest


# ============================================================================
# Main
# ============================================================================

def main():
    """Build and save interactive 3D column visualization."""

    print("Building v0.3.7 Interactive 3D Source/Field/Probe Column...")
    print()

    # Create output directory
    output_dir = Path("docs/assets/interactive")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure and simulate
    print("[1/5] Configuring laminar column...")
    cfg = build_column_config(n_per_layer=12)

    print("[2/5] Simulating column dynamics...")
    model, signals = simulate_column(cfg)

    print("[3/5] Extracting neuron metadata...")
    neuron_metadata = extract_neuron_metadata(signals, n_per_layer=12)

    print("[4/5] Creating 3D visualization...")
    fig_3d = create_3d_column_plot(signals, neuron_metadata, n_per_layer=12)

    # Save 3D column as primary output
    html_path = output_dir / "v037_source_column_3d.html"
    fig_3d.write_html(str(html_path))
    print(f"  ✓ Saved: {html_path}")

    # Also create and embed readout panels
    print("[5/5] Creating readout panels...")
    fig_readouts = create_readout_panels(signals)
    readout_path = output_dir / "v037_readout_panels.html"
    fig_readouts.write_html(str(readout_path))
    print(f"  ✓ Saved: {readout_path}")

    # Create and save manifest
    manifest = create_manifest(cfg, signals, neuron_metadata)
    manifest_path = output_dir / "v037_source_column_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  ✓ Manifest: {manifest_path}")
    print()

    # Print summary
    print("Summary:")
    print(f"  Neurons: {manifest['n_neurons']}")
    print(f"  Mean rate: {manifest['mean_population_rate_hz']} Hz")
    print(f"  All finite: {manifest['all_outputs_finite']}")
    print(f"  Duration: {manifest['duration_ms']} ms")
    print()
    print(f"✓ Build complete. Open {html_path} in a browser.")

    return html_path


if __name__ == "__main__":
    main()
