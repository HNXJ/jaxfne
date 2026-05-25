#!/usr/bin/env python3
"""
v0.3 Tutorial Plotly Utilities

Provides helper functions for generating interactive Plotly figures without
requiring Plotly to be installed. If Plotly is available, generates HTML figures;
otherwise returns JSON-safe dictionaries suitable for manual visualization.

This module ensures tutorials can run without Plotly dependency while still
supporting optional interactive figure generation.

truth_mode: truth_safe_unverified
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import os


# Try to import Plotly; continue if unavailable
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None


def plotly_available() -> bool:
    """Check if Plotly is installed and available.

    Returns:
        bool: True if Plotly can be imported, False otherwise.
    """
    return PLOTLY_AVAILABLE


def create_spike_raster_figure(
    spike_times: List[List[float]],
    neuron_ids: Optional[List[int]] = None,
    duration_ms: float = 1000.0,
    title: str = "Spike Raster",
    as_html: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an interactive spike raster figure.

    Args:
        spike_times: List of spike times (ms) for each neuron.
                    spike_times[i] = array of spike times for neuron i.
        neuron_ids: Optional list of neuron IDs. Defaults to [0, 1, 2, ...].
        duration_ms: Total simulation duration (ms).
        title: Figure title.
        as_html: If True and Plotly available, save as HTML file. Otherwise return dict.
        output_path: Path to save HTML figure (used if as_html=True).

    Returns:
        dict: Figure data (if Plotly unavailable or as_html=False) with keys:
              - 'type': 'spike_raster'
              - 'neurons': number of neurons
              - 'duration_ms': total duration
              - 'spike_counts': array of spike counts per neuron
              - 'mean_firing_rate': mean firing rate (Hz)
              If as_html=True and Plotly available:
              - 'html_path': path to saved HTML file
              - 'hash': SHA256 hash of HTML file
    """
    if neuron_ids is None:
        neuron_ids = list(range(len(spike_times)))

    # Compute basic statistics
    spike_counts = [len(st) for st in spike_times]
    duration_sec = duration_ms / 1000.0
    firing_rates = [count / duration_sec for count in spike_counts]
    mean_firing_rate = sum(firing_rates) / len(firing_rates) if firing_rates else 0.0

    result = {
        'type': 'spike_raster',
        'neurons': len(spike_times),
        'duration_ms': duration_ms,
        'spike_counts': spike_counts,
        'mean_firing_rate': mean_firing_rate,
    }

    # Generate HTML if Plotly available and requested
    if as_html and PLOTLY_AVAILABLE and output_path:
        fig = go.Figure()

        for neuron_id, spike_times_i in enumerate(spike_times):
            for spike_time in spike_times_i:
                fig.add_vline(
                    x=spike_time,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Neuron {neuron_id}",
                    annotation_position="top",
                )

        fig.update_layout(
            title=title,
            xaxis_title="Time (ms)",
            yaxis_title="Neuron",
            hovermode="x unified",
            height=400 + len(spike_times) * 5,
        )

        # Save HTML
        fig.write_html(output_path)

        # Compute hash
        with open(output_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        result['html_path'] = output_path
        result['hash'] = file_hash

    return result


def create_voltage_trace_figure(
    time_ms: List[float],
    voltage_traces: Dict[int, List[float]],
    title: str = "Membrane Voltage",
    as_html: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a voltage trace figure.

    Args:
        time_ms: Time array (ms).
        voltage_traces: Dict mapping neuron_id -> voltage array (mV).
        title: Figure title.
        as_html: If True and Plotly available, save as HTML.
        output_path: Path to save HTML figure.

    Returns:
        dict: Figure data with keys:
              - 'type': 'voltage_trace'
              - 'traces': number of traces
              - 'duration_ms': total duration
              If as_html=True:
              - 'html_path': path to HTML
              - 'hash': SHA256 hash
    """
    result = {
        'type': 'voltage_trace',
        'traces': len(voltage_traces),
        'duration_ms': time_ms[-1] if time_ms else 0.0,
    }

    if as_html and PLOTLY_AVAILABLE and output_path:
        fig = go.Figure()

        for neuron_id, voltages in voltage_traces.items():
            fig.add_trace(go.Scatter(
                x=time_ms,
                y=voltages,
                mode='lines',
                name=f'Neuron {neuron_id}',
                hovertemplate='<b>Neuron %{fullData.name}</b><br>Time: %{x:.1f} ms<br>V_m: %{y:.1f} mV<extra></extra>',
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time (ms)",
            yaxis_title="Membrane Potential (mV)",
            hovermode="x unified",
            height=500,
        )

        fig.write_html(output_path)

        with open(output_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        result['html_path'] = output_path
        result['hash'] = file_hash

    return result


def create_firing_rate_figure(
    time_ms: List[float],
    firing_rate_hz: List[float],
    title: str = "Population Firing Rate",
    as_html: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a firing rate dynamics figure.

    Args:
        time_ms: Time array (ms).
        firing_rate_hz: Firing rate array (Hz).
        title: Figure title.
        as_html: If True and Plotly available, save as HTML.
        output_path: Path to save HTML figure.

    Returns:
        dict: Figure data with keys:
              - 'type': 'firing_rate'
              - 'mean_rate_hz': mean firing rate
              - 'max_rate_hz': peak firing rate
              If as_html=True:
              - 'html_path': path to HTML
              - 'hash': SHA256 hash
    """
    mean_rate = sum(firing_rate_hz) / len(firing_rate_hz) if firing_rate_hz else 0.0
    max_rate = max(firing_rate_hz) if firing_rate_hz else 0.0

    result = {
        'type': 'firing_rate',
        'mean_rate_hz': mean_rate,
        'max_rate_hz': max_rate,
    }

    if as_html and PLOTLY_AVAILABLE and output_path:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_ms,
            y=firing_rate_hz,
            mode='lines',
            name='Population firing rate',
            fill='tozeroy',
            hovertemplate='Time: %{x:.1f} ms<br>Rate: %{y:.1f} Hz<extra></extra>',
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time (ms)",
            yaxis_title="Firing Rate (Hz)",
            hovermode="x unified",
            height=400,
        )

        fig.write_html(output_path)

        with open(output_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        result['html_path'] = output_path
        result['hash'] = file_hash

    return result


def save_artifact_metadata(
    scenario_id: str,
    figures: List[Dict[str, Any]],
    output_dir: str = 'outputs',
) -> str:
    """
    Save artifact metadata (all figures and their hashes) to JSON.

    Args:
        scenario_id: Scenario identifier (e.g., 'v030_01').
        figures: List of figure metadata dicts (output from create_*_figure functions).
        output_dir: Directory to save metadata JSON.

    Returns:
        str: Path to saved metadata JSON file.
    """
    metadata = {
        'scenario': scenario_id,
        'figures': figures,
    }

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, f'{scenario_id}_artifacts.json')

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def validate_artifact_hashes(
    artifact_metadata_path: str,
    verbose: bool = False,
) -> bool:
    """
    Validate SHA256 hashes of artifact files.

    Args:
        artifact_metadata_path: Path to artifacts.json file.
        verbose: If True, print validation results.

    Returns:
        bool: True if all hashes match, False otherwise.
    """
    with open(artifact_metadata_path) as f:
        metadata = json.load(f)

    all_valid = True
    output_dir = os.path.dirname(artifact_metadata_path)

    for figure in metadata.get('figures', []):
        if 'hash' not in figure or 'html_path' not in figure:
            continue

        filepath = os.path.join(output_dir, os.path.basename(figure['html_path']))
        recorded_hash = figure['hash']

        try:
            with open(filepath, 'rb') as f:
                computed_hash = hashlib.sha256(f.read()).hexdigest()

            if computed_hash == recorded_hash:
                if verbose:
                    print(f"✓ {os.path.basename(filepath)}: {recorded_hash}")
            else:
                if verbose:
                    print(f"✗ {os.path.basename(filepath)}: hash mismatch")
                    print(f"  Recorded: {recorded_hash}")
                    print(f"  Computed: {computed_hash}")
                all_valid = False
        except FileNotFoundError:
            if verbose:
                print(f"✗ {filepath}: file not found")
            all_valid = False

    return all_valid


if __name__ == '__main__':
    # Example usage
    print(f"Plotly available: {plotly_available()}")

    # Create a simple spike raster
    spike_times = [
        [10.0, 25.5, 40.2, 55.1],
        [15.3, 30.8, 45.0],
        [5.0, 20.0, 35.0, 50.0, 65.0],
    ]

    raster_data = create_spike_raster_figure(
        spike_times,
        duration_ms=100.0,
        as_html=PLOTLY_AVAILABLE,
        output_path='test_raster.html' if PLOTLY_AVAILABLE else None,
    )

    print("Spike raster figure data:", raster_data)

    if PLOTLY_AVAILABLE:
        print("✓ HTML figure generated (test_raster.html)")
    else:
        print("⊘ Plotly not available; figure data returned as dict")
