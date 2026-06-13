"""Root-level export and visualization APIs for notebooks.

All functions here are designed for direct notebook use with strict call grammar:
  jtfne.save_figure(...)
  jtfne.export_report(...)
  jtfne.plot_raster(...)

No matplotlib calls in notebooks; matplotlib used internally with lazy imports only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple, Optional
import numpy as np


def save_figure(fig, path: str | Path, dpi: int = 150, bbox_inches: str = "tight") -> str:
    """Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    path : str or Path
        Output file path (e.g., 'outputs/my_figure.png')
    dpi : int
        Resolution in dots per inch
    bbox_inches : str
        Bounding box setting ('tight', 'standard')

    Returns
    -------
    str
        Path where figure was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)


def save_figures(figures: Mapping[str, object], output_dir: str | Path,
                dpi: int = 150, prefix: str = "", suffix: str = "") -> Mapping[str, str]:
    """Save multiple figures to an output directory.

    Parameters
    ----------
    figures : dict[str -> matplotlib.figure.Figure]
        Dictionary mapping names to figure objects
    output_dir : str or Path
        Output directory path
    dpi : int
        Resolution in dots per inch
    prefix, suffix : str
        Filename prefix/suffix

    Returns
    -------
    dict[str -> str]
        Mapping of names to saved paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, fig in figures.items():
        path = output_dir / f"{prefix}{name}{suffix}.png"
        paths[name] = save_figure(fig, path, dpi=dpi)

    return paths


def export_report(
    output_dir: str | Path,
    manifest: Optional[Mapping] = None,
    metrics: Optional[Mapping] = None,
    validation: Optional[Mapping] = None,
    figures: Optional[Mapping[str, object]] = None,
    dpi: int = 150,
) -> Mapping[str, str]:
    """Export a complete report with JSON artifacts and figures.

    Parameters
    ----------
    output_dir : str or Path
        Output directory
    manifest : dict or None
        Configuration/metadata dict
    metrics : dict or None
        Metrics and results dict
    validation : dict or None
        Validation report dict
    figures : dict[str -> matplotlib.figure.Figure] or None
        Figures to save
    dpi : int
        Figure resolution

    Returns
    -------
    dict
        Mapping of artifact names to saved paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save JSON artifacts
    for name, data in [
        ("manifest", manifest),
        ("metrics", metrics),
        ("validation_report", validation),
    ]:
        if data is None:
            continue
        path = output_dir / f"{name}.json"
        _safe_json_write(data, path)
        paths[name] = str(path)

    # Save figures
    if figures:
        fig_dir = output_dir / "figures"
        fig_paths = save_figures(figures, fig_dir, dpi=dpi)
        for name, path in fig_paths.items():
            paths[f"figure_{name}"] = path

    return paths


def export_tutorial_artifacts(
    output_dir: str | Path,
    manifest: Optional[Mapping] = None,
    metrics: Optional[Mapping] = None,
    validation: Optional[Mapping] = None,
) -> Mapping[str, str]:
    """Export tutorial artifacts (JSON only, no figures).

    Parameters
    ----------
    output_dir : str or Path
        Output directory
    manifest : dict or None
        Configuration/metadata
    metrics : dict or None
        Results
    validation : dict or None
        Validation report

    Returns
    -------
    dict
        Paths to saved JSON files
    """
    return export_report(output_dir, manifest, metrics, validation, figures=None)


def plot_raster(spike_times: np.ndarray, spike_ids: np.ndarray,
                time_ms: np.ndarray, figsize: Tuple[float, float] = (10, 4),
                title: str = "Spike Raster") -> object:
    """Plot a spike raster.

    Parameters
    ----------
    spike_times : ndarray
        Spike times in milliseconds
    spike_ids : ndarray
        Neuron IDs for each spike
    time_ms : ndarray
        Time axis for axis limits
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Figure object (use jtfne.save_figure(...) to save)
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(spike_times, spike_ids, s=2, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron ID")
    ax.set_title(title)
    ax.set_xlim(time_ms.min(), time_ms.max())
    return fig


def plot_spectrolaminar_suite(signals: object = None, **kwargs) -> object:
    """Plot spectrolaminar suite from signals object.

    Root-level wrapper for jtfne.vis.spectrolaminar_suite. Accepts Signals object
    or dict-like signals, returns matplotlib figure for saving with jtfne.save_figure.

    Parameters
    ----------
    signals : Signals or dict or None
        Signals object (from jtfne.simulate) or dict with signal arrays
    **kwargs
        Additional arguments passed to jtfne.vis.spectrolaminar_suite

    Returns
    -------
    matplotlib.figure.Figure
        Figure object (use jtfne.save_figure(...) to save)
    """
    try:
        from .vis import spectrolaminar_suite as _vis_spectrolaminar_suite
        return _vis_spectrolaminar_suite(signals, **kwargs)
    except ImportError:
        raise RuntimeError("vis module not available; cannot create spectrolaminar plot")


def _safe_json_write(data: object, path: Path) -> None:
    """Write data to JSON, converting to JSON-safe types."""
    def _to_jsonable(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_to_jsonable(x) for x in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    data_safe = _to_jsonable(data)
    text = json.dumps(data_safe, indent=2, allow_nan=False)
    path.write_text(text)
