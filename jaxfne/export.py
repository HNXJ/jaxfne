"""Root-level export and visualization APIs for notebooks.

All functions here are designed for direct notebook use with strict call grammar:
  jtfne.save_figure(...)
  jtfne.export_report(...)
  jtfne.plot_raster(...)

No matplotlib calls in notebooks; matplotlib used internally with lazy imports only.

0.4.7 legacy-thinning note: ``save_figure``/``save_figures`` remain until tutorial
call sites migrate to ``jaxfne.vis.export_figure`` (see F-016).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Mapping, Optional
import numpy as np


def save_figure(fig, path: str | Path, dpi: int = 150, bbox_inches: str = "tight") -> str:
    """DEPRECATED: use ``jaxfne.vis.export_figure`` (handles matplotlib + plotly).

    Thin matplotlib-only wrapper kept for back-compat; closes ``fig`` after
    saving (the canonical exporter does not).
    """
    warnings.warn(
        "jaxfne.export.save_figure is deprecated; use jaxfne.vis.export_figure "
        "instead (handles matplotlib + plotly, does not close the figure).",
        DeprecationWarning,
        stacklevel=2,
    )
    from .vis.exporters import export_figure, close_matplotlib_figure
    path = Path(path)
    fmt = path.suffix.lstrip(".") or "png"
    written = export_figure(fig, path.with_suffix(""), formats=(fmt,), dpi=dpi)
    close_matplotlib_figure(fig)
    return written[fmt]


def save_figures(figures: Mapping[str, object], output_dir: str | Path,
                dpi: int = 150, prefix: str = "", suffix: str = "") -> Mapping[str, str]:
    """DEPRECATED: use ``jaxfne.vis.export_figures`` (handles matplotlib + plotly)."""
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

    NAME COLLISION NOTE: ``jaxfne.tutorial_utils.export_tutorial_artifacts``
    is a DIFFERENT function with a different signature (takes a
    ``LaminarColumnConfig`` as its first positional arg, not ``output_dir``)
    -- it is the actively-used one in tutorials/notebooks. This is the
    root-exported, generic ``jtfne.export_tutorial_artifacts`` (JSON-only,
    config-agnostic thin wrapper around :func:`export_report`).

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
