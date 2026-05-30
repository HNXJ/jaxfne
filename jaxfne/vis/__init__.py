"""Visualization package for jaxfne.

Static, NumPy-isolated graphics, raster, trace, and PSD plots.
Fully de-allocated from the active JAX tracer tree.
"""
from __future__ import annotations

from .core import FigureResult, require_matplotlib, prepare_static_plot_matrix
from .rasters import plot_spike_rasters, raster, raster_with_meta
from .traces import (
    plot_continuous_traces,
    vm,
    vm_with_meta,
    rate,
    rate_with_meta,
    source,
    source_with_meta,
    lfp,
    lfp_with_meta,
    csd,
    csd_with_meta,
    lfp_traces,
    csd_traces,
    eeg,
    meg,
    emm,
    summary,
    summary_with_meta,
)
from .spectra import (
    plot_spectrogram_profiles,
    psd,
    psd_with_meta,
    spectrogram,
    spectrogram_with_meta,
)
from .fields import (
    plot_laminar_field_interpolation,
    spectrolaminar,
    spectrolaminar_suite,
    bandpower,
    laminar_profile,
    layer_celltype_counts,
    connectivity,
    connectivity_matrix,
    multi_area_layout,
    objective_report,
)
from .network3d import (
    circuit3d,
    geometry3d,
    column_geometry,
)

__all__ = [
    "FigureResult",
    "require_matplotlib",
    "prepare_static_plot_matrix",
    "plot_spike_rasters",
    "raster",
    "raster_with_meta",
    "plot_continuous_traces",
    "vm",
    "vm_with_meta",
    "rate",
    "rate_with_meta",
    "source",
    "source_with_meta",
    "lfp",
    "lfp_with_meta",
    "csd",
    "csd_with_meta",
    "lfp_traces",
    "csd_traces",
    "eeg",
    "meg",
    "emm",
    "summary",
    "summary_with_meta",
    "plot_spectrogram_profiles",
    "psd",
    "psd_with_meta",
    "spectrogram",
    "spectrogram_with_meta",
    "plot_laminar_field_interpolation",
    "spectrolaminar",
    "spectrolaminar_suite",
    "bandpower",
    "laminar_profile",
    "layer_celltype_counts",
    "connectivity",
    "connectivity_matrix",
    "multi_area_layout",
    "objective_report",
    "circuit3d",
    "geometry3d",
    "column_geometry",
]
