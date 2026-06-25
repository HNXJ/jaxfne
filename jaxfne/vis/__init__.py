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
    visualize_network_3d,
)
from . import tutorial_panels
from .tutorial_panels import (
    visualize_laminar_column_3d,
    activity_trace_suite,
    spectrolaminar_suite_3panel,
)
from .raster_arrays import raster_from_arrays
from .plasticity_viz import plot_stdp_adaptation_suite
from .layout import cumulative_stack_offsets, cumulative_panel_extents
from .tutorial_array_plots import (
    plot_population_raster,
    plot_population_rate_array,
    plot_voltage_samples_array,
    plot_connectivity_matrix_array,
    plot_laminar_readout_array,
    plot_spectrolaminar_power_array,
)
from . import plotly
from .canonical import (
    plot_network_3d,
    plot_raster,
    plot_population_rate,
    plot_membrane_potentials,
    plot_lfp,
    plot_csd,
    plot_psd,
    plot_spectrogram,
    plot_band_power,
    plot_depth_profile,
    plot_connectivity,
    plot_objective_history,
)
from .exporters import export_figure, export_figures, FigureBundle

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
    "visualize_network_3d",
    # Tutorial panels (Etude No. 1)
    "visualize_laminar_column_3d",
    "activity_trace_suite",
    "spectrolaminar_suite_3panel",
    "raster_from_arrays",
    "plot_stdp_adaptation_suite",
    "cumulative_stack_offsets",
    "cumulative_panel_extents",
    "plot_population_raster",
    "plot_population_rate_array",
    "plot_voltage_samples_array",
    "plot_connectivity_matrix_array",
    "plot_laminar_readout_array",
    "plot_spectrolaminar_power_array",
    # Canonical backend-dispatching entry points (default backend="plotly")
    "plotly",
    "plot_network_3d",
    "plot_raster",
    "plot_population_rate",
    "plot_membrane_potentials",
    "plot_lfp",
    "plot_csd",
    "plot_psd",
    "plot_spectrogram",
    "plot_band_power",
    "plot_depth_profile",
    "plot_connectivity",
    "plot_objective_history",
    "export_figure",
    "export_figures",
    "FigureBundle",
]
