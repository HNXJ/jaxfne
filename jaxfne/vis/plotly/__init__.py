"""Plotly-based visualization pipeline for jaxfne.

Generic, package-native, manuscript-agnostic. Every function takes a
:class:`jaxfne.Signals` (and usually a :class:`jaxfne.Model`) and returns a
:class:`plotly.graph_objects.Figure` — no manuscript-specific preprocessing.

Proxy readouts only: LFP/CSD/source here are ``*_proxy`` fields produced by
the linear laminar-source kernel (``field_solver_status="linear_solver"``,
``field_claim_level="proxy_readout"``); these functions never escalate that
claim, they only plot it.
"""
from .network import plot_network_3d
from .raster import plot_raster, plot_population_rates, plot_membrane_potentials
from .lfp import plot_lfp
from .csd import plot_csd
from .spectra import plot_psd, plot_spectrogram, plot_band_power, plot_depth_profile
from .connectivity import plot_connectivity
from .metrics import plot_objective_history
from .dashboard import create_dashboard
from .exporters import export_figure, export_figures, FigureBundle
from . import manuscript
from .manuscript import manuscript_figures

__all__ = [
    "plot_network_3d",
    "plot_raster",
    "plot_population_rates",
    "plot_membrane_potentials",
    "plot_lfp",
    "plot_csd",
    "plot_psd",
    "plot_spectrogram",
    "plot_band_power",
    "plot_depth_profile",
    "plot_connectivity",
    "plot_objective_history",
    "create_dashboard",
    "export_figure",
    "export_figures",
    "FigureBundle",
    "manuscript_figures",
    "manuscript",
]
