"""jaxfne: JAX Field Neural Equations.

A compact source-to-field neurophysiology skeleton for Tensor-Field Neural Equations.
"""

from .core import (
    Configuration,
    Model,
    Objective,
    Paradigm,
    Probe,
    RuntimeConfig,
    Signal,
    Signals,
    Simulation,
    configuration,
    construct,
    objective,
    paradigm,
    runtime,
    simulation,
)
from .emitters import izhikevich_eig_params, simulate_izhikevich_eig
from .fields import project_sources_to_laminar_field
from .io import manifest, save_json

__all__ = [
    "Configuration",
    "Model",
    "Objective",
    "Paradigm",
    "Probe",
    "RuntimeConfig",
    "Signal",
    "Signals",
    "Simulation",
    "configuration",
    "construct",
    "objective",
    "paradigm",
    "runtime",
    "simulation",
    "izhikevich_eig_params",
    "simulate_izhikevich_eig",
    "project_sources_to_laminar_field",
    "manifest",
    "save_json",
]

__version__ = "0.0.3"
