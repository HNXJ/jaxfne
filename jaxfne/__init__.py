"""jaxfne: JAX Field Neural Equations.

A compact source-to-field neurophysiology engine for Tensor-Field Neural
Equations (TFNE).  Public API is object-oriented; numerical kernels are JAX-first.
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
    operator_status,
    paradigm,
    runtime,
    runtime_report,
    simulation,
)
from .emitters import EIGNetwork, IzhikevichParams, make_eig_network, simulate_eig_izhikevich
from .fields import (
    FieldOutput,
    project_laminar_sources,
    project_sources_to_laminar_field,
    probe_laminar_modes,
    validate_projection_invariants,
    validate_source_field_status,
)
from .io import config_hash, json_safe, manifest, save_json, sha256_file, sha256_text

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
    "operator_status",
    "paradigm",
    "runtime",
    "runtime_report",
    "simulation",
    "EIGNetwork",
    "IzhikevichParams",
    "make_eig_network",
    "simulate_eig_izhikevich",
    "FieldOutput",
    "project_laminar_sources",
    "project_sources_to_laminar_field",
    "probe_laminar_modes",
    "validate_projection_invariants",
    "validate_source_field_status",
    "config_hash",
    "json_safe",
    "manifest",
    "save_json",
    "sha256_file",
    "sha256_text",
]

__version__ = "0.0.4"
