"""jaxfne: JAX Field Neural Equations.

A compact source-to-field neurophysiology engine for Tensor-Field Neural
Equations (TFNE).  Public API is object-oriented; numerical kernels are JAX-first.
"""

from .core import (
    AxisSpec,
    BasisSpec,
    Configuration,
    ConfigValidationResult,
    DatasetSpec,
    JaxFNEConfig,
    LaminarPopulation,
    LaminarSourceGeometry,
    Model,
    Objective,
    Paradigm,
    ParadigmCondition,
    ParadigmEvent,
    Probe,
    ObjectiveReport,
    ReadoutResult,
    ReadoutSpec,
    RunReceipt,
    RuntimeConfig,
    Signal,
    Signals,
    Simulation,
    StimulusSchedule,
    SurrogateConfig,
    TrialBatch,
    TrialBatchResult,
    TrialResult,
    TrialSpec,
    config_to_configuration,
    config_to_geometry,
    config_to_simulation,
    config_to_trial_batch,
    config_truth_boundary,
    configuration,
    construct,
    dataset_spec,
    enable_x64,
    laminar_source_geometry,
    load_config,
    objective,
    operator_status,
    paradigm,
    runtime_report,
    _JAXFNE_VERSION,
    _KNOWN_READOUT_METRICS,
    readout_spec,
    run_receipt,
    run_trials,
    simulate,
    simulation,
    standard_visual_omission,
    stimulus_schedule,
    surrogate_config,
    trial_batch,
    validate_config,
    with_emitter_parameters,
    _KNOWN_METRICS,
    default_basis_spec,
)
from .bridges import BridgeSpec, JaxleyEmitterBridge, JaxleyTraceSpec, jaxley_trace_to_signals, require_jaxley
from . import vis
from .emitters import (
    EdgeList,
    EIGNetwork,
    IzhikevichParams,
    ReceptorSpec,
    SynapseSpec,
    make_edge_list_from_dense,
    make_eig_network,
    simulate_edge_recurrent_izhikevich,
    simulate_eig_izhikevich,
    simulate_receptor_exponential_izhikevich,
    standard_receptor_specs,
    standard_receptor_tau_table,
)
from .presets import (
    CELL_TYPE_PRESETS,
    DEFAULT_SPIKE_IMPULSE_GAIN,
    RECEPTOR_KINETICS,
)
from .fields import (
    FieldOutput,
    compute_conservation_proxy_diagnostics,
    project_laminar_sources,
    project_sources_to_laminar_field,
    probe_laminar_modes,
    validate_projection_invariants,
    validate_source_field_status,
)
from .io import config_hash, json_safe, manifest, save_json, save_receipt, sha256_file, sha256_text
from .optim import (
    AGSDR,
    AGSDRState,
    GSDRState,
    OptimizerSpec,
    SDRState,
    agsdr,
    agsdr_transform,
    gsdr,
    gsdr_transform,
    optax_adam,
    optax_sgd,
    random_search,
    require_optax,
    sdr_transform,
)

__all__ = [
    "AxisSpec",
    "BasisSpec",
    "BridgeSpec",
    "ConfigValidationResult",
    "Configuration",
    "default_basis_spec",
    "DatasetSpec",
    "JaxFNEConfig",
    "JaxleyEmitterBridge",
    "JaxleyTraceSpec",
    "LaminarPopulation",
    "LaminarSourceGeometry",
    "Model",
    "Objective",
    "Paradigm",
    "ParadigmCondition",
    "ParadigmEvent",
    "Probe",
    "ObjectiveReport",
    "ReadoutResult",
    "ReadoutSpec",
    "RunReceipt",
    "RuntimeConfig",
    "Signal",
    "Signals",
    "Simulation",
    "StimulusSchedule",
    "SurrogateConfig",
    "TrialBatch",
    "TrialBatchResult",
    "TrialResult",
    "TrialSpec",

    "config_to_configuration",
    "config_to_geometry",
    "config_to_simulation",
    "config_to_trial_batch",
    "config_truth_boundary",
    "configuration",
    "construct",
    "dataset_spec",
    "enable_x64",
    "laminar_source_geometry",
    "load_config",
    "objective",
    "operator_status",
    "paradigm",
    "readout_spec",
    "require_jaxley",
    "jaxley_trace_to_signals",
    "run_receipt",
    "runtime",
    "runtime_report",
    "run_trials",
    "simulate",
    "simulation",
    "standard_visual_omission",
    "stimulus_schedule",
    "surrogate_config",
    "trial_batch",
    "validate_config",
    "vis",
    "with_emitter_parameters",
    "_KNOWN_METRICS",
    "EdgeList",
    "EIGNetwork",
    "IzhikevichParams",
    "ReceptorSpec",
    "SynapseSpec",
    "make_edge_list_from_dense",
    "make_eig_network",
    "simulate_edge_recurrent_izhikevich",
    "simulate_eig_izhikevich",
    "simulate_receptor_exponential_izhikevich",
    "standard_receptor_specs",
    "standard_receptor_tau_table",
    "CELL_TYPE_PRESETS",
    "DEFAULT_SPIKE_IMPULSE_GAIN",
    "RECEPTOR_KINETICS",
    "compute_conservation_proxy_diagnostics",
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
    "save_receipt",
    "sha256_file",
    "sha256_text",
    "AGSDR",
    "AGSDRState",
    "GSDRState",
    "OptimizerSpec",
    "SDRState",
    "agsdr",
    "agsdr_transform",
    "gsdr",
    "gsdr_transform",
    "optax_adam",
    "optax_sgd",
    "random_search",
    "require_optax",
    "sdr_transform",
]

__version__ = _JAXFNE_VERSION


import sys
from types import ModuleType as _ModuleType


class _RuntimeModuleWrapper(_ModuleType):
    """Custom module wrapper to handle jaxfne.runtime() function / module collision.

    Problem: Python doesn't allow having both a function called 'runtime' and a
    module called 'runtime' in the same package. When 'import jaxfne.runtime' is
    executed, it replaces jaxfne.runtime (the function) with jaxfne.runtime (the
    module), breaking code that calls jaxfne.runtime().

    Solution: Override __setattr__ to prevent Python from storing the jaxfne.runtime
    module in this module's __dict__. Override __getattr__ to return the function
    instead when runtime is accessed.
    """

    def __setattr__(self, name, value):
        """Prevent 'runtime' module from being stored in __dict__."""
        if name == "runtime" and isinstance(value, _ModuleType):
            # Python is trying to add the jaxfne.runtime module to __dict__.
            # We ignore this to prevent the collision.
            return
        # For all other attributes, use normal assignment
        super().__setattr__(name, value)

    def __getattr__(self, name):
        """Dynamically resolve attributes to handle the runtime name collision."""
        if name == "runtime":
            # Return the runtime function from core, not the module
            from .core import runtime as _runtime_fn
            return _runtime_fn
        # Delegate to the original module's __dict__ for other attributes
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


# Replace jaxfne module in sys.modules with the wrapper to handle attribute access
_current_module = sys.modules[__name__]
_wrapper = _RuntimeModuleWrapper(__name__)
_wrapper.__dict__.update(_current_module.__dict__)
_wrapper.__file__ = __file__
sys.modules[__name__] = _wrapper
