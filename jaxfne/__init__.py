"""jaxfne: JAX Field Neural Equations.

A compact source-to-field neurophysiology engine for Tensor-Field Neural
Equations (TFNE).  Public API is object-oriented; numerical kernels are JAX-first.
"""

from .core import (
    AxisSpec,
    BasisSpec,
    Configuration,
    Config,
    MatrixParameterSpec,
    matrix_parameter,
    ConfigValidationResult,
    DatasetSpec,
    JaxFNEConfig,
    LaminarPopulation,
    LaminarSourceGeometry,
    Model,
    NodeIdentity,
    SelectorSpec,
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
    TuneResult,
    config_to_configuration,
    config_to_geometry,
    config_to_simulation,
    config_to_trial_batch,
    config_truth_boundary,
    configuration,
    connect,
    construct,
    migrate_schema,
    dataset_spec,
    enable_x64,
    get_signal,
    evoked_l4_drive_paradigm,
    laminar_source_geometry,
    load_config,
    objective,
    omission_oddball_paradigm,
    coop_omission_oddball_paradigm,
    general_sequential_oddball_paradigm,
    general_delayed_match_to_sample_paradigm,
    operator_status,
    rate_targets,
    runtime_report,
    _JAXFNE_VERSION,
    _KNOWN_READOUT_METRICS,
    readout_spec,
    run_receipt,
    provenance_receipt,
    run_trials,
    simulate,
    simulation,
    standard_visual_omission,
    suite2_celltype_presets,
    suite2_single_neuron_config,
    suite2_four_celltype_config,
    suite2_net1_config,
    suite2_v1_v4_config,
    suite2_simulation,
    suite2_tune_noise_agsdr_adam,
    suite2_run_bundle,
    stimulus_schedule,
    surrogate_config,
    trial_batch,
    validate_config,
    with_emitter_parameters,
    default_basis_spec,
)

from . import paradigm
from . import tutorial_utils
from . import export as _export_module
from .tutorial_utils import (
    select_neurons,
    kappa_synchrony,
    rate_synchrony_targets,
    # Etude No. 1 laminar column helpers.
    # The tutorial scaffold builder lives at jtfne.tutorial_utils.build_laminar_column
    # and is also exposed at root as the unambiguous alias build_tutorial_laminar_column
    # (defined below). Root-level build_laminar_column remains the established
    # builders.build_laminar_column API.
    LaminarColumnConfig,
    CellTypePreset,
    make_cell_dist,
    make_cell_type_catalog,
    cell_catalog_frame,
    make_laminar_column_config,
    config_summary_frame,
    make_izhikevich_control_panel,
    collect_izhikevich_control,
    make_stimulus,
    build_laminar_connections,
    select_cells,
    simulate_laminar_trials,
    spectrolaminar_from_trials,
    spectrolaminar_motif_score,
    summarize_spectrolaminar_similarity,
)
from .export import (
    save_figure,
    save_figures,
    export_report,
    export_tutorial_artifacts,
    plot_raster,
    plot_spectrolaminar_suite,
)

# Unambiguous root-level alias for the tutorial scaffold builder. Root-level
# build_laminar_column (from builders, imported below) is the established API;
# this alias avoids confusion for tutorial users.
build_tutorial_laminar_column = tutorial_utils.build_laminar_column

from .bridges import BridgeSpec, JaxleyEmitterBridge, JaxleyTraceSpec, jaxley_trace_to_signals, jaxley_to_signals, require_jaxley, JaxleyBridge, hh_numpy_reference_trace
from . import analysis
from . import vis
from .emitters import (
    EdgeList,
    EIGNetwork,
    IzhikevichParams,
    ReceptorSpec,
    SynapseSpec,
    make_edge_list_from_dense,
    make_eig_network,
    izhikevich_params_from_labels,
    simulate_edge_recurrent_izhikevich,
    simulate_eig_izhikevich,
    simulate_receptor_exponential_izhikevich,
    standard_receptor_specs,
    standard_receptor_tau_table,
    synaptic_tau_from_mechanism,
    synaptic_current_tensor,
    synaptic_tensor_report,
    Emitter,
    IzhikevichEmitter,
    GLIFEmitter,
    LIFEmitter,
    SynapseState,
    SynapseLayer,
)
from .presets import (
    CELL_TYPE_PRESETS,
    DEFAULT_SPIKE_IMPULSE_GAIN,
    RECEPTOR_KINETICS,
)
from .builders import (
    default_cortical_column_config,
    default_spectrolaminar_config,
    default_nuclei_config,
    default_complete_configuration,
    laminar_cortex_config,
    build_laminar_column,
    build_multi_area_columns,
    CANONICAL_LAYER_CELL_TYPE_FRACTIONS,
    CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L,
    CANONICAL_Z_BANDS,
    CANONICAL_Z_BANDS_5L,
    CANONICAL_LAYERS_6L,
    FLAT_CELL_TYPE_FRACTIONS,
    DEFAULT_LAYERS,
    connect_columns,
    sparse_intercolumn_connectivity,
    all_to_all_intercolumn_connectivity,
    layer_celltype_count_table,
    column_density_table,
    configuration_table,
    validate_configuration,
)
from .connectivity import compile_connection_rules, ConnectionCompileResult, compile_connection_rules_jax
from .fields import (
    FieldOutput,
    compute_conservation_proxy_diagnostics,
    construct_source_tensor,
    eeg_proxy_transform,
    emm_proxy_transform,
    meg_proxy_transform,
    project_laminar_sources,
    project_sources_to_laminar_field,
    probe_laminar_modes,
    validate_projection_invariants,
    validate_source_field_status,
    LinearReadout,
    cable_filter_tau,
    cable_filter_sources,
    cable_filter_report,
    csd_tensor,
)
from .io import config_hash, json_safe, manifest, save_json, save_receipt, sha256_file, sha256_text, validation_report, probe_report, asset_hashes
from .optim import (
    AGSDR,
    AGSDROptimizerSpec,
    AGSDRState,
    GSDRState,
    GSGDState,
    OptimizerSpec,
    SDRState,
    agsdr,
    agsdr_transform,
    gsdr,
    gsdr_transform,
    gsgd,
    optax_adam,
    optax_sgd,
    random_search,
    require_optax,
    sdr_transform,
    step_gsgd_transform,
)
# v0.3.18: sharding stubs — imported lazily so single-device users have no overhead.
from .sharding_utils import (
    get_sharding_context,
    make_candidate_sharding,
    make_population_mesh,
    make_replicated_sharding,
)
# v0.3.20: compilation registry
from .validation import compilation_registry, is_valid_signal
# v0.3.31: state integrators
from .solvers import (
    euler_step,
    euler_scan,
    SolverConfig,
    EulerSolver,
    DiffraxSolver,
    solve_ode,
    solve_volume_conductor_experimental,
)
# v0.3.31: PyNWB placeholder (not yet implemented)
from .pynwb_compat import write_nwb, read_nwb
# v0.3.32: Hierarchical global-local oddball API hardening
from .sanity_delta import (
    SanityDeltaConfig,
    SanityDeltaModel,
    HierarchicalOddballParadigm,
    BehaviorGate,
    BackupState,
    TaskEpisode,
    Manifest,
)
from . import sanity_runtime

# v0.3.34: STDP network plasticity API
from .plasticity import (
    STDPPlasticityConfig,
    STDPState,
    summarize_stdp_adaptation,
    plot_stdp_adaptation_suite,
    update_stdp_weights_jax,
)
from .analysis.spectral import (
    spectrolaminar_psd_jax,
    bandpower_jax,
    spectrolaminar_readout_kernel_jax,
    spectrolaminar_similarity_kernel_jax,
    spectrolaminar_similarity_candidates_jax,
    spectrolaminar_similarity_candidates_seeds_jax,
)
from .geometry import make_ei_cloud_network
from .stimulus import triangular_drive
from .streaming import run_stdp_stream

__all__ = [
    "STDPPlasticityConfig",
    "STDPState",
    "summarize_stdp_adaptation",
    "plot_stdp_adaptation_suite",
    "make_ei_cloud_network",
    "triangular_drive",
    "run_stdp_stream",
    "compilation_registry",
    "AxisSpec",
    "BasisSpec",
    "BridgeSpec",
    "ConfigValidationResult",
    "Configuration",
    "Config",
    "default_basis_spec",
    "default_cortical_column_config",
    "default_spectrolaminar_config",
    "default_nuclei_config",
    "default_complete_configuration",
    "laminar_cortex_config",
    "build_laminar_column",
    "build_multi_area_columns",
    "CANONICAL_LAYER_CELL_TYPE_FRACTIONS",
    "CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L",
    "CANONICAL_Z_BANDS",
    "CANONICAL_Z_BANDS_5L",
    "CANONICAL_LAYERS_6L",
    "FLAT_CELL_TYPE_FRACTIONS",
    "DEFAULT_LAYERS",
    "DatasetSpec",
    "JaxFNEConfig",
    "JaxleyEmitterBridge",
    "JaxleyBridge",
    "JaxleyTraceSpec",
    "hh_numpy_reference_trace",
    "LaminarPopulation",
    "LaminarSourceGeometry",
    "MatrixParameterSpec",
    "matrix_parameter",
    "Model",
    "Net",
    "NodeIdentity",
    "SelectorSpec",
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
    "TuneResult",

    "config_to_configuration",
    "config_to_geometry",
    "config_to_simulation",
    "config_to_trial_batch",
    "config_truth_boundary",
    "configuration",
    "connect",
    "construct",
    "migrate_schema",
    "compile_connection_rules",
    "ConnectionCompileResult",
    "dataset_spec",
    "enable_x64",
    "get_signal",
    "laminar_source_geometry",
    "load_config",
    "objective",
    "omission_oddball_paradigm",
    "coop_omission_oddball_paradigm",
    "general_sequential_oddball_paradigm",
    "general_delayed_match_to_sample_paradigm",
    "operator_status",
    "paradigm",
    "rate_targets",
    "readout_spec",
    "require_jaxley",
    "jaxley_trace_to_signals",
    "jaxley_to_signals",
    "run_receipt",
    "provenance_receipt",
    "runtime",
    "runtime_report",
    "run_trials",
    "simulate",
    "simulation",
    "standard_visual_omission",
    'suite2_celltype_presets',
    'suite2_single_neuron_config',
    'suite2_four_celltype_config',
    'suite2_net1_config',
    'suite2_v1_v4_config',
    'suite2_simulation',
    'suite2_tune_noise_agsdr_adam',
    'suite2_run_bundle',
    "stimulus_schedule",
    "surrogate_config",
    "trial_batch",
    "validate_config",
    "is_valid_signal",
    "vis",
    "with_emitter_parameters",
    # Etude No. 1 tutorial scaffold builder (unambiguous alias)
    "build_tutorial_laminar_column",
    # Phase 5b: AGSDR tuning helpers
    "select_neurons",
    "kappa_synchrony",
    "rate_synchrony_targets",
    "EdgeList",
    "EIGNetwork",
    "IzhikevichParams",
    "ReceptorSpec",
    "SynapseSpec",
    "Emitter",
    "IzhikevichEmitter",
    "GLIFEmitter",
    "LIFEmitter",
    "SynapseState",
    "SynapseLayer",
    "make_edge_list_from_dense",
    "make_eig_network",
    "izhikevich_params_from_labels",
    "simulate_edge_recurrent_izhikevich",
    "simulate_eig_izhikevich",
    "simulate_receptor_exponential_izhikevich",
    "standard_receptor_specs",
    "standard_receptor_tau_table",
    "synaptic_tau_from_mechanism",
    "synaptic_current_tensor",
    "synaptic_tensor_report",
    "CELL_TYPE_PRESETS",
    "DEFAULT_SPIKE_IMPULSE_GAIN",
    "RECEPTOR_KINETICS",
    "compute_conservation_proxy_diagnostics",
    "eeg_proxy_transform",
    "emm_proxy_transform",
    "meg_proxy_transform",
    "FieldOutput",
    "project_laminar_sources",
    "project_sources_to_laminar_field",
    "probe_laminar_modes",
    "validate_projection_invariants",
    "validate_source_field_status",
    "construct_source_tensor",
    "LinearReadout",
    "cable_filter_tau",
    "cable_filter_sources",
    "cable_filter_report",
    "csd_tensor",
    "config_hash",
    "json_safe",
    "manifest",
    "save_json",
    "save_receipt",
    "sha256_file",
    "sha256_text",
    "validation_report",
    "probe_report",
    "asset_hashes",
    # v0.3.37: root-level export grammar (lazy matplotlib; see jaxfne.export)
    "save_figure",
    "save_figures",
    "export_report",
    "export_tutorial_artifacts",
    "plot_raster",
    "plot_spectrolaminar_suite",
    "AGSDR",
    "AGSDROptimizerSpec",
    "AGSDRState",
    "GSDRState",
    "GSGDState",
    "OptimizerSpec",
    "SDRState",
    "agsdr",
    "agsdr_transform",
    "gsdr",
    "gsdr_transform",
    "gsgd",
    "optax_adam",
    "optax_sgd",
    "random_search",
    "require_optax",
    "sdr_transform",
    "step_gsgd_transform",
    # v0.3.18: sharding stubs
    "get_sharding_context",
    "make_candidate_sharding",
    "make_population_mesh",
    "make_replicated_sharding",
    # v0.3.31: state integrators
    "euler_step",
    "euler_scan",
    "SolverConfig",
    "EulerSolver",
    "DiffraxSolver",
    "solve_ode",
    # v0.3.31: PyNWB placeholder
    "write_nwb",
    "read_nwb",
    # v0.3.32: Hierarchical global-local oddball API
    "SanityDeltaConfig",
    "SanityDeltaModel",
    "HierarchicalOddballParadigm",
    "BehaviorGate",
    "BackupState",
    "TaskEpisode",
    "Manifest",
    "spectrolaminar_psd_jax",
    "bandpower_jax",
    "spectrolaminar_readout_kernel_jax",
    "spectrolaminar_similarity_kernel_jax",
    "spectrolaminar_similarity_candidates_jax",
    "spectrolaminar_similarity_candidates_seeds_jax",
    "spectrolaminar_motif_score",
    "compile_connection_rules_jax",
    "update_stdp_weights_jax",
    "solve_volume_conductor_experimental",
]

__version__ = _JAXFNE_VERSION
Net = Model


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
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


# Replace jaxfne module in sys.modules with the wrapper to handle attribute access
_current_module = sys.modules[__name__]
_wrapper = _RuntimeModuleWrapper(__name__)
_wrapper.__dict__.update(_current_module.__dict__)
_wrapper.__file__ = __file__
sys.modules[__name__] = _wrapper
