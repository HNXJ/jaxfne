"""Core object model for :mod:`jaxfne`.

Docs: ``docs/api/core.md`` (https://jaxfne.readthedocs.io/en/latest/api/core/) —
update that page when this module's public API changes.

Design target: object-oriented public API, pure-JAX computational core.  The
current package is an honest TFNE scaffold: reduced emitters plus laminar proxy
source/readout status, a proxy field/readout scaffold.
"""

from __future__ import annotations

import contextlib
import json
import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from .emitters import (
    EdgeList,
    EIGNetwork,
    IzhikevichParams,
    make_edge_list_from_dense,
    make_eig_network,
    izhikevich_params_from_labels,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_homeostatic,
    simulate_edge_recurrent_izhikevich_hdp,
    simulate_eig_izhikevich,
    simulate_receptor_exponential_izhikevich,
)

from .fields import FieldOutput, probe_laminar_modes, project_laminar_sources
from .io import config_hash, json_safe, load_json, manifest as build_manifest
from .presets import DEFAULT_SPIKE_IMPULSE_GAIN
from ._runtime_config import (
    RuntimeConfig,
    SurrogateConfig,
    _ALLOWED_DTYPES,
    _ALLOWED_SYNAPTIC_KERNELS,
    _device_scope,
    _jaxlib_version,
    _resolve_backend_device,
)
from ._config import (
    Configuration,
    Config,
    _default_operator_status,
    _circuit_json_safe,
    _default_metadata,
    _counts_from_fractions,
    _reject_retired_like,
    _ProbeDeclarations,
)

# v0.3.29: canonical selector/identity types live in experimental_hpc.contracts.
# Re-export (do not duplicate) so the stable API exposes one definition.
from .experimental_hpc.contracts import NodeIdentity, SelectorSpec
from ._signals import (
    Simulation,
    Probe,
    Signals,
    Signal,
    Objective,
    DatasetSpec,
    TrialSpec,
    TrialBatch,
    TrialResult,
    TrialBatchResult,
    ReadoutSpec,
    ReadoutResult,
    ObjectiveReport,
    RunReceipt,
    StimulusSchedule,
    LaminarPopulation,
    LaminarSourceGeometry,
    AxisSpec,
    BasisSpec,
    default_basis_spec,
    ParadigmEvent,
    ParadigmCondition,
    Paradigm,
    paradigm,
    evoked_l4_drive_paradigm,
    omission_oddball_paradigm,
    coop_omission_oddball_paradigm,
    general_sequential_oddball_paradigm,
    general_delayed_match_to_sample_paradigm,
    _make_poisson_drive,
    _finite_or_none,
    _compute_kappa_synchrony_metric,
    _compute_all_metrics,
    _check_gate_criterion,
    _evaluate_loss_spec,
    _evaluate_regularizer_spec,
    _evaluate_gate_spec,
    _default_basis_dict,
    _normalize_manifest_readout,
    _KNOWN_METRICS,
    _KNOWN_LAYERS,
    _KNOWN_CONFIG_GATE_METRICS,
    _SIGNALS_GET_NEURON_AXIS_KEYS,
    _SIGNALS_GET_FIELD_KEYS,
    _SIGNALS_GET_KEY_ALIASES,
    _AXIS_STATUS_VALUES,
    _SPACE_BASIS_VALUES,
    _TIME_BASIS_VALUES,
    _FIELD_REGIME_VALUES,
    _FUTURE_FIELD_REGIMES,
    _SOURCE_MODE_BASIS_VALUES,
    _PROBE_BASIS_VALUES,
    _CONSERVATIVE_TRUTH_DEFAULTS,
)
from ._model import (
    Model,
    MatrixParameterSpec,
    matrix_parameter,
    TuneResult,
    stimulus_schedule,
    with_emitter_parameters,
    _model_with_scalar_parameter,
    _mask_for_parameter,
    _model_with_matrix_parameter,
    _model_with_parameters,
    _evaluate_soft_rate_targets,
    _JAXFNE_VERSION,
    _RECEIPT_SCHEMA_VERSION,
    _MANIFEST_SCHEMA_VERSION,
    _SOURCE_PROXY_METADATA,
    _KNOWN_READOUT_METRICS,
)

# MatrixParameterSpec, matrix_parameter, TuneResult, Model, stimulus_schedule,
# _JAXFNE_VERSION, _RECEIPT_SCHEMA_VERSION, _MANIFEST_SCHEMA_VERSION,
# _SOURCE_PROXY_METADATA, _KNOWN_READOUT_METRICS moved to jaxfne/_model.py and
# re-exported above (slice 4 of the core.py monolith split).
from ._construct import (
    _CONNECTIONS_EXACT_PRODUCT_CAP,
    _DENSE_CONNECTIVITY_WARN_N,
    _JAXFNE_CONFIG_SCHEMA_VERSION,
    _OBJECTIVE_REPORT_SCHEMA_VERSION,
    _RECOGNIZED_OPTIONAL_CONFIG_SECTIONS,
    _REQUIRED_CONFIG_SECTIONS,
    _SPARSE_DIRECT_N,
    _SUITE2_LAYER_CELL_TYPES_V1,
    _SUITE2_LAYER_CELL_TYPES_V4,
    _SUITE2_LAYER_FRACTIONS,
    _SUITE2_PROXY_MODES,
    _SUPPORTED_EMITTER_FAMILIES,
    _VALID_FIELD_SOLVER_STATUS,
    _all_connection_rules_declare_resolvable_mechanism,
    _apply_canonical_biophysics,
    _apply_edge_sign_policy,
    _area_layer_cell_type_map,
    _area_layer_count_frac,
    _compile_connection_rules,
    _compile_mechanism_aware_connection_rules,
    _concat_edge_lists,
    _connect_compile_cross_edges,
    _connect_merge_cfg,
    _connect_merge_edges,
    _connect_merge_emitter,
    _connect_merge_neuron_metadata,
    _connect_merge_positions,
    _connect_merge_static,
    _connect_reconcile_runtime,
    _connect_resolve_namespace,
    _connect_validate_models,
    _connection_selector_mask,
    _construct_apply_geometry_override,
    _construct_build_network,
    _construct_build_static,
    _construct_compile_connections,
    _construct_from_configuration,
    _construct_resolve_edge_list,
    _construct_validate_config,
    _empty_edge_list,
    _homeostasis_params_cache_fingerprint,
    _interarea_W,
    _interarea_layer_set,
    _layer_ranges_for,
    _make_sparse_within_area_edges,
    _mark_connections_compiled,
    _model_edge_list,
    _np_isscalar_param,
    _resolve_homeostasis_k_gain,
    _runtime_config_from_metadata,
    _simulate_hdp_metadata,
    _simulate_homeostasis_metadata,
    _suite2_apply_connectivity,
    _suite2_default_layer_cell_types,
    _suite2_neuron_population_from_config,
    compute_fields,
    configuration,
    connect,
    construct,
    dataset_spec,
    enable_x64,
    get_signal,
    laminar_source_geometry,
    migrate_schema,
    objective,
    operator_status,
    provenance_receipt,
    rate_targets,
    readout_spec,
    run_receipt,
    run_trials,
    runtime,
    runtime_report,
    simulate,
    simulation,
    standard_visual_omission,
    suite2_celltype_presets,
    suite2_four_celltype_config,
    suite2_net1_config,
    suite2_run_bundle,
    suite2_simulation,
    suite2_single_neuron_config,
    suite2_tune_noise_agsdr_adam,
    suite2_v1_v4_config,
    surrogate_config,
    trial_batch,
)

# All remaining core.py content (construct()/connect() hub, group-1
# connectivity helpers, group-8 paradigm/receipt/manifest functions) moved
# to jaxfne/_construct.py and re-exported above (slice 5/5, final slice, of
# the core.py monolith split).
