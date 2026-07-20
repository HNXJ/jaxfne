"""construct()/connect() hub: builds Model from Configuration, merges Models,
plus the remaining group-1 connectivity compiler helpers and group-8 paradigm/
receipt/manifest-adjacent module-level functions.

Split (Phase 2 defragmentation, 2026-07-20, part of the 0.4.8-0.4.48 roadmap's
Defragmentation wave 1) into 5 submodules -- ``_construct_population.py``,
``_construct_connectivity.py``, ``_construct_presets.py``, ``_construct_core.py``,
``_construct_extras.py`` -- each re-exported below for backward compatibility.
This mirrors the same "monolith -> thin re-export aggregator" pattern
``jaxfne/core.py`` already uses for its own split (see that file's docstring).
Import from ``jaxfne.core``, not this module, unless you are working on the
split itself.

Internal dependency graph among the 5 submodules (acyclic by construction):
``_construct_connectivity`` and ``_construct_extras`` are leaves (no
dependency on any other ``_construct_*`` submodule); ``_construct_population``
depends on ``_construct_connectivity``; ``_construct_presets`` depends on
``_construct_population``; ``_construct_core`` depends on
``_construct_connectivity``, ``_construct_population``, and
``_construct_extras``. None of the 5 import from this aggregator file.
"""

from __future__ import annotations

from ._construct_population import (
    _DENSE_CONNECTIVITY_WARN_N,
    _SPARSE_DIRECT_N,
    _SUITE2_LAYER_CELL_TYPES_V1,
    _SUITE2_LAYER_CELL_TYPES_V4,
    _SUITE2_LAYER_FRACTIONS,
    _SUITE2_PROXY_MODES,
    _area_layer_cell_type_map,
    _area_layer_count_frac,
    _layer_ranges_for,
    _make_sparse_within_area_edges,
    _apply_connectivity,
    _default_layer_cell_types,
    _neuron_population_from_config,
)
from ._construct_connectivity import (
    _CONNECTIONS_EXACT_PRODUCT_CAP,
    _all_connection_rules_declare_resolvable_mechanism,
    _apply_edge_sign_policy,
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
    _empty_edge_list,
    _interarea_W,
    _interarea_layer_set,
    _mark_connections_compiled,
    _model_edge_list,
    connect,
)
from ._construct_presets import (
    configuration,
    objective,
    rate_targets,
    runtime,
    runtime_report,
    simulation,
    suite2_celltype_presets,
    suite2_four_celltype_config,
    suite2_net1_config,
    suite2_run_bundle,
    suite2_simulation,
    suite2_single_neuron_config,
    suite2_tune_noise_agsdr_adam,
    suite2_v1_v4_config,
)
from ._construct_core import (
    _HOMEOSTATIC_EI_CANONICAL_DEFAULTS,
    _SUPPORTED_EMITTER_FAMILIES,
    _apply_canonical_biophysics,
    _construct_apply_geometry_override,
    _construct_build_network,
    _construct_build_static,
    _construct_compile_connections,
    _construct_from_configuration,
    _construct_homeostatic_ei_model,
    _construct_resolve_edge_list,
    _construct_validate_config,
    _homeostasis_params_cache_fingerprint,
    _homeostatic_ei_cell_type_split,
    _np_isscalar_param,
    _resolve_homeostasis_k_gain,
    _runtime_config_from_metadata,
    _simulate_hdp_metadata,
    _simulate_homeostasis_metadata,
    compute_fields,
    construct,
    simulate,
)
from ._construct_extras import (
    _JAXFNE_CONFIG_SCHEMA_VERSION,
    _OBJECTIVE_REPORT_SCHEMA_VERSION,
    _RECOGNIZED_OPTIONAL_CONFIG_SECTIONS,
    _REQUIRED_CONFIG_SECTIONS,
    _VALID_FIELD_SOLVER_STATUS,
    dataset_spec,
    enable_x64,
    get_signal,
    laminar_source_geometry,
    migrate_schema,
    operator_status,
    provenance_receipt,
    readout_spec,
    run_receipt,
    run_trials,
    standard_visual_omission,
    surrogate_config,
    trial_batch,
)
