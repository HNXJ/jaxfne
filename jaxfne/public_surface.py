"""0.4.13 public semantic / API contract (Pass 1).

Classifies root exports into CANONICAL, ADVANCED, COMPATIBILITY, and
EXPERIMENTAL_INTERNAL tiers. ``hdp_params`` is a compatibility transport;
semantic groups are validated separately from flat key presence.

This module is the authority for ``jaxfne.__all__`` after 0.4.13 Pass 1.
"""

from __future__ import annotations

from typing import Any, Final, Literal

Tier = Literal["CANONICAL", "ADVANCED", "COMPATIBILITY", "EXPERIMENTAL_INTERNAL"]

PUBLIC_SURFACE_VERSION: Final[str] = "0.4.13-pass1"

# Internal dispatch identifiers — not public vocabulary.
INTERNAL_HDP_RULE_IDS: Final[frozenset[str]] = frozenset({"population_vector_restoring"})

# --- hdp_params semantic groups (compatibility transport, not conceptual API) ---

HDP_PARAM_GROUP_H_STATE: Final[frozenset[str]] = frozenset(
    {
        "h_state_dim",
        "h_state_locality",
        "h_state_readout",
        "h_state_coupling",
    }
)

HDP_PARAM_GROUP_H_DYNAMICS: Final[frozenset[str]] = frozenset(
    {
        "hdp_rule",
        "K_HDP",
        "tau_0_ms",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "C_spike",
        "K_ctrl",
        "K_w_ctrl",
        "rho_passive",
        "barrier_c",
        "barrier_d",
        "barrier_eps",
        "H_min",
        "H_max",
        "H_boost_gain",
        "size_scale_by_cell_type",
        "size_scale_override",
        "noise_scale",
        "record_dH_components",
        "record_edge_current",
        "record_weight_trace",
        "w_floor",
        "w_ceiling",
        "v_floor",
        "v_ceiling",
        "u_abs_max",
        "syn_abs_max",
    }
)

HDP_PARAM_GROUP_THETA_ADAPTATION: Final[frozenset[str]] = frozenset(
    {
        "controller_B",
        "controller_lambda",
        "controller_tau_H_s",
        "controller_tau_theta_s",
        "controller_rate_setpoint_E_hz",
        "controller_rate_setpoint_I_hz",
        "controller_theta_S_init",
        "m_ei_edge_mask",
        "e_neuron_mask",
        "theta_m_EI_bounds",
        "theta_eta_a_bounds",
    }
)

HDP_PARAM_GROUPS: Final[dict[str, frozenset[str]]] = {
    "h_state": HDP_PARAM_GROUP_H_STATE,
    "h_dynamics": HDP_PARAM_GROUP_H_DYNAMICS,
    "theta_adaptation": HDP_PARAM_GROUP_THETA_ADAPTATION,
}

KNOWN_HDP_PARAM_KEYS: Final[frozenset[str]] = frozenset().union(*HDP_PARAM_GROUPS.values())

# Population-H public semantics use locality, not MVC dispatch names.
PUBLIC_H_STATE_LOCALITIES: Final[frozenset[str]] = frozenset({"node", "population"})

COMPATIBILITY_DEPRECATIONS: Final[dict[str, str]] = {
    "Net": "Use Model instead; Net is a compatibility alias removed after 0.4.14.",
    "Config": "Use Configuration instead; Config is a compatibility alias removed after 0.4.14.",
    "AGSDR": "Use agsdr() instead; AGSDR is a legacy adapter.",
    "construct_neuronal_tensor": "Use construct(tensor, RuntimeConfiguration(...)) instead.",
    "load_neuronal_tensor": "Use load(path) instead.",
    "agsdr_transform": "Advanced Optax transform; prefer agsdr() spec at root.",
    "gsdr_transform": "Advanced Optax transform; prefer gsdr() spec at root.",
    "sdr_transform": "Advanced Optax transform.",
    "step_gsgd_transform": "Advanced Optax transform; prefer gsgd() spec at root.",
}

# --- Tier classification (259 symbols at 0.4.13 baseline) ---

_CANONICAL: Final[frozenset[str]] = frozenset(
    {
        # CircuitSpec → Model → Signals
        "AxisSpec",
        "BasisSpec",
        "Configuration",
        "ContinuationState",
        "DatasetSpec",
        "DynamicState",
        "LaminarPopulation",
        "LaminarSourceGeometry",
        "Model",
        "NodeIdentity",
        "Objective",
        "ObjectiveReport",
        "Paradigm",
        "ParadigmCondition",
        "ParadigmEvent",
        "Probe",
        "ReadoutResult",
        "ReadoutSpec",
        "RunReceipt",
        # Runtime ownership (2026-08-22 W4 scoping): RuntimeConfig is the single
        # owner of execution-policy semantics (backend/dtype/jit/vmap/kernels)
        # for the Configuration path; RuntimeConfiguration is the tensor-workflow
        # execution config consumed by construct(tensor, runtime) and bridged
        # into the same resolved semantics. One concept, one owner per path;
        # structural merge deferred past the 0.4.17 freeze.
        "RuntimeConfig",
        "RuntimeConfiguration",
        "SelectorSpec",
        "Signal",
        "Signals",
        "Simulation",
        "StimulusSchedule",
        "TrialBatch",
        "TrialBatchResult",
        "TrialResult",
        "TrialSpec",
        "TuneResult",
        "checkpoint_state",
        "compile_step_fn",
        "configuration",
        "connect",
        "construct",
        "dataset_spec",
        "default_basis_spec",
        "dynamic_state_from_model",
        "enable_x64",
        "evoked_l4_drive_paradigm",
        "get_signal",
        "laminar_source_geometry",
        "migrate_schema",
        "objective",
        "operator_status",
        "paradigm",
        "rate_targets",
        "rate_synchrony_targets",
        "readout_spec",
        "restore_state",
        "run_receipt",
        "provenance_receipt",
        "run_trials",
        "runtime",
        "runtime_report",
        "scan_network",
        "simulate",
        "simulation",
        "stimulus_schedule",
        "trial_batch",
        "with_emitter_parameters",
        # NeuronalTensor
        "Area",
        "AreaConnection",
        "Geometry3D",
        "InterConnection",
        "Layer",
        "NEURONAL_TENSOR_SCHEMA_VERSION",
        "NeuronType",
        "NeuronalTensor",
        "PlasticParams",
        "Pose3D",
        "StaticParams",
        "configs_dir",
        "default_relative_size",
        "list_canonical_neuronal_tensors",
        "load",
        "load_canonical_neuronal_tensor",
        "make_minimal_ei_tensor",
        "merge_neuronal_tensors",
        "neuronal_tensor_to_configuration",
        "save_neuronal_tensor",
        # JDNA (PseudoGenome / development)
        "PseudoGenome",
        "develop",
        "load_pseudogenome",
        "load_canonical_pseudogenome",
        "list_canonical_pseudogenomes",
        # Source / field / probe
        "FieldOutput",
        "LinearReadout",
        "cable_filter_report",
        "cable_filter_sources",
        "cable_filter_tau",
        "compute_fields",
        "construct_source_tensor",
        "csd_tensor",
        "eeg_proxy_transform",
        "emm_proxy_transform",
        "meg_proxy_transform",
        "probe_laminar_modes",
        "project_laminar_sources",
        "project_sources_to_laminar_field",
        "validate_projection_invariants",
        "validate_source_field_status",
        # Objectives / optimizers
        "AGSDROptimizerSpec",
        "EdgeParameterSpec",
        "MatrixParameterSpec",
        "OptimizerSpec",
        "agsdr",
        "edge_parameter",
        "gsdr",
        "gsgd",
        "matrix_parameter",
        "optax_adam",
        "optax_sgd",
        "random_search",
        "require_optax",
        # HDP family (RBS / h_state_* are compatibility API names for hidden state)
        "DEFAULT_HDP",
        # Emitters (family types, not low-level kernels)
        "Emitter",
        "IzhikevichEmitter",
        # Builders / presets used in grammar tutorials
        "CANONICAL_LAYERS_6L",
        "CANONICAL_LAYER_CELL_TYPE_FRACTIONS",
        "CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L",
        "CANONICAL_Z_BANDS",
        "CANONICAL_Z_BANDS_5L",
        "CELL_TYPE_PRESETS",
        "DEFAULT_LAYERS",
        "DEFAULT_SPIKE_IMPULSE_GAIN",
        "FLAT_CELL_TYPE_FRACTIONS",
        "RECEPTOR_KINETICS",
        "build_laminar_column",
        "build_multi_area_columns",
        "build_tutorial_laminar_column",
        "compile_connection_rules",
        "ConnectionCompileResult",
        "default_complete_configuration",
        "default_cortical_column_config",
        "laminar_cortex_config",
        # Paradigms
        "coop_omission_oddball_for_model",
        "coop_omission_oddball_for_neuronal_tensor",
        "coop_omission_oddball_paradigm",
        "general_delayed_match_to_sample_paradigm",
        "general_sequential_oddball_paradigm",
        "omission_oddball_paradigm",
        "paradigm_target_indices_from_model",
        "standard_visual_omission",
        # Suite 2 (tutorial grammar presets — canonical tutorial surface)
        "suite2_celltype_presets",
        "suite2_four_celltype_config",
        "suite2_net1_config",
        "suite2_run_bundle",
        "suite2_simulation",
        "suite2_single_neuron_config",
        "suite2_tune_noise_agsdr_adam",
        "suite2_v1_v4_config",
        # Evidence / I/O
        "asset_hashes",
        "config_hash",
        "export_report",
        "export_tutorial_artifacts",
        "json_safe",
        "manifest",
        "probe_report",
        "save_figure",
        "save_figures",
        "save_json",
        "save_receipt",
        "sha256_file",
        "sha256_text",
        "validation_report",
        # Tutorial / analysis helpers on critical paths
        "kappa_synchrony",
        "select_neurons",
        "spectrolaminar_motif_score",
        # Validation (runtime contract checks)
        "validate_runtime_config",
        "validate_neuronal_tensor",
        "validate_model",
        # Lazy / optional graphics (discoverable, optional dep)
        "plot_raster",
        "plot_spectrolaminar_suite",
        "vis",
    }
)

_ADVANCED: Final[frozenset[str]] = frozenset(
    {
        # Low-level emitter kernels and wiring primitives
        "EdgeList",
        "EIGNetwork",
        "IzhikevichParams",
        "ReceptorSpec",
        "SynapseLayer",
        "SynapseSpec",
        "SynapseState",
        "izhikevich_params_from_labels",
        "make_edge_list_from_dense",
        "make_eig_network",
        "simulate_edge_recurrent_izhikevich",
        "simulate_eig_izhikevich",
        "simulate_receptor_exponential_izhikevich",
        "standard_receptor_specs",
        "standard_receptor_tau_table",
        "synaptic_current_tensor",
        "synaptic_tau_from_mechanism",
        "synaptic_tensor_report",
        # Bridges / optional interop
        "BridgeSpec",
        "JaxFemFieldBridge",
        "JaxleyBridge",
        "JaxleyEmitterBridge",
        "JaxleyTraceSpec",
        "hh_jaxley_reference_trace",
        "hh_numpy_reference_trace",
        "jaxley_to_signals",
        "jaxley_trace_to_signals",
        "require_jax_fem",
        "require_jaxley",
        # Solvers / integrators
        "DiffraxSolver",
        "EulerSolver",
        "SolverConfig",
        "euler_scan",
        "euler_step",
        "solve_ode",
        # Spectral analysis kernels
        "bandpower_jax",
        "spectrolaminar_psd_jax",
        "spectrolaminar_readout_kernel_jax",
        "spectrolaminar_similarity_candidates_jax",
        "spectrolaminar_similarity_candidates_seeds_jax",
        "spectrolaminar_similarity_kernel_jax",
        # STDP / streaming plasticity
        "STDPPlasticityConfig",
        "STDPState",
        "plot_stdp_adaptation_suite",
        "run_stdp_stream",
        "summarize_stdp_adaptation",
        "update_stdp_weights_jax",
        # Dev / QA utilities
        "compilation_registry",
        "compile_connection_rules_jax",
        "compute_conservation_proxy_diagnostics",
        "configuration_diff",
        "is_valid_signal",
        "make_ei_cloud_network",
        "model_diff",
        "merge_runtime_configs",
        "runtime_config_diff",
        "tensor_summary",
        "triangular_drive",
    }
)

_COMPATIBILITY: Final[frozenset[str]] = frozenset(
    {
        "AGSDR",
        "AGSDRState",
        "Config",
        "GSDRState",
        "GSGDState",
        "Net",
        "SDRState",
        "agsdr_transform",
        "construct_neuronal_tensor",
        "gsdr_transform",
        "load_neuronal_tensor",
        "sdr_transform",
        "step_gsgd_transform",
    }
)

_EXPERIMENTAL_INTERNAL: Final[frozenset[str]] = frozenset(
    {
        "BackupState",
        "BehaviorGate",
        "GLIFEmitter",
        "HierarchicalOddballParadigm",
        "LIFEmitter",
        "Manifest",
        "SanityDeltaConfig",
        "SanityDeltaModel",
        "SurrogateConfig",
        "surrogate_config",
        "TaskEpisode",
        "get_sharding_context",
        "make_candidate_sharding",
        "make_population_mesh",
        "make_replicated_sharding",
        "read_nwb",
        "solve_volume_conductor_experimental",
        "write_nwb",
    }
)

_ALL_CLASSIFIED: Final[frozenset[str]] = (
    _CANONICAL | _ADVANCED | _COMPATIBILITY | _EXPERIMENTAL_INTERNAL
)

PUBLIC_EXPORTS: Final[tuple[str, ...]] = tuple(sorted(_CANONICAL | _COMPATIBILITY))

ADVANCED_NAMESPACE: Final[dict[str, str]] = {
    "simulate_edge_recurrent_izhikevich": "jaxfne.emitters",
    "simulate_eig_izhikevich": "jaxfne.emitters",
    "simulate_receptor_exponential_izhikevich": "jaxfne.emitters",
    "EdgeList": "jaxfne.emitters",
    "EIGNetwork": "jaxfne.emitters",
    "IzhikevichParams": "jaxfne.emitters",
    "BridgeSpec": "jaxfne.bridges",
    "JaxleyEmitterBridge": "jaxfne.bridges",
    "STDPPlasticityConfig": "jaxfne.plasticity",
    "update_stdp_weights_jax": "jaxfne.plasticity",
    "spectrolaminar_psd_jax": "jaxfne.analysis.spectral",
    "get_sharding_context": "jaxfne.sharding_utils",
    "write_nwb": "jaxfne.pynwb_compat",
    "read_nwb": "jaxfne.pynwb_compat",
    "SanityDeltaConfig": "jaxfne.sanity_delta",
    "SurrogateConfig": "jaxfne._runtime_config",
    "surrogate_config": "jaxfne._runtime_config",
    "GLIFEmitter": "jaxfne.emitters",
    "LIFEmitter": "jaxfne.emitters",
    "solve_volume_conductor_experimental": "jaxfne.solvers",
}


def symbol_tier(name: str) -> Tier | None:
    if name in _CANONICAL:
        return "CANONICAL"
    if name in _ADVANCED:
        return "ADVANCED"
    if name in _COMPATIBILITY:
        return "COMPATIBILITY"
    if name in _EXPERIMENTAL_INTERNAL:
        return "EXPERIMENTAL_INTERNAL"
    return None


def validate_hdp_params_semantics(
    hdp_params: dict[str, Any],
    *,
    strict: bool = False,
) -> list[str]:
    """Validate ``hdp_params`` semantic grouping and public RBS (``h_state_*``) contracts."""
    issues: list[str] = []
    if not isinstance(hdp_params, dict):
        msg = "hdp_params must be a dict (compatibility transport for RBS/h_state groups)"
        return [msg] if strict else issues

    unknown = set(hdp_params) - KNOWN_HDP_PARAM_KEYS
    if unknown:
        issues.append(
            "hdp_params contains unrecognized keys "
            f"(not in semantic groups): {sorted(unknown)}"
        )

    rule = hdp_params.get("hdp_rule")
    if rule in INTERNAL_HDP_RULE_IDS:
        issues.append(
            f"hdp_rule={rule!r} is an internal dispatch identifier; "
            "public population-H semantics should use "
            "h_state_locality='population' with theta-adaptation coefficients"
        )

    locality = hdp_params.get("h_state_locality")
    if locality is not None and locality not in PUBLIC_H_STATE_LOCALITIES:
        issues.append(
            f"h_state_locality must be one of {sorted(PUBLIC_H_STATE_LOCALITIES)}; "
            f"got {locality!r}"
        )

    if locality == "population":
        missing_theta = sorted(
            k for k in ("controller_B", "m_ei_edge_mask")
            if k not in hdp_params
        )
        if missing_theta:
            issues.append(
                "population h_state_locality requires theta-adaptation keys: "
                f"{missing_theta}"
            )

    node_rules = {"signed_linear", "signed_quadratic", "hebbian_product"}
    if locality in (None, "node") and rule is not None and rule not in node_rules:
        if rule not in INTERNAL_HDP_RULE_IDS:
            issues.append(
                f"node hdp_rule must be one of {sorted(node_rules)}; got {rule!r}"
            )

    if strict and issues:
        raise ValueError(f"validate_hdp_params_semantics: {issues[0]}")
    return issues


def classify_hdp_params(hdp_params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Partition ``hdp_params`` into semantic groups for receipts/manifests."""
    grouped: dict[str, dict[str, Any]] = {k: {} for k in HDP_PARAM_GROUPS}
    grouped["unclassified"] = {}
    for key, value in hdp_params.items():
        placed = False
        for group_name, keys in HDP_PARAM_GROUPS.items():
            if key in keys:
                grouped[group_name][key] = value
                placed = True
                break
        if not placed:
            grouped["unclassified"][key] = value
    return grouped


def public_surface_summary() -> dict[str, Any]:
    """Machine-readable contract summary for 0.4.13 receipts."""
    return {
        "schema": "jaxfne.public_surface_contract.v0.4.13-pass1",
        "version": PUBLIC_SURFACE_VERSION,
        "counts": {
            "canonical": len(_CANONICAL),
            "advanced": len(_ADVANCED),
            "compatibility": len(_COMPATIBILITY),
            "experimental_internal": len(_EXPERIMENTAL_INTERNAL),
            "public_exports": len(PUBLIC_EXPORTS),
            "baseline_all": len(_ALL_CLASSIFIED),
        },
        "public_exports": list(PUBLIC_EXPORTS),
        "compatibility_deprecations": dict(COMPATIBILITY_DEPRECATIONS),
        "hdp_param_groups": {k: sorted(v) for k, v in HDP_PARAM_GROUPS.items()},
        "internal_hdp_rule_ids": sorted(INTERNAL_HDP_RULE_IDS),
        "public_h_state_localities": sorted(PUBLIC_H_STATE_LOCALITIES),
    }
