# Operator Inventory (generated)

Generated from the live `jaxfne.__all__` export surface (246 entries) by `scripts/generate_operator_inventory.py`. Grouped by each export's real defining submodule, not a hand-maintained category list — do not hand-edit; regenerate after any export change.

## `jaxfne (unresolved)` (12)

| Name | Kind | Signature |
|---|---|---|
| `CANONICAL_LAYERS_6L` | value |  |
| `CANONICAL_LAYER_CELL_TYPE_FRACTIONS` | value |  |
| `CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L` | value |  |
| `CANONICAL_Z_BANDS` | value |  |
| `CANONICAL_Z_BANDS_5L` | value |  |
| `CELL_TYPE_PRESETS` | value |  |
| `DEFAULT_LAYERS` | value |  |
| `DEFAULT_SPIKE_IMPULSE_GAIN` | value |  |
| `FLAT_CELL_TYPE_FRACTIONS` | value |  |
| `NEURONAL_TENSOR_SCHEMA_VERSION` | value |  |
| `RECEPTOR_KINETICS` | value |  |
| `vis` | value |  |

## `jaxfne._pipeline` (4)

| Name | Kind | Signature |
|---|---|---|
| `DynamicState` | class | `(v: ForwardRef('jax.Array'), u: ForwardRef('jax.Array'), prev_spikes: ForwardRef('jax.Array'), syn_state: ForwardRef('jax.Array'), H: ForwardRef('jax.Array'), w: ForwardRef('jax.Array'))` |
| `checkpoint_state` | function | `(model: 'Model', path: 'str | Path') -> 'Path'` |
| `dynamic_state_from_model` | function | `(model: 'Model') -> 'DynamicState'` |
| `restore_state` | function | `(path: 'str | Path') -> 'tuple[list, dict]'` |

## `jaxfne.analysis.spectral` (6)

| Name | Kind | Signature |
|---|---|---|
| `bandpower_jax` | function | `(psd: jax.Array, freqs: jax.Array, band_range: jax.Array) -> jax.Array` |
| `spectrolaminar_psd_jax` | function | `(signal: jax.Array, fs: float = 1000.0, freqs: jax.Array | None = None) -> jax.Array` |
| `spectrolaminar_readout_kernel_jax` | function | `(psd: jax.Array, freqs: jax.Array, alpha_beta_range: jax.Array, gamma_range: jax.Array) -> Dict[str, jax.Array]` |
| `spectrolaminar_similarity_candidates_jax` | function | `(alpha_beta: jax.Array, gamma: jax.Array, target_alpha_beta: jax.Array, target_gamma: jax.Array) -> jax.Array` |
| `spectrolaminar_similarity_candidates_seeds_jax` | function | `(alpha_beta: jax.Array, gamma: jax.Array, target_alpha_beta: jax.Array, target_gamma: jax.Array) -> jax.Array` |
| `spectrolaminar_similarity_kernel_jax` | function | `(alpha_beta: jax.Array, gamma: jax.Array, target_alpha_beta: jax.Array, target_gamma: jax.Array) -> jax.Array` |

## `jaxfne.bridges` (8)

| Name | Kind | Signature |
|---|---|---|
| `BridgeSpec` | class | `(name: 'str', backend: 'str', status: 'str' = 'schema_only_no_backend_constructed', source_calibration_status: 'str' = 'uncalibrated_bridge_output', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `JaxleyBridge` | class | `(model: 'Any', source_mode: 'str' = 'transmembrane_current', compartment_axis: 'str' = 'last')` |
| `JaxleyEmitterBridge` | class | `(morphology: 'str | None' = None, mechanisms: 'tuple[str, ...]' = (), source_calibration_status: 'str' = 'uncalibrated_jaxley_bridge', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `JaxleyTraceSpec` | class | `(trace_name: 'str' = 'jaxley_trace', backend: 'str' = 'jaxley', layout: 'str' = 'time_by_unit', state_name: 'str' = 'v', dt_ms: 'float' = 0.025, units_or_status: 'str' = 'mV_or_declared', source_mode: 'str' = 'voltage_proxy', source_calibration_status: 'str' = 'uncalibrated_jaxley_voltage_proxy', source_projection_mode: 'str' = 'external_trace_proxy', source_decomposition: 'str' = 'proxy_voltage_trace_not_current', spike_threshold: 'float | None' = 0.0, physical_amplitude_calibrated: 'bool' = False, claim_level: 'str' = 'computational_scaffold', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `hh_numpy_reference_trace` | function | `(duration_ms: 'float' = 500.0, dt_ms: 'float' = 0.1, current_amplitude: 'float' = 10.0) -> 'tuple[Any, Any, Any]'` |
| `jaxley_to_signals` | function | `(module: 'Any', recordings: 'Any', *, dt_ms: 'float' = 0.025, state: 'str' = 'v', spec: "'JaxleyTraceSpec | None'" = None, source: 'Any' = None) -> 'Any'` |
| `jaxley_trace_to_signals` | function | `(trace: 'Any', *, spec: 'JaxleyTraceSpec | None' = None, dt_ms: 'float | None' = None, layout: 'str | None' = None, source: 'Any' = None) -> 'Any'` |
| `require_jaxley` | function | `()` |

## `jaxfne.builders` (5)

| Name | Kind | Signature |
|---|---|---|
| `build_laminar_column` | function | `(name: 'str' = 'V1', n: 'int' = 1000, layers: 'Sequence[str] | None' = None, layer_fractions: 'Mapping[str, tuple] | None' = None, cell_type_fractions: 'Mapping[str, float] | None' = None, layer_cell_type_fractions: 'Mapping[str, Mapping[str, float]] | None' = None, *, ei_profile: "Literal['flat', 'canonical']" = 'flat', geometry: "Literal['auto', 'uniform3d', 'laminar']" = 'auto', within_connectivity: 'str' = 'all_to_all_uniform_random', within_gain: 'float' = 0.45, radius_mm: 'float' = 0.25, height_mm: 'float' = 1.6, edge_seed: 'int | None' = None) -> 'Configuration'` |
| `build_multi_area_columns` | function | `(areas: 'Sequence[str]' = ('V1', 'V4', 'PFC'), n_per_area: 'int' = 200, layers: 'Sequence[str] | None' = None, connectivity_mode: "Literal['sparse', 'all_to_all']" = 'sparse', *, ei_profile: "Literal['flat', 'canonical']" = 'flat', cell_type_fractions: 'Mapping[str, float] | None' = None, within_connectivity: 'str' = 'all_to_all_uniform_random', within_gain: 'float' = 0.35, p_feedforward: 'float' = 0.3, p_feedback: 'float' = 0.2) -> 'Configuration'` |
| `default_complete_configuration` | function | `(column_name: 'str' = 'V1', nucleus_name: 'str' = 'thalamus', n_column: 'int' = 100, n_nucleus: 'int' = 60, layers: 'Sequence[str] | None' = None, seed: 'int | None' = None, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1) -> 'Configuration'` |
| `default_cortical_column_config` | function | `(column_name: 'str' = 'single_column', n: 'int' = 100, layers: 'Sequence[str] | None' = None, seed: 'int | None' = None, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, *, synaptic_kernel: "Literal['exponential', 'receptor_exponential']" = 'exponential') -> 'Configuration'` |
| `laminar_cortex_config` | function | `(*, seed: 'int' = 0, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, areas: 'Sequence[str] | None' = None, layers: 'Sequence[str] | None' = None, cell_types: 'Mapping[str, float] | None' = None, n: 'int' = 128, emitter: 'str' = 'izhikevich', baseline_drive_by_cell_type: 'Mapping[str, float] | None' = None) -> 'Configuration'` |

## `jaxfne.connectivity` (3)

| Name | Kind | Signature |
|---|---|---|
| `ConnectionCompileResult` | class | `(edge_pre: 'jax.Array', edge_post: 'jax.Array', edge_weight: 'jax.Array', edge_mechanism: 'jax.Array', edge_rule_id: 'jax.Array', connection_table: 'list[dict[str, Any]]', mechanism_table: 'list[dict[str, Any]]', diagnostics: 'dict[str, Any]' = <factory>) -> None` |
| `compile_connection_rules` | function | `(neurons: 'Sequence[Mapping[str, Any]]', connections: 'Sequence[Mapping[str, Any]]', mechanisms: 'Sequence[Mapping[str, Any]]', *, seed: 'int' = 0, allow_empty: 'bool' = False, allow_self_connections: 'bool' = False, artifacts: 'Optional[Mapping[str, Any]]' = None, dtype: 'str' = 'float32') -> 'ConnectionCompileResult'` |
| `compile_connection_rules_jax` | function | `(pre_indices: 'jax.Array', post_indices: 'jax.Array', probability: 'float', key: 'jax.Array', max_edges: 'int', weight_val: 'float' = 1.0) -> 'tuple[jax.Array, jax.Array, jax.Array]'` |

## `jaxfne.core` (62)

| Name | Kind | Signature |
|---|---|---|
| `AxisSpec` | class | `(name: 'str', status: 'str' = 'active', size: 'Optional[int]' = None, units_or_status: 'str' = 'declared') -> None` |
| `BasisSpec` | class | `(space_basis: 'str' = 'laminar_depth', time_basis: 'str' = 'continuous_ms', field_regime: 'str' = 'laminar_proxy', source_mode: 'str' = 'proxy_no_field_solve', probe_basis: 'str' = 'multimodal_proxy', axes: 'tuple[Any, ...]' = <factory>) -> None` |
| `Config` | class | `(networks: 'list[dict[str, Any]]' = <factory>, emitters: 'list[dict[str, Any]]' = <factory>, fields: 'list[dict[str, Any]]' = <factory>, probes: 'list[dict[str, Any]]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `Configuration` | class | `(networks: 'list[dict[str, Any]]' = <factory>, emitters: 'list[dict[str, Any]]' = <factory>, fields: 'list[dict[str, Any]]' = <factory>, probes: 'list[dict[str, Any]]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `DatasetSpec` | class | `(name: 'str' = 'unnamed_dataset', modality: 'str' = 'unspecified', source_format: 'str' = 'unspecified', comparison_label: 'str' = 'p1', comparison_code: 'int' = 101, sampling_rate_hz: 'Optional[float]' = None, units: 'str' = 'unspecified', trial_filter: 'dict[str, Any]' = <factory>, condition_map: 'dict[str, list[int]]' = <factory>, quality_gates: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `LaminarPopulation` | class | `(name: 'str', cell_type: 'str', layer: 'str', depth_min: 'float', depth_max: 'float', n_units: 'int', source_calibration_status: 'str' = 'uncalibrated_izhikevich_native_current', physical_amplitude_calibrated: 'bool' = False, claim_level: 'str' = 'computational_scaffold') -> None` |
| `LaminarSourceGeometry` | class | `(populations: 'tuple[LaminarPopulation, ...]', n_units_total: 'int', position_units: 'str' = 'relative_laminar_depth_proxy', source_calibration_status: 'str' = 'uncalibrated_izhikevich_native_current', physical_amplitude_calibrated: 'bool' = False, claim_level: 'str' = 'computational_scaffold') -> None` |
| `MatrixParameterSpec` | class | `(mask: 'str', bounds: 'tuple', init: 'str' = 'current', trainable: 'bool' = True) -> None` |
| `Model` | class | `(cfg: 'Configuration', params: 'dict[str, Any]', static: 'dict[str, Any]') -> None` |
| `Net` | class | `(cfg: 'Configuration', params: 'dict[str, Any]', static: 'dict[str, Any]') -> None` |
| `Objective` | class | `(name: 'str' = 'anonymous', kind: 'str' = 'generic', losses: 'list[dict[str, Any]]' = <factory>, regularizers: 'list[dict[str, Any]]' = <factory>, gates: 'list[dict[str, Any]]' = <factory>) -> None` |
| `ObjectiveReport` | class | `(objective_name: 'str', evaluation_status: 'str', total_loss: 'Optional[float]', all_gates_pass: 'bool', losses: 'tuple[dict[str, Any], ...]', regularizers: 'tuple[dict[str, Any], ...]', gates: 'tuple[dict[str, Any], ...]', readout_results: "tuple['ReadoutResult', ...]" = <factory>, truth: 'dict[str, Any]' = <factory>, warnings: 'tuple[str, ...]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `Probe` | class | `(name: 'str', modes: 'Sequence[str]', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `ReadoutResult` | class | `(spec_name: 'str', metric: 'str', value: 'Optional[float]', status: 'str' = 'computed', claim_level: 'str' = 'computational_scaffold', physical_amplitude_calibrated: 'bool' = False, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `ReadoutSpec` | class | `(name: 'str', metric: 'str', time_window_ms: 'Optional[tuple[float, float]]' = None, n_contacts_slice: 'Optional[tuple[int, int]]' = None, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `RunReceipt` | class | `(receipt_id: 'str', jaxfne_version: 'str', config_hash: 'str', simulation: 'dict[str, Any]', signals_summary: 'dict[str, Any]', truth: 'dict[str, Any]', claim_labels: 'dict[str, Any]', backend: 'dict[str, Any]', tags: 'dict[str, Any]' = <factory>) -> None` |
| `RuntimeConfig` | class | `(backend: 'str' = 'auto', dtype: 'str' = 'float32', jit: 'bool | str' = False, vmap: 'bool | str' = False, precision: 'str' = 'default', seed: 'int' = 0, n_steps: 'int' = 0, recurrent_backend: 'str' = 'dense', synaptic_kernel: 'str' = 'exponential', recompilation_guard: 'str' = 'warning', enable_homeostasis: 'bool' = False, homeostasis_params: 'dict' = <factory>, enable_hdp: 'bool' = False, hdp_params: 'dict' = <factory>, device_type: 'Optional[str]' = None, dtype_primary: 'Optional[str]' = None, x64_enabled: 'Optional[bool]' = None) -> None` |
| `Signal` | class | `(time_ms: 'jax.Array', V_m: 'jax.Array', spikes: 'jax.Array', sources: 'Optional[jax.Array]', field: 'Optional[FieldOutput]', metadata: 'dict[str, Any]') -> None` |
| `Signals` | class | `(time_ms: 'jax.Array', V_m: 'jax.Array', spikes: 'jax.Array', sources: 'Optional[jax.Array]', field: 'Optional[FieldOutput]', metadata: 'dict[str, Any]') -> None` |
| `Simulation` | class | `(duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.05, plasticity: 'float' = 0.0, seed: 'int' = 0, record_sources: 'bool' = True, record_fields: 'bool' = True, poisson_drive: 'Optional[dict]' = None, runtime: 'RuntimeConfig | None' = None, ablation: 'Optional[str]' = None) -> None` |
| `StimulusSchedule` | class | `(events: 'tuple[dict[str, Any], ...]', n_neurons: 'int', source_calibration_status: 'str' = 'uncalibrated_izhikevich_native_current', physical_amplitude_calibrated: 'bool' = False, claim_level: 'str' = 'computational_scaffold') -> None` |
| `SurrogateConfig` | class | `(method: 'str' = 'none', beta: 'float' = 10.0, applies_to: 'str' = 'izhikevich_reset', status: 'str' = 'declaration_only_v0.0.8') -> None` |
| `TrialBatch` | class | `(trials: 'tuple[TrialSpec, ...]', batch_id: 'str' = 'anonymous_batch', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `TrialBatchResult` | class | `(batch_id: 'str', results: 'tuple[TrialResult, ...]', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `TrialResult` | class | `(trial_id: 'str', condition_label: 'Optional[str]' = None, signals: 'Optional[Signals]' = None, success: 'bool' = True, error_message: 'Optional[str]' = None, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `TrialSpec` | class | `(trial_id: 'str', condition: 'Optional[ParadigmCondition]' = None, seed: 'int' = 0, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `TuneResult` | class | `(best_parameters: 'dict[str, float]', best_score: 'float', history: 'list[dict[str, Any]]', summary: 'dict[str, Any]', model: 'Any' = None) -> None` |
| `compute_fields` | function | `(model: "'Model'", signals: "'Signals'") -> "'FieldOutput'"` |
| `configuration` | function | `() -> 'Configuration'` |
| `connect` | function | `(*models: "'Model'", edges: "'Sequence[Mapping[str, Any]] | None'" = None, namespace: "'Sequence[str] | None'" = None, layout: 'str' = 'offset_x', strict: 'bool' = True, name: "'str | None'" = None) -> "'Model'"` |
| `construct` | function | `(cfg: "'Configuration | Any'", runtime: "'Any | None'" = None, *, geometry: "'LaminarSourceGeometry | None'" = None) -> 'Model'` |
| `dataset_spec` | function | `(**kwargs: 'Any') -> 'DatasetSpec'` |
| `default_basis_spec` | function | `() -> 'BasisSpec'` |
| `enable_x64` | function | `() -> 'dict[str, Any]'` |
| `get_signal` | function | `(obj: 'Any', key: 'str', **kwargs: 'Any') -> 'Any'` |
| `laminar_source_geometry` | function | `(populations: "Sequence['LaminarPopulation']") -> "'LaminarSourceGeometry'"` |
| `matrix_parameter` | function | `(*, mask: 'str', bounds: 'tuple', init: 'str' = 'current', trainable: 'bool' = True) -> 'MatrixParameterSpec'` |
| `migrate_schema` | function | `(meta: 'dict[str, Any]') -> 'dict[str, Any]'` |
| `objective` | function | `() -> 'Objective'` |
| `operator_status` | function | `() -> 'dict[str, str]'` |
| `provenance_receipt` | function | `(branch: 'str' = 'unknown', sha: 'str' = 'unknown', dirty: 'bool' = False) -> 'dict[str, Any]'` |
| `rate_targets` | function | `(groups: 'dict[str, Any]', targets_hz: 'dict[str, float]', weights: 'Optional[dict[str, float]]' = None) -> 'Objective'` |
| `readout_spec` | function | `(name: 'str', metric: 'str', *, time_window_ms: 'Optional[tuple[float, float]]' = None, n_contacts_slice: 'Optional[tuple[int, int]]' = None, metadata: 'Optional[dict[str, Any]]' = None) -> 'ReadoutSpec'` |
| `run_receipt` | function | `(model: "'Model'", signals: 'Signals', *, tags: 'Optional[dict[str, Any]]' = None) -> 'RunReceipt'` |
| `run_trials` | function | `(model: 'Model', batch: 'TrialBatch', sim: 'Simulation', *, collect_errors: 'bool' = False) -> 'TrialBatchResult'` |
| `runtime` | function | `(backend: 'str' = 'auto', dtype: 'str' = 'float32', jit: 'bool' = False, vmap: 'bool' = False, precision: 'str' = 'default', seed: 'int' = 0, n_steps: 'int' = 0, recurrent_backend: 'str' = 'dense', synaptic_kernel: 'str' = 'exponential', device_type: 'str | None' = None, dtype_primary: 'str | None' = None, x64_enabled: 'bool | None' = None) -> 'RuntimeConfig'` |
| `runtime_report` | function | `(runtime_config: 'RuntimeConfig | None' = None) -> 'dict[str, Any]'` |
| `simulate` | function | `(model: 'Model', sim: 'Optional[Simulation]' = None, paradigm: 'Optional[Any]' = None, **kwargs: 'Any') -> 'Signals'` |
| `simulation` | function | `(**kwargs: 'Any') -> 'Simulation'` |
| `standard_visual_omission` | function | `() -> 'Paradigm'` |
| `stimulus_schedule` | function | `(events: 'Sequence[Any]', n_neurons: 'int', *, drive_amplitude: 'float' = 5.0, event_duration_ms: 'float' = 50.0) -> 'StimulusSchedule'` |
| `suite2_celltype_presets` | function | `() -> 'dict[str, dict[str, float | str]]'` |
| `suite2_four_celltype_config` | function | `(*, seed: 'int' = 7, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1) -> 'Configuration'` |
| `suite2_net1_config` | function | `(*, seed: 'int' = 7, n: 'int' = 100, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, drives: 'Mapping[str, float] | None' = None) -> 'Configuration'` |
| `suite2_run_bundle` | function | `(model: 'Model', *, seed: 'int' = 7, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, noise_amplitude: 'float | None' = None) -> 'dict[str, Any]'` |
| `suite2_simulation` | function | `(*, seed: 'int' = 7, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, noise_amplitude: 'float | None' = None, noise_rate_hz: 'float' = 2.0, target: 'str' = 'all', jit: 'bool' = False, recurrent_backend: 'str' = 'dense') -> 'Simulation'` |
| `suite2_single_neuron_config` | function | `(*, seed: 'int' = 7, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, cell_type: 'str' = 'E') -> 'Configuration'` |
| `suite2_tune_noise_agsdr_adam` | function | `(model: 'Model', *, simulation: 'Simulation | None' = None, target_rate_hz: 'tuple[float, float]' = (5.0, 10.0), amplitudes: 'Sequence[float]' = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0), poisson_rate_hz: 'float' = 2.0, adam_steps: 'int' = 2, learning_rate: 'float' = 0.2, finite_difference_eps: 'float' = 0.05, seed: 'int' = 7) -> 'TuneResult'` |
| `suite2_v1_v4_config` | function | `(*, seed: 'int' = 7, n_per_area: 'int' = 400, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, v1_layer_cell_types: 'Mapping[str, Mapping[str, float]] | None' = None, v4_layer_cell_types: 'Mapping[str, Mapping[str, float]] | None' = None) -> 'Configuration'` |
| `surrogate_config` | function | `(**kwargs: 'Any') -> 'SurrogateConfig'` |
| `trial_batch` | function | `(conditions: 'Sequence[ParadigmCondition]', n_reps: 'int' = 1, seed: 'int' = 0, seed_policy: 'str' = 'paired_by_replicate', batch_id: 'Optional[str]' = None, metadata: 'Optional[dict[str, Any]]' = None) -> 'TrialBatch'` |
| `with_emitter_parameters` | function | `(model: 'Model', *, a: "'float | None'" = None, b: "'float | None'" = None, c: "'float | None'" = None, d: "'float | None'" = None, drive_scale: "'float | None'" = None, a_per_neuron: "'jax.Array | None'" = None, b_per_neuron: "'jax.Array | None'" = None, c_per_neuron: "'jax.Array | None'" = None, d_per_neuron: "'jax.Array | None'" = None, drive_per_neuron: "'jax.Array | None'" = None) -> 'Model'` |

## `jaxfne.emitters` (22)

| Name | Kind | Signature |
|---|---|---|
| `EIGNetwork` | class | `(params: 'IzhikevichParams', positions: 'jax.Array', metadata: 'dict') -> None` |
| `EdgeList` | class | `(pre: 'jax.Array', post: 'jax.Array', weight: 'jax.Array', receptor_index: 'jax.Array', tau_ms: 'jax.Array', source_calibration_status: 'str' = 'uncalibrated_izhikevich_native_current') -> None` |
| `Emitter` | class | `()` |
| `GLIFEmitter` | class | `(*args, **kwargs)` |
| `IzhikevichEmitter` | class | `(n: 'int | None' = None, *, n_neurons: 'int | None' = None, dtype: 'str' = 'float32', cell_type_fractions: 'Mapping[str, float] | None' = None)` |
| `IzhikevichParams` | class | `(a: 'jax.Array', b: 'jax.Array', c: 'jax.Array', d: 'jax.Array', drive: 'jax.Array', sign: 'jax.Array', W: 'jax.Array', v0: 'jax.Array', u0: 'jax.Array', source_scale: 'jax.Array', labels: 'tuple[str, ...]', layer_labels: 'tuple[str, ...] | None' = None, source_calibration_status: 'str' = 'uncalibrated_izhikevich_native_current') -> None` |
| `LIFEmitter` | class | `(*args, **kwargs)` |
| `ReceptorSpec` | class | `(name: 'str', receptor_index: 'int', sign: 'int', tau_ms: 'float', reversal_mV: 'float | None', source_calibration_status: 'str' = 'metadata_only_uncalibrated', claim_level: 'str' = 'computational_scaffold') -> None` |
| `SynapseLayer` | class | `(n: 'int', W: 'jax.Array', tau_ms: 'float' = 5.0, dtype: 'str' = 'float32')` |
| `SynapseSpec` | class | `(receptors: 'tuple[ReceptorSpec, ...]', source_calibration_status: 'str' = 'metadata_only_uncalibrated', physical_amplitude_calibrated: 'bool' = False) -> None` |
| `SynapseState` | class | `(trace: ForwardRef('jax.Array'))` |
| `izhikevich_params_from_labels` | function | `(labels: 'tuple[str, ...] | list[str]', *, layer_labels: 'tuple[str, ...] | list[str] | None' = None, dtype: 'str' = 'float32', drive_overrides: 'Mapping[str, float] | None' = None, source_scale: 'float' = 1.0) -> 'IzhikevichParams'` |
| `make_edge_list_from_dense` | function | `(weights: 'jax.Array', *, threshold: 'float' = 1e-12, dtype: 'str' = 'float32') -> 'EdgeList'` |
| `make_eig_network` | function | `(n: 'int' = 128, cell_type_fractions: 'Mapping[str, float] | None' = None, *, dtype: 'str' = 'float32') -> 'EIGNetwork'` |
| `simulate_edge_recurrent_izhikevich` | function | `(params: 'IzhikevichParams', edges: 'EdgeList', n_steps: 'int', dt_ms: 'float', key: 'jax.Array', *, dtype: 'str' = 'float32', drive_schedule: "'jax.Array | None'" = None, silence_mask: "'jax.Array | None'" = None, noise_scale: "'jax.Array | float | None'" = None) -> 'tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]'` |
| `simulate_eig_izhikevich` | function | `(params: 'IzhikevichParams', n_steps: 'int', dt_ms: 'float', key: 'jax.Array', *, dtype: 'str' = 'float32', drive_schedule: "'jax.Array | None'" = None, silence_mask: "'jax.Array | None'" = None, noise_scale: "'jax.Array | float | None'" = None) -> 'tuple[jax.Array, jax.Array, jax.Array]'` |
| `simulate_receptor_exponential_izhikevich` | function | `(params: 'IzhikevichParams', edges: 'EdgeList', n_steps: 'int', dt_ms: 'float', key: 'jax.Array', *, dtype: 'str' = 'float32', drive_schedule: "'jax.Array | None'" = None, silence_mask: "'jax.Array | None'" = None) -> 'tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]'` |
| `standard_receptor_specs` | function | `() -> 'dict[str, ReceptorSpec]'` |
| `standard_receptor_tau_table` | function | `(dtype: 'str' = 'float32') -> 'jax.Array'` |
| `synaptic_current_tensor` | function | `(spikes_pre: 'jax.Array', tau_ms: 'jax.Array', dt_ms: 'float') -> 'jax.Array'` |
| `synaptic_tau_from_mechanism` | function | `(mechanism: 'Sequence[str]', *, dtype: 'str' = 'float32') -> 'jax.Array'` |
| `synaptic_tensor_report` | function | `(tau_ms: 'jax.Array', mechanism: "'Sequence[str] | None'" = None) -> 'dict[str, Any]'` |

## `jaxfne.experimental_hpc.contracts` (2)

| Name | Kind | Signature |
|---|---|---|
| `NodeIdentity` | class | `(global_id: 'int', area: 'str', area_id: 'str', local_id: 'int', layer: 'str', cell_type: 'str') -> None` |
| `SelectorSpec` | class | `(area: 'Optional[str]' = None, area_id: 'Optional[str]' = None, layer: 'Optional[str]' = None, cell_type: 'Optional[str]' = None, ids: 'Optional[tuple[int, ...]]' = None) -> None` |

## `jaxfne.export` (4)

| Name | Kind | Signature |
|---|---|---|
| `export_report` | function | `(output_dir: 'str | Path', manifest: 'Optional[Mapping]' = None, metrics: 'Optional[Mapping]' = None, validation: 'Optional[Mapping]' = None, figures: 'Optional[Mapping[str, object]]' = None, dpi: 'int' = 150) -> 'Mapping[str, str]'` |
| `export_tutorial_artifacts` | function | `(output_dir: 'str | Path', manifest: 'Optional[Mapping]' = None, metrics: 'Optional[Mapping]' = None, validation: 'Optional[Mapping]' = None) -> 'Mapping[str, str]'` |
| `save_figure` | function | `(fig, path: 'str | Path', dpi: 'int' = 150, bbox_inches: 'str' = 'tight') -> 'str'` |
| `save_figures` | function | `(figures: 'Mapping[str, object]', output_dir: 'str | Path', dpi: 'int' = 150, prefix: 'str' = '', suffix: 'str' = '') -> 'Mapping[str, str]'` |

## `jaxfne.fields.diagnostics` (1)

| Name | Kind | Signature |
|---|---|---|
| `validate_projection_invariants` | function | `(*, sources: 'jax.Array', positions: 'jax.Array', kernel: 'jax.Array', source_proxy: 'jax.Array', phi_e_proxy: 'jax.Array', csd_proxy: 'jax.Array', lfp_proxy: 'jax.Array', mode: 'str' = 'row_normalize') -> 'dict[str, Any]'` |

## `jaxfne.fields.probes` (3)

| Name | Kind | Signature |
|---|---|---|
| `eeg_proxy_transform` | function | `(source: 'jax.Array', leadfield: 'jax.Array') -> 'jax.Array'` |
| `emm_proxy_transform` | function | `(spike_rate: 'jax.Array', source: 'jax.Array', field_potential: 'jax.Array', lambda_spk: 'float' = 1.0, lambda_src: 'float' = 1.0, lambda_field: 'float' = 1.0) -> 'jax.Array'` |
| `meg_proxy_transform` | function | `(source_oriented: 'jax.Array', leadfield: 'jax.Array') -> 'jax.Array'` |

## `jaxfne.fields.proxy` (12)

| Name | Kind | Signature |
|---|---|---|
| `FieldOutput` | class | `(source_proxy: 'jax.Array', phi_e_proxy: 'jax.Array', csd_proxy: 'jax.Array', lfp_proxy: 'jax.Array', kernel: 'jax.Array', contact_depths: 'jax.Array', diagnostics: 'dict[str, Any]') -> None` |
| `LinearReadout` | class | `(name: 'str', W: 'jax.Array', leadfield_status: 'str' = 'toy_or_declared_proxy', operator_status: 'str' = 'simulated_proxy', units_or_status: 'str' = 'relative_proxy_units') -> None` |
| `cable_filter_report` | function | `(tau_s: 'jax.Array', order: 'int' = 2) -> 'dict[str, Any]'` |
| `cable_filter_sources` | function | `(sources: 'jax.Array', tau_s: 'jax.Array', dt_ms: 'float', *, order: 'int' = 2) -> 'jax.Array'` |
| `cable_filter_tau` | function | `(cell_type: 'Sequence[str] | np.ndarray', depth_z: 'jax.Array', *, tau_e_superficial_ms: 'float' = 1.0, tau_e_deep_ms: 'float' = 5.0, tau_pv_ms: 'float' = 0.5, tau_sst_ms: 'float' = 2.0, tau_vip_ms: 'float' = 2.0) -> 'jax.Array'` |
| `compute_conservation_proxy_diagnostics` | function | `(*, source: 'jax.Array | None' = None, phi_e: 'jax.Array | None' = None, csd: 'jax.Array | None' = None, lfp: 'jax.Array | None' = None, field_solution: 'FieldOutput | None' = None, source_calibration_status: 'str' = 'uncalibrated_izhikevich_native_current', field_solver_status: 'str' = 'linear_solver', field_claim_level: 'str' = 'proxy_readout') -> 'dict[str, Any]'` |
| `construct_source_tensor` | function | `(*, mode: 'str' = 'total_membrane_current_proxy', total_membrane_current: 'jax.Array | None' = None, decomposed_cap_ion: 'jax.Array | None' = None, synaptic_current: 'jax.Array | None' = None, spike_proxy: 'jax.Array | None' = None, scale: 'float' = 1.0) -> 'tuple[jax.Array, dict[str, Any]]'` |
| `csd_tensor` | function | `(phi_e_proxy: 'jax.Array', dz: 'jax.Array | float') -> 'jax.Array'` |
| `probe_laminar_modes` | function | `(field_output: 'FieldOutput', modes: 'Sequence[str]' = ('source', 'phi_e', 'CSD', 'LFP')) -> 'dict[str, Any]'` |
| `project_laminar_sources` | function | `(sources: 'jax.Array', positions: 'jax.Array', *, n_contacts: 'int' = 16, width: 'float' = 0.1, mode: 'str' = 'density_preserving', dtype: 'str' = 'float32') -> 'FieldOutput'` |
| `project_sources_to_laminar_field` | function | `(sources: 'jax.Array', positions: 'jax.Array', n_contacts: 'int' = 16, *, mode: 'str' = 'density_preserving', dtype: 'str' = 'float32') -> 'FieldOutput'` |
| `validate_source_field_status` | function | `(field_output: 'FieldOutput | None' = None, cfg_metadata: 'Mapping[str, Any] | None' = None, *, requested_modes: 'Sequence[str] | None' = None) -> 'dict[str, Any]'` |

## `jaxfne.geometry` (1)

| Name | Kind | Signature |
|---|---|---|
| `make_ei_cloud_network` | function | `(n_neurons: 'int' = 100, seed: 'int' = 42) -> 'Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]'` |

## `jaxfne.io` (10)

| Name | Kind | Signature |
|---|---|---|
| `asset_hashes` | function | `(assets: 'dict[str, str | Path]') -> 'dict[str, str]'` |
| `config_hash` | function | `(cfg: 'Any') -> 'str'` |
| `json_safe` | function | `(obj: 'Any') -> 'Any'` |
| `manifest` | function | `(cfg: 'Any', signals: 'Optional[Any]' = None, readout: 'Optional[dict[str, Any]]' = None, runtime_config: 'Optional[Any]' = None, paradigm: 'Optional[dict[str, Any]]' = None, objective: 'Optional[dict[str, Any]]' = None, evaluation: 'Optional[dict[str, Any]]' = None, tuning: 'Optional[dict[str, Any]]' = None, dataset: 'Optional[dict[str, Any]]' = None) -> 'dict[str, Any]'` |
| `probe_report` | function | `(n_probes: 'int', probe_types: 'dict[str, int] | None' = None, metadata: 'dict[str, Any] | None' = None) -> 'dict[str, Any]'` |
| `save_json` | function | `(obj: 'Any', path: 'str | Path') -> 'None'` |
| `save_receipt` | function | `(receipt: 'Any', path: "'str | Path'", *, overwrite: 'bool' = False) -> 'None'` |
| `sha256_file` | function | `(path: 'str | Path') -> 'str'` |
| `sha256_text` | function | `(text: 'str') -> 'str'` |
| `validation_report` | function | `(config_valid: 'bool', issues: 'list[str] | None' = None, metadata: 'dict[str, Any] | None' = None) -> 'dict[str, Any]'` |

## `jaxfne.neuronal_tensor` (21)

| Name | Kind | Signature |
|---|---|---|
| `Area` | class | `(name: 'str', layers: 'Sequence[Layer]' = <factory>, inter_connections: 'Sequence[InterConnection]' = <factory>, pose: 'Pose3D' = <factory>) -> None` |
| `AreaConnection` | class | `(source_area: 'str', source_layer: 'str', source_neuron_type: 'str', target_area: 'str', target_layer: 'str', target_neuron_type: 'str', mechanism: 'str' = 'monotonic_cable_synapse', static: 'StaticParams' = <factory>, plastic: 'PlasticParams' = <factory>) -> None` |
| `Geometry3D` | class | `(distribution: 'str' = 'uniform_random', x_range: 'tuple[float, float]' = (0.0, 1.0), y_range: 'tuple[float, float]' = (0.0, 1.0), z_range: 'tuple[float, float]' = (0.0, 1.0), value_tag: 'ValueTag' = 'relative') -> None` |
| `InterConnection` | class | `(source_layer: 'str', source_neuron_type: 'str', target_layer: 'str', target_neuron_type: 'str', mechanism: 'str', static: 'StaticParams' = <factory>, plastic: 'PlasticParams' = <factory>) -> None` |
| `Layer` | class | `(name: 'str', neuron_types: 'Sequence[NeuronType]' = <factory>, geometry: 'Geometry3D' = <factory>, n_neurons: 'int' = 0) -> None` |
| `NeuronType` | class | `(name: 'str', relative_size: 'float' = 1.0, fraction: 'Optional[float]' = None, value_tag: 'ValueTag' = 'relative') -> None` |
| `NeuronalTensor` | class | `(areas: 'Sequence[Area]' = <factory>, area_connections: 'Sequence[AreaConnection]' = <factory>, name: 'str' = 'untitled') -> None` |
| `PlasticParams` | class | `(w_mech: 'float' = 1.0, H: 'float' = 0.0, value_tag: 'ValueTag' = 'relative') -> None` |
| `Pose3D` | class | `(plane: 'Plane' = 'xy', rotation_deg: 'float' = 0.0, translation: 'tuple[float, float, float]' = (0.0, 0.0, 0.0), value_tag: 'ValueTag' = 'relative') -> None` |
| `RuntimeConfiguration` | class | `(duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, seed: 'int' = 0, dtype: 'str' = 'float32', emitter: 'str' = 'izhikevich', device: 'str' = 'auto', jit: "'bool | str'" = False, vmap: "'bool | str'" = False, solver: "'str | None'" = None, probes: "'Sequence[str] | None'" = None, n_contacts: 'int' = 16, outputs: "'dict | None'" = None, optimizer: "'Any | None'" = None) -> None` |
| `StaticParams` | class | `(g_mech: 'dict' = <factory>, reversal_potentials_mV: 'dict' = <factory>, dT_ms: 'float' = 0.1, value_tag: 'ValueTag' = 'relative') -> None` |
| `configs_dir` | function | `() -> 'Path'` |
| `construct_neuronal_tensor` | function | `(tensor: 'NeuronalTensor', *, seed: 'int' = 0, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, emitter: 'str' = 'izhikevich') -> 'Model'` |
| `default_relative_size` | function | `(neuron_type: 'str') -> 'float'` |
| `list_canonical_neuronal_tensors` | function | `() -> 'list[str]'` |
| `load` | function | `(path: 'str | Path') -> 'NeuronalTensor'` |
| `load_canonical_neuronal_tensor` | function | `(name: 'str') -> 'NeuronalTensor'` |
| `load_neuronal_tensor` | function | `(path: 'str | Path') -> 'NeuronalTensor'` |
| `merge_neuronal_tensors` | function | `(tensors: 'Sequence[NeuronalTensor]', poses: 'Optional[Sequence[Pose3D]]' = None, *, name: 'str' = 'merged') -> 'NeuronalTensor'` |
| `neuronal_tensor_to_configuration` | function | `(tensor: 'NeuronalTensor', *, seed: 'int' = 0, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, emitter: 'str' = 'izhikevich') -> 'Configuration'` |
| `save_neuronal_tensor` | function | `(tensor: 'NeuronalTensor', path: 'str | Path') -> 'str'` |

## `jaxfne.optim.agsdr` (1)

| Name | Kind | Signature |
|---|---|---|
| `AGSDRState` | class | `(step: 'int' = 0, best_loss: 'float' = inf, best_param: 'Optional[Any]' = None, reset_counter: 'int' = 0, var_sup_ema: 'float' = 0.0, var_unsup_ema: 'float' = 0.0, ema_decay: 'float' = 0.99, deselection_counter: 'int' = 0, alpha_adaptive: 'float' = 0.7) -> None` |

## `jaxfne.optim.core` (13)

| Name | Kind | Signature |
|---|---|---|
| `AGSDR` | class | `(alpha: 'float' = 0.7, exploration: 'float' = 0.05, deselect_factor: 'float' = 2.0) -> None` |
| `AGSDROptimizerSpec` | class | `(parameters: 'dict', generations: 'int' = 8, population_size: 'int' = 6, alpha: 'float' = 0.65, exploration: 'float' = 0.18, deselect_factor: 'float' = 2.0, seed: 'int' = 0, inner_optimizer: 'Any' = None, inner_steps: 'int' = 0, inner_objective: 'Optional[str]' = None) -> None` |
| `OptimizerSpec` | class | `(optimizer: 'str', optimizer_class: 'str', differentiability_status: 'str', surrogate_status: 'str', alpha: 'float' = 0.7, exploration: 'float' = 0.05, deselect_factor: 'float' = 2.0, learning_rate: 'Optional[float]' = None, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `agsdr` | function | `(alpha: 'float' = 0.7, exploration: 'float' = 0.05, deselect_factor: 'float' = 2.0, metadata: 'Optional[dict[str, Any]]' = None, parameters: 'Optional[dict]' = None, generations: 'Optional[int]' = None, population_size: 'Optional[int]' = None, seed: 'int' = 0, inner_optimizer: 'Any' = None, inner_steps: 'int' = 0, inner_objective: 'Optional[str]' = None) -> 'Any'` |
| `agsdr_transform` | function | `(inner_optimizer: 'Optional[Any]' = None, stochastic_scale: 'float' = 0.1, global_scale: 'float' = 1.0, checkpoint_n_steps: 'int' = 50, deselection_threshold: 'int' = 10, epsilon: 'float' = 1e-06, alpha_min: 'float' = 0.0, alpha_max: 'float' = 1.0) -> 'Any'` |
| `gsdr` | function | `(alpha: 'float' = 0.7, exploration: 'float' = 0.05, deselect_factor: 'float' = 2.0, metadata: 'Optional[dict[str, Any]]' = None) -> 'OptimizerSpec'` |
| `gsdr_transform` | function | `(inner_optimizer: 'Optional[Any]' = None, stochastic_scale: 'float' = 0.1, checkpoint_n_steps: 'int' = 50, deselection_threshold: 'int' = 10) -> 'Any'` |
| `gsgd` | function | `(learning_rate: 'float' = 0.01, differentiability_status: 'str' = 'not_checked', surrogate_status: 'str' = 'none', metadata: 'Optional[dict[str, Any]]' = None) -> 'OptimizerSpec'` |
| `optax_adam` | function | `(learning_rate: 'float' = 0.001, differentiability_status: 'str' = 'not_checked', surrogate_status: 'str' = 'none', metadata: 'Optional[dict[str, Any]]' = None) -> 'OptimizerSpec'` |
| `optax_sgd` | function | `(learning_rate: 'float' = 0.001, differentiability_status: 'str' = 'not_checked', surrogate_status: 'str' = 'none', metadata: 'Optional[dict[str, Any]]' = None) -> 'OptimizerSpec'` |
| `random_search` | function | `(metadata: 'Optional[dict[str, Any]]' = None) -> 'OptimizerSpec'` |
| `require_optax` | function | `() -> 'Any'` |
| `sdr_transform` | function | `(inner_optimizer: 'Optional[Any]' = None, stochastic_scale: 'float' = 0.1, checkpoint_n_steps: 'int' = 50, alpha_min: 'float' = 0.0, alpha_max: 'float' = 1.0) -> 'Any'` |

## `jaxfne.optim.gsdr` (1)

| Name | Kind | Signature |
|---|---|---|
| `GSDRState` | class | `(step: 'int' = 0, best_loss: 'float' = inf, best_param: 'Optional[Any]' = None, reset_counter: 'int' = 0, var_sup_ema: 'float' = 0.0, var_unsup_ema: 'float' = 0.0, ema_decay: 'float' = 0.99, deselection_counter: 'int' = 0) -> None` |

## `jaxfne.optim.gsgd` (2)

| Name | Kind | Signature |
|---|---|---|
| `GSGDState` | class | `(count: ForwardRef('jnp.ndarray'), step_size: ForwardRef('jnp.ndarray'))` |
| `step_gsgd_transform` | function | `(u_t: 'jnp.ndarray', grad_l: 'jnp.ndarray', state: 'Any', hyperparams: 'dict') -> 'tuple[jnp.ndarray, Any]'` |

## `jaxfne.optim.sdr` (1)

| Name | Kind | Signature |
|---|---|---|
| `SDRState` | class | `(step: 'int' = 0, best_loss: 'float' = inf, best_param: 'Optional[Any]' = None, reset_counter: 'int' = 0, var_sup_ema: 'float' = 0.0, var_unsup_ema: 'float' = 0.0, ema_decay: 'float' = 0.99) -> None` |

## `jaxfne.paradigm` (8)

| Name | Kind | Signature |
|---|---|---|
| `Paradigm` | class | `(name: 'str' = 'none', blocks: 'list[dict[str, Any]]' = <factory>, conditions: 'tuple[ParadigmCondition, ...]' = (), comparison_code: 'int' = 101, comparison_label: 'str' = 'p1', pre_stimulus_buffer_ms: 'float' = 1000.0, analysis_windows: 'dict[str, tuple[float, float]]' = <factory>, event_codes: 'dict[str, int]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `ParadigmCondition` | class | `(name: 'str', sequence: 'tuple[str, ...]', omission_position: 'Optional[str]' = None, probability: 'Optional[float]' = None, condition_numbers: 'tuple[int, ...]' = (), events: 'tuple[ParadigmEvent, ...]' = (), metadata: 'dict[str, Any]' = <factory>) -> None` |
| `ParadigmEvent` | class | `(label: 'str', onset_ms: 'Optional[float]' = None, duration_ms: 'Optional[float]' = None, code: 'Optional[int]' = None, stimulus: 'Optional[str]' = None, is_omission: 'bool' = False, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `coop_omission_oddball_paradigm` | function | `(duration_ms: 'float' = 10000.0, dt_ms: 'float' = 0.5, freq_hz: 'float' = 6.0, omission_prob: 'float' = 0.1, standard_amplitude: 'float' = 5.0, pre_stimulus_buffer_ms: 'float' = 200.0, target_indices: 'Optional[Sequence[int]]' = None, seed: 'int' = 42, name: 'str' = 'coop_omission_oddball') -> 'Paradigm'` |
| `general_delayed_match_to_sample_paradigm` | function | `(*, name: 'str' = 'general_delayed_match_to_sample', event_windows: 'Optional[Mapping[str, Sequence[float]]]' = None, event_codes: 'Optional[Mapping[str, Sequence[float]]]' = None, sequence_event_labels: 'Optional[Sequence[str]]' = None, conditions: 'Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]]' = None, presentations: 'Optional[Mapping[str, Any]]' = None, static_events: 'Optional[Mapping[str, Any]]' = None, blocks: 'Optional[Sequence[Mapping[str, Any]]]' = None, analysis_windows: 'Optional[Mapping[str, Sequence[float]]]' = None, comparison_label: 'str' = 'sample', comparison_code: 'int' = 101, pre_stimulus_buffer_ms: 'float' = 1000.0, omission_tokens: 'Sequence[str]' = ('X', 'omit', 'omission'), metadata: 'Optional[dict[str, Any]]' = None) -> 'Paradigm'` |
| `general_sequential_oddball_paradigm` | function | `(*, name: 'str' = 'general_sequential_oddball', event_windows: 'Optional[Mapping[str, Sequence[float]]]' = None, event_codes: 'Optional[Mapping[str, Sequence[float]]]' = None, sequence_event_labels: 'Optional[Sequence[str]]' = None, conditions: 'Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]]' = None, presentations: 'Optional[Mapping[str, Any]]' = None, static_events: 'Optional[Mapping[str, Any]]' = None, blocks: 'Optional[Sequence[Mapping[str, Any]]]' = None, analysis_windows: 'Optional[Mapping[str, Sequence[float]]]' = None, comparison_label: 'str' = 'p1', comparison_code: 'int' = 101, pre_stimulus_buffer_ms: 'float' = 1000.0, omission_tokens: 'Sequence[str]' = ('X', 'omit', 'omission'), metadata: 'Optional[dict[str, Any]]' = None) -> 'Paradigm'` |
| `omission_oddball_paradigm` | function | `(standard_onset_ms: 'float' = 500.0, standard_duration_ms: 'float' = 100.0, deviant_onset_ms: 'Optional[float]' = None, deviant_duration_ms: 'float' = 100.0, deviant_label: 'str' = 'deviant', omission_position: 'str' = 'standard', pre_stimulus_buffer_ms: 'float' = 200.0, post_stimulus_buffer_ms: 'float' = 500.0, name: 'str' = 'omission_oddball') -> 'Paradigm'` |
| `paradigm` | function | `(*args, **kwargs)` |

## `jaxfne.plasticity` (4)

| Name | Kind | Signature |
|---|---|---|
| `STDPPlasticityConfig` | class | `(A_plus: 'float' = 0.01, A_minus: 'float' = 0.012, tau_plus: 'float' = 20.0, tau_minus: 'float' = 20.0, w_min: 'float' = 0.0, w_max: 'float' = 1.5) -> None` |
| `STDPState` | class | `(W: 'jnp.ndarray', trace_pre: 'jnp.ndarray', trace_post: 'jnp.ndarray') -> None` |
| `summarize_stdp_adaptation` | function | `(W_before: 'np.ndarray', W_after: 'np.ndarray') -> 'Dict[str, Any]'` |
| `update_stdp_weights_jax` | function | `(W: 'jax.Array', trace_pre: 'jax.Array', trace_post: 'jax.Array', spiked: 'jax.Array', exc_mask: 'jax.Array', A_plus: 'float', A_minus: 'float', plasticity_scale: 'float', w_min: 'float', w_max: 'float') -> 'jax.Array'` |

## `jaxfne.pynwb_compat` (2)

| Name | Kind | Signature |
|---|---|---|
| `read_nwb` | function | `(*args, **kwargs)` |
| `write_nwb` | function | `(*args, **kwargs)` |

## `jaxfne.sanity_delta` (7)

| Name | Kind | Signature |
|---|---|---|
| `BackupState` | class | `(paradigm: 'HierarchicalOddballParadigm', time_ms: 'float', vm: 'jnp.ndarray', recovery: 'jnp.ndarray', synapse_traces: 'dict[str, jnp.ndarray]', plasticity_traces: 'dict[str, jnp.ndarray]', weights: 'jnp.ndarray', task_state: 'dict[str, Any]', fixation_counter: 'int', reward_state: 'dict[str, Any]', prng_key: 'jax.Array', history_buffer: 'jnp.ndarray', runtime_metadata: 'dict[str, Any]') -> None` |
| `BehaviorGate` | class | `(paradigm: 'HierarchicalOddballParadigm', area: 'str', layer_group: 'str', target_rate_hz: 'float', tolerance_hz: 'float', window_ms: 'float') -> None` |
| `HierarchicalOddballParadigm` | class | `(config: 'SanityDeltaConfig', name: 'str', sequence: 'tuple[str, ...]', prefix_ms: 'float', fixation_required_ms: 'float', presentation_ms: 'float', delay_ms: 'float', review_ms: 'float') -> None` |
| `Manifest` | class | `(config: 'SanityDeltaConfig', paradigm: 'HierarchicalOddballParadigm', backup: 'BackupState', episode_metadata: 'dict[str, Any]', generated_at_utc: 'str', strict_json: 'bool' = True) -> None` |
| `SanityDeltaConfig` | class | `(seed: 'int', duration_ms: 'float', dt_ms: 'float', neurons_per_area: 'int', areas: 'tuple[str, ...]', hierarchy: 'tuple[tuple[str, str], ...]', cell_counts: 'dict[str, int]', stimulus_frequency_hz: 'float', stimulus_map: 'dict[str, dict[str, float]]', claim_level: "Literal['computational_scaffold']" = 'computational_scaffold', field_solver_status: "Literal['linear_solver']" = 'linear_solver', physical_amplitude_calibrated: 'bool' = False, biological_learning_claim: 'bool' = False, runtime_mode: "Literal['scaffold', 'full']" = 'scaffold') -> None` |
| `SanityDeltaModel` | class | `(config: 'SanityDeltaConfig', model_state: 'dict[str, Any]', n_neurons: 'int', n_steps: 'int', plasticity_enabled: 'bool' = False, plasticity_config: 'dict[str, Any] | None' = None) -> None` |
| `TaskEpisode` | class | `(config: 'SanityDeltaConfig', paradigm: 'HierarchicalOddballParadigm', model: 'SanityDeltaModel', backup: 'BackupState', spikes: 'jnp.ndarray', vm: 'jnp.ndarray', recovery_arr: 'jnp.ndarray', synapse_arr: 'jnp.ndarray', source_terms: 'dict[str, jnp.ndarray]', t_ms: 'jnp.ndarray', task_schedule: 'dict[str, Any]', runtime_mode: 'str' = 'scaffold', final_weights: 'Optional[np.ndarray]' = None, initial_weights: 'Optional[np.ndarray]' = None, segment_weights: 'Optional[dict[str, np.ndarray]]' = None, clip_count: 'int' = 0, updated_connection_count: 'int' = 0, excitatory_sign_preserved: 'bool' = True, inhibitory_sign_preserved: 'bool' = True, inhibitory_modified: 'bool' = False) -> None` |

## `jaxfne.sharding_utils` (4)

| Name | Kind | Signature |
|---|---|---|
| `get_sharding_context` | function | `() -> 'Optional[dict]'` |
| `make_candidate_sharding` | function | `(mesh: 'Mesh') -> 'NamedSharding'` |
| `make_population_mesh` | function | `() -> 'Optional[Mesh]'` |
| `make_replicated_sharding` | function | `(mesh: 'Mesh') -> 'NamedSharding'` |

## `jaxfne.solvers` (7)

| Name | Kind | Signature |
|---|---|---|
| `DiffraxSolver` | class | `(dt: float, rtol: float = 0.001, atol: float = 1e-06, solver_type: Optional[str] = None)` |
| `EulerSolver` | class | `(dt: float)` |
| `SolverConfig` | class | `(method: str = 'euler', dt: float = 0.1, rtol: float = 0.001, atol: float = 1e-06, solver_type: Optional[str] = None) -> None` |
| `euler_scan` | function | `(y_init: jax.Array, t_start: float, dt: float, n_steps: int, dydt_fn: Callable[[jax.Array, float], jax.Array]) -> Tuple[jax.Array, jax.Array]` |
| `euler_step` | function | `(y: jax.Array, t: float, dt: float, dydt_fn: Callable[[jax.Array, float], jax.Array]) -> jax.Array` |
| `solve_ode` | function | `(config: jaxfne.solvers.SolverConfig, dydt_fn: Callable[[jax.Array, float], jax.Array], y_init: jax.Array, t_start: float, t_end: float) -> Tuple[jax.Array, jax.Array]` |
| `solve_volume_conductor_experimental` | function | `(*args, **kwargs)` |

## `jaxfne.stimulus` (1)

| Name | Kind | Signature |
|---|---|---|
| `triangular_drive` | function | `(duration_ms: 'float', dt_ms: 'float', freq_hz: 'float' = 6.0, amplitude: 'float' = 5.0) -> 'jnp.ndarray'` |

## `jaxfne.streaming` (1)

| Name | Kind | Signature |
|---|---|---|
| `run_stdp_stream` | function | `(v_init: 'jnp.ndarray', u_init: 'jnp.ndarray', s_init: 'jnp.ndarray', stdp_state: 'STDPState', stim_drive: 'jnp.ndarray', noise: 'jnp.ndarray', solver_config: 'SolverConfig', plasticity_config: 'STDPPlasticityConfig', plasticity_scale: 'float', exc_mask: 'jnp.ndarray', inh_mask: 'jnp.ndarray', a: 'jnp.ndarray', b: 'jnp.ndarray', c: 'jnp.ndarray', d: 'jnp.ndarray', chunk_size_ms: 'float' = 10000.0, downsample_factor: 'int' = 10) -> 'Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, STDPState], Dict[str, Any]]'` |

## `jaxfne.tutorial_utils` (5)

| Name | Kind | Signature |
|---|---|---|
| `build_tutorial_laminar_column` | function | `(cfg: 'LaminarColumnConfig') -> 'dict'` |
| `kappa_synchrony` | function | `(spikes: 'np.ndarray', dt_ms: 'float' = 0.1) -> 'float'` |
| `rate_synchrony_targets` | function | `(target_rate_hz: 'float' = 10.0, target_kappa_synchrony: 'float' = 0.0, rate_weight: 'float' = 1.0, synchrony_weight: 'float' = 0.25)` |
| `select_neurons` | function | `(model, area: 'str | None' = None, layer: 'str | None' = None, cell_type: 'str | None' = None) -> 'np.ndarray'` |
| `spectrolaminar_motif_score` | function | `(alpha_beta, gamma) -> 'float'` |

## `jaxfne.util` (8)

| Name | Kind | Signature |
|---|---|---|
| `configuration_diff` | function | `(a: 'Configuration', b: 'Configuration') -> 'dict[str, Any]'` |
| `merge_runtime_configs` | function | `(*cfgs: 'RuntimeConfig', **overrides: 'Any') -> 'RuntimeConfig'` |
| `model_diff` | function | `(a: 'Model', b: 'Model', *, atol: 'float' = 1e-06) -> 'dict[str, Any]'` |
| `runtime_config_diff` | function | `(a: 'RuntimeConfig', b: 'RuntimeConfig') -> 'dict[str, tuple[Any, Any]]'` |
| `tensor_summary` | function | `(nt: 'NeuronalTensor') -> 'dict[str, Any]'` |
| `validate_model` | function | `(model: 'Model', *, strict: 'bool' = False) -> 'list[str]'` |
| `validate_neuronal_tensor` | function | `(nt: 'NeuronalTensor', *, strict: 'bool' = False) -> 'list[str]'` |
| `validate_runtime_config` | function | `(cfg: 'RuntimeConfig', *, strict: 'bool' = False) -> 'list[str]'` |

## `jaxfne.validation` (2)

| Name | Kind | Signature |
|---|---|---|
| `compilation_registry` | value |  |
| `is_valid_signal` | function | `(signals: 'Any') -> 'bool'` |

## `jaxfne.vis.canonical` (1)

| Name | Kind | Signature |
|---|---|---|
| `plot_raster` | function | `(signals, model=None, *, backend: 'str' = 'plotly', title: 'Optional[str]' = None, xlabel: 'Optional[str]' = 'Time (ms)', ylabel: 'Optional[str]' = 'Neuron index', legend: 'bool' = True, colors: 'Optional[Sequence[str]]' = None, width: 'Optional[int]' = None, height: 'Optional[int]' = None, **kwargs) -> 'Any'` |

## `jaxfne.vis.fields` (1)

| Name | Kind | Signature |
|---|---|---|
| `plot_spectrolaminar_suite` | function | `(signals: 'Signals | dict[str, Any]', **kwargs: 'Any') -> 'matplotlib.figure.Figure'` |

## `jaxfne.vis.plasticity_viz` (1)

| Name | Kind | Signature |
|---|---|---|
| `plot_stdp_adaptation_suite` | function | `(trajectories: 'Dict[str, Any]', W_before: 'np.ndarray', W_after: 'np.ndarray', stimulus: 'np.ndarray', fig_dir: 'str', prefix: 'str' = '') -> 'None'` |

