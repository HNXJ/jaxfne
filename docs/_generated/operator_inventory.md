# Operator Inventory (generated)

Generated from the live `jaxfne.__all__` export surface (189 entries) by `scripts/generate_operator_inventory.py`. Deterministic dense table Operator|Input|Output|State effect|Public — do not hand-edit; regenerate after any export change.

| Operator | Input | Output | State effect | Public |
|---|---|---|---|---|
| `AGSDR` | `(alpha: 'float' = 0.7, exploration: 'float' = 0.05, deselect_factor: 'float' = 2.0)` | `None` | constructs | COMPATIBILITY |
| `AGSDROptimizerSpec` | `(parameters: 'dict', generations: 'int' = 8, population_size: 'int' = 6, alpha: 'float' = 0.65, exploration: 'float'...` | `None` | constructs | CANONICAL |
| `AGSDRState` | `(step: 'int' = 0, best_loss: 'float' = inf, best_param: 'Optional[Any]' = None, reset_counter: 'int' = 0, var_sup_em...` | `None` | constructs | CANONICAL |
| `Area` | `(name: 'str', layers: 'Sequence[Layer]' = <factory>, inter_connections: 'Sequence[InterConnection]' = <factory>, pos...` | `None` | constructs | CANONICAL |
| `AreaConnection` | `(source_area: 'str', source_layer: 'str', source_neuron_type: 'str', target_area: 'str', target_layer: 'str', target...` | `None` | constructs | CANONICAL |
| `AxisSpec` | `(name: 'str', status: 'str' = 'active', size: 'Optional[int]' = None, units_or_status: 'str' = 'declared')` | `None` | constructs | CANONICAL |
| `BasisSpec` | `(space_basis: 'str' = 'laminar_depth', time_basis: 'str' = 'continuous_ms', field_regime: 'str' = 'laminar_proxy', s...` | `None` | constructs | CANONICAL |
| `CANONICAL_LAYERS_6L` | — | — | — | CANONICAL |
| `CANONICAL_LAYER_CELL_TYPE_FRACTIONS` | — | — | — | CANONICAL |
| `CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L` | — | — | — | CANONICAL |
| `CANONICAL_Z_BANDS` | — | — | — | CANONICAL |
| `CANONICAL_Z_BANDS_5L` | — | — | — | CANONICAL |
| `CELL_TYPE_PRESETS` | — | — | — | CANONICAL |
| `Config` | `(networks: 'list[dict[str, Any]]' = <factory>, emitters: 'list[dict[str, Any]]' = <factory>, fields: 'list[dict[str,...` | `None` | constructs | COMPATIBILITY |
| `Configuration` | `(networks: 'list[dict[str, Any]]' = <factory>, emitters: 'list[dict[str, Any]]' = <factory>, fields: 'list[dict[str,...` | `None` | constructs | CANONICAL |
| `ConnectionCompileResult` | `(edge_pre: 'jax.Array', edge_post: 'jax.Array', edge_weight: 'jax.Array', edge_mechanism: 'jax.Array', edge_rule_id:...` | `None` | constructs | CANONICAL |
| `ContinuationState` | `(dynamic: DynamicState, prng_key: jax.Array, step_index: int = 0, delay_state: jax.Array | None = None)` | — | constructs | CANONICAL |
| `DEFAULT_HDP` | — | — | — | CANONICAL |
| `DEFAULT_LAYERS` | — | — | — | CANONICAL |
| `DEFAULT_SPIKE_IMPULSE_GAIN` | — | — | — | CANONICAL |
| `DatasetSpec` | `(name: 'str' = 'unnamed_dataset', modality: 'str' = 'unspecified', source_format: 'str' = 'unspecified', comparison_...` | `None` | constructs | CANONICAL |
| `DynamicState` | `(v: jax.Array, u: jax.Array, prev_spikes: jax.Array, syn_state: jax.Array, H: jax.Array, w: jax.Array)` | — | constructs | CANONICAL |
| `EdgeParameterSpec` | `(pre: 'Optional[SelectorSpec | Mapping[str, Any]]' = None, post: 'Optional[SelectorSpec | Mapping[str, Any]]' = None...` | `None` | constructs | CANONICAL |
| `Emitter` | `()` | — | constructs | CANONICAL |
| `FLAT_CELL_TYPE_FRACTIONS` | — | — | — | CANONICAL |
| `FieldOutput` | `(source_proxy: 'jax.Array', phi_e_proxy: 'jax.Array', csd_proxy: 'jax.Array', lfp_proxy: 'jax.Array', kernel: 'jax.A...` | `None` | constructs | CANONICAL |
| `GSDRState` | `(step: 'int' = 0, best_loss: 'float' = inf, best_param: 'Optional[Any]' = None, reset_counter: 'int' = 0, var_sup_em...` | `None` | constructs | CANONICAL |
| `GSGDState` | `(count: jnp.ndarray, step_size: jnp.ndarray)` | — | constructs | CANONICAL |
| `Geometry3D` | `(distribution: 'str' = 'uniform_random', x_range: 'tuple[float, float]' = (0.0, 1.0), y_range: 'tuple[float, float]'...` | `None` | constructs | CANONICAL |
| `InterConnection` | `(source_layer: 'str', source_neuron_type: 'str', target_layer: 'str', target_neuron_type: 'str', mechanism: 'str', s...` | `None` | constructs | CANONICAL |
| `IzhikevichEmitter` | `(n: 'int | None' = None, *, n_neurons: 'int | None' = None, dtype: 'str' = 'float32', cell_type_fractions: 'Mapping[...` | — | constructs | CANONICAL |
| `LaminarPopulation` | `(name: 'str', cell_type: 'str', layer: 'str', depth_min: 'float', depth_max: 'float', n_units: 'int', source_calibra...` | `None` | constructs | CANONICAL |
| `LaminarSourceGeometry` | `(populations: 'tuple[LaminarPopulation, ...]', n_units_total: 'int', position_units: 'str' = 'relative_laminar_depth...` | `None` | constructs | CANONICAL |
| `Layer` | `(name: 'str', neuron_types: 'Sequence[NeuronType]' = <factory>, geometry: 'Geometry3D' = <factory>, n_neurons: 'int'...` | `None` | constructs | CANONICAL |
| `LinearReadout` | `(name: 'str', W: 'jax.Array', leadfield_status: 'str' = 'toy_or_declared_proxy', operator_status: 'str' = 'simulated...` | `None` | constructs | CANONICAL |
| `MatrixParameterSpec` | `(mask: 'str', bounds: 'tuple', init: 'str' = 'current', trainable: 'bool' = True, target: 'str' = 'W')` | `None` | constructs | CANONICAL |
| `Model` | `(cfg: 'Configuration', params: 'dict[str, Any]', static: 'dict[str, Any]')` | `None` | constructs | CANONICAL |
| `NEURONAL_TENSOR_SCHEMA_VERSION` | — | — | — | CANONICAL |
| `Net` | `(cfg: 'Configuration', params: 'dict[str, Any]', static: 'dict[str, Any]')` | `None` | constructs | COMPATIBILITY |
| `NeuronType` | `(name: 'str', relative_size: 'float' = 1.0, fraction: 'Optional[float]' = None, value_tag: 'ValueTag' = 'relative')` | `None` | constructs | CANONICAL |
| `NeuronalTensor` | `(areas: 'Sequence[Area]' = <factory>, area_connections: 'Sequence[AreaConnection]' = <factory>, name: 'str' = 'untit...` | `None` | constructs | CANONICAL |
| `NodeIdentity` | `(global_id: 'int', area: 'str', area_id: 'str', local_id: 'int', layer: 'str', cell_type: 'str')` | `None` | constructs | CANONICAL |
| `Objective` | `(name: 'str' = 'anonymous', kind: 'str' = 'generic', losses: 'list[dict[str, Any]]' = <factory>, regularizers: 'list...` | `None` | constructs | CANONICAL |
| `ObjectiveReport` | `(objective_name: 'str', evaluation_status: 'str', total_loss: 'Optional[float]', all_gates_pass: 'bool', losses: 'tu...` | `None` | constructs | CANONICAL |
| `OptimizerSpec` | `(optimizer: 'str', optimizer_class: 'str', differentiability_status: 'str', surrogate_status: 'str', alpha: 'float' ...` | `None` | constructs | CANONICAL |
| `Paradigm` | `(name: 'str' = 'none', blocks: 'list[dict[str, Any]]' = <factory>, conditions: 'tuple[ParadigmCondition, ...]' = (),...` | `None` | constructs | CANONICAL |
| `ParadigmCondition` | `(name: 'str', sequence: 'tuple[str, ...]', omission_position: 'Optional[str]' = None, probability: 'Optional[float]'...` | `None` | constructs | CANONICAL |
| `ParadigmEvent` | `(label: 'str', onset_ms: 'Optional[float]' = None, duration_ms: 'Optional[float]' = None, code: 'Optional[int]' = No...` | `None` | constructs | CANONICAL |
| `PlasticParams` | `(w_mech: 'float' = 1.0, H: 'float' = 0.0, value_tag: 'ValueTag' = 'relative')` | `None` | constructs | CANONICAL |
| `Pose3D` | `(plane: 'Plane' = 'xy', rotation_deg: 'float' = 0.0, translation: 'tuple[float, float, float]' = (0.0, 0.0, 0.0), va...` | `None` | constructs | CANONICAL |
| `Probe` | `(name: 'str', modes: 'Sequence[str]', metadata: 'dict[str, Any]' = <factory>)` | `None` | constructs | CANONICAL |
| `PseudoGenome` | `(name: 'str', schema_version: 'str' = 'pseudogenome_v1', description: 'str' = '', areas: 'Sequence[AreaGenome]' = <f...` | `None` | constructs | CANONICAL |
| `RECEPTOR_KINETICS` | — | — | — | CANONICAL |
| `ReadoutResult` | `(spec_name: 'str', metric: 'str', value: 'Optional[float]', status: 'str' = 'computed', claim_level: 'str' = 'comput...` | `None` | constructs | CANONICAL |
| `ReadoutSpec` | `(name: 'str', metric: 'str', time_window_ms: 'Optional[tuple[float, float]]' = None, n_contacts_slice: 'Optional[tup...` | `None` | constructs | CANONICAL |
| `RunReceipt` | `(receipt_id: 'str', jaxfne_version: 'str', config_hash: 'str', simulation: 'dict[str, Any]', signals_summary: 'dict[...` | `None` | constructs | CANONICAL |
| `RuntimeConfig` | `(backend: 'str' = 'auto', dtype: 'str' = 'float32', jit: 'bool | str' = False, vmap: 'bool | str' = False, precision...` | `None` | constructs | CANONICAL |
| `RuntimeConfiguration` | `(duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, seed: 'int' = 0, dtype: 'str' = 'float32', emitter: 'str' = 'i...` | `None` | constructs | CANONICAL |
| `SDRState` | `(step: 'int' = 0, best_loss: 'float' = inf, best_param: 'Optional[Any]' = None, reset_counter: 'int' = 0, var_sup_em...` | `None` | constructs | CANONICAL |
| `SelectorSpec` | `(area: 'Optional[str]' = None, area_id: 'Optional[str]' = None, layer: 'Optional[str]' = None, cell_type: 'Optional[...` | `None` | constructs | CANONICAL |
| `Signal` | `(time_ms: 'jax.Array', V_m: 'jax.Array', spikes: 'jax.Array', sources: 'Optional[jax.Array]', field: 'Optional[Field...` | `None` | constructs | CANONICAL |
| `Signals` | `(time_ms: 'jax.Array', V_m: 'jax.Array', spikes: 'jax.Array', sources: 'Optional[jax.Array]', field: 'Optional[Field...` | `None` | constructs | CANONICAL |
| `Simulation` | `(duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.05, plasticity: 'float' = 0.0, seed: 'int' = 0, record_sources: '...` | `None` | constructs | CANONICAL |
| `StaticParams` | `(g_mech: 'dict' = <factory>, reversal_potentials_mV: 'dict' = <factory>, dT_ms: 'float' = 0.1, value_tag: 'ValueTag'...` | `None` | constructs | CANONICAL |
| `StimulusSchedule` | `(events: 'tuple[dict[str, Any], ...]', n_neurons: 'int', source_calibration_status: 'str' = 'uncalibrated_izhikevich...` | `None` | constructs | CANONICAL |
| `TrialBatch` | `(trials: 'tuple[TrialSpec, ...]', batch_id: 'str' = 'anonymous_batch', metadata: 'dict[str, Any]' = <factory>)` | `None` | constructs | CANONICAL |
| `TrialBatchResult` | `(batch_id: 'str', results: 'tuple[TrialResult, ...]', metadata: 'dict[str, Any]' = <factory>)` | `None` | constructs | CANONICAL |
| `TrialResult` | `(trial_id: 'str', condition_label: 'Optional[str]' = None, signals: 'Optional[Signals]' = None, success: 'bool' = Tr...` | `None` | constructs | CANONICAL |
| `TrialSpec` | `(trial_id: 'str', condition: 'Optional[ParadigmCondition]' = None, seed: 'int' = 0, metadata: 'dict[str, Any]' = <fa...` | `None` | constructs | CANONICAL |
| `TuneResult` | `(best_parameters: 'dict[str, float]', best_score: 'float', history: 'list[dict[str, Any]]', summary: 'dict[str, Any]...` | `None` | constructs | CANONICAL |
| `agsdr` | `(alpha: 'float' = 0.7, exploration: 'float' = 0.05, deselect_factor: 'float' = 2.0, metadata: 'Optional[dict[str, An...` | `'Any'` | pure | CANONICAL |
| `agsdr_transform` | `(inner_optimizer: 'Optional[Any]' = None, stochastic_scale: 'float' = 0.1, global_scale: 'float' = 1.0, checkpoint_n...` | `'Any'` | pure | COMPATIBILITY |
| `asset_hashes` | `(assets: 'dict[str, str | Path]')` | `'dict[str, str]'` | pure | CANONICAL |
| `build_laminar_column` | `(name: 'str' = 'V1', n: 'int' = 1000, layers: 'Sequence[str] | None' = None, layer_fractions: 'Mapping[str, tuple] |...` | `'Configuration'` | stateful | CANONICAL |
| `build_multi_area_columns` | `(areas: 'Sequence[str]' = ('V1', 'V4', 'PFC'), n_per_area: 'int' = 200, layers: 'Sequence[str] | None' = None, conne...` | `'Configuration'` | stateful | CANONICAL |
| `build_tutorial_laminar_column` | `(cfg: 'LaminarColumnConfig')` | `'dict'` | stateful | CANONICAL |
| `cable_filter_report` | `(tau_s: 'jax.Array', order: 'int' = 2)` | `'dict[str, Any]'` | pure | CANONICAL |
| `cable_filter_sources` | `(sources: 'jax.Array', tau_s: 'jax.Array', dt_ms: 'float', *, order: 'int' = 2)` | `'jax.Array'` | pure | CANONICAL |
| `cable_filter_tau` | `(cell_type: 'Sequence[str] | np.ndarray', depth_z: 'jax.Array', *, tau_e_superficial_ms: 'float' = 1.0, tau_e_deep_m...` | `'jax.Array'` | pure | CANONICAL |
| `checkpoint_state` | `(model: 'Model', path: 'str | Path')` | `'Path'` | stateful | CANONICAL |
| `compile_connection_rules` | `(neurons: 'Sequence[Mapping[str, Any]]', connections: 'Sequence[Mapping[str, Any]]', mechanisms: 'Sequence[Mapping[s...` | `'ConnectionCompileResult'` | pure | CANONICAL |
| `compile_step_fn` | `(model: 'Model', *, dt_ms: 'float', kernel: 'str' = 'hdp', record_dH_components: 'bool' = False, record_edge_current...` | `"'tuple[callable, ContinuationState]'"` | pure | CANONICAL |
| `compute_fields` | `(model: "'Model'", signals: "'Signals'")` | `"'FieldOutput'"` | pure | CANONICAL |
| `config_hash` | `(cfg: 'Any')` | `'str'` | pure | CANONICAL |
| `configs_dir` | `()` | `'Path'` | pure | CANONICAL |
| `configuration` | `()` | `'Configuration'` | pure | CANONICAL |
| `connect` | `(*models: "'Model'", edges: "'Sequence[Mapping[str, Any]] | None'" = None, namespace: "'Sequence[str] | None'" = Non...` | `"'Model'"` | pure | CANONICAL |
| `construct` | `(cfg: "'Configuration | Any'", runtime: "'Any | None'" = None, *, geometry: "'LaminarSourceGeometry | None'" = None)` | `'Model'` | stateful | CANONICAL |
| `construct_neuronal_tensor` | `(tensor: 'NeuronalTensor', *, seed: 'int' = 0, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, emitter: 'str' =...` | `'Model'` | stateful | COMPATIBILITY |
| `construct_source_tensor` | `(*, mode: 'str' = 'total_membrane_current_proxy', total_membrane_current: 'jax.Array | None' = None, decomposed_cap_...` | `'tuple[jax.Array, dict[str, Any]]'` | stateful | CANONICAL |
| `coop_omission_oddball_for_model` | `(model, *, target_area: 'Optional[str]' = None, target_layer: 'Optional[str]' = None, target_cell_type: 'Optional[st...` | `'Paradigm'` | pure | CANONICAL |
| `coop_omission_oddball_for_neuronal_tensor` | `(tensor, *, construct_seed: 'int' = 0, construct_duration_ms: 'float' = 1000.0, construct_dt_ms: 'float' = 0.1, targ...` | `'Paradigm'` | pure | CANONICAL |
| `coop_omission_oddball_paradigm` | `(duration_ms: 'float' = 10000.0, dt_ms: 'float' = 0.5, freq_hz: 'float' = 6.0, omission_prob: 'float' = 0.1, standar...` | `'Paradigm'` | pure | CANONICAL |
| `csd_tensor` | `(phi_e_proxy: 'jax.Array', dz: 'jax.Array | float')` | `'jax.Array'` | pure | CANONICAL |
| `dataset_spec` | `(**kwargs: 'Any')` | `'DatasetSpec'` | pure | CANONICAL |
| `default_basis_spec` | `()` | `'BasisSpec'` | pure | CANONICAL |
| `default_complete_configuration` | `(column_name: 'str' = 'V1', nucleus_name: 'str' = 'thalamus', n_column: 'int' = 100, n_nucleus: 'int' = 60, layers: ...` | `'Configuration'` | pure | CANONICAL |
| `default_cortical_column_config` | `(column_name: 'str' = 'single_column', n: 'int' = 100, layers: 'Sequence[str] | None' = None, seed: 'int | None' = N...` | `'Configuration'` | pure | CANONICAL |
| `default_relative_size` | `(neuron_type: 'str')` | `'float'` | pure | CANONICAL |
| `develop` | `(genome: 'PseudoGenome', seed: 'int' = 0, *, development_parameters: 'Optional[Mapping[str, Any]]' = None)` | `'NeuronalTensor'` | pure | CANONICAL |
| `dynamic_state_from_model` | `(model: 'Model', *, h_state_dim: 'int' = 1, h_state_locality: 'str | None' = None)` | `'DynamicState'` | pure | CANONICAL |
| `edge_parameter` | `(*, pre: 'Optional[SelectorSpec | Mapping[str, Any]]' = None, post: 'Optional[SelectorSpec | Mapping[str, Any]]' = N...` | `'EdgeParameterSpec'` | pure | CANONICAL |
| `eeg_proxy_transform` | `(source: 'jax.Array', leadfield: 'jax.Array')` | `'jax.Array'` | pure | CANONICAL |
| `emm_proxy_transform` | `(spike_rate: 'jax.Array', source: 'jax.Array', field_potential: 'jax.Array', lambda_spk: 'float' = 1.0, lambda_src: ...` | `'jax.Array'` | pure | CANONICAL |
| `enable_x64` | `()` | `'dict[str, Any]'` | pure | CANONICAL |
| `evoked_l4_drive_paradigm` | `(l4_onset_ms: 'float' = 100.0, l4_duration_ms: 'float' = 200.0, l4_amplitude: 'float' = 1.0, pre_stimulus_buffer_ms:...` | `'Paradigm'` | pure | CANONICAL |
| `export_report` | `(output_dir: 'str | Path', manifest: 'Optional[Mapping]' = None, metrics: 'Optional[Mapping]' = None, validation: 'O...` | `'Mapping[str, str]'` | pure | CANONICAL |
| `export_tutorial_artifacts` | `(output_dir: 'str | Path', manifest: 'Optional[Mapping]' = None, metrics: 'Optional[Mapping]' = None, validation: 'O...` | `'Mapping[str, str]'` | pure | CANONICAL |
| `general_delayed_match_to_sample_paradigm` | `(*, name: 'str' = 'general_delayed_match_to_sample', event_windows: 'Optional[Mapping[str, Sequence[float]]]' = None...` | `'Paradigm'` | pure | CANONICAL |
| `general_sequential_oddball_paradigm` | `(*, name: 'str' = 'general_sequential_oddball', event_windows: 'Optional[Mapping[str, Sequence[float]]]' = None, eve...` | `'Paradigm'` | pure | CANONICAL |
| `get_signal` | `(obj: 'Any', key: 'str', **kwargs: 'Any')` | `'Any'` | pure | CANONICAL |
| `gsdr` | `(alpha: 'float' = 0.7, exploration: 'float' = 0.05, deselect_factor: 'float' = 2.0, metadata: 'Optional[dict[str, An...` | `'OptimizerSpec'` | pure | CANONICAL |
| `gsdr_transform` | `(inner_optimizer: 'Optional[Any]' = None, stochastic_scale: 'float' = 0.1, checkpoint_n_steps: 'int' = 50, deselecti...` | `'Any'` | pure | COMPATIBILITY |
| `gsgd` | `(learning_rate: 'float' = 0.01, differentiability_status: 'str' = 'not_checked', surrogate_status: 'str' = 'none', m...` | `'OptimizerSpec'` | pure | CANONICAL |
| `json_safe` | `(obj: 'Any')` | `'Any'` | pure | CANONICAL |
| `kappa_synchrony` | `(spikes: 'np.ndarray', dt_ms: 'float' = 0.1)` | `'float'` | pure | CANONICAL |
| `laminar_cortex_config` | `(*, seed: 'int' = 0, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, areas: 'Sequence[str] | None' = None, laye...` | `'Configuration'` | pure | CANONICAL |
| `laminar_source_geometry` | `(populations: "Sequence['LaminarPopulation']")` | `"'LaminarSourceGeometry'"` | pure | CANONICAL |
| `list_canonical_neuronal_tensors` | `()` | `'list[str]'` | pure | CANONICAL |
| `list_canonical_pseudogenomes` | `()` | `'list[str]'` | pure | CANONICAL |
| `load` | `(path: 'str | Path')` | `'NeuronalTensor'` | pure | CANONICAL |
| `load_canonical_neuronal_tensor` | `(name: 'str')` | `'NeuronalTensor'` | stateful | CANONICAL |
| `load_canonical_pseudogenome` | `(name: 'str')` | `'PseudoGenome'` | stateful | CANONICAL |
| `load_neuronal_tensor` | `(path: 'str | Path')` | `'NeuronalTensor'` | stateful | COMPATIBILITY |
| `load_pseudogenome` | `(path: 'str | Path | Mapping[str, Any]')` | `'PseudoGenome'` | stateful | CANONICAL |
| `make_minimal_ei_tensor` | `(n: 'int' = 8, e_fraction: 'float' = 0.75, *, layer_name: 'str' = 'L1', area_name: 'str' = 'minimal', h: 'float' = 1...` | `'NeuronalTensor'` | stateful | CANONICAL |
| `manifest` | `(cfg: 'Any', signals: 'Optional[Any]' = None, readout: 'Optional[dict[str, Any]]' = None, runtime_config: 'Optional[...` | `'dict[str, Any]'` | pure | CANONICAL |
| `matrix_parameter` | `(*, mask: 'str', bounds: 'tuple', init: 'str' = 'current', trainable: 'bool' = True, target: 'str' = 'W')` | `'MatrixParameterSpec'` | pure | CANONICAL |
| `meg_proxy_transform` | `(source_oriented: 'jax.Array', leadfield: 'jax.Array')` | `'jax.Array'` | pure | CANONICAL |
| `merge_neuronal_tensors` | `(tensors: 'Sequence[NeuronalTensor]', poses: 'Optional[Sequence[Pose3D]]' = None, *, name: 'str' = 'merged')` | `'NeuronalTensor'` | pure | CANONICAL |
| `migrate_schema` | `(meta: 'dict[str, Any]')` | `'dict[str, Any]'` | pure | CANONICAL |
| `neuronal_tensor_to_configuration` | `(tensor: 'NeuronalTensor', *, seed: 'int' = 0, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, emitter: 'str' =...` | `'Configuration'` | pure | CANONICAL |
| `objective` | `()` | `'Objective'` | pure | CANONICAL |
| `omission_oddball_paradigm` | `(standard_onset_ms: 'float' = 500.0, standard_duration_ms: 'float' = 100.0, deviant_onset_ms: 'Optional[float]' = No...` | `'Paradigm'` | pure | CANONICAL |
| `operator_status` | `()` | `'dict[str, str]'` | pure | CANONICAL |
| `optax_adam` | `(learning_rate: 'float' = 0.001, differentiability_status: 'str' = 'not_checked', surrogate_status: 'str' = 'none', ...` | `'OptimizerSpec'` | pure | CANONICAL |
| `optax_sgd` | `(learning_rate: 'float' = 0.001, differentiability_status: 'str' = 'not_checked', surrogate_status: 'str' = 'none', ...` | `'OptimizerSpec'` | pure | CANONICAL |
| `paradigm` | `(*args, **kwargs)` | — | pure | CANONICAL |
| `paradigm_target_indices_from_model` | `(model, *, area: 'Optional[str]' = None, layer: 'Optional[str]' = None, cell_type: 'Optional[str]' = None)` | `'list[int]'` | pure | CANONICAL |
| `plot_raster` | `(signals, model=None, *, backend: 'str' = 'plotly', title: 'Optional[str]' = None, xlabel: 'Optional[str]' = 'Time (...` | `'Any'` | pure | CANONICAL |
| `plot_spectrolaminar_suite` | `(signals: 'Signals | dict[str, Any]', **kwargs: 'Any')` | `'matplotlib.figure.Figure'` | pure | CANONICAL |
| `probe_laminar_modes` | `(field_output: 'FieldOutput', modes: 'Sequence[str]' = ('source', 'phi_e', 'CSD', 'LFP'))` | `'dict[str, Any]'` | pure | CANONICAL |
| `probe_report` | `(n_probes: 'int', probe_types: 'dict[str, int] | None' = None, metadata: 'dict[str, Any] | None' = None)` | `'dict[str, Any]'` | pure | CANONICAL |
| `project_laminar_sources` | `(sources: 'jax.Array', positions: 'jax.Array', *, n_contacts: 'int' = 16, width: 'float' = 0.1, mode: 'str' = 'densi...` | `'FieldOutput'` | pure | CANONICAL |
| `project_sources_to_laminar_field` | `(sources: 'jax.Array', positions: 'jax.Array', n_contacts: 'int' = 16, *, mode: 'str' = 'density_preserving', dtype:...` | `'FieldOutput'` | pure | CANONICAL |
| `provenance_receipt` | `(branch: 'str' = 'unknown', sha: 'str' = 'unknown', dirty: 'bool' = False)` | `'dict[str, Any]'` | pure | CANONICAL |
| `random_search` | `(metadata: 'Optional[dict[str, Any]]' = None)` | `'OptimizerSpec'` | pure | CANONICAL |
| `rate_synchrony_targets` | `(target_rate_hz: 'float' = 10.0, target_kappa_synchrony: 'float' = 0.0, rate_weight: 'float' = 1.0, synchrony_weight...` | — | pure | CANONICAL |
| `rate_targets` | `(groups: 'dict[str, Any]', targets_hz: 'dict[str, float]', weights: 'Optional[dict[str, float]]' = None, *, burn_in_...` | `'Objective'` | pure | CANONICAL |
| `readout_spec` | `(name: 'str', metric: 'str', *, time_window_ms: 'Optional[tuple[float, float]]' = None, n_contacts_slice: 'Optional[...` | `'ReadoutSpec'` | pure | CANONICAL |
| `require_optax` | `()` | `'Any'` | pure | CANONICAL |
| `restore_state` | `(path: 'str | Path')` | `'tuple[list, dict]'` | stateful | CANONICAL |
| `run_receipt` | `(model: "'Model'", signals: 'Signals', *, tags: 'Optional[dict[str, Any]]' = None)` | `'RunReceipt'` | stateful | CANONICAL |
| `run_trials` | `(model: 'Model', batch: 'TrialBatch', sim: 'Simulation', *, collect_errors: 'bool' = False)` | `'TrialBatchResult'` | stateful | CANONICAL |
| `runtime` | `(backend: 'str' = 'auto', dtype: 'str' = 'float32', jit: 'bool' = False, vmap: 'bool' = False, precision: 'str' = 'd...` | `'RuntimeConfig'` | pure | CANONICAL |
| `runtime_report` | `(runtime_config: 'RuntimeConfig | None' = None)` | `'dict[str, Any]'` | pure | CANONICAL |
| `save_figure` | `(fig, path: 'str | Path', dpi: 'int' = 150, bbox_inches: 'str' = 'tight')` | `'str'` | stateful | CANONICAL |
| `save_figures` | `(figures: 'Mapping[str, object]', output_dir: 'str | Path', dpi: 'int' = 150, prefix: 'str' = '', suffix: 'str' = '')` | `'Mapping[str, str]'` | stateful | CANONICAL |
| `save_json` | `(obj: 'Any', path: 'str | Path')` | `'None'` | stateful | CANONICAL |
| `save_neuronal_tensor` | `(tensor: 'NeuronalTensor', path: 'str | Path')` | `'str'` | stateful | CANONICAL |
| `save_receipt` | `(receipt: 'Any', path: "'str | Path'", *, overwrite: 'bool' = False)` | `'None'` | stateful | CANONICAL |
| `scan_network` | `(step_fn: "'callable'", init: "'DynamicState | ContinuationState'", drive_schedule: 'jax.Array', keys: 'jax.Array')` | `"'tuple[DynamicState | ContinuationState, tuple]'"` | pure | CANONICAL |
| `sdr_transform` | `(inner_optimizer: 'Optional[Any]' = None, stochastic_scale: 'float' = 0.1, checkpoint_n_steps: 'int' = 50, alpha_min...` | `'Any'` | pure | COMPATIBILITY |
| `select_neurons` | `(model, area: 'str | None' = None, layer: 'str | None' = None, cell_type: 'str | None' = None)` | `'np.ndarray'` | pure | CANONICAL |
| `sha256_file` | `(path: 'str | Path')` | `'str'` | pure | CANONICAL |
| `sha256_text` | `(text: 'str')` | `'str'` | pure | CANONICAL |
| `simulate` | `(model: 'Model', sim: 'Optional[Simulation]' = None, paradigm: 'Optional[Any]' = None, *, continuation: 'Any' = None...` | `"'Signals | tuple[Signals, Any]'"` | stateful | CANONICAL |
| `simulation` | `(**kwargs: 'Any')` | `'Simulation'` | pure | CANONICAL |
| `spectrolaminar_motif_score` | `(alpha_beta, gamma)` | `'float'` | pure | CANONICAL |
| `standard_visual_omission` | `()` | `'Paradigm'` | pure | CANONICAL |
| `step_gsgd_transform` | `(u_t: 'jnp.ndarray', grad_l: 'jnp.ndarray', state: 'Any', hyperparams: 'dict')` | `'tuple[jnp.ndarray, Any]'` | pure | COMPATIBILITY |
| `stimulus_schedule` | `(events: 'Sequence[Any]', n_neurons: 'int', *, drive_amplitude: 'float' = 5.0, event_duration_ms: 'float' = 50.0)` | `'StimulusSchedule'` | pure | CANONICAL |
| `suite2_celltype_presets` | `()` | `'dict[str, dict[str, float | str]]'` | pure | CANONICAL |
| `suite2_four_celltype_config` | `(*, seed: 'int' = 7, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1)` | `'Configuration'` | pure | CANONICAL |
| `suite2_net1_config` | `(*, seed: 'int' = 7, n: 'int' = 100, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, drives: 'Mapping[str, floa...` | `'Configuration'` | pure | CANONICAL |
| `suite2_run_bundle` | `(model: 'Model', *, seed: 'int' = 7, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, noise_amplitude: 'float | ...` | `'dict[str, Any]'` | stateful | CANONICAL |
| `suite2_simulation` | `(*, seed: 'int' = 7, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, noise_amplitude: 'float | None' = None, no...` | `'Simulation'` | pure | CANONICAL |
| `suite2_single_neuron_config` | `(*, seed: 'int' = 7, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, cell_type: 'str' = 'E')` | `'Configuration'` | pure | CANONICAL |
| `suite2_tune_noise_agsdr_adam` | `(model: 'Model', *, simulation: 'Simulation | None' = None, target_rate_hz: 'tuple[float, float]' = (5.0, 10.0), amp...` | `'TuneResult'` | stateful | CANONICAL |
| `suite2_v1_v4_config` | `(*, seed: 'int' = 7, n_per_area: 'int' = 400, duration_ms: 'float' = 1000.0, dt_ms: 'float' = 0.1, v1_layer_cell_typ...` | `'Configuration'` | pure | CANONICAL |
| `trial_batch` | `(conditions: 'Sequence[ParadigmCondition]', n_reps: 'int' = 1, seed: 'int' = 0, seed_policy: 'str' = 'paired_by_repl...` | `'TrialBatch'` | pure | CANONICAL |
| `validate_model` | `(model: 'Model', *, strict: 'bool' = False)` | `'list[str]'` | pure | CANONICAL |
| `validate_neuronal_tensor` | `(nt: 'NeuronalTensor', *, strict: 'bool' = False)` | `'list[str]'` | pure | CANONICAL |
| `validate_projection_invariants` | `(*, sources: 'jax.Array', positions: 'jax.Array', kernel: 'jax.Array', source_proxy: 'jax.Array', phi_e_proxy: 'jax....` | `'dict[str, Any]'` | pure | CANONICAL |
| `validate_runtime_config` | `(cfg: 'RuntimeConfig', *, strict: 'bool' = False)` | `'list[str]'` | pure | CANONICAL |
| `validate_source_field_status` | `(field_output: 'FieldOutput | None' = None, cfg_metadata: 'Mapping[str, Any] | None' = None, *, requested_modes: 'Se...` | `'dict[str, Any]'` | pure | CANONICAL |
| `validation_report` | `(config_valid: 'bool', issues: 'list[str] | None' = None, metadata: 'dict[str, Any] | None' = None)` | `'dict[str, Any]'` | pure | CANONICAL |
| `vis` | — | — | — | CANONICAL |
| `with_emitter_parameters` | `(model: 'Model', *, a: "'float | None'" = None, b: "'float | None'" = None, c: "'float | None'" = None, d: "'float |...` | `'Model'` | pure | CANONICAL |

