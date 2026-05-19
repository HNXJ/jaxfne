## v0.1.0

- Declared practical OOP core freeze for the compact JAX-native TFNE scaffold.
- Preserved canonical workflow: `run_receipt`, `compute_readout`, `evaluate_report`.
- Includes all v0.0.23 fixes: `manifest(signals, readouts)` readout compatibility,
  MIT LICENSE, normalized examples 00-06, full packaging validation.
- Validated wheel and sdist install smokes from `/tmp`; canonical workflow passes
  from installed package (site-packages, not repo).
- Preserved truth status at `truth_safe_unverified`.
- Preserved field status as `laminar_proxy_no_pde`.
- Preserved source calibration as `uncalibrated_izhikevich_native_current`.
- Preserved `physical_amplitude_claim_allowed=False` across all outputs.

## v0.0.23

- **Fixed `Model.manifest()` readout compatibility:** `manifest(signals, readouts)` now
  accepts any of: `None`, `dict` (legacy), `list[ReadoutResult]` (canonical v0.1 output
  of `compute_readout()`), `tuple[ReadoutResult]`, `list[dict]`, or a single
  `ReadoutResult`. Previously raised `AttributeError: 'list' object has no attribute 'get'`
  when passed the canonical `compute_readout()` return value.
- Added `_normalize_manifest_readout()` normaliser; surfaces readout results under
  `readout_results` key in manifest with `n_results`, `requested_metrics`, and frozen
  `physical_amplitude_claim_allowed=False` guard.
- Added 8 tests in `tests/test_manifest_readout_compat.py` covering all argument forms,
  JSON strictness, and truth-gate non-escalation.
- Added MIT LICENSE file.
- Normalized examples directory to 00-06 naming convention.
- Validated wheel and sdist build via `python -m build`; `twine check dist/*` passes.
- Confirmed fresh venv wheel and sdist install smokes from `/tmp`; import path confirmed
  as `site-packages` (not repo).
- Confirmed canonical workflow from installed wheel:
  `compute_readout(...)` → `manifest(signals, readouts)` → `json.dumps(allow_nan=False)`.
- Preserved truth status at `truth_safe_unverified`.
- Preserved `physical_amplitude_claim_allowed=False` across all outputs.

## v0.0.22
- Added packaging, release, and Colab installation documentation.
- Validated wheel and sdist builds with twine check.
- Validated fresh virtual-environment install smoke tests for wheel and sdist.
- Added a minimal Colab spectrolaminar proxy scaffold example.
- Preserved truth status at truth_safe_unverified.

# Changelog

All entries reflect `truth_mode: truth_safe_unverified`. No biological claims
are made at any version. Receipts, reports, and manifests are computational
validation artifacts, not empirical evidence.

## v0.0.21

- **Config/runtime fidelity:** Added `_SUPPORTED_RUNTIME_SPEC_KEYS` and
  `_runtime_from_spec()` to validate runtime declarations from `.jcfg.json`;
  unknown keys now warn; invalid known values (e.g. bad `synaptic_kernel`) raise.
- **Truth escalation guard:** Implemented `_conservative_truth_transfer()` to
  force user-declared truth claims back to conservative defaults
  (`truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`);
  escalations trigger warnings; non-scalar unknown keys skipped.
- **Unsupported config warnings:** Added `_config_section_warnings()` to detect
  unsupported emitter families, field domains, conductivities, boundaries, and gauges;
  warnings merged into `Configuration.metadata["unsupported_config_warnings"]`.
- **Runtime config warnings registry:** Added `_CONFIG_RUNTIME_WARNINGS` module-level
  dict to surface `runtime_spec` warnings without mutating frozen `RuntimeConfig`.
- **Backend reporting fidelity:** Enhanced `RuntimeConfig.runtime_report()` to
  distinguish `requested_backend` vs `actual_backend` and report enforced status
  and mismatches (e.g. requested GPU on CPU-only JAX device).
- **vmap behavioral semantics:** Made `simulate_batch()` respect `runtime.vmap` flag:
  `vmap=True` uses `jax.vmap` over seed batch; `vmap=False` uses Python loop with
  `jnp.stack`; mode reported in metadata as `batch_execution_mode`.
- **Source proxy metadata:** Added `_SOURCE_PROXY_METADATA` constant documenting
  source model: `izhikevich_native_current_plus_spike_impulse_proxy` with spike
  impulse gain 20.0; injected into `simulate()`, `simulate_batch()`, and
  `manifest()` under `source_model` and `backend_metadata.source_model`.
- **Receptor/tau source documentation:** Enhanced `manifest()` to document
  `receptor_tau_source` distinction: exponential kernel uses default tau;
  receptor_exponential kernel looks up tau by receptor_index; results equivalent
  for current default flow.
- **Schema version bumps:** Updated `_RECEIPT_SCHEMA_VERSION` to
  `"run_receipt_v0.0.21"`, `_MANIFEST_SCHEMA_VERSION` to `"manifest.v0.0.21"`.
- Preserved all truth gates at `truth_safe_unverified / computational_scaffold /
  laminar_proxy_no_pde / proxy_readout_only / physical_amplitude_claim_allowed=False`.

## v0.0.20

- Fixed `RunReceipt` completeness: `duration_ms`, `dt_ms`, `n_steps`, and
  record flags now propagate into receipt simulation metadata.
- Strengthened `receipt_id` hashing to include run-level metadata
  (recurrent backend, synaptic kernel, condition name, stimulus schedule).
- Fixed `Model.manifest()` to report executed backend separately from
  available infrastructure.
- Honored probe `n_contacts` from `.jcfg.json` in field/readout construction.
- Applied readout `time_window_ms` slicing to field-backed CSD/LFP metrics.
- Added safe empty/negative-window handling to avoid NaN in readout output.
- Added `Simulation.__post_init__` validation (duration/dt must be positive finite).
- Clarified `record_sources` semantics in metadata.
- Centralized version/schema constants near `_JAXFNE_VERSION`.
- Preserved truth status at `truth_safe_unverified`.
- Preserved canonical v0.1 workflow: `run_receipt`, `compute_readout`,
  `evaluate_report`. Compatibility aliases (`manifest`, `probe`) unchanged.

## v0.0.19

- Clarified canonical v0.1 API wording: `run_receipt`, `compute_readout`,
  and `evaluate_report` are the canonical workflow methods.
- Documented `manifest()` and `probe()` as compatibility aliases retained
  from v0.0.4–v0.0.14; not removed.
- Added docstring notes to `config_truth_boundary()`: passthrough helper,
  call `validate_config()` first.
- Added docstring notes to `JaxFNEConfig.config_hash`: unknown `.jcfg.json`
  keys enter the hash; hash equality is structural identity, not biological
  equivalence.
- Added docstring notes to `Model.run_receipt()`: `receipt_id` is tied to
  `_JAXFNE_VERSION`; upgrading the package changes IDs for same cfg/seed.
- Added `CHANGELOG.md`.
- Version bump `0.0.18` → `0.0.19`.

## v0.0.18

- Added `ObjectiveReport` (frozen dataclass) and `Model.evaluate_report()`.
- `ObjectiveReport` embeds `ReadoutResult` items when `readout_specs` are
  provided; carries frozen truth gates.

## v0.0.17

- Added `ReadoutSpec`, `ReadoutResult`, `readout_spec()` factory.
- Added `Model.compute_readout(signals, specs)`.
- Six supported metrics: `spike_rate_hz`, `spike_count`, `mean_V_m`,
  `csd_abs_mean`, `lfp_abs_mean`, `source_abs_mean`.

## v0.0.16

- Added `RunReceipt` (frozen dataclass) and `Model.run_receipt()`.
- Added module-level `run_receipt()` factory and `save_receipt()`.
- `receipt_id` is deterministic: `sha256(config_hash:seed:version)[:16]`.
- Fixed `truth_mode` absent from `_CONSERVATIVE_TRUTH_DEFAULTS` (blocking
  defect: `validate_config` now checks all 8 required truth keys).

## v0.0.15

- Added `JaxFNEConfig`, `ConfigValidationResult`, `load_config()`,
  `validate_config()`, and `.jcfg.json` declarative config standard.
- Added `config_to_simulation`, `config_to_geometry`,
  `config_to_configuration`, `config_to_trial_batch`, `config_truth_boundary`.
- Truth boundary fields are required in every config; any escalation is a
  blocking validation error.

## v0.0.14

- Added sequential trial runner: `TrialSpec`, `TrialBatch`, `TrialResult`,
  `TrialBatchResult`, `trial_batch()`, `Model.run_trials()`.
- Deterministic seed policy; JSON-safe compact results via `Signals.summary()`.

## v0.0.13

- Added `LaminarPopulation`, `LaminarSourceGeometry`, `laminar_source_geometry()`.
- Geometry depths are normalized proxy coordinates in `[0, 1]`; no physical
  spatial units (mm, µm) introduced.

## v0.0.12

- Added `StimulusSchedule`, `stimulus_schedule()`, and event-aligned native-
  drive injection into all recurrent kernels.

## v0.0.11

- Added receptor-indexed exponential synaptic kernel
  (`synaptic_kernel="receptor_exponential"`).
- `syn_state.shape == (n_edges,)` with per-edge tau lookup by
  `receptor_index` against standard `ReceptorSpec` table.
