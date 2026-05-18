# Changelog

All entries reflect `truth_mode: truth_safe_unverified`. No biological claims
are made at any version. Receipts, reports, and manifests are computational
validation artifacts, not empirical evidence.

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
