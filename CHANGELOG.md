## v0.2.26

- **Computation-basis contracts release.**
- **Added `AxisSpec` and `BasisSpec` dataclasses:** Typed, frozen contract objects in `jaxfne/core.py` describing the spatial basis (`laminar_depth`, `xy`, `xyz`, `collapsed`, `graph`), time basis (`continuous_ms`, `discrete_steps`, `slow_proxy`), field regime (`laminar_proxy`, `quasi_static_resistive`, `solved_poisson`, `future_admittive`, `future_maxwell`), source mode, and probe basis for each TFNE run. Default matches current laminar-proxy scaffold.
- **Added `default_basis_spec()` factory:** Returns the default `BasisSpec` for v0.2.25 laminar-proxy behavior.
- **Added basis validation:** `validate_basis_spec()` and `basis_claim_gate()` in `jaxfne/validation.py`. Validates enum values, axis-space consistency (xy/xyz/laminar_depth/collapsed rules), future-regime gating, and source mode eligibility. Returns JSON-safe dicts. `physical_amplitude_claim_allowed` is always `False`.
- **Future regimes are doctrine-only:** `future_maxwell` and `future_admittive` serialize as `implemented=False, claim_allowed=False`. Cannot be escalated. Structural enforcement tested.
- **Added manifest `basis` block:** `Model.manifest()` now includes a nested `basis` field with space/time/field/source/probe basis metadata and dimension status (x/y collapsed, z active by default).
- **Public API:** `AxisSpec`, `BasisSpec`, `default_basis_spec` exported from `jaxfne`.
- **55 new tests:** `tests/test_computation_basis_v026.py` covers JSON safety, laminar proxy defaults, axis validation, space basis rules (xy/xyz/laminar_depth/collapsed), future regime gating, manifest integration, claim gate, and public export verification.
- **Updated `docs/computation_basis.md`:** Added "Implemented in v0.2.26" section listing contracts, allowed field regimes, and status table.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`, `field_solver_status: laminar_proxy_no_pde`.
- **No new solver, no conservation diagnostics, no Maxwell/stress-energy implementation.** v0.2.26 is contracts-only.

## v0.2.25

- **Mathematical glossary flow and docs-first upgrade.**
- **Added mathematical glossary flow doctrine:** New `docs/mathematical_glossary_flow.md` documents seven core TFNE equations (emitter dynamics, source projection, ohmic current, field compatibility, CSD, probe operator, EMM-proxy) with formal definitions, complete term glossaries, worded-equations, critical bridge terms, claim boundaries, and implementation locations. Includes conservation-law doctrine (Poynting's theorem) as future reference.
- **Added source/field bookkeeping guide:** New `docs/source_field_equations.md` specifies source modes (total_membrane_current, decomposed_cap_ion_syn, proxy_no_field_solve), one-source-per-run requirement, **forbidden synaptic double-counting pattern** with audit examples, field metadata (boundary/gauge/CSD convention), calibration labels, and code-to-manifest mappings. Minimal examples tying equations to implementation.
- **Added computation basis doctrine:** New `docs/computation_basis.md` describes TFNE as collapsible tensor-field scaffold with canonical dimensions (time, units, space, features, readout), collapse rules, basis-change philosophy, declared-future field regimes (v0.2.27 diagnostics, v0.3.x physical), extensibility doctrine for new domains, and PRNG/finiteness contracts.
- **Rewrote README.md for compactness:** Reorganized with sections: identity, installation, minimal example, pipeline (4 stages + operators table), readout meanings, validation (fast vs extended), documentation map, roadmap (v0.2.24–v0.3.0), claim status. Explicit "not a biological simulator" statement with guidance on when/when-not to use.
- **Updated docs/index.md:** Added new "Core mathematics & equations" section highlighting glossary flow, source/field equations, and computation basis. Reorganized "Learn more" to prioritize equation documentation.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`, `field_solver_status: laminar_proxy_no_pde`.
- **No code changes, no feature expansion.** v0.2.25 is pure documentation: mathematical grounding, forbidden patterns, equation-to-code mapping, and clear roadmap.
- **Purpose:** Establish solid mathematical and conceptual foundation for v0.2.26–v0.3.x extensibility, diagnostics, and physical-field work.
- **BETA patch:** Bumped package version to `0.2.25` and patched the source-projection glossary to define `a_k` (state-to-source scalar) and `b_k` (input-to-source scalar), resolving the documentation completeness gap identified in BETA review.

## v0.2.24

- **Foundation audit checkpoint release.**
- **Verified v0.2.23 release baseline:** Confirmed tag, version consistency, and clean repo state after v0.2.23 release.
- **Audited calibration/source/field/report contracts:** Verified all required fields present and correct; no double-counting of synaptic current.
- **Confirmed solver status:** Field solver remains `laminar_proxy_no_pde`; field readouts remain proxy-only; boundary/gauge conditions remain metadata-only.
- **Confirmed public language:** Verified no forbidden phrases (real EEG, real MEG, biological metabolism claims, mechanism proof). All public language uses approved proxy/scaffold terminology.
- **Updated version assertions in tests:** Fixed 11 test files with outdated version checks; validation baseline: 806 passed, 5 skipped.
- **Preserved truth status:** `truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`.
- **No new science features, no code rewrites, no biological claims.**
- **Purpose:** Establish stable foundation for v0.2.25–v0.2.28 late-0.2.x bridge block (mathematical glossary, computation-basis contracts, conservation diagnostics, 0.3 bridge hardening).

## v0.2.23

- **Package cleanup and docs polish release.**
- **Added quick API reference table:** README now includes organized table of main API categories 
  (Configuration, Simulation, Emitters, Fields/Probes, Readouts, Bridges, Optimization, I/O).
- **Clarified validation documentation:** Split validation into core (fast, every commit) and extended 
  (manual, release validation). Large examples (02-05, 07) now documented as manual-validation tests 
  with expected exclusion from fast CI.
- **Added CI policy documentation:** New `docs/ci_policy.md` explains smoke-safe CI gate, large tutorial 
  exclusion rationale, and release validation procedures.
- **Improved docs/index.md:** Reorganized to highlight current best practices; added reference to CI policy 
  in "Development & CI" section.
- **Terminology consistency:** User-facing docs use "programmatically generated" instead of "human-readable"; 
  removed casual language ('we' → 'jaxfne', 'you can' → 'workflows can').
- **Preserved all features:** No code changes, no new dependencies, no feature expansion. v0.2.23 is pure 
  documentation and packaging polish.
- **Preserved truth status:** All claim gates remain frozen (`truth_safe_unverified`, `computational_scaffold`, 
  `physical_amplitude_claim_allowed=False`).
- **Known issue documented:** Subprocess tests (test_example_script_runs) excluded from fast CI; documented as 
  release-validation only. Infrastructure issue, not regression.

## v0.2.22

- **Added Jaxley array-first trace bridge:** Minimal optional bridge for converting Jaxley-style voltage 
  trace arrays to jaxfne Signals without Jaxley installation.
- **New API:** `JaxleyTraceSpec` (frozen dataclass with immutable claim gates) and `jaxley_trace_to_signals()` 
  (main conversion function).
- **Layout normalization:** Supports three input layouts: `time_by_unit` [T,N], `unit_by_time` [N,T], 
  `recording_by_time` [R,T]. All normalize to canonical [T,N] internally.
- **Spike proxy derivation:** Configurable voltage threshold (default 0.0 mV) for deriving binary spike proxy 
  from voltage trace.
- **Conservative source handling:** Voltage-proxy source fallback; no ionic current mapping (deferred to v0.2.23).
- **25 new unit tests:** Comprehensive test suite covering spec validation, layout conversion, spike thresholding, 
  source handling, metadata gates, NumPy/JAX conversion, and no-Jaxley-required import.
- **Synthetic example:** `examples/07_jaxley_trace_bridge.py` demonstrates bridge usage with synthetic voltage 
  traces (CPU-only, <10 seconds runtime).
- **Immutable claim gates (frozen):**
  - `claim_level: "computational_scaffold"`
  - `physical_amplitude_claim_allowed: False`
  - `field_solver_status: "not_computed"`
  - `source_calibration_status: "uncalibrated_jaxley_voltage_proxy"`
- **Field computation deferred:** All Signals have `field=None`; field computation reserved for downstream 
  probe/field layer.
- **Scope discipline:** No multi-compartment support, sparse spike format, ionic current mapping, field computation, 
  or simulator wrapper in v0.2.22 (all deferred to later phases).
- **Full test validation:** 806 tests passed, 5 skipped; no regressions from v0.2.21.
- **Preserved truth status:** `truth_safe_unverified`, `computational_scaffold`, no biological claims.

## v0.1.2

- **Documented scan-backed recurrent execution:** Both dense and edge-list paths use `jax.lax.scan` for 
  deterministic, JAX-native time stepping. This release formalizes the existing implementation and adds 
  metadata confirmation.
- **Added `docs/V012_SCAN_PERFORMANCE.md`:** Design document confirming scan-backed architecture and 
  benchmark methodology.
- **Added benchmark harness:** `scripts/benchmark_scan_backends.py` measures wall time for dense and 
  edge-list backends at scales ranging from 50 to 100 neurons across durations up to 1000 ms.
- **Added metadata tests:** `tests/test_scan_backends_v012.py` ensures dense, edge-list, and 
  receptor-exponential paths report correct metadata and preserve truth gates.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`, 
  `laminar_proxy_no_pde`, `uncalibrated_izhikevich_native_current`, `physical_amplitude_claim_allowed=False`.
- **No kernel refactoring:** Dense and edge-list backends remain unchanged; this release validates 
  performance and correctness of the existing scan-backed implementations.

## v0.1.1

- **Corrected VIP/IS Izhikevich preset:** `b` parameter corrected from `+0.20` to `-0.10` to match
  intrinsic-spiking profile (Izhikevich 2003 Table 1).
- **Added per-neuron `layer_labels` support:** `IzhikevichParams` now accepts optional `layer_labels`
  tuple for layer-selective analysis (e.g., L1, L2/3, L4, L5, L6).
- **Added `population_slices()` method:** `LaminarSourceGeometry` now provides programmatic mapping
  from population names to neuron index ranges for layer-specific readouts.
- **Added preset registry:** Introduced `jaxfne.presets` with standardized `CELL_TYPE_PRESETS`,
  `RECEPTOR_KINETICS`, and `DEFAULT_SPIKE_IMPULSE_GAIN` constants for reproducible configuration.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`,
  `laminar_proxy_no_pde`, `uncalibrated_izhikevich_native_current`, `physical_amplitude_claim_allowed=False`.
- **No biological calibration changes:** This is a computational-correctness and API-readiness pass,
  not an empirical validation upgrade.

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
