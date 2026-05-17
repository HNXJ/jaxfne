# jaxfne doctrine

**Version:** 0.0.11-pre  
**Status:** Design scaffold / computational framework, not biological validation.

## Identity

`jaxfne` = **JAX Field Neural Equations**.

A JAX-native engine for Tensor-Field Neural Equations (TFNE) source-to-field modeling.

## Core chain

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

## Claims

**Allowed:**
- v0.1.x provides the compact JAX-native OOP core required to build reproducible TFNE workflows.
- jaxfne provides a JAX-native computational scaffold for declaring emitters, source projections, fields, probes, objectives, and manifests.

**Forbidden:**
- v0.1.x validates spectrolaminar mechanisms.
- v0.1.x produces calibrated LFP/CSD amplitudes.
- v0.1.x is a full simulator.
- jaxfne validates a biological mechanism or physical CSD/LFP amplitude without calibration.

## Reduced emitter rule

Izhikevich native current is not amperes unless calibrated.

Default label:

```text
source_calibration_status = uncalibrated_izhikevich_native_current
```

## Source bookkeeping

Use one source mode per run. Do not double-count synaptic current.

```text
q = chi * I_m_tot + q_ext
```

or

```text
q = q_cap_ion + q_syn + q_ext
```

but never both simultaneously.

## v0.0.3 API naming and runtime discipline

**Signals** (plural) is the canonical container name:
- Holds time_ms, V_m, spikes, sources, field, metadata.
- Returned by Model.simulate().
- Processed by Model.probe() or Model.record().
- Signal (singular) is a backward-compatibility alias.

**probe()** is the canonical TFNE readout method:
- Extracts named arrays from Signals.
- record() is an alias for user convenience.
- Both return a dict of arrays keyed by mode.

**RuntimeConfig** (new in v0.0.3) declares runtime environment:
- device_type: "cpu", "gpu", or "tpu"
- dtype_primary: "float32" (default) or "float64" when JAX x64 enabled
- x64_enabled: boolean flag for JAX precision mode
- seed: PRNG seed for reproducibility
- n_steps: actual number of timesteps executed
- runtime_report() method returns dict for manifest inclusion

## v0.0.3 metadata gates and runtime reporting

Configuration includes these immutable metadata fields (defaults):

| Gate | Default | Purpose |
|------|---------|---------|
| truth_mode | "truth_safe_unverified" | Scientific claim boundary |
| claim_level | "computational_scaffold" | Implementation status |
| source_calibration_status | "uncalibrated_izhikevich_native_current" | Source units warning |
| source_projection_mode | "proxy_no_field_solve" | Declares laminar proxy solver |
| source_decomposition | "proxy_reduced_emitter" | Reduced source model |
| boundary_condition | "mean_zero_neumann" | Field boundary (declared, not enforced in v0.0.3) |
| gauge | "mean_zero" | Field gauge (declared, not enforced in v0.0.3) |
| csd_sign_convention | "proxy_positive_equals_extracellular_source_like" | CSD interpretation |
| field_solver_status | "laminar_proxy_no_pde" | Solver type (proxy, not PDE) |
| manifest_schema_version | "0.0.3" | Manifest format version |
| operator_status | {...} | Status of TFNE operators (E, S, C, Q, F, P, A, O, C) |

All gates propagate to manifest.json for downstream audit and validation.

**source_field_status** (new in v0.0.3) added to manifest when signals are present:
- field_claim_level: "laminar_proxy_uncalibrated" or "laminar_proxy_calibrated"
- physical_amplitude_claim_allowed: false (until calibration/validation complete)
- is_proxy: boolean flag indicating proxy vs. full PDE solver
- is_calibrated: boolean flag for source calibration status
- warnings: list of issues (non_finite_phi_e, non_finite_J_e, non_finite_CSD, etc.)

## Placeholder policies (Deferred to Post-Paper 1.0)

Do not implement these in the v0.1.x-v0.4.0 (Paper 1.0) sequence. They remain API scaffolds:

- Omission and global/local oddball paradigm execution (Paper 2.0/3.0)

- Paradigm.batch() — specification only; execution deferred to Paper 2.0
- Model.tune() — API stub; optimization planned for v0.0.5+
- Objective losses/regularizers/gates — builder only; not used in v0.0.3
- AGSDR optimizer — placeholder class; no real training
- Jaxley emitter bridge — deferred to post-Paper 1.0
- Optax adapter — guard only; no gradient-based training
- Full field PDE solver — laminar proxy only in v0.0.3
- MEG/EEG readouts — deferred to post-Paper 1.0
- Source conservation tests — framework planned for v0.0.4
- SPD tensor validation — planned for v0.0.4
- Passivity/causality tests — planned for v0.0.4

These APIs exist so designs can be experimented with before implementation locks the behavior.

## v0.0.3 vs. v0.0.2 changes

**New in v0.0.3:**
- RuntimeConfig dataclass and runtime() factory for documenting execution environment
- validate_source_field_status() helper for field solver claim assessment
- manifest_schema_version bumped to "0.0.3"
- source_field_status dict added to manifest (field_claim_level, is_proxy, is_calibrated, warnings)
- Runtime environment (device, dtype, x64_enabled, seed, n_steps) can be passed to manifest() for full context

**Preserved from v0.0.2:**
- All metadata gates and defaults unchanged
- API shape (configuration(), construct(), simulate(), probe(), record(), manifest())
- Izhikevich EIG and laminar field proxy architecture
- JSON-safe serialization with allow_nan=False
- Optional dependency guards (require_jaxley, require_optax)
