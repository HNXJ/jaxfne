# jaxfne doctrine

**Version:** 0.0.2  
**Status:** Design scaffold / computational framework, not biological validation.

## Identity

`jaxfne` = **JAX Field Neural Equations**.

A JAX-native engine for Tensor-Field Neural Equations (TFNE) source-to-field modeling.

## Core chain

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

## Claims

Allowed early claim:

```text
jaxfne provides a JAX-native computational scaffold for declaring emitters, source projections, fields, probes, objectives, and manifests.
```

Forbidden early claim:

```text
jaxfne validates a biological mechanism or physical CSD/LFP amplitude without calibration.
```

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

## v0.0.2 API naming

**Signals** (plural) is the canonical container name:
- Holds time_ms, V_m, spikes, sources, field, metadata.
- Returned by Model.simulate().
- Processed by Model.probe() or Model.record().
- Signal (singular) is a backward-compatibility alias.

**probe()** is the canonical TFNE readout method:
- Extracts named arrays from Signals.
- record() is an alias for user convenience.
- Both return a dict of arrays keyed by mode.

## v0.0.2 metadata gates

Configuration includes these immutable metadata fields (defaults):

| Gate | Default | Purpose |
|------|---------|---------|
| truth_mode | "truth_safe_unverified" | Scientific claim boundary |
| claim_level | "computational_scaffold" | Implementation status |
| source_calibration_status | "uncalibrated_izhikevich_native_current" | Source units warning |
| source_projection_mode | "proxy_no_field_solve" | Declares laminar proxy solver |
| source_decomposition | "proxy_reduced_emitter" | Reduced source model |
| boundary_condition | "mean_zero_neumann" | Field boundary (declared, not enforced in v0.0.2) |
| gauge | "mean_zero" | Field gauge (declared, not enforced in v0.0.2) |
| csd_sign_convention | "proxy_positive_equals_extracellular_source_like" | CSD interpretation |
| field_solver_status | "laminar_proxy_no_pde" | Solver type (proxy, not PDE) |
| manifest_schema_version | "0.0.2" | Manifest format version |
| operator_status | {...} | Status of TFNE operators (E, S, C, Q, F, P, A, O, C) |

All gates propagate to manifest.json for downstream audit and validation.

## Placeholder policies (v0.0.2)

Do not implement these in v0.0.2. They remain API scaffolds:

- Paradigm.batch() — specification only; execution planned for v0.0.4
- Model.tune() — API stub; optimization planned for v0.0.4
- Objective losses/regularizers/gates — builder only; not used in v0.0.2
- AGSDR optimizer — placeholder class; no real training
- Jaxley emitter bridge — connector only; no compartment simulations
- Optax adapter — guard only; no gradient-based training
- Full field PDE solver — laminar proxy only in v0.0.2
- MEG/EEG readouts — not supported in v0.0.2

These APIs exist so designs can be experimented with before implementation locks the behavior.
