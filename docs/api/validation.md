# Validation API

Invariant checks and validation functions for numerical consistency and structural integrity.

**Module note (2026-07-24):** the functions in "Signal & Field Validation" and "Conservation
Diagnostics" below (`validate_source_field_status`, `validate_projection_invariants`,
`compute_conservation_proxy_diagnostics`) are physically implemented in
`jaxfne/fields/proxy.py`/`jaxfne/fields/diagnostics.py`, not `jaxfne/validation.py` — they're
documented here (and in [Fields](fields.md#validation-diagnostics)) because they're the
signal/field-facing validation surface a caller reaches for alongside `is_valid_signal`.
The rest of this page (Conductivity Validators onward) documents `jaxfne/validation.py`'s own
API, which this page did not previously cover at all.

## Overview

The validation module provides tools to:
1. Check configuration validity
2. Verify numerical consistency (finite values, conservation)
3. Validate field and source relationships
4. Report configuration and operator status

---

## Configuration Validation

> **REMOVED (2026-06-30).** `validate_config`, `ConfigValidationResult`,
> `config_truth_boundary`, `load_config`, `JaxFNEConfig`, and the rest of the
> dict-backed `JaxFNEConfig`/`.jcfg.json` config-path API (`config_to_configuration`,
> `config_to_simulation`, `config_to_geometry`, `config_to_trial_batch`) were
> deliberately deleted along with their 21 dependent tests. See the resolved note
> in `jaxfne/_pipeline.py` (top of file) for the removal record. None of these
> names exist in the current package
> (`hasattr(jaxfne, 'JaxFNEConfig')` → `False`, confirmed 2026-07-03).
>
> There is no direct replacement for `validate_config`/`ConfigValidationResult`
> for the fluent `Configuration` object — `jaxfne.builders.validate_configuration(cfg: Configuration, strict: bool = True) -> dict`
> exists in `jaxfne/builders.py` but is a different name/signature/return type,
> not a drop-in substitute, and is not yet re-exported/documented at the root
> level here. Until it is, validate a `Configuration` structurally via
> `jtfne.construct(cfg)` (raises on structural problems) and the signal/field
> checks below (`is_valid_signal`, `validate_projection_invariants`,
> `compute_conservation_proxy_diagnostics`).

---

## Signal & Field Validation

### `validate_source_field_status(field_output: FieldOutput) -> dict`

Validate field output for numerical consistency.

**Parameters:**
- `field_output` (FieldOutput): Field computation results

**Returns:** Dictionary of validation checks

**Checks:**
- All arrays are finite (no NaN/Inf)
- LFP and CSD have expected shapes
- Mean-zero constraint (if applicable)
- Sign conventions respected

**Example:**
```python
field = jtfne.project_sources_to_laminar_field(sources, positions)
status = jtfne.validate_source_field_status(field)

if status["all_finite"]:
    print("✓ Field values are finite")
else:
    print("✗ NaN or Inf detected in field")
    print(f"  finite_LFP: {status['finite_LFP']}")
    print(f"  finite_CSD: {status['finite_CSD']}")
```

---

### `validate_projection_invariants(*, sources, positions, kernel, source_proxy, phi_e_proxy, csd_proxy, lfp_proxy, mode="row_normalize") -> dict`

Check structural invariants of the laminar proxy projection: kernel
row-normalization (row-stochastic to `tol=1e-6`, skipped when
`mode="density_preserving"`) and finiteness/shape consistency of the proxy
arrays. This checks the proxy operator's internal consistency only — it makes
no claim of physical correctness.

**Parameters (all keyword-only):**
- `sources` (jax.Array): Original source signals `[time, n_emitters]`
- `positions` (jax.Array): Emitter positions `[n_emitters, 3]`
- `kernel` (jax.Array): Projection kernel `[n_contacts, n_emitters]`
- `source_proxy`, `phi_e_proxy`, `csd_proxy`, `lfp_proxy` (jax.Array): Proxy outputs to check
- `mode` (str): Projection mode used (`"row_normalize"` or `"density_preserving"`)

**Returns:** JSON-safe `dict` of per-invariant pass/fail diagnostics (not a bool)

**Example:**
```python
report = jtfne.validate_projection_invariants(
    sources=sources, positions=positions, kernel=kernel,
    source_proxy=source_proxy, phi_e_proxy=phi_e_proxy,
    csd_proxy=csd_proxy, lfp_proxy=lfp_proxy, mode="density_preserving",
)
assert not report["warnings"], f"Projection invariants failed: {report['warnings']}"
```

---

## Conservation Diagnostics

### `compute_conservation_proxy_diagnostics(*, source=None, phi_e=None, csd=None, lfp=None, field_solution=None, source_calibration_status=..., field_solver_status=..., field_claim_level=...) -> dict`

Compute conservation-inspired proxy diagnostics over existing source/field
arrays. All array parameters are optional and keyword-only — pass whichever of
`source`/`phi_e`/`csd`/`lfp` you have, or a `field_solution: FieldOutput`.

**Parameters (all keyword-only, all optional):**
- `source`, `phi_e`, `csd`, `lfp` (jax.Array): Arrays to diagnose
- `field_solution` (FieldOutput, optional): Alternative to passing the arrays individually
- `source_calibration_status`, `field_solver_status`, `field_claim_level` (str): Truth-gate metadata carried through into the report, conservative defaults

**Returns:** Dictionary of diagnostic metrics (raises `ValueError` on a non-finite diagnostic value rather than returning one)

**Example:**
```python
diag = jtfne.compute_conservation_proxy_diagnostics(source=sources, lfp=signals.LFP, csd=signals.CSD)
print(diag["source_norm_l1"], diag["warnings"])
```

---

## Operator Status

### `operator_status() -> dict`

Get status declarations for all computational operators.

**Returns:** Dictionary mapping operator names to status strings

**Status values (current registry):**
- `"prototype_api"`: implemented, preliminary interface, subject to change
- `"not_implemented"`: declared in the registry, no implementation yet (e.g. `C_mu_nu`)

**Example:**
```python
status = jtfne.operator_status()
print(f"E (emitter): {status['E_theta']}")
print(f"S (source): {status['S_WDR']}")
print(f"F (field): {status['F_field']}")
print(f"P (probe): {status['P_probe']}")
```

---

## Finite-Value Checking

### Automatic NaN/Inf Detection

All jaxfne simulations automatically check for NaN/Inf:

```python
try:
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
except ValueError as e:
    print(f"Simulation produced NaN/Inf: {e}")
```

### Manual Check

`jtfne.is_valid_signal(signals)` checks `V_m` and `spikes` for finite values and
returns `False` on any missing array, NaN, or Inf:

```python
if jtfne.is_valid_signal(signals):
    print("✓ All required signal values are finite")
else:
    print("✗ Signal contains NaN, Inf, or a missing required array")
```

---

## Metadata Validation

> **REMOVED (2026-06-30).** `config_truth_boundary(cfg: JaxFNEConfig)` was
> deleted along with the rest of the `JaxFNEConfig` config-path API — see the
> note under "Configuration Validation" above. There is no direct root-level
> replacement; truth-gate/claim-level metadata for the fluent `Configuration`
> pipeline is carried on the objects themselves (e.g. `FieldOutput`,
> `RunReceipt`) rather than fetched via a standalone boundary-report call.

---

## Conductivity Validators

`jaxfne/validation.py`'s own API starts here — each returns a JSON-safe `dict` with
`is_valid`/`is_finite`/`status`/`evidence` keys, never raising on bad input:

- `validate_scalar_conductivity(sigma, *, tolerance=1e-10)` (`:104`) — finite, positive scalar.
- `validate_diagonal_conductivity(sigma, *, tolerance=1e-10)` (`:138`) — per-axis diagonal
  conductivity tensor.
- `validate_full_spd_conductivity(sigma, *, tolerance=1e-10)` (`:204`) — full symmetric
  positive-definite conductivity tensor.
- `validate_field_arrays_finite(**arrays)` (`:300`) — finiteness check over an arbitrary set
  of named field arrays.

## Field Admissibility Report

### `build_field_admissibility_report(field_output=None, cfg_metadata=None, signals_field=None) -> dict`

The field-admissibility report — always sets
`physical_amplitude_calibrated=False` and `conductivity_status="proxy_not_solved"` for the
laminar proxy path; pulls `field_solver_status`/`field_claim_level`/`boundary_condition`/
`gauge` from `cfg_metadata` or the field output's own diagnostics, falling back to the
proxy defaults (`linear_solver`/`proxy_readout`/`mean_zero_neumann`/`mean_zero`) otherwise.

## Calibration

### `CalibrationSpec`

Declares a calibration state without upgrading a physical-amplitude claim (`:417`,
v0.2.5 contract). Constructor: `CalibrationSpec(*, name, target, mode="uncalibrated_native",
scale=None, units=None, reference=None, description=None)` — `target` is one of
`source`/`field`/`probe`/`readout`/`objective`; `mode` must be one of `ALLOWED_MODES`
(`uncalibrated_native`, `toy_scale`, `relative_normalized`, `empirical_gain_candidate`,
`physical_units_candidate`, `calibrated_empirical`) or the constructor raises `ValueError`.
`.to_dict()` returns a JSON-safe representation.

### `make_calibration_report(spec, *, readout_kind=None, ...) -> dict` (`:497`)

Builds a report around a `CalibrationSpec` (or an equivalent dict).

## Diagnostic Builders

Each returns a JSON-safe `dict`, all under the same truth-gate discipline:

- `make_source_balance_diagnostic(...)` (`:575`)
- `make_gauge_diagnostic(...)` (`:622`)
- `make_boundary_diagnostic(...)` (`:675`)
- `make_manufactured_residual_diagnostic(...)` (`:708`)
- `make_field_operator_status(...)` (`:753`) — **not** the same function as
  `jtfne.operator_status()` documented above; this one builds a per-field-operator status
  dict scoped to a single validation report.

## Poisson Admissibility Gates

Specification-only gates (nothing solved the Poisson equation numerically when these were written; the experimental
solver that now exists, `jaxfne.fields.experimental_poisson_1d`, is documented in
[Fields](fields.md#field-solvers) — these gates are not yet wired to it):

- `validate_poisson_spd_conductivity(...)` (`:805`)
- `validate_poisson_source_conservation(...)` (`:850`)
- `validate_poisson_gauge_condition(...)` (`:898`) — only `gauge="mean_zero"` is implemented;
  any other value raises `NotImplementedError`.
- `validate_poisson_field_arrays(...)` (`:942`)
- `build_poisson_admissibility_report(...) -> dict` (`:988`) — assembles the above into one report.

## Basis Spec Validation

- `validate_basis_spec(spec) -> dict` (`:1115`) — lazy-imports `BasisSpec`/
  `_FUTURE_FIELD_REGIMES` from `.core` inside the function body specifically to avoid a
  circular import (`core` imports `validation`, not vice versa).
- `basis_claim_gate(...)` (`:1177`).

## JAX Re-compilation Guard

### `CompilationRegistry` / `compilation_registry` (`:1240`, singleton at `:1317`)

Tracks shape signatures `(B, Z, C, T)` across JIT traces and warns/raises on an unexpected
retrace. `.reset()` clears counters; `.set_mode("warning"|"exception"|"off")` controls
alert behavior.

### `make_recompilation_guard(fn, name, recompilation_guard, B, Z, C, T) -> wrapped` (`:1320`)

Wraps `fn` to count trace/compile events. **Constraint (verified 2026-07-05):** the returned
`wrapped` must be passed directly into `jax.jit(...)` before ever being called, and only the
jitted result invoked — calling `wrapped` directly makes every Python call count as a
"compile," producing false-positive re-compilation warnings. All current call sites
(`jaxfne/_model_simulate.py`) follow `jax.jit(make_recompilation_guard(...))`.

---

## Best Practices

1. **Always validate configuration:** Check before construction
2. **Verify simulation output:** Check for finite values
3. **Validate field relationships:** Ensure source-field consistency
4. **Document operator status:** Include in published results
5. **Check conservation properties:** Use diagnostics for validation

**Example: Complete Validation Workflow**

```python
import jaxfne as jtfne

# 1. Build and construct (fluent Configuration -> Model)
cfg = jtfne.Configuration()
# ... configure ...
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

# 2. Validate signals are finite
assert jtfne.is_valid_signal(signals), "Signals contain NaN/Inf"

# 3. Validate field relationships (if field was computed) — validate_projection_invariants
#    operates on the raw proxy arrays, not a FieldOutput, and is keyword-only
if signals.source is not None and signals.LFP is not None:
    report = jtfne.validate_projection_invariants(
        sources=signals.source, positions=positions, kernel=kernel,
        source_proxy=signals.source, phi_e_proxy=signals.phi_e,
        csd_proxy=signals.CSD, lfp_proxy=signals.LFP, mode="density_preserving",
    )
    assert not report["warnings"], f"Projection invariants failed: {report['warnings']}"

# 4. Check conservation diagnostics
if signals.source is not None and signals.LFP is not None:
    diag = jtfne.compute_conservation_proxy_diagnostics(
        source=signals.source, lfp=signals.LFP, csd=signals.CSD,
    )
    print(diag["warnings"])

print("✓ All validation checks passed")
```

Note: the dict-backed `JaxFNEConfig`/`load_config(...)` config-path API
(distinct from the fluent `Configuration` object used above with `construct()`)
was removed 2026-06-30 — see the "Configuration Validation" section above.

---

## Common Validation Errors

### NaN in Signals

**Cause:** Unstable dynamics, extreme parameter values, or numerical issues

**Solution:**
- Reduce timestep (dt_ms)
- Check parameter ranges (Izhikevich parameters)
- Use float64 precision for long simulations
- Reduce external input current

### Invalid Field

**Cause:** Source-field mismatch, incorrect geometry, or solver issue

**Solution:**
- Verify source projection (validate_projection_invariants)
- Check geometry consistency
- Ensure source values are reasonable

### Configuration Errors

**Cause:** Missing fields, invalid parameters, or conflicting declarations

**Solution:**
- Call `construct(cfg)`, which raises on structural problems (`validate_config`
  no longer exists — see "Configuration Validation" above)
- Check error messages for specific issues
- Verify all required fields are present

---

## See also

- [Core API](core.md) — Configuration and Model
- [Fields API](fields.md) — Field validation functions
- [API reference](index.md)
