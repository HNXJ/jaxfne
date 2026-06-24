# Validation API

Invariant checks and validation functions for numerical consistency and structural integrity.

## Overview

The validation module provides tools to:
1. Check configuration validity
2. Verify numerical consistency (finite values, conservation)
3. Validate field and source relationships
4. Report configuration and operator status

---

## Configuration Validation

### `validate_config(cfg: JaxFNEConfig) -> ConfigValidationResult`

Validate a `JaxFNEConfig` (the dict-backed config produced by `load_config`/the
config-path APIs — not the fluent `Configuration` builder) for structural
consistency and completeness. Does not raise for normal validation failures;
escalated truth claims and missing required sections are returned as blocking
issues.

**Parameters:**
- `cfg` (JaxFNEConfig): Config to validate

**Returns:** `ConfigValidationResult` with validation status and issue messages

**Example:**
```python
import jaxfne as jtfne

cfg = jtfne.JaxFNEConfig(...)
result = jtfne.validate_config(cfg)
if result.valid:
    print("✓ Configuration is valid")
else:
    print(f"✗ Validation failed: {result.issues}")
```

---

## ConfigValidationResult

```python
jaxfne.ConfigValidationResult
```

Frozen dataclass container from configuration validation.

### Attributes

- `valid` (bool): True if configuration passes all checks
- `issues` (tuple[str, ...]): Blocking issue messages (empty if valid)
- `warnings` (tuple[str, ...]): Non-critical warnings
- `truth_boundary` (dict): Declared truth/claim-level section
- `schema_version` (str): Schema version the config was validated against

### Methods

#### `to_dict() -> dict`

Return a JSON-safe dict of the result (`valid`, `issues`, `warnings`,
`truth_boundary`, `schema_version`).

**Example:**
```python
result = jtfne.validate_config(cfg)
print(result.to_dict())
```

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

### `config_truth_boundary(cfg: JaxFNEConfig) -> dict`

Reporting/passthrough helper — returns a JSON-safe copy of `cfg.truth` exactly
as stored, without re-validating it. Call `validate_config` first to confirm
the truth section is structurally correct.

**Parameters:**
- `cfg` (JaxFNEConfig): Config to check

**Returns:** Dictionary with status for each operator

**Example:**
```python
boundaries = jtfne.config_truth_boundary(cfg)
print(f"Model status: {boundaries['model_status']}")
print(f"Run status: {boundaries['run_status']}")
print(f"Field solver status: {boundaries['field_solver_status']}")
```

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

Note: `validate_config(cfg: JaxFNEConfig)` validates the separate dict-backed
`JaxFNEConfig` produced by `load_config(...)`, not the fluent `Configuration`
object used above with `construct()` — the two config representations are not
interchangeable; see `AGENTS.md` for the config-path vs fluent-builder
distinction.

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
- Use validate_config before construction
- Check error messages for specific issues
- Verify all required fields are present

---

## See also

- [Core API](core.md) — Configuration and Model
- [Fields API](fields.md) — Field validation functions
- [API reference](index.md)
