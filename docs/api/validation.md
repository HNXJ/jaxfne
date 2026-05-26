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

### `validate_config(cfg: Configuration) -> ConfigValidationResult`

Validate a Configuration for structural consistency and completeness.

**Parameters:**
- `cfg` (Configuration): Configuration to validate

**Returns:** `ConfigValidationResult` with validation status and error messages

**Checks:**
- All required fields are present (networks, emitters, fields, probes)
- Parameter ranges are valid
- No conflicting declarations
- Geometry is consistent

**Example:**
```python
import jaxfne as jtfne

cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column("V1", layers=["L2/3"], n=100)
cfg = cfg.cell_types({"E": 0.8, "I": 0.2})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["SPK", "Vm", "LFP-proxy"])

result = jtfne.validate_config(cfg)
if result.valid:
    print("✓ Configuration is valid")
else:
    print(f"✗ Validation failed: {result.errors}")
```

---

## ConfigValidationResult

```python
jaxfne.ConfigValidationResult
```

Result container from configuration validation.

### Attributes

- `valid` (bool): True if configuration passes all checks
- `errors` (list[str]): List of error messages (empty if valid)
- `warnings` (list[str]): Non-critical warnings
- `claim_status` (dict): Operator status declarations

### Methods

#### `print_summary()`

Print human-readable validation summary.

**Example:**
```python
result = jtfne.validate_config(cfg)
result.print_summary()
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
field = jtfne.project_sources_to_laminar_field(sources, geometry)
status = jtfne.validate_source_field_status(field)

if status["all_finite"]:
    print("✓ Field values are finite")
else:
    print("✗ NaN or Inf detected in field")
    print(f"  finite_LFP: {status['finite_LFP']}")
    print(f"  finite_CSD: {status['finite_CSD']}")
```

---

### `validate_projection_invariants(sources: jax.Array, field: FieldOutput) -> bool`

Check that field respects source-field relationships.

**Parameters:**
- `sources` (jax.Array): Original source signals [time, locations]
- `field` (FieldOutput): Computed field output

**Returns:** Boolean (True if valid)

**Invariants checked:**
- Field magnitude proportional to source magnitude
- No unphysical amplification/attenuation
- CSD sign convention consistent
- Spatial relationships preserved

**Example:**
```python
is_valid = jtfne.validate_projection_invariants(sources, field)
assert is_valid, "Field does not respect source relationship"
```

---

## Conservation Diagnostics

### `compute_conservation_proxy_diagnostics(sources, field) -> dict`

Compute conservation-inspired diagnostic metrics.

**Parameters:**
- `sources` (jax.Array): Source signals [time, locations]
- `field` (FieldOutput): Field output

**Returns:** Dictionary of diagnostic metrics

**Metrics:**
- `total_source_power`: Sum of |source|² over all locations and times
- `field_energy`: Sum of |LFP|² + |CSD|² over all locations and times
- `energy_ratio`: field_energy / total_source_power
- `source_moments`: Spatial center of mass and variance over time
- `field_moments`: Field center of mass over time

**Example:**
```python
diag = jtfne.compute_conservation_proxy_diagnostics(sources, field)

print(f"Source power: {diag['total_source_power']:.3e}")
print(f"Field energy: {diag['field_energy']:.3e}")
print(f"Energy ratio: {diag['energy_ratio']:.3f}")

# Should be <1.0 (field has lower energy than source in proxy mode)
assert diag['energy_ratio'] < 1.0
```

---

## Operator Status

### `operator_status() -> dict`

Get status declarations for all computational operators.

**Returns:** Dictionary mapping operator names to status strings

**Status values:**
- `"prototype_api"`: Preliminary implementation, subject to change
- `"stable_api"`: Core functionality, stable interface
- `"deprecated"`: Scheduled for removal; migration path provided
- `"planned"`: Not yet implemented

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

```python
import jax.numpy as jnp

def is_valid_signal(signals):
    """Check if signals contain only finite values."""
    finite_checks = [
        jnp.isfinite(signals.V_m).all(),
        jnp.isfinite(signals.spikes).all(),
    ]
    if signals.source is not None:
        finite_checks.append(jnp.isfinite(signals.source).all())
    return all(finite_checks)

if is_valid_signal(signals):
    print("✓ All signal values are finite")
else:
    print("✗ Signal contains NaN or Inf")
```

---

## Metadata Validation

### `config_truth_boundary(cfg: Configuration) -> dict`

Get truth/claim boundaries for a configuration.

**Parameters:**
- `cfg` (Configuration): Configuration to check

**Returns:** Dictionary with claim status for each operator

**Example:**
```python
boundaries = jtfne.config_truth_boundary(cfg)
print(f"Claim level: {boundaries['claim_level']}")
print(f"Truth mode: {boundaries['truth_mode']}")
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

# 1. Validate configuration
cfg = jtfne.Configuration()
# ... configure ...
cfg_result = jtfne.validate_config(cfg)
assert cfg_result.valid, f"Config invalid: {cfg_result.errors}"

# 2. Construct and simulate
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

# 3. Validate signals are finite
assert jtfne.is_valid_signal(signals), "Signals contain NaN/Inf"

# 4. Validate field relationships (if field was computed)
if signals.source is not None and signals.LFP is not None:
    field_valid = jtfne.validate_projection_invariants(
        signals.source, 
        FieldOutput(LFP=signals.LFP, CSD=signals.CSD, source=signals.source)
    )
    assert field_valid, "Field does not respect source relationship"

# 5. Check conservation diagnostics
if signals.source is not None and signals.LFP is not None:
    diag = jtfne.compute_conservation_proxy_diagnostics(
        signals.source,
        FieldOutput(LFP=signals.LFP, CSD=signals.CSD, source=signals.source)
    )
    print(f"Energy ratio: {diag['energy_ratio']:.3f}")

print("✓ All validation checks passed")
```

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
