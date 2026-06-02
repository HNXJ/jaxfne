# Skill: Validating Probe Reports

**When to use**: You're working with probe operators and need to understand or validate probe report contracts.

**Goal**: Learn what probe reports are, how to inspect them, and how to avoid common mistakes.

## What Are Probe Reports?

A **probe report** is JSON-safe metadata attached to every `ProbeReadout`. It documents:

- **What**: Operator kind (spk, vm, source, lfp_proxy, csd_proxy, eeg_proxy, meg_proxy, emm_proxy)
- **How**: Method used to compute the readout
- **Status**: Calibration state, field solver status, claim levels
- **Shape**: Data shape of the output
- **Assumptions**: List of assumptions made

Probe reports are designed for **inspection and validation**, not for hiding the limitations or uncertainties of proxy readouts.

## Required Fields

Every probe report must include **14 required fields**:

```python
required_fields = {
    "name",                           # operator name (string)
    "kind",                           # operator kind (spk, vm, source, lfp_proxy, etc.)
    "operator_status",                # simulated_proxy, physical_forward_model, etc.
    "method",                         # description of method (e.g., "threshold_or_emitter_spike_array")
    "data_shape",                     # output shape as string (e.g., "(100, 8)")
    "units_or_status",                # units or status (e.g., "proxy_units", "dimensionless")
    "calibration_status",             # calibration state (e.g., "uncalibrated_proxy")
    "source_calibration_status",      # source calibration (e.g., "uncalibrated_izhikevich_native_current")
    "source_projection_mode",         # how sources are projected (e.g., "proxy_no_field_solve")
    "source_decomposition",           # source decomposition (e.g., "proxy_reduced_emitter")
    "field_solver_status",            # field solver (e.g., "laminar_proxy_no_pde")
    "field_claim_level",              # claim level (e.g., "proxy_readout_only")
    "physical_amplitude_claim_allowed", # boolean: False for proxy operators
    "assumptions",                    # list of assumptions (e.g., ["assumption_1", "assumption_2"])
}
```

## Probe Operator Kinds

Valid kinds are:

- `spk` — Spike detection
- `vm` — Membrane voltage
- `source` — Laminar source density
- `lfp_proxy` — LFP proxy readout
- `csd_proxy` — CSD proxy readout
- `eeg_proxy` — EEG proxy readout
- `meg_proxy` — MEG proxy readout
- `emm_proxy` — Electrokinetic/movement proxy

## Operator-Specific Fields

Some operators include additional fields:

| Operator | Extra Fields |
|----------|--------------|
| `csd_proxy` | `CSD_sign_convention` (e.g., "positive_equals_extracellular_source") |
| `eeg_proxy` | `leadfield_status`, `sensor_geometry_status` |
| `meg_proxy` | `leadfield_status`, `sensor_geometry_status`, `orientation_convention` |

## How to Inspect a Probe Report

```python
import jaxfne as jtfne
from jaxfne.fields import csd_proxy_probe
import json

# Run a probe operator
csd = 42.0  # simulated CSD data shape (10, 16)
readout = csd_proxy_probe(csd)

# Inspect the report
print("Report keys:", readout.report.keys())
print("Kind:", readout.report["kind"])
print("Shape:", readout.report["data_shape"])
print("CSD convention:", readout.report["CSD_sign_convention"])

# Pretty-print the report
print(json.dumps(readout.report, indent=2))
```

## JSON Strictness

All probe reports must be **JSON-serializable with `allow_nan=False`**:

```python
from jaxfne.io import json_safe
import json

readout = csd_proxy_probe(...)
safe_report = json_safe(readout.report)
json.dumps(safe_report, allow_nan=False)  # must not raise
```

This ensures:
- No NaN or Inf values in the report
- No non-serializable types (numpy arrays, complex, etc.)
- Clean, deterministic output

## Common Mistakes and How to Avoid Them

### Mistake 1: Using `*_like` Terminology

❌ **Wrong:**
```python
"field_method": "lfp_like",
"CSD_sign_convention": "proxy_positive_equals_extracellular_source_like",
```

✅ **Correct:**
```python
"field_method": "lfp_proxy",
"CSD_sign_convention": "positive_equals_extracellular_source",
```

### Mistake 2: Putting Arrays in Report Metadata

❌ **Wrong:**
```python
{
    "leadfield_matrix": leadfield_array,  # numpy array
    "sensor_positions": [[0, 0, 0], [1, 1, 1]],  # nested lists
}
```

✅ **Correct:**
```python
{
    "leadfield_status": "available_but_not_serialized",
    "sensor_geometry_status": "32_contacts_at_1mm_spacing",
    "leadfield_matrix_shape": "(32, n_sources)",
}
```

### Mistake 3: Allowing NaN or Inf in Report Values

❌ **Wrong:**
```python
import math
report = {
    "default_value": float("nan"),  # NaN
    "infinity_flag": math.inf,      # Inf
}
```

✅ **Correct:**
```python
report = {
    "default_value": "unknown",
    "infinity_flag": "unbounded",
}
```

### Mistake 4: Omitting `data_shape`

❌ **Wrong:**
```python
report = {
    "kind": "lfp_proxy",
    # missing data_shape
}
```

✅ **Correct:**
```python
report = {
    "kind": "lfp_proxy",
    "data_shape": "(100, 16)",  # shape as string
}
```

### Mistake 5: Claiming Calibration from Proxy Readouts

❌ **Wrong:**
```python
report = {
    "kind": "eeg_proxy",
    "calibration_status": "empirically_calibrated",  # Not true for proxies
    "physical_amplitude_claim_allowed": True,        # False for proxies
}
```

✅ **Correct:**
```python
report = {
    "kind": "eeg_proxy",
    "calibration_status": "uncalibrated_proxy",
    "physical_amplitude_claim_allowed": False,
}
```

## Validation Commands

**Check if a report has all required fields:**
```python
from jaxfne.fields import csd_proxy_probe

readout = csd_proxy_probe(...)
required = {
    "name", "kind", "operator_status", "method", "data_shape",
    "units_or_status", "calibration_status", "source_calibration_status",
    "source_projection_mode", "source_decomposition", "field_solver_status",
    "field_claim_level", "physical_amplitude_claim_allowed", "assumptions",
}

missing = required - set(readout.report.keys())
if missing:
    print(f"Missing fields: {missing}")
else:
    print("✓ All required fields present")
```

**Check for JSON safety:**
```python
import json
from jaxfne.io import json_safe

readout = csd_proxy_probe(...)
safe = json_safe(readout.report)
try:
    json.dumps(safe, allow_nan=False)
    print("✓ Report is JSON-safe")
except ValueError as e:
    print(f"✗ JSON serialization failed: {e}")
```

**Check for forbidden terminology:**
```python
report_str = json.dumps(readout.report)
forbidden = ["_like", "-like", "lfp_like", "csd_like", "eeg_like", "meg_like"]
issues = [f for f in forbidden if f in report_str]
if issues:
    print(f"✗ Found forbidden terms: {issues}")
else:
    print("✓ No forbidden *_like terminology")
```

**Check that data_shape matches output:**
```python
import ast

readout = csd_proxy_probe(...)
reported_shape = ast.literal_eval(readout.report["data_shape"])
actual_shape = readout.data.shape
if reported_shape == actual_shape:
    print(f"✓ Shape matches: {actual_shape}")
else:
    print(f"✗ Shape mismatch: reported={reported_shape}, actual={actual_shape}")
```

## Examples to Study

| File | Operator(s) | Learn |
|------|-------------|-------|
| `examples/03_single_neuron_multimodal_probe.py` | spk, vm, source | Basic probe operators |
| `examples/04_two_neuron_ei_multimodal.py` | All 8 operators | Full probe range |
| `examples/05_network_100_ei_multimodal.py` | All 8 operators with network | Large-scale validation |

## Testing Probe Reports

The test suite validates probe reports:

```bash
# Run probe operator tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/test_probe_operators_v021.py -v

# Run v0.2.12 contract tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/test_probe_report_contract_v0212.py -v
```

## Summary Checklist

Before accepting a probe report:

- [ ] All 14 required fields present
- [ ] `kind` is one of: spk, vm, source, lfp_proxy, csd_proxy, eeg_proxy, meg_proxy, emm_proxy
- [ ] No `truth_mode`, `claim_level`, or other internal fields in public reports
- [ ] No `*_like` or `*_same` terminology
- [ ] `data_shape` matches actual output shape (as string)
- [ ] `physical_amplitude_claim_allowed` is False for proxy operators
- [ ] Operator-specific fields (CSD, EEG, MEG) present and correct
- [ ] Report is JSON-serializable with `allow_nan=False`
- [ ] No arrays, complex numbers, or non-serializable types
- [ ] `assumptions` list is complete and accurate

## Next Steps

- Read [Probe Operators Reference](../probe_operators.md)
- Study `tests/test_probe_report_contract_v0212.py` for examples
- Inspect examples with `python -c "from examples import <name>; readout = ...; print(readout.report)"`
