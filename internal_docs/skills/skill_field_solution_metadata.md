# Field Solution Metadata Validation Skill

This guide teaches you how to validate field solution metadata contracts for proxy and future solved physical fields in jaxfne.

## What Are Field Solution Reports?

A field solution report is a JSON-safe dictionary containing 18+ required fields that describe:

1. **Solver Status** — Is this a proxy readout or a solved PDE?
2. **Solver Properties** — Boundary conditions, gauge, convergence metrics
3. **Field Admissibility** — Are the arrays finite? Is current conserved?
4. **Amplitude Claims** — Can we claim physical units?

Unlike probe operators (which measure signals), field solutions describe the computational path from source currents to extracellular fields.

## When to Use Field Solution Metadata

Use field solution metadata reports when you:

- Generate field outputs from `project_laminar_sources()` or future solvers
- Need to declare field admissibility and claim constraints
- Are building new field solvers (Poisson, Bidomain, etc.)
- Want to document solver convergence and accuracy
- Need to prevent downstream misuse of proxy readouts as physical fields

## The 18 Required Fields

All field solution reports must include:

| Field | Type | Proxy Value | Purpose |
|-------|------|-------------|---------|
| `field_solver_status` | str | `"laminar_proxy_no_pde"` | Solver type (proxy, solved, future) |
| `solver_name` | str | `"laminar_proxy"` | Human-readable solver identifier |
| `boundary_condition` | str | `"declared_metadata_only"` | BC declaration for proxy; computed for real solvers |
| `gauge` | str | `"declared_metadata_only"` | Gauge convention declaration |
| `csd_sign_convention` | str | `"positive_equals_extracellular_source"` | CSD polarity (canonical form, no `_like` suffix) |
| `current_density_layout` | str | `"not_applicable"` | J_e computation status |
| `solver_residual_l2_relative` | float or null | `null` | Relative L2 residual (null for proxy) |
| `n_iterations` | int or null | `null` | Solver iteration count (null for proxy) |
| `converged` | bool or null | `null` | Convergence flag (null for proxy) |
| `finite_phi_e` | bool | `true` (if arrays exist) | Extracellular potential finitude |
| `finite_J_e` | bool | `false` (proxy) | Current density finitude (false: not computed) |
| `finite_CSD` | bool | `true` (if arrays exist) | CSD finitude |
| `field_claim_level` | str | `"proxy_readout_only"` | Claim authority (proxy, admissible, empirical) |
| `physical_amplitude_claim_allowed` | bool | `false` | Can we use physical units? |
| `source_projection_mode` | str | `"proxy_no_field_solve"` | How sources map to field |
| `source_current_conservation_status` | str | `"not_applicable_proxy_mode"` | Conservation test status |
| `source_conservation_tested` | bool | `false` | Was conservation validated? |
| `source_conservation_claim_allowed` | bool | `false` | Can we claim conserved sources? |

## Field Solver Status Values

Use these standardized values for `field_solver_status`:

- **`"laminar_proxy_no_pde"`** — Laminar readout without PDE solve (current: v0.2.13)
- **`"solved_resistive_poisson"`** — Resistive Poisson solver (future: v0.2.14+)
- **`"solved_bidomain"`** — Bidomain PDE solver (future: v0.2.15+)
- **`"specified_future_module"`** — Reserved for future physical solvers

## Claim Levels (Conservative)

Field claim levels express confidence in field admissibility:

- **`"proxy_readout_only"`** — Proxy output; no physical claim
- **`"physical_admissible_candidate"`** — Solver output; passes admissibility checks (finite, SPD, conservative) but not empirically validated
- **`"empirical_candidate"`** — Solver output validated against experimental benchmarks (future phase)

**Rule:** Never claim `"empirical_candidate"` or higher without external validation evidence (receipts).

## Proxy vs. Solved Fields

| Aspect | Proxy (v0.2.13) | Solved (v0.2.14+) |
|--------|-----------------|-------------------|
| `field_solver_status` | `laminar_proxy_no_pde` | `solved_resistive_poisson` |
| `boundary_condition` | `declared_metadata_only` | Computed; e.g., `dirichlet_zero` |
| `gauge` | `declared_metadata_only` | Computed; e.g., `mean_zero` |
| `solver_residual_l2_relative` | `null` | Float ≥ 0 |
| `n_iterations` | `null` | Int > 0 |
| `converged` | `null` | Boolean |
| `physical_amplitude_claim_allowed` | `false` | `true` if admissible, `false` if not |
| `field_claim_level` | `proxy_readout_only` | `physical_admissible_candidate` or higher |
| `finite_J_e` | `false` | `true` if computed |

## Code Examples

### Inspecting a Field Solution Report

```python
from jaxfne.fields import project_laminar_sources
import jax.numpy as jnp

# Create a proxy field output
sources = jnp.ones((100, 10))
positions = jnp.random.normal(size=(10, 3))
field_out = project_laminar_sources(sources, positions, n_contacts=16)

# Access the report
report = field_out.diagnostics

# Check key fields
print(f"Solver: {report['field_solver_status']}")
print(f"Claim level: {report['field_claim_level']}")
print(f"Physical amplitude allowed: {report['physical_amplitude_claim_allowed']}")
print(f"Finite CSD: {report['finite_CSD']}")
```

### Validating Field Metadata

```python
import json

# Ensure all 18 required fields are present
required_fields = {
    "field_solver_status",
    "solver_name",
    "boundary_condition",
    "gauge",
    "csd_sign_convention",
    "current_density_layout",
    "solver_residual_l2_relative",
    "n_iterations",
    "converged",
    "finite_phi_e",
    "finite_J_e",
    "finite_CSD",
    "field_claim_level",
    "physical_amplitude_claim_allowed",
    "source_projection_mode",
    "source_current_conservation_status",
    "source_conservation_tested",
    "source_conservation_claim_allowed",
}

missing = required_fields - set(report.keys())
assert not missing, f"Missing fields: {missing}"

# Ensure JSON strictness (no NaN/Inf)
json.dumps(report, allow_nan=False)  # Raises if NaN or Inf present

# Validate CSD sign convention (canonical form, no _like)
assert report["csd_sign_convention"] == "positive_equals_extracellular_source"

print("✓ Field report passes validation")
```

### Checking Proxy Status

```python
# Verify proxy-specific constraints
if report["field_solver_status"] == "laminar_proxy_no_pde":
    # These must be null for proxy
    assert report["n_iterations"] is None
    assert report["converged"] is None
    assert report["solver_residual_l2_relative"] is None
    
    # Proxy cannot claim physical amplitude
    assert report["physical_amplitude_claim_allowed"] is False
    assert report["field_claim_level"] == "proxy_readout_only"
    
    # Conservation is untested in proxy
    assert report["source_conservation_tested"] is False
    assert report["source_conservation_claim_allowed"] is False
    
    print("✓ Proxy constraints verified")
```

### Creating a Future Solved Field Report

(This is a template for v0.2.14+; do NOT implement Poisson in v0.2.13)

```python
from jaxfne.fields import _make_field_solution_report

# Example: Resistive Poisson solver output
report = _make_field_solution_report(
    field_solver_status="solved_resistive_poisson",
    solver_name="resistive_poisson_cg",
    boundary_condition="dirichlet_zero_boundary",
    gauge="mean_zero_enforced",
    csd_sign_convention="positive_equals_extracellular_source",
    current_density_layout="computed_j_e",
    solver_residual_l2_relative=1.2e-5,  # Actual residual
    n_iterations=47,  # Actual iteration count
    converged=True,  # Actual convergence status
    finite_phi_e=True,
    finite_J_e=True,
    finite_CSD=True,
    field_claim_level="physical_admissible_candidate",  # Not "empirical" yet
    physical_amplitude_claim_allowed=True,  # Only if admissible
    source_projection_mode="solved_conservation_enforced",
    source_current_conservation_status="conservation_tested_pass",
    source_conservation_tested=True,
    source_conservation_claim_allowed=True,
)
```

## Common Mistakes

### 1. Using `_like` Terminology

**WRONG:**
```python
report["csd_sign_convention"] = "proxy_positive_equals_extracellular_source_like"
```

**CORRECT:**
```python
report["csd_sign_convention"] = "positive_equals_extracellular_source"
```

**Why:** The `_like` suffix is forbidden in public reports. Use the canonical form.

---

### 2. Allowing Claim Mismatch

**WRONG:**
```python
# Proxy field but claiming physical amplitude
report["field_solver_status"] = "laminar_proxy_no_pde"
report["physical_amplitude_claim_allowed"] = True  # ✗ False for proxy!
```

**CORRECT:**
```python
# Proxy fields never claim physical amplitude
report["field_solver_status"] = "laminar_proxy_no_pde"
report["physical_amplitude_claim_allowed"] = False
```

**Why:** Proxy fields are dimensionless. Only solved fields that pass admissibility checks can claim physical units.

---

### 3. Non-Null Solver Metrics for Proxy

**WRONG:**
```python
# Proxy, but filling in solver metrics
report["field_solver_status"] = "laminar_proxy_no_pde"
report["n_iterations"] = 0  # Should be null!
report["converged"] = False  # Should be null!
```

**CORRECT:**
```python
# Proxy leaves solver metrics null
report["field_solver_status"] = "laminar_proxy_no_pde"
report["n_iterations"] = None
report["converged"] = None
```

**Why:** Proxy has no solver; these fields make no sense and cause confusion downstream.

---

### 4. Computing J_e in Proxy Mode

**WRONG:**
```python
# Proxy but claiming J_e is computed
report["field_solver_status"] = "laminar_proxy_no_pde"
report["finite_J_e"] = True
report["current_density_layout"] = "computed_j_e"
```

**CORRECT:**
```python
# Proxy does not compute J_e
report["field_solver_status"] = "laminar_proxy_no_pde"
report["finite_J_e"] = False
report["current_density_layout"] = "not_applicable"
```

**Why:** Proxy readouts lack PDE solve; current density is undefined.

---

### 5. Testing Claim Without Test

**WRONG:**
```python
# Claiming conservation was tested, but field is proxy
report["field_solver_status"] = "laminar_proxy_no_pde"
report["source_conservation_tested"] = True  # Untrue for proxy
report["source_conservation_claim_allowed"] = True  # False for proxy
```

**CORRECT:**
```python
# Proxy never tests conservation
report["field_solver_status"] = "laminar_proxy_no_pde"
report["source_conservation_tested"] = False
report["source_conservation_claim_allowed"] = False
```

**Why:** Proxy has no conserved solution; claims are invalid.

---

## Validation Commands

### Syntax and Import Check

```bash
cd /Users/hamednejat/workspace/main/jaxfne
python -m py_compile jaxfne/fields.py
python -c "from jaxfne.fields import _make_field_solution_report; print('✓ Helper imports OK')"
```

### Unit Test for Field Metadata

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests/test_field_solution_metadata_v0213.py -v
```

### Full Test Suite

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
```

Expected: 627+ tests pass (609 baseline + 7 v0.2.11 + 11 v0.2.12 + new v0.2.13 field tests)

### Example Validation

```bash
python examples/03_single_neuron_multimodal_probe.py
# Check outputs/v023_single_neuron_multimodal/probe_report.json includes hardened field metadata
```

## Vocabulary Audit

Field solution reports must follow strict vocabulary rules:

### Forbidden

- ❌ `proxy_positive_equals_extracellular_source_like` (old; contains `_like`)
- ❌ `lfp_like`, `csd_like`, `eeg_like`, `meg_like` (forbidden in any context)
- ❌ `truth_mode` (internal only; not in public reports)
- ❌ `biological_metabolism` without `not_` prefix (proxy cannot claim metabolism)
- ❌ Claims of "validated", "calibrated", or "empirical" without receipts

### Canonical (Required)

- ✓ `positive_equals_extracellular_source` (CSD sign convention)
- ✓ `laminar_proxy_no_pde` (proxy solver status)
- ✓ `proxy_readout_only` (proxy claim level)
- ✓ `not_applicable_proxy_mode` (conservation status in proxy)
- ✓ `declared_metadata_only` (proxy boundary/gauge status)

## Summary Checklist

After creating or modifying a field solution report:

- [ ] All 18 required fields present
- [ ] JSON-safe (no NaN/Inf; `json.dumps(..., allow_nan=False)` succeeds)
- [ ] No `truth_mode` in field report
- [ ] No `_like` terminology anywhere
- [ ] `csd_sign_convention` equals canonical value
- [ ] Proxy fields: `n_iterations`, `converged`, `solver_residual_l2_relative` are all null
- [ ] Proxy fields: `physical_amplitude_claim_allowed` is false
- [ ] Proxy fields: `field_claim_level` is `"proxy_readout_only"`
- [ ] Proxy fields: `current_density_layout` is `"not_applicable"`
- [ ] Future solved fields: all metrics (iterations, residual, converged) are non-null
- [ ] Future solved fields: `physical_amplitude_claim_allowed` matches admissibility
- [ ] Examples validate with correct field metadata
- [ ] Version remains 0.2.10 (no bump for field hardening)

## Next Steps

If implementing a new field solver (v0.2.14+):

1. Inherit `_make_field_solution_report()` pattern
2. Update `field_solver_status` and `solver_name` appropriately
3. Compute and fill solver metrics (`n_iterations`, `converged`, `solver_residual_l2_relative`)
4. Test that claim levels match admissibility (finite arrays, conservation, SPD)
5. Add solver-specific validation tests
6. Create skill documentation for new solver
7. Update examples to use new solver option

---

**Version:** v0.2.13  
**Last Updated:** 2026-05-20  
**Status:** Field solution metadata contract hardened; no Poisson solver in this phase.
