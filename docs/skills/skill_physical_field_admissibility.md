# Skill: Physical Field Admissibility Validation

## When to Use This Skill

Use this skill when you need to:
- Validate that a field solver output is mathematically and physically sound
- Check Poisson solver admissibility before integrating into jaxfne
- Ensure conductivity tensors are well-posed (SPD)
- Verify source/flux conservation
- Test gauge conditions
- Debug solver convergence issues

**Not for:** Claiming biological calibration or physical amplitude without separate empirical validation.

---

## Quick Start

### Example: Validate a Field Solution

```python
import jaxfne
from jaxfne.validation import build_poisson_admissibility_report

# Assume you have a solver output:
phi_e = ... # extracellular potential
J_e = ... # current density
CSD = ... # current source density (derived)

# Compute required diagnostics
integrated_source = ... # sum of emitter sources
integrated_boundary_flux = ... # boundary integral
mean_potential = phi_e.mean()

# Build admissibility report
report = build_poisson_admissibility_report(
    conductivity=conductivity_tensor,
    integrated_source=integrated_source,
    integrated_boundary_flux=integrated_boundary_flux,
    mean_potential=mean_potential,
    phi_e=phi_e,
    J_e=J_e,
    CSD=CSD,
    solver_residual_l2=5e-7,
    n_iterations=200,
    converged=True
)

# Check if admissible
if report["admissibility_status"] == "admissible":
    print("✓ Solution is admissible for physical field modeling")
else:
    print("✗ Solution failed admissibility gates")
    for gate, result in report["gates"].items():
        if not result.get("passed"):
            print(f"  Failed: {gate}")
            print(f"  {result.get('message')}")
```

---

## Five Admissibility Gates

### Gate 1: Conductivity Symmetric Positive Definite (SPD)

**What it checks:** Conductivity tensor is symmetric and all eigenvalues are positive.

**Why it matters:** SPD is required for well-posedness. Non-SPD conductivity leads to ill-posed problems with no unique solution.

**How to use:**

```python
from jaxfne.validation import validate_poisson_spd_conductivity

is_spd, msg = validate_poisson_spd_conductivity(conductivity_tensor)
print(msg)  # e.g., "SPD verified; min eigenvalue 1.2e-02"
assert is_spd, "Conductivity must be SPD"
```

**Common issues:**
- Conductivity has negative eigenvalues → conductivity tensor is wrong or unphysical
- Conductivity is not symmetric → matrix error or read-in issue
- Conductivity is scalar when matrix expected → shape mismatch

**Fix:** Re-examine conductivity definition. For laminar column, use diagonal or full symmetric tensor. Check reference conductivity values from literature.

### Gate 2: Source Conservation

**What it checks:** Integrated source over domain equals (negative) boundary flux.

$$\int I_{\mathrm{src}} = -\int \sigma \nabla \phi_e \cdot \hat{n}$$

**Why it matters:** Ensures charge conservation. Imbalance indicates solver error, incorrect boundary conditions, or flux computation bug.

**How to use:**

```python
from jaxfne.validation import validate_poisson_source_conservation

is_conserved, msg, residual = validate_poisson_source_conservation(
    integrated_source,
    integrated_boundary_flux,
    tolerance=1e-6
)
print(msg)  # e.g., "source conserved; residual 1.1e-07"
assert is_conserved, f"Source not conserved: {msg}"
```

**Parameters:**
- `integrated_source`: total current source (e.g., sum over all neurons)
- `integrated_boundary_flux`: flux integral at boundary  (compute from $-\int \sigma \nabla \phi_e \cdot \hat{n}$)
- `tolerance`: allowed absolute residual (default 1e-6; typically 1e-8 for production solvers)

**Common issues:**
- Residual > tolerance → solver did not converge fully, or boundary condition not correctly applied
- Source and flux have opposite signs → sign convention error
- One integral is much larger → likely computation error on one side

**Fix:**
1. Verify integrated_source computation (sum all emitter currents)
2. Verify boundary flux computation (check normal direction sign)
3. Reduce solver tolerance; re-solve with more iterations
4. Check boundary condition implementation

### Gate 3: Gauge Condition

**What it checks:** If mean-zero gauge is specified, mean of potential is ≈ 0.

$$\overline{\phi_e} = \frac{1}{|\Omega|} \int_{\Omega} \phi_e \approx 0$$

**Why it matters:** Removes additive constant indeterminacy. Makes potential well-defined and aids numerical stability.

**How to use:**

```python
from jaxfne.validation import validate_poisson_gauge_condition

is_satisfied, msg = validate_poisson_gauge_condition(
    mean_potential=phi_e.mean(),
    gauge="mean_zero",
    tolerance=1e-6
)
print(msg)  # e.g., "gauge mean_zero satisfied; |mean(phi_e)| 3.2e-08"
assert is_satisfied, msg
```

**Parameters:**
- `mean_potential`: computed as `phi_e.mean()` or spatial average
- `gauge`: 'mean_zero' or 'other' (others not yet implemented)
- `tolerance`: allowed deviation from zero (default 1e-6)

**Common issues:**
- Gauge not applied during solve → mean ≠ 0; subtract mean before validation
- Solver applied different gauge → update gauge parameter to match
- Numerical error in averaging → increase solver precision or domain size

**Fix:**
1. Explicitly subtract mean: `phi_e = phi_e - phi_e.mean()`
2. Re-solve with explicit mean-zero projection
3. Reduce solver tolerance

### Gate 4: Field Array Finiteness

**What it checks:** All output arrays (phi_e, J_e, CSD) contain only finite values (no NaN, no Inf).

**Why it matters:** NaN/Inf indicate numerical divergence. Signals solver failure or malformed input.

**How to use:**

```python
from jaxfne.validation import validate_poisson_field_arrays

finiteness = validate_poisson_field_arrays(
    phi_e=phi_e,
    J_e=J_e,
    CSD=CSD
)
print(finiteness)  # {'finite_phi_e': True, 'finite_J_e': True, 'finite_CSD': True}
assert all(finiteness.values()), f"Non-finite output: {finiteness}"
```

**Common issues:**
- `finite_phi_e=False` → solver diverged; check conductivity SPD or source conservation
- `finite_J_e=False` → gradient computation failed; check domain or conductivity
- `finite_CSD=False` → second derivative (Laplacian) unstable; may need smoother potential

**Fix:**
1. Check previous gates (SPD, source conservation)
2. Increase solver iterations or decrease tolerance
3. Regularize conductivity (ensure well-conditioned)
4. Use higher floating-point precision (float64)

### Gate 5: Solver Convergence

**What it checks:** Solver residual is below tolerance and iteration limit reached convergence.

$$\frac{\|\nabla \cdot (\sigma \nabla \phi_e) + I_{\mathrm{src}}\|_2}{\|I_{\mathrm{src}}\|_2} < \epsilon_{\mathrm{tol}}$$

**Why it matters:** Ensures solution is accurate to stated tolerance. Lack of convergence means solution may be inaccurate.

**Parameters in admissibility report:**
- `solver_residual_l2_relative`: relative residual (typically 1e-7 to 1e-5 for production)
- `n_iterations`: number of iterations (e.g., 200)
- `converged`: boolean flag (True if residual < tolerance)

**Example validation:**

```python
report = build_poisson_admissibility_report(...)
assert report["solver_metadata"]["converged"], "Solver did not converge"
assert report["solver_metadata"]["solver_residual_l2_relative"] < 1e-6, \
    f"Residual {report['solver_metadata']['solver_residual_l2_relative']} > 1e-6"
```

**Common issues:**
- `converged=False` → Insufficient iterations; increase max_iterations
- `solver_residual_l2` large → Solver is sluggish; check conditioning
- Residual plateaus early → Premature convergence criteria; relax tolerance

---

## Full Admissibility Report Structure

An admissible solution produces a report like:

```python
report = {
    "diagnostic_kind": "poisson_admissibility",
    "admissibility_status": "admissible",  # or "not_admissible"
    "gates": {
        "conductivity_spd": {"passed": True, "message": "..."},
        "source_conservation": {"passed": True, "message": "...", "residual": 1.1e-07},
        "gauge_condition": {"passed": True, "message": "...", "gauge": "mean_zero"},
        "field_finiteness": {"passed": True, "phi_e_finite": True, ...},
    },
    "solver_metadata": {
        "solver_residual_l2_relative": 5.3e-07,
        "n_iterations": 187,
        "converged": True,
        "boundary_condition": "dirichlet",
        "gauge": "mean_zero",
        "csd_sign_convention": "positive_equals_extracellular_source"
    },
    "physical_amplitude_claim_allowed": True,
}
```

**Only if all gates pass AND `physical_amplitude_claim_allowed=True` can you claim the solution is physical.**

---

## Validation Checklist

Before declaring a field solution admissible:

- [ ] **SPD gate:** Conductivity is symmetric and all eigenvalues > 0
- [ ] **Conservation gate:** Source integral ≈ boundary flux integral (residual < 1e-6)
- [ ] **Gauge gate:** Mean potential ≈ 0 (or declared gauge satisfied)
- [ ] **Finiteness gate:** All fields (phi_e, J_e, CSD) are finite
- [ ] **Convergence gate:** Solver converged and residual < tolerance
- [ ] **Report:** `admissibility_status == "admissible"` and `physical_amplitude_claim_allowed == True`

---

## Code Examples

### Example 1: Validate Conductivity

```python
from jaxfne.validation import validate_poisson_spd_conductivity

# Create a test conductivity
sigma = jnp.eye(3) * 0.3  # Diagonal, isotropic, conductivity = 0.3 S/m

is_spd, msg = validate_poisson_spd_conductivity(sigma)
print(f"SPD: {is_spd}, {msg}")
# Output: SPD: True, SPD verified; min eigenvalue 3.0e-01

# Non-SPD conductivity (one negative eigenvalue)
sigma_bad = jnp.array([[1.0, 0, 0], [0, -0.1, 0], [0, 0, 1.0]])
is_spd, msg = validate_poisson_spd_conductivity(sigma_bad)
print(f"SPD: {is_spd}, {msg}")
# Output: SPD: False, conductivity not positive definite; min eigenvalue -1.0e-01
```

### Example 2: Full Admissibility Workflow

```python
import jaxfne
from jaxfne.validation import build_poisson_admissibility_report

# Assume you have:
# - phi_e: potential field [T, L] (time, laminar depth)
# - sigma: conductivity tensor [3, 3]
# - source_currents: emitter source projection [T, L]
# - boundary_flux: flux at domain boundary

# Compute integrals and fields
integrated_source = source_currents.sum()
integrated_boundary_flux = boundary_flux.sum()
mean_phi = phi_e.mean()

# Compute current density and CSD
J_e = -sigma @ grad(phi_e)  # Current density (example)
CSD = laplacian(phi_e)  # Current source density

# Build report
report = build_poisson_admissibility_report(
    conductivity=sigma,
    integrated_source=integrated_source,
    integrated_boundary_flux=integrated_boundary_flux,
    mean_potential=mean_phi,
    phi_e=phi_e,
    J_e=J_e,
    CSD=CSD,
    solver_residual_l2=3e-7,
    n_iterations=250,
    converged=True,
    gauge="mean_zero"
)

# Validate
assert report["admissibility_status"] == "admissible", "Solution not admissible"
print("✓ Field solution is admissible for physical modeling")
print(f"  Residual: {report['solver_metadata']['solver_residual_l2_relative']:.2e}")
print(f"  Iterations: {report['solver_metadata']['n_iterations']}")
```

---

## Common Mistakes and Fixes

### WRONG: Not checking SPD

```python
# ✗ This assumes conductivity is well-posed
phi_e = solve_poisson(conductivity, source)
```

### CORRECT: Validate SPD first

```python
# ✓ Check SPD before solving
from jaxfne.validation import validate_poisson_spd_conductivity
is_spd, msg = validate_poisson_spd_conductivity(conductivity)
assert is_spd, f"Conductivity must be SPD: {msg}"
phi_e = solve_poisson(conductivity, source)
```

---

### WRONG: Ignoring source conservation

```python
# ✗ No check that source and flux balance
report = build_poisson_admissibility_report(conductivity=sigma)
```

### CORRECT: Verify conservation

```python
# ✓ Compute and check conservation
integrated_source = source_currents.sum()
integrated_flux = boundary_integral(J_e)
is_conserved, msg, residual = validate_poisson_source_conservation(
    integrated_source, integrated_flux
)
assert is_conserved, f"Source not conserved: {msg}"
```

---

### WRONG: Not applying gauge after solving

```python
# ✗ Potential has arbitrary additive constant
phi_e = solve_poisson(conductivity, source)
# Later: gauge validation fails
```

### CORRECT: Apply gauge explicitly

```python
# ✓ Subtract mean to enforce mean-zero gauge
phi_e = solve_poisson(conductivity, source)
phi_e = phi_e - phi_e.mean()  # Enforce gauge

# Validate
is_satisfied, msg = validate_poisson_gauge_condition(
    phi_e.mean(), gauge="mean_zero"
)
assert is_satisfied, msg
```

---

### WRONG: Claiming physics without admissibility

```python
# ✗ This is not validated
manifest["field_claim_level"] = "physical"  # No, don't do this
```

### CORRECT: Only claim physics if admissible

```python
# ✓ Check admissibility first
report = build_poisson_admissibility_report(...)
if report["physical_amplitude_claim_allowed"]:
    manifest["field_claim_level"] = "physical_admissible"
else:
    manifest["field_claim_level"] = "proxy_readout_only"
```

---

## Related Documentation

- **[Poisson Admissibility Spec](../poisson_admissibility.md)** — Mathematical specification of all five gates
- **[Field Solution Metadata](./skill_field_solution_metadata.md)** — Field solution report contract and diagnostics
- **[Tensor-Field Workflows](../tensor_field_workflows.md)** — Pipeline overview and math notation

---

## Validation Commands (for development)

```bash
# Run admissibility tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/test_poisson_admissibility_v0215.py -v

# Check that all gates are defined and tested
grep -r "validate_poisson\|build_poisson_admissibility" jaxfne/validation.py

# Quick import check
python -c "from jaxfne.validation import build_poisson_admissibility_report; print('OK')"
```

---

**Skill version:** v0.2.15  
**Last updated:** 2026-05-20  
**Audience:** Agents, developers implementing or validating Poisson solvers.
