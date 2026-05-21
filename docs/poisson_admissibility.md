# Poisson Field Solver Admissibility Specification

## Overview

This document specifies the mathematical contract for future Poisson field solvers (v0.2.16+) in jaxfne. It defines what constitutes an "admissible" solution: a solution that is mathematically well-posed, numerically accurate, and physically consistent.

**Status:** Specification only (v0.2.15). No Poisson solver is implemented yet. v0.2.16 will implement a concrete solver and validate against this contract.

⚠️ **v0.2.15 Critical Invariant:**
- **No physical amplitude claims are allowed in v0.2.15**, even if all admissibility gates pass synthetically on test data.
- `physical_amplitude_claim_allowed` is **always false** in v0.2.15 reports.
- Only v0.2.16+ (after solver implementation and calibration/units validation) may allow physical amplitude claims.

## Mathematical Problem

A resistive forward-field problem seeks extracellular potential $\phi_e$ and current density $\mathbf{J}_e$ satisfying:

$$\nabla \cdot (\sigma \nabla \phi_e) = -I_{\mathrm{src}}$$

where:
- $\sigma$ is the conductivity tensor (must be **Symmetric Positive Definite**)
- $I_{\mathrm{src}}$ is the source term (integrated neuronal currents from emitter)
- Boundary condition: Dirichlet (essential) or Neumann (natural flux)
- Gauge condition: Often mean-zero potential $\int \phi_e = 0$ (optional but recommended)

## Admissibility Gates

A Poisson solver output is admissible if and only if **all five gates pass**:

### Gate 1: Conductivity Symmetric Positive Definite (SPD)

**Mathematical requirement:**
- Conductivity $\sigma$ must be symmetric: $\sigma = \sigma^T$
- Conductivity must be positive definite: $\lambda_{\min}(\sigma) > 0$ (all eigenvalues positive)

**Why:** SPD ensures the Poisson problem is well-posed (unique solution exists). Non-SPD conductivity leads to ill-posed or unstable solutions.

**Check in v0.2.15+:**

```python
from jaxfne.validation import validate_poisson_spd_conductivity

is_spd, msg = validate_poisson_spd_conductivity(conductivity_tensor)
assert is_spd, msg
```

**Parameters:**
- `conductivity`: conductivity tensor or matrix
- `tolerance`: numerical tolerance for near-zero eigenvalues (default 1e-8)

### Gate 2: Source Conservation

**Mathematical requirement:**

$$\int_{\Omega} I_{\mathrm{src}} = -\int_{\partial \Omega} \sigma \nabla \phi_e \cdot \hat{n}$$

Integrated source over domain equals (negative) integrated boundary flux.

**Why:** Ensures charge/current conservation. If source and flux do not balance, solver may diverge or solution may be unphysical.

**Check in v0.2.15+:**

```python
from jaxfne.validation import validate_poisson_source_conservation

is_conserved, msg, residual = validate_poisson_source_conservation(
    integrated_source,
    integrated_boundary_flux,
    tolerance=1e-6
)
assert is_conserved, msg
```

**Parameters:**
- `integrated_source`: $\int I_{\mathrm{src}} d\Omega$
- `integrated_boundary_flux`: $-\int \sigma \nabla \phi_e \cdot \hat{n} d\partial\Omega$
- `tolerance`: absolute residual tolerance (default 1e-6)

### Gate 3: Gauge Condition (Optional but Recommended)

**Mathematical requirement (mean-zero gauge):**

$$\frac{1}{|\Omega|} \int_{\Omega} \phi_e = 0$$

Mean of potential over domain is zero (removes arbitrary additive constant).

**Why:** Removes indeterminacy in solution (any constant added to $\phi_e$ also solves Poisson). Mean-zero gauge is physically motivated for extracellular recording.

**Check in v0.2.15+:**

```python
from jaxfne.validation import validate_poisson_gauge_condition

is_satisfied, msg = validate_poisson_gauge_condition(
    mean_potential,
    gauge="mean_zero",
    tolerance=1e-6
)
assert is_satisfied, msg
```

**Parameters:**
- `mean_potential`: $\overline{\phi_e} = \frac{1}{|\Omega|} \int \phi_e d\Omega$
- `gauge`: gauge type ('mean_zero' or 'other')
- `tolerance`: tolerance on mean value (default 1e-6)

### Gate 4: Field Array Finiteness

**Mathematical requirement:**

All outputs must be finite (no NaN, no Inf):
- $\phi_e$ finite everywhere in domain
- $\mathbf{J}_e = -\sigma \nabla \phi_e$ finite everywhere
- CSD (derived from $\phi_e$) finite everywhere

**Why:** Ensures numerical stability. NaN/Inf indicate solver divergence or malformed input.

**Check in v0.2.15+:**

```python
from jaxfne.validation import validate_poisson_field_arrays

finiteness = validate_poisson_field_arrays(
    phi_e=phi_e,
    J_e=J_e,
    CSD=CSD
)
assert all(finiteness.values()), f"Non-finite output: {finiteness}"
```

**Parameters:**
- `phi_e`: extracellular potential field
- `J_e`: extracellular current density
- `CSD`: current source density (or derivatives)

### Gate 5: Solver Convergence

**Mathematical requirement:**

Solver must satisfy a relative residual criterion:

$$\frac{\|\nabla \cdot (\sigma \nabla \phi_e^{\mathrm{iter}}) + I_{\mathrm{src}}\|_2}{\|I_{\mathrm{src}}\|_2} < \epsilon_{\mathrm{tol}}$$

**Parameters:**
- `solver_residual_l2_relative`: relative L2 residual
- `n_iterations`: number of iterations performed
- `converged`: boolean flag (True if residual < tolerance)
- `tolerance`: typical target $\epsilon_{\mathrm{tol}} \approx 10^{-6}$ to $10^{-8}$

**Why:** Ensures solution is accurate to stated tolerance. Lack of convergence means solver may not have found true solution.

## Admissibility Report

When all five gates pass, the solver emits an admissibility report:

```python
from jaxfne.validation import build_poisson_admissibility_report

report = build_poisson_admissibility_report(
    conductivity=sigma_tensor,
    integrated_source=sum_of_sources,
    integrated_boundary_flux=-sum_of_boundary_flux,
    mean_potential=mean_of_phi,
    phi_e=potential_field,
    J_e=current_density,
    CSD=current_source_density,
    solver_residual_l2=relative_residual,
    n_iterations=200,
    converged=True,
    gauge="mean_zero",
    boundary_condition="dirichlet",
    csd_sign_convention="positive_equals_extracellular_source"
)

# Example report structure (v0.2.15)
assert report["admissibility_status"] == "admissible"
# v0.2.15 specification-only: physical_amplitude_claim_allowed is ALWAYS false
assert report["physical_amplitude_claim_allowed"] is False
assert "specification-only" in report.get("v0215_note", "").lower()
```

**Report structure:**

```json
{
  "diagnostic_kind": "poisson_admissibility",
  "admissibility_status": "admissible",
  "gates": {
    "conductivity_spd": {
      "passed": true,
      "message": "SPD verified; min eigenvalue 1.2e-02"
    },
    "source_conservation": {
      "passed": true,
      "message": "source conserved; residual 1.1e-07",
      "residual": 1.1e-07
    },
    "gauge_condition": {
      "passed": true,
      "message": "gauge mean_zero satisfied; |mean(phi_e)| 3.2e-08",
      "gauge": "mean_zero"
    },
    "field_finiteness": {
      "passed": true,
      "phi_e_finite": true,
      "J_e_finite": true,
      "CSD_finite": true
    }
  },
  "solver_metadata": {
    "solver_residual_l2_relative": 5.3e-07,
    "n_iterations": 187,
    "converged": true,
    "boundary_condition": "dirichlet",
    "gauge": "mean_zero",
    "csd_sign_convention": "positive_equals_extracellular_source"
  },
  "physical_amplitude_claim_allowed": false,
  "v0215_note": "v0.2.15 is specification-only (no solver yet). physical_amplitude_claim_allowed is ALWAYS false. v0.2.16+ will enable claims after solver implementation and calibration/units validation."
}
```

## Usage in v0.2.16+

When a Poisson solver is implemented in v0.2.16:

1. **Solve:** Compute $\phi_e$, $\mathbf{J}_e$ from Poisson equation
2. **Validate:** Run all five admissibility gates
3. **Report:** Build and emit admissibility report
4. **Gate:** Only pass solution downstream if `admissibility_status == "admissible"`
5. **Claim:** Only allow physical amplitude claims if `physical_amplitude_claim_allowed == True`

## Current Status (v0.2.15)

- ✓ Admissibility specification defined
- ✓ Five gates mathematically specified
- ✓ Validation helpers implemented in `jaxfne.validation`
- ✗ Poisson solver not yet implemented (v0.2.16)
- ✗ Physical amplitude claims not yet validated

**Public-output status:** Specification-only; no solver implementation; no calibrated physical-amplitude claims.

---

## Mathematical References

- **Well-posedness:** Lions & Magenes (1972), Variational Methods in Mathematical Physics
- **SPD conductivity:** Standard elliptic PDE theory; required for uniqueness
- **Source conservation:** Follows from divergence theorem; fundamental for physics
- **Gauge conditions:** Standard practice in computational electrodynamics and neuroscience forward modeling

---

**Document version:** v0.2.15  
**Last updated:** 2026-05-20
