# v0.3 Readiness Bridge (v0.2.30)

**Preparing jaxfne for the v0.3 tutorial-scenario line: locked APIs, doctrine constraints, and future regimes.**

**Version:** v0.2.30  
**Last updated:** 2026-05-23  
**truth_mode:** truth_safe_unverified

---

## Overview

The v0.2 series is feature-frozen with locked APIs. v0.2.30 is the stable toolbox for the
v0.3 tutorial-scenario line. **v0.3.x phases are primarily tutorial, documentation, and
scenario work** — not a constant package-mutation line. The package version only bumps
if a tutorial reveals a real bug or missing required public API.

Future technical work deferred to separate approved phases:
- Poisson solver integration (with approval)
- Maxwell/admittive solvers (future doctrine)
- Stress-energy tensor computation (future doctrine)
- Poynting flux analysis (future doctrine)

This document describes what is locked in v0.2.30 and what will change in v0.3+.

---

## Locked APIs (v0.2.30 → v0.3+)

### Emitters (LOCKED)

**Izhikevich:**
```python
from jaxfne.emitters import IzhikevichEmitter

# API frozen. No breaking changes expected.
emitter = IzhikevichEmitter(
    n_neurons=100,
    a=0.02, b=0.2, c=-65.0, d=8.0,
    v_init=-65.0, u_init=-15.0
)
```

**Hodgkin-Huxley:**
```python
from jaxfne.emitters import HodgkinHuxleyEmitter

# API frozen. No breaking changes expected.
emitter = HodgkinHuxleyEmitter(n_neurons=50)
```

**Guarantee:** Both emitter classes will maintain their current parameter API in v0.3.

---

### Core Pipeline (LOCKED)

**Configuration and construction:**
```python
from jaxfne.core import configuration, construct, simulate

# API frozen. No breaking changes expected.
config = configuration()
model = construct(config, emitters=[...])
signals = simulate(model, duration_ms=100, dt_ms=0.1)
```

**Signals object:**
```python
# Guaranteed fields (v0.3+):
# - time_ms: [T]
# - V_m: [T, N]
# - spikes: [T, N]
# - source: [T, N] or None
# - field: FieldOutput or None
# - metadata: dict
```

**Manifest API:**
```python
manifest = model.manifest(signals)

# Guaranteed blocks (v0.3+):
# - "basis": basis specification (v0.2.26+)
# - "probe_report": 8 probe operator results
# - "validation_report": validation gates
# - "conservation_proxy_diagnostics": diagnostics (v0.2.27+)
```

**Guarantee:** Pipeline structure will not change in v0.3.

---

### Probe Operators (LOCKED)

**Eight probe operators (v0.2.27):**
1. probe_spikes
2. probe_membrane_voltage
3. probe_source_projection
4. probe_lfp_proxy
5. probe_csd_proxy
6. probe_eeg_proxy
7. probe_meg_proxy
8. probe_emm_proxy

**Guarantee:** All 8 operators will maintain their current semantics in v0.3.

---

### Claim Gates (LOCKED AND IMMUTABLE)

**Hardcoded in v0.2.27 and v0.3+:**
```python
physical_amplitude_claim_allowed: False
biological_metabolism_claim_allowed: False
```

**Guarantee:** These gates cannot be set to True without major structural redesign.

---

## Future Regimes (Gated for v0.3+)

### Poisson Solver (Declared Future)

**Current status (v0.2.30):**
```python
# compute_conservation_proxy_diagnostics() returns:
{
    "poisson_solver_status": "not_implemented",
    ...
}
```

**Planned for v0.3.x (requires separate approval):**
- Implement direct or iterative CG/MINRES Poisson solver
- Accept conductivity tensor (σ) and boundary conditions
- Enforce ∇·(-σ∇φ) = q with proper PDE formulation
- Update claim gates if physical validation evidence provided

**What changes in v0.3 (if approved):**
- New function: `solve_poisson(source, conductivity, boundary_conditions)`
- New FieldOutput fields: `phi_e_solved`, `phi_e_error`, `solver_iterations`
- Updated manifest block: `"poisson_solution"` with residual/convergence metrics

**What does NOT change:**
- Izhikevich emitter (still phenomenological)
- Probe operators (still multimodal)
- Claim gates (still computational_scaffold, physical_amplitude_claim_allowed=False)

---

### Maxwell Solver (Future Doctrine)

**Current status (v0.2.30):**
```python
{
    "maxwell_solver_status": "not_implemented",
}
```

**Planned for v0.3.x+ (far future, requires major structural work):**
- Time-varying electromagnetic fields
- Magnetic susceptibility and conductivity coupling
- Freqeuncy-domain or time-domain Maxwell solver

**Guarantee:** v0.3 will NOT include Maxwell solver (v0.4+ candidate).

---

### Admittive Field Solver (Future Doctrine)

**Current status (v0.2.30):**
```python
{
    "admittive_solver_status": "not_implemented",
}
```

**Planned for v0.3.x+ (future):**
- Coupled voltage/conductance field equations
- Non-linear tissue response (Ohmic heating, conductivity saturation)

**Guarantee:** v0.3 will NOT include admittive solver (v0.3.x+ candidate).

---

### Stress-Energy Tensor (Future Doctrine)

**Current status (v0.2.30):**
```python
{
    "stress_energy_tensor_status": "not_implemented",
}
```

**Planned for v0.4+ (highly speculative):**
- Energy dissipation analysis
- Heat flow in neural tissue

**Guarantee:** v0.3 will NOT compute stress-energy.

---

### Poynting Flux (Future Doctrine)

**Current status (v0.2.30):**
```python
{
    "poynting_flux_proxy": None,
    "poynting_flux_status": "not_implemented",
}
```

**Planned for v0.4+ (Maxwell+, future):**
- Electromagnetic energy flow (S = E × H)
- Only meaningful with Maxwell solver

**Guarantee:** v0.3 will NOT compute Poynting flux.

---

## Migration Path: v0.2.30 → v0.3

### For Users

**Installation:**
```bash
# v0.2.30 (current stable toolbox for v0.3 tutorials)
pip install jaxfne==0.2.30

# v0.3 (future package release, when and if required)
pip install jaxfne==0.3.0
# or (to get latest)
pip install --upgrade jaxfne
```

**API changes (expected):**
- ✓ All existing code should work unchanged
- ? Minor additions (e.g., new Poisson solver functions)
- ? New manifest blocks (if new solvers added)

**Deprecations (unlikely):**
- Do not expect removal of any v0.2.30 APIs in v0.3

---

### For Contributors

**Before v0.3 Release:**

1. **API Stability Review**
   - Verify all locked APIs are correctly documented
   - Check for any unintended breaking changes

2. **Test Coverage**
   - All 8 probe operators must remain tested
   - Locked emitters must pass baseline tests
   - Manifest generation must produce same structure

3. **Documentation**
   - Update `docs/index.md` with v0.3 new features
   - Preserve all v0.2.30 documentation (backward compatibility)
   - Add v0.3 migration guide if any changes affect users

---

## What v0.3 Will Still NOT Include

| Feature | Status | Reason |
|----|----|----|
| Jaxley simulator wrapper | Not included | Too narrow; Jaxley is external |
| Optax integration in core | Not included | Optional layer; keep separate |
| Multi-compartment soma mapping | Not included | Biological territory; requires validation |
| Sparse spike format | Not included | Performance concern; dense OK for now |
| GPU-specific optimizations | Not included | Works on CPU; no GPU guarantee needed |
| Whole-brain simulation | Not included | Out of scope (too large) |

---

## Versioning Policy (v0.3+)

**Semantic Versioning:**
- Major: Breaking API changes (e.g., rewrite Core pipeline)
- Minor: New features (e.g., Poisson solver added, backward compatible)
- Patch: Bug fixes (e.g., numerical stability improvement)

**Examples:**
- v0.2.27 → v0.3.0: Major (if Poisson solver requires API changes)
- v0.2.27 → v0.3.1: Minor (if new solver is additive)
- v0.3.1 → v0.3.2: Patch (bug fix)

**Backward compatibility:**
- v0.2.27 code should work on v0.3.x without modification (if v0.3.x is minor bump)
- v0.2.27 code may need updates for v1.0+ (if major structural redesign)

---

## v0.3 Release Checklist (Template)

When v0.3 is ready:

- [ ] All locked APIs tested and documented
- [ ] New solver(s) implemented and tested
- [ ] Manifest schema documented
- [ ] Version bumped in pyproject.toml and core.py
- [ ] CHANGELOG.md updated with v0.3.0 entry
- [ ] Migration guide written (if needed)
- [ ] ReadTheDocs updated
- [ ] PyPI package released
- [ ] GitHub tag v0.3.0 created

---

## See Also

- [docs/computation_basis.md](computation_basis.md) — Locked computation contract
- [docs/tensor_network_ancestry.md](tensor_network_ancestry.md) — v0.2.29 basis-transform doctrine and extensibility philosophy
- [docs/poisson_admissibility.md](poisson_admissibility.md) — Poisson solver spec (future)
- [docs/conservation_proxy_diagnostics.md](conservation_proxy_diagnostics.md) — Current diagnostics
- [CHANGELOG.md](../CHANGELOG.md) — Version history
- [README.md](../README.md) — Current installation and quick start
