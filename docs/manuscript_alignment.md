# Manuscript Alignment (v0.2.27)

**How the current jaxfne codebase maps to the manuscript draft.**

**Version:** v0.2.27  
**Last updated:** 2026-05-22  
**Manuscript target:** v0.2.24-aligned draft  
**truth_mode:** truth_safe_unverified

---

## Overview

The jaxfne codebase evolved from the v0.2.24-aligned manuscript draft. The following sections map code sections, scope contracts, and key features to manuscript content.

**Key changes since manuscript draft:**
- v0.2.25: Added AxisSpec / BasisSpec computation-basis contracts
- v0.2.26: Froze computation-basis with immutable scope contracts
- v0.2.27: Added conservation-inspired proxy diagnostics

---

## Core Pipeline: Emitter → Source → Field → Probe → Objective

**Manuscript Section:** "TFNE Source-Field-Probe Framework"

**Current Implementation:**
```python
from jaxfne.core import configuration, construct, simulate, Signals, Objective
from jaxfne.emitters import IzhikevichEmitter
from jaxfne.fields import FieldOutput
from jaxfne.probes import probe_laminar_modes

# (1) Emitters: Generate spiking dynamics
emitter = IzhikevichEmitter(n_neurons=100)

# (2) Construction: Assemble model
model = construct(configuration(), emitters=[emitter])

# (3) Simulation: Get source/voltage signals
signals = simulate(model, duration_ms=100, dt_ms=0.1)

# (4) Field: Compute extracellular potentials (if applicable)
# (Inline in simulate() or via custom field computation)

# (5) Probe: Extract multimodal readouts (spikes, voltage, CSD proxy, etc.)
# (Inline in probe operators or via manifest)

# (6) Objective: Fitness/loss computation
obj = Objective(...)  # Custom fitness function
```

**Scope contracts (immutable):**
- physical_amplitude_allowed: False
- biological_mechanism_claimed: False
- computational_scaffold: True

---

## Izhikevich Emitter

**Manuscript Section:** "Izhikevich Phenomenological Neuron Model"

**Current Implementation:**
```python
from jaxfne.emitters import IzhikevichEmitter

emitter = IzhikevichEmitter(
    n_neurons=5,
    a=0.02,      # recovery time scale
    b=0.2,       # recovery coupling
    c=-65.0,     # reset voltage
    d=8.0,       # reset recovery
    v_init=-65.0,
    u_init=-15.0
)

# Model equations (jaxfne/emitters/izhikevich.py):
# dv/dt = 0.04*v^2 + 5*v + 140 - u + I_in
# du/dt = a*(b*v - u)
# if v >= 30: v := c, u := u + d
```

**Manuscript alignment:**
- ✓ Parameter set preserved (a, b, c, d)
- ✓ Reset behavior unchanged (spike detection at v >= 30)
- ✓ Recovery variable (u) maintained
- ✓ No biological calibration claimed (truth_safe_unverified)

---

## Field Operators and CSD/LFP Proxies

**Manuscript Section:** "Forward-Field Modeling: Source Projection and Current-Source Density"

**Current Implementation (v0.2.27):**
```python
from jaxfne.fields import FieldOutput, probe_laminar_modes, compute_conservation_proxy_diagnostics

# Field outputs (if computed)
# structure: FieldOutput with source_proxy[T,N], phi_e_proxy[T,X], csd_proxy[T,X], lfp_proxy[T,X]

# Diagnostics over field outputs
diag = compute_conservation_proxy_diagnostics(
    source=signals.field.source_proxy,
    phi_e=signals.field.phi_e_proxy,
    csd=signals.field.csd_proxy,
    lfp=signals.field.lfp_proxy,
)

# Claim gates (hardcoded, immutable):
# - field_solver_status: "laminar_proxy_no_pde"
# - Poisson solver: planned for future release
# - Maxwell solver: planned for future release
# - j_dot_e_proxy: None
# - poynting_flux_proxy: None
```

**Manuscript alignment:**
- ✓ Forward-field framing preserved (source → phi_e → CSD/LFP chain)
- ✓ Laminar projection operator intact
- ◐ Poisson solver: v0.2.27 uses proxy diagnostics, full solver implementation planned for future release
- ◐ Maxwell solver: planned for future release with separate approval
- ✓ Proxy terminology enforced (all outputs labeled "proxy")

**Important divergence:**
- Manuscript may describe future Poisson solver
- v0.2.27 implements diagnostics **instead of** solver
- Poisson solver deferred to v0.2.28+ with separate approval

---

## Probe Operators (Eight Readout Modalities)

**Manuscript Section:** "Multimodal Probe Operators"

**Current Implementation:**
```python
# Eight probe operators (frozen API, immutable behavior):
from jaxfne.probes import (
    probe_spikes,              # (1) Spike raster from voltage threshold
    probe_membrane_voltage,    # (2) Membrane voltage trace
    probe_source_projection,   # (3) Source trace (if available)
    probe_lfp_proxy,          # (4) LFP proxy (summed/filtered field)
    probe_csd_proxy,          # (5) CSD proxy (spatial derivative of field)
    probe_eeg_proxy,          # (6) EEG proxy (distant far-field approximation)
    probe_meg_proxy,          # (7) MEG proxy (magnetic field approximation)
    probe_emm_proxy,          # (8) EMM proxy (electrode-to-membrane mapping)
)

# All operators embedded in manifest["probe_report"]
manifest = model.manifest(signals)
probe_report = manifest["probe_report"]  # Contains all 8 operator results
```

**Manuscript alignment:**
- ✓ All 8 operators preserved (SPK, Vm, source, LFP, CSD, EEG, MEG, EMM)
- ✓ Operator semantics unchanged
- ✓ Multimodal contract enforced
- ✓ All outputs validated as JSON-safe

---

## Computation Basis and Claim Boundary (v0.2.26–v0.2.27)

**NEW in v0.2.26:** Explicit computation-basis contract  
**EXTENDED in v0.2.27:** Conservation proxy diagnostics

**Current Implementation:**
```python
from jaxfne.core import default_basis_spec

basis = default_basis_spec()
# Returns frozen BasisSpec with scope properties:
# - scope_level: "computational_scaffold"
# - physical_amplitude_allowed: False
# - biological_mechanism_claimed: False
# - field_solver_status: "laminar_proxy_no_pde"
# - maxwell_solver_status: "planned_future_module"
# - admittance_solver_status: "planned_future_module"
# - poisson_solver_status: "planned_future_module"

# Manifest integration (v0.2.26)
manifest["basis"] = _default_basis_dict()

# Diagnostics integration (v0.2.27)
if signals.field is not None:
    manifest["conservation_proxy_diagnostics"] = compute_conservation_proxy_diagnostics(...)
```

**Manuscript alignment:**
- ✓ New computation-basis chapter aligns with v0.2.26+ codebase
- ✓ Claim gates frozen (cannot be violated at runtime)
- ✓ Future solver regimes explicitly gated
- ✗ Manuscript may predate computation-basis contract; update required

---

## Conservation-Inspired Proxy Diagnostics (v0.2.27)

**NEW in v0.2.27:** Safe scalar summaries over existing field arrays

**Current Implementation:**
```python
from jaxfne.fields import compute_conservation_proxy_diagnostics

diag = compute_conservation_proxy_diagnostics(
    field_solution=signals.field,  # or pass arrays directly
)

# Returns dict with:
# - source_norm_l1, source_norm_l2: source magnitude proxies
# - source_conservation_proxy_residual: spatial-mean balance proxy
# - phi_gradient_proxy_norm2: field gradient magnitude
# - csd_abs_mean, csd_norm_l2: CSD magnitude proxies
# - lfp_abs_mean, lfp_norm_l2: LFP magnitude proxies
# - field_energy_like_proxy: alias for phi_gradient_proxy_norm2
# - j_dot_e_proxy: null (J_e not computed)
# - poynting_flux_proxy: null (not implemented)
# - poisson_solver_status: "not_implemented"
# - maxwell_solver_status: "not_implemented"
# - stress_energy_tensor_status: "not_implemented"
```

**Manuscript alignment:**
- ✗ **Not in original manuscript** (added v0.2.27)
- ✓ Aligns with "Conservation" section if manuscript discusses proxy diagnostics
- ✗ If manuscript claims Poisson solver, that is **deferred** to future phase

**Important:** This is a **diagnostic-only** addition. No solvers implemented.

---

## Optimization and Fitness (GSDR/AGSDR)

**Manuscript Section:** "Optimization: Fitness and Plasticity"

**Current Implementation:**
```python
from jaxfne.optimization import GSDR, AGSDR

# Custom optimization workflows (outside core TFNE pipeline)
# GSDR: Genetic algorithm for network parameters
# AGSDR: Adaptive genetic algorithm variant

# Example: Optimize firing rate
def fitness(population_params):
    # Decode parameters
    # Simulate circuit
    # Measure spike rates
    # Return error
    pass

optimizer = GSDR(fitness, population_size=32, generations=10)
best_params, best_fitness = optimizer.run()
```

**Manuscript alignment:**
- ✓ Optimization layer preserved (separate from core TFNE)
- ✓ Fitness function interface unchanged
- ✓ No integration with Optax (optional, separate layer)

---

## CRITICAL: Poisson Solver Status

**Manuscript may state:** "v0.2.27 includes Poisson solver with CG/MINRES"

**ACTUAL v0.2.27 state:** **Poisson solver NOT implemented**

**Timeline:**
- v0.2.25: Core pipeline complete
- v0.2.26: Computation-basis contract frozen; Poisson gated as future
- v0.2.27: Conservation proxy diagnostics (diagnostic-only); Poisson still gated

**If manuscript claims Poisson solver in v0.2.27:**
- ❌ That is **incorrect** for the released v0.2.27 codebase
- ✓ Update manuscript to reflect diagnostic-only status
- ✓ Mark Poisson solver as "gated future work"

---

## Truth Status and Manuscript Framing

**Manuscript truth_mode:** truth_safe_unverified  
**Current code truth_mode:** truth_safe_unverified  
**Scientific claims:** None (exploratory framework)

**Manuscript should state:**
- jaxfne is a **computational scaffold** for neuroscience research
- No biological validation
- No empirical calibration
- Teaching/exploration tool only
- Izhikevich model is phenomenological (not biophysical)
- Field outputs are proxies (no physical claims)

---

## Mapping: Manuscript Chapters to Code Locations

| Manuscript Chapter | Code Location | Status |
|----|----|----|
| TFNE Overview | jaxfne/core.py | ✓ Current |
| Izhikevich Model | jaxfne/emitters/izhikevich.py | ✓ Current |
| Hodgkin-Huxley Model | jaxfne/emitters/hodgkin_huxley.py | ✓ Current |
| Forward-Field Modeling | jaxfne/fields.py | ✓ Current (no Poisson solver) |
| Probe Operators | jaxfne/probes.py | ✓ Current (8 operators) |
| Computation Basis | jaxfne/core.py (BasisSpec) | ✓ New (v0.2.26+) |
| Conservation Diagnostics | jaxfne/fields.py (diagnostics fn) | ✓ New (v0.2.27) |
| Optimization | jaxfne/optimization.py | ✓ Current |

---

## Required Manuscript Updates

If manuscript was finalized before v0.2.27:

1. **Add "v0.2.26–v0.2.27 Updates" section:**
   - ✓ Computation-basis contract (v0.2.26)
   - ✓ Conservation proxy diagnostics (v0.2.27)
   - ✓ Hardened scope contracts

2. **Clarify Poisson solver status:**
   - Change: "v0.2.27 implements Poisson solver"
   - To: "v0.2.27 implements conservation proxy diagnostics; Poisson solver deferred to future phase"

3. **Document frozen APIs:**
   - 8 probe operators (immutable)
   - Izhikevich / HH emitters (immutable)
   - TFNE pipeline (immutable)

4. **Add truth status section:**
   - Explicitly state "computational_scaffold"
   - Explicitly state no biological calibration
   - Explicitly state proxy-field terminology

---

## See Also

- [Computation Basis](computation_basis.md) — Computation contract details
- [Tensor Network Ancestry](tensor_network_ancestry.md) — v0.2.29 basis-transform doctrine and historical context
- [Conservation Proxy Diagnostics](conservation_proxy_diagnostics.md) — Diagnostics reference
- [Poisson Admissibility](poisson_admissibility.md) — Poisson solver future doctrine
