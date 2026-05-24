# v0.3 Tutorial Atlas and Scenario Scaffolds

**Version:** v0.3.0+  
**Last updated:** 2026-05-23  
**truth_mode:** truth_safe_unverified  

---

## Overview

The v0.3 tutorial-scenario line is a comprehensive set of 15 executable Jupyter notebooks (scenarios) designed to teach TFNE (Tensor-Field Neural Equations) from first principles through complete multi-area cortical simulations. The v0.3 line is **not a constant package-mutation line** — the package version `jaxfne` only bumps when a tutorial reveals a real bug or missing required public API.

**Key principle:** v0.3 is tutorial/documentation work on a stable v0.2.30 toolbox. Scientific claims are explicitly bounded as computational scaffolds (no biological validation, no Maxwell/Poisson solvers, no proof-of-mechanism).

---

## Quick Links

- **[Scenario Index](scenario_index.md)** — 15 scenarios, learning path, prerequisites
- **[v0.3.3–v0.3.31 Planning Handoff](v0303_to_v0331_planning_handoff.md)** — Complete roadmap, scenario definitions, audit phases, PyPI strategy, open questions
- **[Visualization Doctrine](visualization_doctrine.md)** — PNG mandatory, Plotly optional, figure panels, manifest contracts
- **[Tutorial Template](template.md)** — Required 13-section structure for all v0.3 notebooks
- **[Acceptance Gates](acceptance_gates.yaml)** — Hard validation criteria (firing rate, finite values, JSON-safe, JAX-native)
- **[Plotly Artifact Policy](plotly_policy.md)** — PNG figures required; Plotly optional
- **[Canonical Import Guide](canonical_imports.md)** — `import jaxfne as jtfne` only

---

## v0.3 Phases and Scope

### 31-Phase Roadmap (v0.3.0 → v0.3.31)

| Phase | Scenario | Type | Status | Purpose |
|-------|----------|------|--------|---------|
| v0.3.0 | Atlas scaffold | Framework | [TBD] | 15-scenario spine, acceptance gates, artifact system |
| v0.3.1 | Single neuron I | Core | [TBD] | Izhikevich phenomenology, membrane dynamics |
| v0.3.2 | Single neuron II | Core | [TBD] | Hodgkin-Huxley model, ionic currents |
| v0.3.3 | Synaptic dynamics | Core | [TBD] | Receptor kinetics, exponential kernels |
| v0.3.4 | Two-neuron E/I | Core | [TBD] | Excitatory/inhibitory connectivity, synchrony |
| v0.3.5 | Laminar population | Core | [TBD] | Layer structure, population current |
| v0.3.6 | Three-area hierarchy | Core | [TBD] | Feedforward, feedback, lateral connectivity |
| v0.3.7 | Field proxy I | Core | [TBD] | Voltage dipole, kernel projection |
| v0.3.8 | Field proxy II | Core | [TBD] | CSD/LFP proxy, multimodal readout |
| v0.3.9 | Oddball stimulus | Advanced | [TBD] | Global vs. oddball conditions, behavioral latency |
| v0.3.10 | Omission response | Advanced | [TBD] | Prediction error, active sensing |
| v0.3.11 | Plasticity I | Advanced | [TBD] | Spike-timing-dependent plasticity (STDP) |
| v0.3.12 | Plasticity II | Advanced | [TBD] | Homeostatic scaling, learning rules |
| v0.3.13 | Optimization I | Advanced | [TBD] | GSDR/AGSDR objective fitting |
| v0.3.14 | Optimization II | Advanced | [TBD] | Multi-objective, evolutionary search |
| v0.3.15 | Whole-scenario review | Review | [TBD] | Integration, benchmarks, next steps |
| v0.3.16–v0.3.31 | Audit phases | Validation | [TBD] | Performance baseline, CI/CD, docs audit |

### Scenario Categories

- **Core (v0.3.1–v0.3.8):** Foundation models (neurons, synapses, fields)
- **Advanced (v0.3.9–v0.3.14):** Behavioral tasks, learning, optimization
- **Review & Audit (v0.3.15–v0.3.31):** Integration validation, benchmarks, documentation

---

## Hard Acceptance Gates

All v0.3 scenarios must pass strict validation gates before acceptance:

### Gate Category: Firing Rate
- **Requirement:** Each population must fire at 2–25 Hz (mean over simulation)
- **Failure:** Flat (0 Hz), dead (< 2 Hz), or explosive (> 25 Hz) activity rejected
- **Rationale:** Physiological range ensures model is not trivial or pathological

### Gate Category: Numerical Stability
- **Requirement:** All signals finite (no NaN, no Inf)
- **Failure:** Any NaN/Inf in V_m, spikes, or field proxies → reject
- **Rationale:** Ensures JAX-native numerical paths work correctly

### Gate Category: JSON Safety
- **Requirement:** All outputs JSON-serializable; no NaN/Inf in JSON
- **Failure:** Cannot parse manifest or metrics as valid JSON → reject
- **Rationale:** Machine-readable outputs for downstream analysis

### Gate Category: JAX Native
- **Requirement:** Numerical simulation path jittable; plotting/I/O separate
- **Failure:** Simulation loop has non-JAX operations (print, file I/O) → reject
- **Rationale:** Enables future GPU/TPU acceleration

### Gate Category: Geometry Metadata
- **Requirement:** dx=dy=dz=0.010 mm declared (laminar_proxy_no_pde mode)
- **Failure:** Geometry inconsistent with laminar proxy semantics → reject
- **Rationale:** Prevents false 3D PDE claims without solver

### Gate Category: Figure Artifacts
- **Requirement:** PNG figures required; Plotly optional; both with SHA256 hashes
- **Failure:** Missing PNG or corrupted hash → reject
- **Rationale:** Reproducible visual validation and integrity

### Gate Category: Claim Gates
- **Requirement:** physical_amplitude_claim_allowed=False, claim_level="computational_scaffold"
- **Failure:** Any claim gate violated → reject
- **Rationale:** Enforces truthful, bounded scientific framing

---

## Canonical Import Guide

**Required import alias:**
```python
import jaxfne as jtfne
```

**Forbidden aliases:**
- ❌ `import jaxfne` (bare)
- ❌ `from jaxfne import *`
- ❌ `import jaxfne as tfne`
- ❌ `import jaxfne as jtnfe`
- ❌ `import jaxfne as jtFNE`

**Rationale:** Canonical alias ensures consistent global naming in all v0.3 notebooks. Readers will see `jtfne.` prefix everywhere.

**Guarded Plotly import (optional):**
```python
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
```

---

## Package Mutation Policy

**Default:** v0.3 does NOT bump package version.

**Trigger for version bump (v0.2.30 → v0.2.31 or v0.3.0):**
1. Tutorial execution reveals a real **bug** in v0.2.30 (not tutorial error)
2. Tutorial requires a **missing required public API** (not workaround-able)
3. **Core gate violation** that cannot be worked around (e.g., claim gate false positive)

**Not a trigger:**
- ❌ Tutorial documentation improvements
- ❌ New example scripts or notebooks
- ❌ Figure regeneration
- ❌ Plotly integration
- ❌ Performance optimization (same API)

**Decision process:**
1. Tutorial author identifies potential bug/missing API
2. Opens GitHub issue with evidence
3. Core maintainer reviews and gates decision
4. If approved: create targeted fix commit + increment version
5. If not approved: close issue + continue with v0.3 workaround

**Result:** v0.3 is a stable documentation/tutorial line, not a constant package-mutation pipeline.

---

## Truth Status and Claim Boundaries

**Explicit truth declarations for all v0.3 tutorials:**

- **truth_mode:** truth_safe_unverified
- **claim_level:** computational_scaffold
- **physical_amplitude_claim_allowed:** False
- **field_solver_status:** laminar_proxy_no_pde
- **source_calibration_status:** uncalibrated (teaching proxy)

**What we claim:**
- ✓ JAX-native Izhikevich and HH neuron models (phenomenological, not biophysical)
- ✓ TFNE forward-field framework with kernel-based CSD/LFP proxies
- ✓ Multimodal probe readouts (spikes, voltage, field proxies)
- ✓ Executable demonstrations of oddball/omission tasks
- ✓ Optimization framework for fitting model parameters to synthetic data

**What we do NOT claim:**
- ❌ Biological calibration or validation
- ❌ Physical amplitude claims for membrane voltage or field potentials
- ❌ Maxwell solver or full PDE solutions
- ❌ Proof of neural mechanisms or pathophysiology
- ❌ Whole-brain simulation or scale-free architecture

---

## Learning Path Recommendations

### Beginner (v0.3.1–v0.3.4)
1. v0.3.1: Single neuron Izhikevich
2. v0.3.2: Single neuron Hodgkin-Huxley
3. v0.3.3: Synaptic dynamics and receptors
4. v0.3.4: Two-neuron E/I circuit

### Intermediate (v0.3.5–v0.3.8)
5. v0.3.5: Laminar population structure
6. v0.3.6: Three-area hierarchical network
7. v0.3.7: Field proxy I (voltage dipole)
8. v0.3.8: Field proxy II (CSD/LFP multimodal)

### Advanced (v0.3.9–v0.3.14)
9. v0.3.9: Oddball stimulus and global context
10. v0.3.10: Omission response and prediction error
11. v0.3.11: Spike-timing-dependent plasticity
12. v0.3.12: Homeostatic scaling and learning
13. v0.3.13: Optimization I (GSDR fitting)
14. v0.3.14: Optimization II (multi-objective, evolution)

### Capstone
15. v0.3.15: Whole-scenario review and benchmarks

---

## Development Status

**v0.3.0 phase:** Atlas scaffold and acceptance gate framework  
**Status:** [IN PROGRESS — Phases A–L]

Expected completion of full 15-scenario atlas: Q3 2026 (dependent on phase sequencing and approval gates).

---

## See Also

- [Scenario Index](scenario_index.md) — Detailed 15-scenario spine
- [Tutorial Template](template.md) — 13-section required structure
- [Acceptance Gates (YAML)](acceptance_gates.yaml) — Validation criteria
- [Canonical Imports](canonical_imports.md) — Import conventions
- [Plotly Policy](plotly_policy.md) — PNG/interactive figure system
- [v0.3 Readiness Bridge](../v03_bridge.md) — Locked APIs, future solvers, migration path
- [v0.3 Tutorial-Scenario Plan](../v030_tutorial_scenario_plan.md) — v0.3 doctrine (existing)
- [Tutorial Template (v0.3)](../tutorial_template_v030.md) — 13-section template (existing)
