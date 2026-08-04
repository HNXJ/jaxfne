# jaxfne Execution Plan

Every scenario follows the same operator grammar:

  Configuration
  → Construct
  → Simulate
  → Source
  → Field
  → Probe
  → Objective
  → Optimizer
  → Manifest

---

## H(t) Implementation

H(t) is implemented as an emitter-level state variable updated at each simulation step:

    τ_H · dH/dt = r_target - r(t)
    b_eff(t) = b_0 + α · H(t)       # intrinsic threshold coupling (primary mode)

Alternative coupling (ablation distinguishes):
    I_eff(t) = I(t) · (1 + β · H(t))  # multiplicative gain on input current

Ablation modes:
  - H(t) decoupled:  set α = 0 (H evolves but has no effect)
  - H(t) frozen:     set τ_H → ∞ (H held at initial value)
  - Fast timescale:  set τ_H = 100 ms (Abbott/Varela regime comparison)
  - Full ablation:   α = 0, τ_H → ∞
Document which ablation mode is used per experiment in validation_report.json.

---

## Artifact Exports (per experiment)

  - manifest.json
  - validation_report.json          (includes ablation mode, τ_H, r_target, α)
  - metrics.json
  - editable configuration (.yaml or .json)
  - PNG figures (spectrolaminar_suite + raster minimum)
  - optional HTML visualization

---

## Execution Requirements

  - Deterministic seeds (explicit PRNG keys, logged in manifest)
  - Finite outputs verified before manifesting
  - Identical RuntimeConfiguration across conditions within each stage
  - Minimum 5 seeds per ablation condition
