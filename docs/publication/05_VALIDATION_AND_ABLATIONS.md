# Validation and Ablations

## Primary Ablation Set

- H(t) fully removed (α = 0 + τ_H → ∞)
- Intrinsic coupling only (threshold offset, α ≠ 0, no synaptic gain)
- Synaptic gain only (multiplicative input gain, no threshold offset)
- Fast timescale (τ_H = 100 ms, Abbott/Varela short-term depression regime)
- Multiple slow timescales (τ_H = 1 s, 5 s, 10 s, 30 s)
- H(t) present but r_target = r(t=0) with zero homeostatic drive

## Fast vs. Slow Mechanism Separation

Key ablation: replace H(t) with Abbott & Varela (1997) short-term synaptic depression.
This directly tests whether the slow homeostatic mechanism is necessary
beyond what the fast (~100–500 ms) depression mechanism already explains.
Expected: fast mechanism alone does not produce a sustained omission response;
H(t) slow mechanism is required.

## Sensitivity Analysis

- Noise sensitivity (input noise amplitude)
- Connectivity sensitivity (E/I ratio, inter-area weights)
- Cell-type sensitivity (E vs. I proportion)
- Layer sensitivity (superficial vs. deep coupling strength)
- Geometry sensitivity (column diameter, inter-area distance)
- Multi-seed reproducibility (minimum 5 seeds per condition)

## Evidence Standard

Every major claim must include:
  - Repeated seeds (≥5) with mean ± CI reported
  - Null comparison (H(t) ablated)
  - Parameter sensitivity (τ_H, r_target, α at minimum)
  - Effect present across ≥2 neuron model classes
