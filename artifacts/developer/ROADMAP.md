# Roadmap

## Title

Omission Responses in a Multi-Area Laminar Cortical Hierarchy Emerge from
Simple Firing-Rate Homeostasis

---

## Central Question

Does a single slow homeostatic variable H(t) — tracking population firing rate
toward a set-point — suffice to produce omission responses observed in multi-area
laminar in-vivo recordings, without predictive coding circuits or synaptic depression?

---

## H(t) Formulation

    τ_H · dH/dt = r_target - r(t)

    r(t)      — instantaneous population firing rate
    r_target  — homeostatic set-point (free parameter, matched to in-vivo baseline)
    τ_H       — slow timescale (seconds to tens of seconds)
    H(t)      — additive offset on Izhikevich threshold b  [primary mode]
                OR multiplicative gain on input current     [ablation mode]

---

## References

| Reference | Role |
|---|---|
| Turrigiano & Nelson, Nat Rev Neurosci 2004 | Foundational homeostasis motivation |
| Cannon & Miller, PLOS Comp Biol 2016 | Closest formal match to H(t) (τ_θ ~30 s) |
| Abbott & Varela, Science 1997 | Fast-mechanism foil (synaptic depression) |
| Yaron et al., Neuron 2025 | Empirical omission target |

---

## Simulation Stages

Stage 0   Single neuron equilibrium — H(t) convergence to steady-state
Stage 1   Repeated pulse adaptation — firing-rate decay, recovery time
Stage 2   Population adaptation — synchrony, H distribution
Stage 3   Frequency sweep — τ_H calibration (1–40 Hz)
Stage 4   Amplitude sweep
Stage 5   Duration sweep
Stage 6   Random stimulus trains
Stage 7   Classical oddball — SSA index vs. Abbott/Varela null
Stage 8   Omission paradigm  ← PRIMARY RESULT
          Matched to in-vivo protocol; laminar LFP proxy; H(t) trace;
          spectrolaminar motif; ablation null control
Stage 9   Global-local oddball
Stage 10  Long-term adaptation (Colab GPU)
Stage 11  Multi-area propagation
Stage 12  Jaxley/HH emitter generalization
Stage 13  Optimization with AGSDR
Stage 14  Parameter recovery (τ_H, r_target identifiability)
Stage 15  Cross-model comparison (Izhikevich / LIF / HH-Jaxley)

---

## Ablation Set

- H(t) fully removed (α=0, τ_H→∞)
- Intrinsic coupling only (threshold offset, no synaptic gain)
- Synaptic gain only (no threshold offset)
- Fast timescale (τ_H=100 ms — Abbott/Varela regime)
- Slow timescale sweep (τ_H = 1, 5, 10, 30 s)
- H(t) present, zero drive (r_target = r(t=0))

Evidence standard: ≥5 seeds per condition, mean ± CI, null comparison included.

---

## Figures

1.  Conceptual diagram — H(t) in multi-area laminar hierarchy
2.  H(t) equation + TFNE-Izhikevich coupling
3.  jaxfne operator pipeline
4.  Single-neuron H(t) dynamics (Stage 0–1)
5.  Population adaptation (Stage 2)
6.  Frequency dependence (Stage 3)
7.  Classical oddball / SSA (Stage 7)
8.  Omission paradigm (Stage 8) ← PRIMARY FIGURE
    A: in-vivo LFP reference
    B: simulated LFP proxy
    C: laminar profile
    D: H(t) trace at omission onset
    E: ablation null
    F: spectrolaminar motif comparison
9.  Global-local oddball (Stage 9)
10. Long-term adaptation (Stage 10)
11. Ablation summary
12. Robustness sweep
13. Cross-model generalization (Stage 15)
14. Parameter recovery
15. Summary schematic

---

## Phases

A  Theory complete          ← CURRENT
   H(t) locked, references anchored, title decided
B  H(t) emitter implemented in jaxfne (Izhikevich, threshold mode)
   Stage 0 passing, Stage 1 passing
C  All 16 stages reproduced; Stage 8 is the gate
D  Figures frozen (≥5 seeds per condition)
E  Manuscript draft
F  Internal review
G  Submission — PLOS Computational Biology (primary) / eLife (alt)

---

## Claim Level

claim_level: computational_scaffold
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
All field outputs are Relative unless an explicit calibration step is documented.
