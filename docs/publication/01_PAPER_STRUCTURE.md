# Paper Structure

1. Abstract
2. Introduction
3. Background
   - Omission paradigm and in-vivo observations
   - Homeostatic plasticity: Turrigiano & Nelson 2004, Cannon & Miller 2016
   - Short-term depression as competing account (Abbott & Varela 1997)
   - Laminar cortical hierarchy and spectrolaminar readouts
4. H(t) Theory
   - Minimal formulation: τ_H · dH/dt = r_target - r(t)
   - Why this is the simplest sufficient model
   - Relationship to general HDP
5. Mathematical Formulation
   - TFNE-Izhikevich emitter equations
   - H(t) coupling to intrinsic excitability (threshold offset + gain ablation)
   - Source → Field → Probe chain
6. jaxfne Computational Framework
   - Operator grammar: Configuration → Construct → Simulate → Source → Field → Probe → Objective → Optimizer → Manifest
   - Deterministic reproducibility (seeds, artifact exports)
   - Claim levels: Relative vs. Absolute outputs
7. Experimental Methodology
   - In-vivo omission paradigm (matched stimulus schedule)
   - Simulation parameter matching procedure
   - Spectrolaminar readout protocol
8. Progressive Simulation Results
   - Stages 0–15 (see 02_SIMULATION_SCENARIOS.md)
   - Primary result: Stage 8 — omission response emergence from H(t)
9. Ablation Studies
   - H(t) removed vs. present
   - Intrinsic-only vs. synaptic-only vs. full H(t)
   - Fast (Abbott/Varela) vs. slow (H(t)) timescale comparison
10. Robustness
11. Generalization
12. Discussion
    - What H(t) explains and what it does not
    - Comparison to predictive coding accounts
    - Laminar signature: feedforward vs. feedback interpretation (determined from Stage 8 results)
13. Limitations
14. Future Directions
15. Conclusion
16. Supplementary Material
