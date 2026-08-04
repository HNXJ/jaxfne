# Publication Roadmap

Phase A — Theory complete  ← CURRENT
  H(t) equation locked: τ_H · dH/dt = r_target - r(t)
  References anchored (Turrigiano 2004, Cannon 2016, Abbott 1997, Yaron 2025)
  Title locked: "Omission Responses in a Multi-Area Laminar Cortical Hierarchy
                 Emerge from Simple Firing-Rate Homeostasis"
  Summary and novelty statements drafted (see TITLE_AND_ABSTRACT_DECISIONS.md)

Phase B — Simulation engine complete
  H(t) implemented in jaxfne emitter (Izhikevich coupling, threshold mode)
  Stage 0 (single neuron equilibrium) passing
  Stage 1 (repeated pulse) passing

Phase C — All 16 scenarios reproduced
  Stage 8 (omission, PRIMARY) is the gate for proceeding to writing.
  All ablation conditions must pass evidence standard before gate.

Phase D — Figures frozen
  All 15 figures finalized with ≥5 seeds per condition
  Ablation set complete

Phase E — Writing
  Abstract draft (next session)
  Full manuscript draft

Phase F — Internal review

Phase G — Submission
  Target journal: PLOS Computational Biology (primary)
  Alternative: eLife

---

Target outcome:
  One coherent manuscript where H(t) provides the scientific hypothesis,
  jaxfne provides the computational infrastructure, and Stage 8 (omission)
  is the primary empirical result against which the claim is adjudicated.
