# jaxfne arXiv paper plan (Hamm 2026-10-25)

Goal G-0: submit one jaxfne paper to arXiv (q-bio.NC, cross-list cs.NE, CC BY 4.0) with 8 main
figures, 16 supplementary figures, at most 80 references, and checklist C1-C12 at 100/100.
Scope rule: every paper task names its figure (F1-F8, S1-S16) or checklist item (C1-C12). Work that
maps to none waits until Hamm amends this file (dated entry at the end). Releases 0.5.9 to 0.5.11
serve this plan first; 0.6.0 is the acceptance release (amendment 2026-10-26).

Framing (Hamm 2026-10-25): a methods and software paper with validation, theory first. TFNE is a
containment and composition language for neural models of different resolution; geometry, H-state
and plasticity, fast compute and augmentation, and alignment are its tested properties.
Main figures carry no negative results. Every negative, mixed or unresolved verdict is shown in a
supplementary figure and named in the main text.
Alignment means both: spec-to-claim agreement (SPECIFIED, IMPLEMENTED, TESTED, OBSERVED) and fit to data.
Replication set (at least 7): Brunel 2000, Izhikevich 2003, Potjans 2014, CAMINOS, Vogels 2011,
PCL (N'dri 2025), GSDR. Mejias 2016 and Destexhe 2009 are extras.

## Main figures

| Fig | Content | Evidence | Release |
|---|---|---|---|
| F1 | TFNE language: Emitter-Source-Field-Probe; CircuitSpec-construct-Model-simulate-Signals; relative quantities and calibration | doctrine (`tfne_algebra`, containment, relative-quantity grammar) | 0.5.9 |
| F2 | Geometry and containment from one neuron to a column to 20 areas; identity, topology, signs and positions kept through flattening | Atlas AT-10-N20, NeuronalTensor, S3 | 0.5.10 |
| F3 | H-state and HDP: RBS, RBD, HDP; neutral H recovers the base model; typed coupling maps | reduction-identity tests, S5 | 0.5.9 |
| F4 | Fast compute and augmentation: neuron-steps/s against N, compile against run, vmap ensembles, float32/64, Brian2, NEST and Jaxley baselines; observables under augmentation | benchmark prereg and receipts; ensemble study | 0.5.10 |
| F5 | Validation, emitters: Brunel 2000 Fig. 8 regimes; Izhikevich 2003 Fig. 2-3 on the public path | Brunel P3; Izhikevich v7 + E1 | 0.5.9 |
| F6 | Validation, circuits and geometry: Potjans 2014 microcircuit; CAMINOS Fig. 1 on the library path | Potjans P3; CAMINOS P6 v2 | 0.5.9 |
| F7 | Validation, plasticity and fit to data: Vogels 2011 inhibitory balance; PCL; GSDR fits | Vogels P3 (per-synapse rule); PCL criteria that pass; GSDR P3 | 0.5.9 |
| F8 | Alignment: claim coverage by status tier; fit-to-data summary across the replication set | claim ledger, dashboard data | 0.5.11 |

## Supplementary figures

| Fig | Content | Feeds |
|---|---|---|
| S1 | Notation table; each definition mapped to its public symbol and test | F1, C3 |
| S2 | Relative quantities and calibration transforms at semantic boundaries | F1 |
| S3 | Geometry property tests: round trip, permutation invariance, rigid transform | F2, C4 |
| S4 | NeuronalTensor schema and flattening | F2 |
| S5 | HDP reduction identity per registered rule | F3, C5 |
| S6 | Benchmark detail: CPU, GPU, Brian2, NEST, Jaxley | F4, C6 |
| S7 | Augmentation determinism and provenance | F4, C7 |
| S8 | Brunel sensitivity (dt, seeds, S2 and S3 runs) | F5 |
| S9 | Izhikevich Fig. 2 cell classes, eight criteria | F5 |
| S10 | Potjans per-population rates, CV, synchrony | F6 |
| S11 | CAMINOS detail: the R2 PV-SOM lag finding (mixed) | F6 |
| S12 | Vogels segment B and memory recall | F7 |
| S13 | PCL detail, including the signed T2 failure | F7 |
| S14 | GSDR fit detail and held-out seeds | F7 |
| S15 | Negative and unresolved registered results (traveling waves, NO_ADAPTATION, closed loop) | F8 |
| S16 | Extras: Mejias 2016, Destexhe 2009, kernel LFP and Jansen-Rit fields | F8 |

## References (at most 80)

Replicated papers 9; simulators and tools (JAX, Brian2, NEST, NEURON, Jaxley, NetPyNE, LFPy) 12;
plasticity and homeostasis 14; laminar and multi-area anatomy 12; fields and LFP 8; fitting and
optimization 9; reproducibility and preregistration 6; other 10. Each in
`docs/reference/references.md` with a Crossref DOI.

## Checklist (weights sum to 100)

| C | Item | Wt | Check |
|---|---|---|---|
| C1 | Alignment: claim ledger covers every manuscript claim with its status tier | 10 | script: 0 orphan claims |
| C2 | Evidence refrozen at the paper tag; each figure regenerates from a clean checkout | 8 | frozen manifest, one command per figure |
| C3 | Language: definition, symbol and test triples | 12 | S1 complete, 0 orphans |
| C4 | Geometry invariance tests and F2 | 10 | S3 tests fail on a broken invariant |
| C5 | H/HDP reduction identity and typed coupling | 10 | S5 per rule |
| C6 | Benchmarks: prereg, receipts, scaling, Brian2, NEST and Jaxley baselines where the model class applies | 10 | receipts hash-pinned |
| C7 | Augmentation ensemble study with provenance | 6 | two seeds change output; same seed reproduces |
| C8 | The 7 replications: preregs signed, public path, verdicts in F5-F7 or S-figures | 14 | F8 and S15 match the dashboard data |
| C9 | Reproducibility: lock file, Zenodo DOI, license, data and code statement | 8 | fresh-clone build of all figures |
| C10 | Manuscript current; at most 80 references with DOIs; figures pass the visual-style audit | 7 | reference count, DOI check, render audit |
| C11 | Independent science review: second reviewer, then an external reader | 3 | findings fixed or recorded |
| C12 | arXiv mechanics: TeX builds on arXiv TeX Live; category; endorsement; ancillary files | 2 | arXiv preview compiles |

## Sequence

0.5.8: published 2026-10-26. 0.5.9: F3, F5-F7 lanes (Izhikevich E1, Potjans P3, CAMINOS S2 rerun,
Vogels rerun and P3, PCL R2-R5, GSDR prereg and P3), S1. 0.5.10: F2, F4, S3, S6, S7, S16. 0.5.11:
manuscript rewrite, refreeze (C2), review (C11), arXiv (C12). 0.6.0: acceptance. Packages:
`artifacts/etudes/meta/plans/release_plan_0.5.9-0.6.0.md`.

## Amendments

- 2026-10-26 (Hamm): 0.6.0 is the post-peer-review and acceptance release, a hard and complete one.
  Submission moves to 0.5.11, the release after 0.5.10 (the dispatcher's numbering); F8 and the
  0.6.0 sequence items move with it.
- 2026-10-26 (Hamm, H1): F1 is frozen fig01, F3 is frozen fig06, S15 is frozen fig05; fig02–04 and
  fig07 stay frozen evidence outside the paper.
- 2026-10-26 (Hamm, H2): D21 maps to `jaxfne.emitters.resolve_edge_weight`; D34 becomes the public
  `jaxfne.emitters.rbd_recurrent_gain`.
