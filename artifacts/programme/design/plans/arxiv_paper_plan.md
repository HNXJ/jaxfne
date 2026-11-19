# jaxfne arXiv paper plan (Hamm 2026-10-25)

Goal G-0: submit one jaxfne paper to arXiv (q-bio.NC, cross-list cs.NE, CC BY 4.0) with 8 main
figures, 16 supplementary figures, references with DOIs (no cap, Hamm 2026-10-28), and checklist C1-C12 at 100/100.
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
| C10 | Manuscript current; references with DOIs; figures pass the visual-style audit | 7 | DOI check, render audit |
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
- 2026-10-26 (Hamm, H4): the publication ledger keeps CL-1 to CL-26; the E2 V1/V2 claims are CL-27 to
  CL-34. H5: LICENSE names Hamed Nejat; the data and code statement cites third-party inputs by DOI and
  licence and never redistributes them.
- 2026-11-02 (Hamm): a jaxfne short paper, non-archival, goes to the NeuroAI 2027 workshop (AAAI 2027,
  Montreal, 2027-02-22/23; 4 pages, references excluded; double-blind; AAAI-27 author kit; OpenReview;
  deadline 2026-11-20 23:59 AoE, UTC-12, decisions 2026-12-05; per the OpenReview submission form read
  2026-11-02: short papers are non-archival, the bar on concurrent submissions binds archival submissions only,
  every author needs an OpenReview profile and is asked to join the reviewer pool): the TFNE language and the pre-registered replication
  programme with its failures reported. It reuses F1 and the 0.5.9 validation evidence and does not
  replace the arXiv paper. jomission drops its engine row and submits its column paper separately.
  Submission on OpenReview is Hamm's act.
- 2026-11-19 (Hamm, directive "Execute the 15-Stage TFNE Manuscript Programme", Option A: unified
  mathematical and computational framework): the paper follows the fifteen stages below, scored as
  workflow completion (done 1, started 0.5, else 0; score = 100 x credit / 15) beside the unchanged
  C1-C12 weights. The eight main figures become: 1 TFNE theory and architecture; 2 heterogeneous emitter
  composition; 3 geometry and source-field mapping; 4 ephaptic feedback and ablations; 5 H-state
  multiphysics if validated, else a verified numerical or compositional result; 6 multiscale observation
  operators; 7 published-model replications, failures included; 8 scaling and validity boundaries.
  Read by the main session as superseding, for the main figures only, the F1-F8 table and "main figures
  carry no negative results" (Fig 7 shows failures). C1-C12 keep their weights and checks; a check naming
  an old figure reads the new one carrying that content (F2 -> 3, F3 -> 5, F4 -> 8, F5-F7 -> 7, F8 -> 7).
  H1's frozen mappings (F1 = fig01, F3 = fig06, S15 = fig05) are audited at stage 12 before any change.

## Fifteen-stage programme (2026-11-19)

Baseline: jaxfne `_temp_tfne-manuscript` at 022da7a0 (dev c892ba30 plus T-86); main e0160dfc. Status
words: PROPOSED (written), IMPLEMENTED (in code), TESTED (a test that can fail), VERIFIED (independent
review). A stage closes only on its exit check and a non-writing reviewer's ACCEPT.

| Stage | Status | Evidence at baseline | Closes when |
|---|---|---|---|
| 1 Scope, novelty, claims | started | framing above; claim ledger CL-01..CL-34 (`artifacts/publication/publication_claim_ledger.json`) is the wave-estimator paper's | every primary claim has a distinguishing property, evidence requirement and limitation (matrix below), reviewed |
| 2 Mathematical theory | started | `docs/theory/doctrine/` (`tfne_algebra`, `rbs_rbd_hdp`, containment), `docs/theory/source_field_equations.md` | the gaps below resolved or scoped; independent math review finds no material inconsistency |
| 3 Implementation, reproducibility | started | `CircuitSpec -> construct -> Model -> simulate -> Signals`; frozen manifest of the old figure set | each demonstrated feature has a public or documented path and a receipt |
| 4 Numerical verification | started | delay-boundary, determinism (release plan C2), segmented-continuation (methods §4) tests | mandatory numerical gates pass or restrict claims |
| 5 Heterogeneous emitters | started | families registered (inventory pending) | three families validated, heterogeneous composition shown |
| 6 Geometry and fields | started | A3 geometry tests; kernel LFP; P-056 open | field kernels checked against analytic cases within stated domains |
| 7 Ephaptic coupling | started | `jaxfne/hdp_rules_ephaptic.py`, alpha = 0 identity test | an effect attributable to the operator, with zero, shuffled and perturbed controls, or experimental-only |
| 8 H-state | started | RBS/RBD/HDP kernels; reduction-identity tests | only validated components enter Results |
| 9 Multiscale observables | started | linear proxy operators (methods §6); no physical solve | one circuit, several independently checked observables |
| 10 Replications | started | lanes and verdicts in `dashboard/data/replications.json` | at least 7 auditable records, negatives kept, reviewer verdict each |
| 11 Scaling | started | B1 prereg signed, no scored run (release plan B2) | matched-accuracy benchmark receipts |
| 12 Figures | started | frozen fig01-fig07 of the old paper | every figure: generator, provenance, frozen inputs, independent check |
| 13 Manuscript | started | `artifacts/programme/paper/publication/manuscript/` (wave-estimator paper, CL-01..CL-21) | complete draft, no unsupported primary claim |
| 14 Review and lock | not started | | reviews ACCEPT; owner-approved methodological lock |
| 15 Publication verification | not started | | C1-C12 = 100; gates pass; owner authorizes submission |

Workflow 2026-11-19: 13 started, 0 done = 6.5 / 15 = 43/100. C1-C12: 0/100 scored (no item's registered
check run at this baseline). Mandatory gates: broad gate and CI per promoted SHA (`artifacts/state.md`).

### Stage 2 gaps found 2026-11-19 (PROPOSED; each has a todo row)

- Ephaptic operator: the code computes Phi_i = alpha sum_j M_ij (v_j - v_rest) from neighbour voltages
  (`jaxfne/hdp_rules_ephaptic.py:1-30`), not phi = K[s] from the source; written that way it is a
  distance-weighted voltage coupling with the algebra of a gap-junction proxy (no -v_i self term). The
  methods' upstream-feed prohibition (manuscript `methods.md` §1) does not hold once it is on. Either
  derive C_eph from the source-field chain or name the operator for what it is.
- Source Q_t = s_scale (I_native + g_spk s) (`methods.md` Eq. 5) is a relative proxy without
  transmembrane current balance; LFP, CSD and EEG/MEG readouts from it are proxies (`methods.md` §6). Two
  paths differ: `jaxfne/fields/kernel_lfp.py` uses cable kernels whose membrane currents sum to zero
  (`tests/test_kernel_lfp.py:32`), and `jaxfne/fields/head.py` holds spherical-head EEG/MEG leadfields
  (`eeg_sphere_leadfield` :258, `meg_sphere_leadfield` :227). Stage 9 claims physics only on those paths.
- H boundedness is clamp-imposed (`methods.md` §5.3); a stability claim needs the unclamped analysis.
- Notation: the directive's continuous form (x neural state, s source, h H-state, phi field, y_m probes)
  reuses letters the sealed language already binds (`docs/theory/doctrine/tfne_algebra.md:19-27`: x upstream
  input, h mutable carry, s static state; execution normal form (h_t, x_t; s) -> (h_{t+1}, y_t)). The paper
  needs one notation table (S1). Proposed mapping, to check against code at stage 3:

| Directive | Sealed language / code | Note |
|---|---|---|
| x (neural state), its ODE | part of h (kernel carry: v, u, synapses, spikes) | executed as a discrete map, explicit Euler; the ODE is the per-family generator |
| u (input) | x (typed upstream boundary) | drives and stimuli |
| h (H-state), dh/dt = H(...) | H inside h; RBD dH/dt = F_H, HDP dW/dt = F_W (`rbs_rbd_hdp.md`) | typed coupling maps Gamma_H required |
| G (geometry), theta | s (static: topology, parameters, fractional geometry) | geometry is relative, no mm semantics (`tfne_algebra.md:91-100`) |
| s = S(x, h; G) | Q = S_psi(x_t), relative source (`methods.md` Eq. 5) | no current balance (gap above) |
| phi = K[s] | Phi = F_gamma(Q), post hoc on frozen traces (`methods.md` §1) | linear proxy |
| lambda_eph C_eph(x, phi) | `ephaptic_field` HDP rule, alpha (`hdp_rules_ephaptic.py`) | reads v, not Phi (gap above) |
| y_m = P_m(s, phi, h; G_m) | Y = P_eta(Phi) | probes |

## Feasibility and novelty gate (Hamm 2026-11-19)

The first deliverable is this gate, not a draft. Four outputs; no expensive experiment, GPU run or new
replication starts before G4 returns a verdict. H-state, EEG/MEG and ephaptic coupling are secondary: they
enter the paper only as validated, and an unvalidated one never blocks the core framework paper.

| Output | Content | Home | Status |
|---|---|---|---|
| G1 Novelty comparison | TFNE against established frameworks, by mathematical or computational property | below | PROPOSED |
| G2 Evidence matrix | each capability: SPECIFIED, IMPLEMENTED, TESTED or VERIFIED, with its file | below | PROPOSED |
| G3 Replication inventory | each published-model lane: frozen criteria, verdict, discrepancies, missing receipts | below | PROPOSED |
| G4 Stage 1-2 review | independent critic: mathematical consistency, novelty, claim scope | verdict recorded below | running |

Gate verdict options: PUBLISHABLE CORE (name the claims), NARROWED (name what drops), NOT DEFENSIBLE.

### G1 Novelty comparison (PROPOSED; rows about other tools are inferred until checked against their papers)

Several properties are not new and cannot carry the claim: several neuron models in one network (Brian2, NEST,
NetPyNE, BrainPy do this), differentiable simulation in JAX (Jaxley, BrainPy), declarative network specification
(NeuroML, NineML, PyNN, NetPyNE), forward models to LFP/EEG/MEG (LFPy, hnn-core, TVB; physically grounded where
ours are proxies).

| Framework | Already provides |
|---|---|
| Brian2, NEST, GeNN, Arbor | equation- or library-defined networks; mixed neuron models in one network; NEST and Arbor scale |
| NEURON + NetPyNE + LFPy/LFPykit | declarative specs; LFP/EEG from multicompartment transmembrane currents |
| NeuroML, NineML, PyNN | simulator-independent model description |
| Jaxley | differentiable multicompartment simulation in JAX |
| BrainPy / brainstate, Norse, snnTorch | differentiable spiking networks with several neuron models |
| The Virtual Brain; neural mass and field models | population dynamics on connectomes with EEG/MEG forward models |
| hnn-core | laminar model with a calibrated dipole-to-EEG/MEG link |

Candidate distinguishing properties, each with the test that would falsify it:

| ID | Property | Evidence needed | Falsified if |
|---|---|---|---|
| N1 | A sealed specification algebra (`x:A:y`; define, factor, compose, realize, flatten) whose flattening keeps a bidirectional index map (path <-> slice, rule <-> edge) and refuses what it cannot express (`tfne_algebra.md`) | formal statement; tests that the map is a bijection on realized objects | NeuroML/NineML/NetPyNE already give an invertible spec-to-array map with refusal semantics |
| N2 | Reduction identities as contracts: disabling an extension (H, delays, observation, ephaptic alpha = 0) returns the base model bitwise through the public path | per-extension tests that fail on a planted break | a listed tool states and tests the same identities, or ours hold only on private paths |
| N3 | Typed slow state H with typed coupling maps Gamma_H into emitter, source, field and probe (RBS/RBD/HDP), neutral H recovering the base | formal definition; reduction tests; one non-trivial validated use | it reduces to ordinary extra state variables with no added semantics |
| N4 | Relative-quantity grammar: amplitudes stay relative until an explicit calibration at a semantic boundary | formal statement; refusal tests | it is a units convention only, with no checked consequence |

Feasibility reading (inferred, for G4 to test): N1 and N2 are the likeliest defensible core: a methods paper on
compositional semantics with verified contracts and a replication programme. N3 is defensible only with a
validated use. N4 is supporting. If the critic finds N1 and N2 already present elsewhere, the paper narrows to a
software and replication report.

### G2 Evidence matrix (2026-11-19, at 022da7a0; inventory `D:/tmp/_temp_ms/inventory.md`, not tracked)

VERIFIED means an independent reviewer's verdict on the capability itself; none holds one yet.

| Capability | Highest class | Evidence | Missing for the paper |
|---|---|---|---|
| N1 specification algebra | TESTED, partial | sealed `tfne/2` (`docs/theory/doctrine/tfne_algebra.md`); compiler built against `tfne/1`, 7 clauses not conformant (:163-174); `tests/test_tfne_algebra.py`, `tests/test_tfne_parameter_transfer.py` | a bijection test of the index map; conformance or scoped claim |
| N2 reduction identities | TESTED | `tests/test_hdp_reduction_identity.py:398`, `tests/test_hdp_zero_rule_public_identity.py:56`, zero-delay bit equivalence (`methods.md` §4), ephaptic alpha = 0 (`tests/test_hdp_rules_ephaptic.py:115`, `:171`) | one table of every extension with its identity test and a planted-break check |
| N3 typed H-state | TESTED | 8 registered HDP rules (`jaxfne/hdp_rule.py:477-618`, PCL, Vogels iSTDP, ephaptic); reduction tests above | one validated non-trivial use (candidates: Vogels iSTDP, PCL STDP) |
| N4 relative quantities | SPECIFIED, TESTED in part | `docs/theory/doctrine/relative_quantity_grammar.md`; probe semantic status (`methods.md` §6) | refusal tests named per boundary |
| Emitter families | TESTED per family | registry `izhikevich`, `lif`, `glif` (registered, not routed), `adex`, `rate` (`jaxfne/emitter_registry.py:106-110`, `jaxfne/neuronal_tensor.py:93`); HH and Jansen-Rit outside the registry | a test of two families in one circuit (none found); stale comment `neuronal_tensor.py:89-91` |
| Geometry | TESTED | `Geometry3D`, `NeuronalTensor`; `tests/test_geometry_invariants.py:112,136,192`; A3 mutation receipt (release plan) | relative coordinates only (`tfne_algebra.md:91-100`) |
| LFP, CSD proxies | TESTED | `jaxfne/fields/probes.py:536,612`; `jaxfne/fields/proxy.py:1376` | proxy status kept in text |
| Kernel LFP | TESTED | `jaxfne/fields/kernel_lfp.py:286`; current balance `tests/test_kernel_lfp.py:32`; P-056 open (deep-layer power) | P-056 |
| EEG/MEG | TESTED (sphere leadfield) | `jaxfne/fields/head.py:227,258`; `tests/test_head_forward.py` | conductivity and source-physics statement; secondary |
| Ephaptic | TESTED | `jaxfne/hdp_rules_ephaptic.py`; 4 tests (`tests/test_hdp_rules_ephaptic.py:79-171`) | physical derivation (T-98); controls; secondary |
| Differentiability | TESTED | `jaxfne/optim/`, surrogate spike (`jaxfne/emitters.py:318`); `tests/test_gradient_checks_v0417.py` | gradient check against finite differences named in text |
| Scaling | SPECIFIED | B1 prereg signed 2026-11-06; no scored run (release plan B2) | matched-workload receipts; secondary for the gate |

### G3 Replication inventory (2026-11-19; registry `dashboard/data/replications.json` generated at dev d30e7418, 2026-11-12)

| Lane | Frozen criteria | Latest verdict (file) | Path | Discrepancy or missing |
|---|---|---|---|---|
| Brunel 2000 | P3 v4 signed 2026-10-19 | close, Q1-Q3 pass (`brunel2000/RESULT_P3.md:3`) | public API | registry says "reproduced (conditional)"; Q3 reading chosen after a failed pilot |
| Izhikevich 2003 | P3 v7, v8.2, P6 v2 signed | P6 PASS Q1-Q5 (`izhikevich2003/RESULT_P6.md:9`); P3 v7 FAIL, v8.2 passes post hoc | public API; NumPy oracle | post hoc pass counts as partial; registry verdict null |
| Potjans 2014 | P3 v2 signed 2026-11-02 | off (`potjans2014/RESULT_P3.md:3`) | public API | no oracle; registry null |
| CAMINOS | P3 v6, v7; P6 v2 signed | P6 PASS; P3 PASS under v7 after v6 R4 failed; PV-SOM lag 6.47 vs 8.5 ms off | public API; JAX oracle | registry "mixed" |
| Vogels 2011 | P3 v4 signed 2026-11-02 | off, R-E 4.13 Hz (`vogels2011/RESULT_P3.md:3`) | public API | registry null |
| PCL (N'dri 2025) | Fig 3 v4, v5, R4b signed | FAIL at v4, v5, R5-exact, R5-approximate (`pcl/prereg/P3_fig3_r5_approx_result.md:10`) | core equals numba oracle on 16 pilot files | P-055 attributed |
| GSDR | P3 fit signed 2026-11-01 | FAIL, 0 of 5 seeds (`gsdr2026/RESULT_P3_fit.md:10`) | UNKNOWN | no DOI (unpublished source) |
| Destexhe 2009 | P3 v3 signed 2026-11-02 | FAIL, 0 of 5 (`destexhe2009/RESULT_P3.md:3`) | public API | registry null |
| Jansen-Rit 1995 | P3 v1, signed by the keeper agent on Hamm's standing go (erratum) | PASS 5 of 5, conditional on a declared hold and RK4 | lane scripts | signature by agent; approximate integrator |
| Mejias 2016 | P3 unsigned | none | public API | not scorable |
| lee2025, mazzoni2015, pcl_h, qlearning | none | none | | notes or task lanes; not replications |

Nine lanes have signed criteria and a scored verdict: 4 positive (Izhikevich P6, CAMINOS P6, Jansen-Rit
conditional, Brunel close), 5 negative or off (Potjans, Vogels, PCL, GSDR, Destexhe). None has an independent
reviewer verdict yet. Registry fields lag the RESULT files, and `scripts/check_replications.py` reads
`dashboard/sources.json`, not `replications.json`.
