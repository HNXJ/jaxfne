# Introduction — Traceability Map

Companion to `introduction_draft.md` (681-709 words, DRAFT). Maps every
proposition to its manuscript sentence and to its evidence authority
(claim ledger / frozen receipts / verified external citation). All
quantitative claims stay in Results/Methods; the Introduction carries no
empirical quantity.

## Proposition architecture

| # | Proposition | Manuscript paragraph | Evidence authority |
|---|-------------|----------------------|--------------------|
| P1 | Simulation studies span dynamics, fields, and readouts; differentiable/JAX generation and established general-purpose simulators cover this space | Para 1 | Citations 1-8 (verified 2026-08-17); no empirical claim |
| P2 | Stage relationships (frozen-trajectory vs separate machinery; relative vs calibrated; hidden-state influence; pathway ownership) are per-tool conventions, not typed contracts; no surveyed framework exposes them as typed, verifiable elements | Para 2 | Survey R2 working notes (artifacts/developer/introduction_working_notes.md); citations 1,2,9; NOT-FOUND finding |
| P3 | Opacity of claim provenance conflicts with verification-oriented reproducibility practice | Para 3 | Citations 10-12 (TOP 2025; Nosek 2015; Heil 2021) |
| P4 | TFNE typed operator grammar: dynamics → source → field → probe; observations are pure downstream compositions; written hidden state can be computationally latent; relative/calibrated semantics is part of the contract | Para 4 | CL-01 (representational grammar); CL-16 (observation boundary); CL-17 (N0==N1 bit-exact under Gamma_H=I); evidence-index anchor sentence |
| P5 | jaxfne is JAX-native; each extension defined by a reduction identity (neutral state restores base; no delays restores instantaneous; no observations restores substrate) | Para 5 | CL-07 (H_K=1 restores base emitter); CL-14 (tau=0 recovers E1); CL-16 (observation-disabled reduction); Methods layers 2-6 |
| P6 | The framework is validated on its own pipeline: estimator validated on synthetic controls only; preregistered no-wave grid reported as negative; adaptation phenotype not supported (attenuation not attributable to written state; gates not satisfied); hierarchical propagation with bit-exact leakage controls; unresolved closed-loop region reported as unresolved; hash-gated frozen figures reproducible from clean checkout | Para 6 | CL-05 (estimator, synthetic controls only); CL-06 (no-wave, tested regimes only); CL-08 (NO_ADAPTATION, D-N2 null); CL-17/18 (leakage control + HIERARCHICAL_PROPAGATION); CL-11/19 (W3b unresolved); CL-20/21 (provenance, reproduction path) |
| P7 | Contribution: explicit, checkable stage boundaries, state semantics, and claim boundaries, demonstrated through registered deterministic validation including negative and unresolved outcomes; no capability/fidelity/training comparison | Para 7 | Synthesis of CL-01..CL-21; A-1a/A-1b/A-2/A-3 receipts; PEC |

## Terminology consistency (vs finalized Results/Methods)

| Term used in Introduction | Results/Methods authority | Status |
|---------------------------|---------------------------|--------|
| relative computational readout | Results "Relative computational readouts"; CL-03 | MATCH |
| canonical relative source | Results "canonical relative source Q"; Methods S_psi | MATCH |
| validated on synthetic controls only | Results "validity is established on synthetic controls only"; CL-05 carve | MATCH |
| tested regimes | Results "no claim ... outside the tested geometry/delay domain"; CL-06 | MATCH |
| pure downstream compositions | Results "pure downstream compositions of the frozen substrate state"; CL-16 | MATCH |
| computationally latent | Evidence-index anchor sentence; CL-17 allowed language | MATCH |
| bit-identical / bit-exact | Results bit-exact N0==N1; CL-17 | MATCH |
| reduction identity | Methods reduction ladder (E2->E1, E3->E2, observation-disabled); CL-07/14/16 | MATCH |
| attenuation cannot be attributed to the written state | Results D3 paragraph (D minus N2 null identical); CL-08 | MATCH |
| closed-loop dynamics remains unresolved | Results W3b paragraph; CL-11/19 (unresolved, not negative) | MATCH |

## Banned-word / overclaim audit

- first/unique/unprecedented/novel: 0 occurrences (grep-verified).
- No "differentiable simulator" contribution framing; differentiability appears
  only as background (para 1).
- No repository jargon (construct(), HDP, RBS, protocol tags, grammar-doctrine
  phrasing); "typed" defined inline (para 4).
- No protocol chronology; no feature lists; no capability/training claims.
- No comparison to named tools beyond neutral landscape description; R-6
  (TVB/NEST/Jansen-Rit comparison demos) honored.

## Citation verification ledger (2026-08-17)

| Ref | Status | Verification basis |
|-----|--------|--------------------|
| 1 LFPy 2018;12:92 | VERIFIED | LFPy README/official docs (GitHub, readthedocs) |
| 2 Sanz Leon 2013;7:10 | VERIFIED | Front Neuroinform record (7:10, doi 10.3389/fninf.2013.00010); author list corrected (Woodman MM, Domide L, Mersmann J, McIntosh AR, Jirsa V) |
| 3 Jaxley Nat Methods 2025 | VERIFIED | nature.com article page; DOI resolves |
| 4 BrainPy eLife 2023;12:e86365 | VERIFIED | eLife page; bibtex in repo README |
| 5 TVB-Optim bioRxiv 2025 | VERIFIED | bioRxiv DOI 10.1101/2025.11.18.689003; active v0.4.0 2026 |
| 6 Hines & Carnevale 1997;9:1179-1209 | VERIFIED | Neural Comput 9(6):1179-1209, doi 10.1162/neco.1997.9.6.1179 |
| 7 Gewaltig & Diesmann 2007;2(4):1430 | VERIFIED | Scholarpedia record; doi 10.4249/scholarpedia.1430 |
| 8 Stimberg 2019;8:e47314 | VERIFIED | eLife record; doi 10.7554/eLife.47314 |
| 9 Huh & Sejnowski NeurIPS 2018;31:1433-1443 | VERIFIED | ML Anthology record |
| 10 Nosek 2015;348(6242):1422-1425 | VERIFIED | Science record; doi 10.1126/science.aab2374; volume corrected (348, not 349) |
| 11 TOP 2025 | VERIFIED | Res Integr Peer Rev 2026; DOI 10.1186/s41073-026-00223-0 |
| 12 Heil 2021;18:1132-1135 | VERIFIED | Nat Methods 18:1132-1135; DOI 10.1038/s41592-021-01256-7 |
| 13 Ermentrout & Kleinfeld 2001;29:33-44 | VERIFIED | Neuron 29:33-44, 2001 |
| 14 Muller 2018;19:255-268 | VERIFIED | Nat Rev Neurosci 19(5):255-268; doi 10.1038/nrn.2018.20 |

Re-verification TODO before submission: ref 5 (check for a journal version of
TVB-Optim beyond bioRxiv).

## R5 findings disposition (see 10_introduction_adversarial_review.md)

- H1/I5 FATAL (D3 mischaracterization) -> REPAIRED (G2): sentence now states
  attenuation not attributable to written state + gates not satisfied.
- C5 MAJOR (typed gloss) -> REPAIRED: inline definition added (para 4).
- L2 MAJOR (readout ambiguity phrasing) -> REPAIRED: "single frozen trajectory
  or separate machinery".
- L1 MINOR -> REPAIRED ("routinely" -> "commonly").
- L6, C4, H5, H6, H7 MINOR -> recorded; L6/H7 re-verified at acceptance.