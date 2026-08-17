# Abstract traceability map — A = Π_abstract(M)

The Abstract is a lossy projection of the finalized proposition graph M (user directive
2026-08-17): every sentence compresses frozen propositions already present in the
manuscript; no new scientific proposition is introduced. Draft: `abstract_draft.md`.

| Sentence | Compressed content | Source propositions (frozen authorities) |
|----------|--------------------|------------------------------------------|
| S1 | Motivation: stage relationships set by per-tool convention, not contract; claim boundaries hard to verify | Introduction ¶1–2 (convention-not-contract opacity), Introduction ¶7 / P15 (claim boundaries explicit and checkable as the contribution) |
| S2 | jaxfne, JAX-native; typed stages dynamics/source/field/probe with ownership and disablement contracts; quantities declared relative or calibrated | Introduction ¶4 (typed stages, ownership/disablement contract), Introduction ¶4 (quantities declared relative or calibrated), Results P1 (one object grammar), Discussion D |
| S3 | Reduction identity: neutral written state, removed delays, disabled observations recover base semantics exactly | Introduction ¶5 (reduction identities verbatim), Results P12 (H_K = 1 restores base emitter), Results P16 (delay zero recovers E1; H = H* recovers E2), Results P18 (disabling observation recovers substrate state) |
| S4a | Registered deterministic validation found no traveling waves across preregistered geometries and delay policies | Results P6–P10 (grid 60-cell, search 45-cell, sensitivity floor; Fig. 5 outcome C), Discussion ¶2 (bounded negative), CL-06 |
| S4b | by an estimator validated on synthetic controls only | Results P5 (CL-01..CL-07; 48/48 recovery, 5/5 negative controls), Introduction ¶6 ("validated on synthetic controls only" verbatim) |
| S4c | candidate adaptation phenotype reported as attenuation, not adaptation | Results P13 (D3; CL-20; "attenuation is reported as attenuation, not adaptation"), Introduction ¶6, Discussion ¶3 |
| S4d | propagation of a written state through an existing hierarchical feedback pathway | Results P19 (E5: 3/3 HIERARCHICAL_PROPAGATION, bit-exact leakage controls; CL-16..CL-18), Discussion ¶3 |
| S5 | closed-loop region remains unresolved | Results P15 (W3b: 1944 active unresolved, not a negative result; CL-21), Discussion ¶3, Limitations ¶2 |
| S6 | figures from a single frozen, hash-gated evidence set, reproducible from a clean checkout | Results P20 (Q27 7/7, Q28 pinned dependencies, clean-checkout entrypoint), Introduction ¶6 |

## Compression contract compliance

- Word count 149 ≤ 150 (Nature Methods limit). Verified by script (2026-08-17).
- Single paragraph; no citations; no undefined abbreviations (JAX standard usage).
- Forbidden words (first/unique/unprecedented/novel/robust): 0.
- Language gates: "validated on synthetic controls only" (allowed exact draft language);
  "relative or calibrated" (Relative/Calibrated framing); "attenuation, not adaptation"
  (no mechanism claim); "remains explicitly unresolved" (unresolved stays unresolved).
- No comparison of speed, fidelity, or training capability; no biological fidelity claim;
  no per-constituent novelty statement ("first", "novel" absent).
- Omitted by projection (lossy by design): all quantitative markers {Qnn}, figure numbers,
  internal tags, HDP acronym, estimator revision SHA, five-regime taxonomy, W3b lattice
  counts, CIs statement — each lives in Results/Supplement, not the Abstract.