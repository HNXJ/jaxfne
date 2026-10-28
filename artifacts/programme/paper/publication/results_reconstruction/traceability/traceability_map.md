# Results draft — evidence traceability map

Checkpoint `publication_results_reconstruction` · baseline `dev@f30c2d5` (revised
`dev@<post-4R>`) · audited by `scripts/audit_results_draft.py`.

Paragraph labels P1–P20 correspond to the twenty `##` sections of
`results_draft.md` in order (titles below). All Q markers are inline in the prose;
claim tags `{CL-xx}` and receipt paths were removed from the prose in the 4R
revision and are carried exclusively by this map.

## Sentence/paragraph-level trace

### [P1] Grammar map — `{CL-01}`
| Claim | Artifact |
|---|---|
| CL-01 (representational_document) | `artifacts/publication/fig01_grammar_spec.json`, `artifacts/publication/fig01_generation_receipt.json`, `artifacts/figures/publication/fig01_tfne_grammar.png` |

No empirical quantity; forbidden: presenting Fig. 1 as an empirical/demonstrated result.

### [P2]–[P3] Canonical source — `{CL-02, CL-03, CL-04, Q01}`
| Claim | Artifact |
|---|---|
| CL-02 (deterministic_numerical_property) | `artifacts/publication/fig02_04_experiment_a_spec.json` (cross_figure_invariant), `artifacts/etudes/experiment_a/canonical_source.npz`, `b1_canonical_receipt.json` |
| CL-03 (deterministic_numerical_property) | Fig. 3 receipts; `fig03_generation_receipt.json` |
| CL-04 (method_boundary_statement) | Fig. 4 receipts; `fig04_generation_receipt.json` |

- Q01: `cross_figure_invariant == "hash(Q_Fig2) = hash(Q_Fig3) = hash(Q_Fig4)"`

### [P4]–[P9] Wave coverage — `{CL-05, CL-06}`

| Claim | Artifact |
|---|---|
| CL-05 estimator validity on synthetic controls only | `artifacts/protocol_c/c0_wave_protocol_spec.json` (preregistered_estimator_parameters); estimator module SHA `684859a98da51de79887ec26ba8d7134e2fa0e97` |
| CL-06 no TW across preregistered ring regimes (scope-qualified) | `artifacts/protocol_c/c3_execution_receipt.json`, `c3_condition_summary.json`, `artifacts/publication/fig05_generation_receipt.json` (polarity NEGATIVE, DEMONSTRATED, outcome C) |

Quantitative markers (post-freeze validation, reported separately):

| Q | Value quoted | Source |
|---|---|---|
| Q02 | 48 positive cases, all detected | `p2v_a1a_receipt.json` `n_positive_cases`, `summary.all_positives_pass` |
| Q03 | zero frequency error | `p2v_a1a_receipt.json` cases → estimator errors |
| Q04 | wave-number error ≤ 0.00831 relative | cases → estimator errors (measured max) |
| Q05 | zero direction error | cases → errors |
| Q06 | velocity error ≤ 0.00824 relative | cases → errors (measured max) |
| Q07 | 5/5 negatives rejected | `n_negative_cases`, `summary.all_negatives_pass` |
| Q08 | freq 8.5–12.5 Hz; v 53.407 → 39.27 mm/s | `summary.frequency_range_recovered` / `velocity_range_recovered_mm_per_s` |
| Q09 | 60 cells; delay 4 uniform | `c3_execution_receipt.json` `n_cells`, per-cell `delay_steps` |
| Q10 | 60/60 NO_WAVE; 52/4/4 quality mix | per-cell estimator `classification` / `quality_reasons` |
| Q11 | 15 points × 3 seeds = 45 cells | `p2v_a1b_receipt.json` `n_points`, `n_cells` |
| Q12 | 45/45 NO_WAVE, 0 invalid | `point_outcomes` |
| Q13 | NO_POSITIVE_DOMAIN_IN_TESTED_RANGE | `domain_outcome` |
| Q14 | anchor vc0.131_k1 bitwise (3 seeds, max\|ΔV_m\|=0.0) | `anchor_identity` |
| Q15 | reasons 27/11/7; 2 spiking neurons, max 0.5 Hz; no adaptive extension | cells `quality_reasons` / `activity_summary`, `no_adaptive_extension_observed` |
| Q16 | S1 40 cases | `p2v_a2_receipt.json` `stage_S1.n_cases` |
| Q17 | γ\* = 1.0 on 3 cells × 2 phases | `stage_S2.cells[*].gamma_star` |
| Q18 | S3 12 cases; 6-site parity | `stage_S3.n_cases`, per-case classification |

### [P10]–[P16] RBS ladder and nulls — `{CL-07..CL-12, CL-19}`

| Claim | Artifact |
|---|---|
| CL-07 typed vector RBS; H_K=1 restores emitter | `artifacts/protocol_d_biological_rbs/` D0/D1 receipts |
| CL-09 state memory P1/P2; no P3 | D2a/D2b receipts; fig06 spec (panel A/B) |
| CL-12 written ω frozen during expression (W1/W2); no W3 | D/W receipts; fig06 spec (panel D/E) |
| CL-10 H4 negative | `artifacts/protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json` (FROZEN_NEGATIVE_RESULT) |
| CL-08 D3 NO_ADAPTATION | `d3_interpretation_receipt.json`, `d3_execution_receipt.json` |
| CL-11 / CL-19 W3b unresolved | `artifacts/protocol_w/w3b_parameter_domain/w3b_domain_receipt.json` (FROZEN_ANALYSIS) |

| Q | Value quoted | Source |
|---|---|---|
| Q19 | M_X 0.0 / 0.0521 / 0.0 / 0.0; α_length = 0.0 | `h4_interpretation_receipt.json` `primary_endpoint_results` / `factorial_estimates` |
| Q29 | α_heterogeneity = +0.0521; α_interaction = −0.0521 (cancellation) | `h4_interpretation_receipt.json` `factorial_estimates` (mu 0.0, alpha_length 0.0) |
| Q20 | D: 9/9 attenuation, 0/9 formal; counts 9/9/9 vs 9; D−N2 0.2857 vs 0.2857 | `d3_interpretation_receipt.json` `questions.Q1_mechanism`, `classification_counts_all_arms`, `primary_contrast_D_minus_N2` |
| Q25 | 36 cells × 4 arms | `d3_execution_receipt.json` `n_cells` |
| Q21 | 2187 lattice; D=243, S=0, C=0, U=0, X=1944; N_S=0 | `w3b_domain_receipt.json` `regime_counts` / `aggregate_quantities` |
| Q22 | DEFAULT_HDP H∈[1.0000,1.0008], \|w\|=6.0, growth 1.00000; DESYNC H∈[1.0000,1.0310], \|w\|∈[5.8692,6.1684], growth 1.00004; invariants pass | `p2v_a3_receipt.json` `runs`, `all_hard_bound_invariants_pass`, `scoped_statement` |

### [P17]–[P19] Hierarchy, delays, composition, causal propagation — `{CL-13..CL-18}`

| Claim | Artifact |
|---|---|
| CL-13 two-area hierarchy FF/FB | `artifacts/protocol_e_integration/` E1 receipts |
| CL-14 delays compose; zero → E1 | E2 receipts |
| CL-15 sparse H_K composes; H\* → E2 | E3 receipts |
| CL-16 observations downstream | E4 receipts |
| CL-17 negative control (Γ_H=I, X bit-exact) | E5 `sanity_checks` (N0≡N1 bit-exact) |
| CL-18 propagation 3/3 seeds | E5 `quality_gates.G7_classification_applied` |

| Q | Value quoted | Source |
|---|---|---|
| Q24 | 9 trajectories; N0≡N1 bit-exact (V_m, spikes, Q) 3/3; G1 H_K(N1)≡H_K(D) 3/3; G7 per-seed ×3 | `e5_execution_receipt.json` `design.trajectory_count`, `sanity_checks`, `quality_gates` |
| Q30 | owner 9.263 mV / +7 spikes; A2 non-owner 2.430 mV / 0; A1 3.162 mV / +9; gates G_O..G_Y true, d_propagation Y, identical seeds 11-13 | `e5_execution_receipt.json` `quality_gates.G3_owner_contrast_measurable.per_seed` |
| Q23 | — (same evidence as Q24; used for the Fig. 7E bit-exact sentence) | as above |

### [P20] Provenance — `{CL-20, CL-21}`

| Claim | Artifact |
|---|---|
| CL-20 single frozen evidence set, per-file SHA-256 | `artifacts/publication/fig01..fig07_generation_receipt.json` (`figure_sha256`), `equivalence_report.json` |
| CL-21 reproducible from clean checkout | `pyproject.toml` pins; documented entrypoints |

| Q | Value quoted | Source |
|---|---|---|
| Q27 | 7/7 receipts' `figure_sha256` match shipped PNGs | computed by auditor over `artifacts/figures/publication/*.png` |
| Q26 | equivalence gate 7/7 byte-sha-equal | `equivalence_report.json` cases `byte_sha_equal` |
| Q28 | pins present (matplotlib ≥3.10.9,<3.11; scipy ==1.17.1) | parsed from `pyproject.toml` |

## Claim-coverage tally

| Claim | Paragraph | Checked |
|---|---|---|
| CL-01 | P1 | ✓ |
| CL-02 | P2 | ✓ |
| CL-03 | P3 | ✓ |
| CL-04 | P3 | ✓ |
| CL-05 | P4, P5, P8, P9 | ✓ |
| CL-06 | P6, P7, P9 | ✓ |
| CL-07 | P10 | ✓ |
| CL-08 | P14 | ✓ |
| CL-09 | P11 | ✓ |
| CL-10 | P12 | ✓ |
| CL-11 | P15 | ✓ |
| CL-12 | P13 | ✓ |
| CL-13 | P17 | ✓ |
| CL-14 | P17 | ✓ |
| CL-15 | P17 | ✓ |
| CL-16 | P18 | ✓ |
| CL-17 | P19 | ✓ |
| CL-18 | P19 | ✓ |
| CL-19 | P15, P16 | ✓ |
| CL-20 | P20 | ✓ |
| CL-21 | P20 | ✓ |

## Forbidden-overclaim audit (implemented in the auditor)

| Regime | Term(s) | Allowed paragraphs only |
|---|---|---|
| CL-01 | figure 1 empirical/demonstrated result | none — must not occur |
| CL-03 | proxy equals experimental LFP/CSD | none |
| CL-04 | physical/quantitative EEG/MEG amplitude interpretation | none |
| CL-05 | "no traveling wave exists outside the tested regime" | none — only tested-regime claim allowed |
| CL-06 | cortex / cortical / in vivo | none |
| CL-07 | new biological mechanism / new neuron type | none |
| CL-08 | attenuation ≡ adaptation | P14 only (with NO_ADAPTATION); "fatigue" as formal mechanism: none |
| CL-09 | recurrent geometry/delays extend memory (P3) | none; "cognitive/cognition/predictive" : none |
| CL-10 | "length has no effect on memory" generalised | none; only "no positive effect detected by preregistered assay" |
| CL-11/CL-19 | "robust-active" as demonstrated domain | P15 only; "stability ... classified" positive: none |
| CL-12 | closed-loop feedback demonstrated | none; "closed-loop" only with "not demonstrated"/"remains open" |
| CL-13 | functional FF/FB spectral claim; biological specialization | none |
| CL-14 | delay-dependent spectral/oscillatory function | none |
| CL-15 | HDP/plasticity claim in composition | none (P17 must not pair "plasticity" with E3) |
| CL-16 | observations feed back into substrate | none |
| CL-18 | feedback suppresses/enhances a band; cognition/predictive processing | none |
| CL-20/CL-21 | byte-for-byte reproducibility beyond equivalence gate; "external" | P20 only (equivalence-gate-scoped) |

The auditor scans the draft for each term and fails unless every hit lies in a listed
allowed paragraph; the receipt of the run is
`artifacts/publication/results_reconstruction/results_audit_receipt.json`.