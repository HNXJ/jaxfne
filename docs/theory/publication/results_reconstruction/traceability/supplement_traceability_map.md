# Supplement traceability map (final assembly, 2026-08-16)

Every section of `supplement.md` maps to the frozen receipts it quotes. The audit
`scripts/audit_supplement_draft.py` re-derives every quantitative table value from these
receipts and fails if the draft and the receipts disagree.

| Section | Content | Authority (frozen receipt / sealed artifact) |
|---|---|---|
| Header | authority phase; evidence boundary; corrections to pre-material | Methods seal lineage; `supplement_material.md` (historical) |
| S.1 | RFFT bin spacing; S3 frequency quantization; A-1a integer-bin claim | `artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json` (stage_S3); `artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json` |
| S.2 | Units/magnitude conventions | `scripts/p2v_a1a_synthetic_control.py`, C3 spec, `estimate_traveling_wave` source; Methods §2, §11 |
| S.3 | RNG usage note | executor sources (`c3_neural_experiment.py`, A-series executors); Methods §2 |
| S.4 | C0-registered vs implemented branches | `artifacts/protocol_c/c0_wave_protocol_receipt.json`, estimator source (SHA 684859a…), C3/A-series receipts |
| S.5 | Protocol/regime definitions | Methods §2, §4, §10, §11; `c3_neural_experiment_spec.json`, D3/E5 specs, `artifacts/etudes/experiment_a/` |
| S.6 | C3 per-seed classification (60 cells) | `artifacts/protocol_c/c3_execution_receipt.json`; `c4_interpretation_receipt.json` |
| S.7.1–S.7.3 | A-1a tolerances, 48 positives, 5 negatives | `artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json` |
| S.7.4 | A-1b dynamic search (45 cells, anchor) | `artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json` |
| S.8.1 | A-2 S1 grid (40 cases) | `artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json` (stage_S1) |
| S.8.2 | A-2 S2 γ threshold (3 cells × 12) | same receipt (stage_S2) |
| S.8.3 | A-2 S3 duration/sites (12 cases) | same receipt (stage_S3) |
| S.9 | A-3 HDP boundedness (6 runs, C-HDP-1..8) | `artifacts/protocol_c/p2v_a3_hdp_boundedness/p2v_a3_receipt.json` |
| S.10 | D3 per-arm/seed/interval (36 cells) | `artifacts/protocol_d_biological_rbs/d3_execution_receipt.json`, `d3_interpretation_receipt.json` |
| S.11 | H4 matrix (4 cells, 5 lags) | `artifacts/protocol_h_rbd/h4_matrix/h4_matrix_receipt.json`, `h4_interpretation_receipt.json` |
| S.12 | E5 per-seed contrasts | `artifacts/protocol_e_integration/e5_interpretation_receipt.json` |
| S.13 | W3b domain map (2187 cells) | `artifacts/protocol_w/w3b_parameter_domain/w3b_domain_receipt.json`, `w3b_interpretation_receipt.json` |
| S.14 | Negative/unresolved register | `artifacts/publication/publication_evidence_index.json`; per-protocol receipts |
| S.15 | Reproducibility/provenance | Methods §14; `REVIEW_NAVIGATION.md`; `artifacts/publication/publication_evidence_index.json` (fig SHAs); receipt package heads |
| S.16 | Displaced material register | results_draft.md, methods_draft.md, receipts above |
| S.17 | Claim × evidence-regime classification (21 claims) | `artifacts/publication/publication_claim_ledger.json` (evidence_regime, verbatim); Methods §12 family mapping |
| S.18 | E2 V1/V2 confirmatory negatives (supplement-first) | `artifacts/e2/E2_SYNTHESIS.md:43-61` (V1/V2/Joint boundary sentences), `artifacts/e2/S_PULSE_CHARACTERIZATION.md` (pulse-regime reuse), `artifacts/e2/preregistration/e2_ping_prereg.json` (spec_hash b89a09c...), `artifacts/e2/preregistration/e2_ssa_spec.v6.json` (spec_hash 0df9bfe...), `artifacts/e2/preregistration/E2a_search/e2a_search_receipt.json` (theta* six-way tie), `artifacts/e2/preregistration/E2b_confirmatory/v1_ping_receipt.json` + `v1_rescored_frozen_only.json` + `v1_corrigendum_and_adjudication.json`, `artifacts/e2/preregistration/E2b_confirmatory/v2_ssa_confirmatory_receipt.json` + `v2_rescored_frozen_only.json`, `artifacts/e2/audit/claim_ledger_delta.json` (CL-22..29) |
| S.18-1 | V1/V2 adequacy 5/5 & 20/20 + six-way tie | `e2_ping_prereg.json:12-17` G_adequate_PING, `e2_ssa_spec.v6.json:14-24` SSA.adequacy, `E2a_search/e2a_search_receipt.json:13-22,62-829` |
| S.18-2 | V1 pulse f0 7.22 Hz harmonic comb | `E2b_confirmatory/v1_ping_receipt.json:74-1591`, `v1_corrigendum_and_adjudication.json:1-44`, `S_PULSE_CHARACTERIZATION.md:27-55` |
| S.18-3 | V1 phase outside window | `e2_ping_prereg.json:classified G_phase`, `e2_exec/e2_exec_lib.py:13-31` typed units, receipt phase metrics |
| S.18-4 | V2 SSA pooled SI −0.084 swap 0.426 | `E2b_confirmatory/v2_ssa_confirmatory_receipt.json:1-364`, `v2_rescored_frozen_only.json`, `v2_runs/rep_*.json` |
| S.18-5 | R-A..R-D disclosure + H3 correction | `artifacts/e2/E2_SYNTHESIS.md:34-42`, `v1_corrigendum_and_adjudication.json:18-20`, `e2_ssa_spec_v6_amendment_receipt_CORRECTION.json:4-6` (176000/88 s) |

## Verification commands (authority-level)

```bash
PYTHONPATH=. python3 scripts/audit_supplement_draft.py          # re-derives all table values
PYTHONPATH=. python3 scripts/audit_public_docs_language.py --check
PYTHONPATH=. python3 scripts/check_docs_orphans.py
python3 -m compileall -q scripts/audit_supplement_draft.py
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_agent_context_hygiene.py tests/test_public_docs_hygiene.py -q --tb=short
```