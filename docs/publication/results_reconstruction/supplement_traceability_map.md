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