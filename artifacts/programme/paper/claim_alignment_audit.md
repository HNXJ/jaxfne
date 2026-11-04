# Claim alignment audit (C1), 2026-10-28

Manuscript `artifacts/programme/paper/publication/manuscript/` against `artifacts/publication/publication_claim_ledger.json`.
results.md carries its CL ids in `artifacts/programme/paper/publication/results_reconstruction/traceability/traceability_map.md`
(sections P1-P21 of results.md; the A-1a to A-3 numbers as Q markers checked by
`scripts/audit_results_draft.py`, 70 of 70 checks pass). `python scripts/check_manuscript_claims.py` reads
the map: 0 unknown ids, 0 statuses outside the tier set, 13 uncited ledger ids, 0 orphan paragraphs.

## Resolved 2026-10-28 (Hamm's rulings; the 2026-10-27 list is in git history)

| Item | Fix |
|---|---|
| H4 ring size | results.md: rings of 3 and 12 neurons, as methods.md and the H4 matrix receipt |
| CL-29 0.426 | ledger claim: maximum swap asymmetry over repetitions (`swap_max_observed`) |
| E5 seeds | results.md: reproduced identically in all three protocol runs, seeds as nominal metadata (CL-18) |
| CL-05 revision | ledger claim names the one post-registration revision, 684859a98da51de79887ec26ba8d7134e2fa0e97 |
| CL-06 class | ledger: uniformly NO_WAVE (60/60 cells), the 52, 4, 4 split as kinds of NO_WAVE cell (C3 receipt) |
| S18 lag and phase | discussion.md: E leads I by 6.8 ms (dt_lag_ms -6.8), delta_phi -85.27 to -90.48 deg |
| Fig. equivalence | results.md: decoded-pixel identity 7/7, byte-equal on the reference environment (CL-21) |
| abstract.md:3 | cites CL-01 to CL-21 |
| results appendix | traceability section P21 maps it to CL-05 and CL-06 |

Uncited rows CL-22 to CL-34 are E2 and AT-10 claims the 0.5.5 manuscript does not state; a read-only sweep
places CL-27 and CL-29 in discussion.md:39-47, CL-28 in methods.md:141, CL-30 in supplement.md:767, CL-32 in
discussion.md:21 and CL-33 in methods.md:238, uncited. CL-34 cites its two render receipts (UNRESOLVED: one
render, no reproduction run).
