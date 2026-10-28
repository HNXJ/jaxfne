# Claim alignment audit (C1), 2026-10-27

Manuscript `artifacts/programme/paper/publication/manuscript/` against `artifacts/publication/publication_claim_ledger.json`.
results.md carries its CL ids in `artifacts/programme/paper/publication/results_reconstruction/traceability/traceability_map.md`
(the 4R revision moved them out of the prose; sections P1-P20 of results.md, the A-1a to A-3 numbers as
Q markers checked by `scripts/audit_results_draft.py`, 70 of 70 checks pass; results.md is the draft without its
Q markers). `python scripts/check_manuscript_claims.py` reads the
map: 0 unknown ids, 0 statuses outside the tier set, 13 uncited ledger ids, 2 orphan paragraphs.

## Left for 0 orphans

| Paragraph | Gap |
|---|---|
| abstract.md:3 | summarizes CL-01 to CL-21 and cites none; its first sentence (per-tool conventions) has no row |
| results.md:229 | the appendix index of the A-1a, A-1b, A-2, A-3 receipts sits outside P1-P20 |

Uncited rows CL-22 to CL-34 are E2 and AT-10 claims the 0.5.5 manuscript does not state; a read-only sweep
places CL-27 and CL-29 in discussion.md:39-47, CL-28 in methods.md:141, CL-30 in supplement.md:767, CL-32 in
discussion.md:21 and CL-33 in methods.md:238, uncited. CL-34 cites its two render receipts (UNRESOLVED: one
render, no reproduction run).

## Conflicts that need a ruling (checked against both files)

1. results.md:134 runs H4 "on the 24-neuron ring"; methods.md:202 runs it on rings of 3 and 12 neurons.
2. CL-29 `claim` calls 0.426 the pooled swap asymmetry; its `forbidden_overclaim` forbids substituting
   max_over_reps 0.426 for the pooled mean, and `e2_fig09_ssa_receipt.json` records 0.426 as
   `swap_max_observed`. discussion.md:45 writes swap_max 0.426.
3. results.md:204 "classifies every seed ... (3/3)"; CL-18 allows "reproduced identically in all three
   protocol runs" and calls the seeds nominal run metadata.
4. CL-05 names a frozen estimator; results.md:35 says the module was revised once after the primary analyses
   were registered.
5. CL-06 reads the 4 structured-but-fails cells as a minority outside NO_WAVE; results.md:59 counts all 60
   as NO_WAVE (52 + 4 + 4). The counts agree; the class name differs.
6. discussion.md:45: "E-leads-I +6.8 ms" against `dt_lag_ms_median` -6.8 (sign convention unstated), and
   "-85 to -91 deg" against the ledger's -85.27 to -90.48.
7. results.md:215 claims byte equality for 7/7 figures; CL-21 `null_or_control` says decoded-pixel identical.
   The Linux publication job (Release CI 40346237854) matched RGBA for 4 of 7 and bytes for none.
