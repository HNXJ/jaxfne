# Claim alignment audit (C1), 2026-10-27

Manuscript `docs/theory/publication/manuscript/` against `artifacts/publication/publication_claim_ledger.json`
at 7cf4e2b4. `python scripts/check_manuscript_claims.py`: 0 unknown ids, 0 statuses outside the tier set,
13 uncited ledger ids, 17 orphan paragraphs. Mapping from a read-only sweep; rows marked checked were
re-read against both files, the rest are proposals.

## Orphan paragraphs and the rows that state them

| Paragraph | Rows | Gap |
|---|---|---|
| abstract.md:3 | summary of CL-01 to CL-21 | first sentence (per-tool conventions) has no row |
| results.md:5 | CL-01 | architecture sentences are method text |
| results.md:17 | CL-02 | |
| results.md:24 | CL-03, CL-04 | |
| results.md:32 | CL-05 | conflict 4 |
| results.md:44 | CL-05 (controls) | A-1a numbers (48 cases, 0.00831, 0.00824) have no row |
| results.md:57 | CL-06 (checked) | conflict 5 |
| results.md:68 | CL-06 | A-1b numbers have no row |
| results.md:83 | none | A-2 sensitivity floor has no row |
| results.md:94 | CL-06 | "insufficient to generate" reads causal beyond the ring scope |
| results.md:120 | CL-07 | |
| results.md:134 | CL-10 (checked) | conflict 1 |
| results.md:151 | CL-08 | per-arm ADAPTATION counts have no row |
| results.md:165 | CL-11, CL-19 | |
| results.md:185 | CL-13, CL-14, CL-15 | |
| results.md:200 | CL-17, CL-18 (checked) | conflict 3 |
| results.md:229 | none | index of the A-1a, A-1b, A-2, A-3 receipts |

The checker misses results.md:171-181 (A-3, numbers, no row) and results.md:213-219 (conflict 7).

Uncited rows (sweep proposals): CL-22 to CL-26, CL-31 and CL-34 appear nowhere in the manuscript; CL-27 and CL-29 are stated in
discussion.md:39-47, CL-28 in methods.md:141, CL-30 in supplement.md:767, CL-32 in discussion.md:21, CL-33 in
methods.md:238. CL-34 now cites the two render receipts (UNRESOLVED: one render, no reproduction run).

## Conflicts that need a ruling (checked)

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

## To reach 0 orphans

Rows for the A-1a, A-1b, A-2 and A-3 receipts; a ruling on conflicts 1-7; then a CL id on each paragraph
above. The id format in prose is open (discussion.md cites ids inline, supplement.md in a table).
