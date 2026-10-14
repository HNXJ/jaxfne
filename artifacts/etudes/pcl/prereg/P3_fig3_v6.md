# L1-P3 preregistration v6: PCL Fig. 3 T2 amendment (post hoc)

Status: **withdrawn draft** (2026-10-13), never signed. Review (Muse, REVISE) found the column
term mis-scaled: one 32-cell column against the 4-column pool has expected squared difference
1.25 σ²_col. Recomputed with that term on the v4 runs (even phases):
- Complex: the tolerances become 0.14–0.49 and the guard limit 0.528. A flat line at the figure's
  mean (SSE 0.393) would pass, so the criterion no longer discriminates. With the port's own column
  spread, one figure column cannot test the complex sf shape.
- Simple: v6 still fails, on fraction (0.625) and on level (ours 0.095 above the figure,
  against ε = 0.04).

Read: the complex failure is a resolution limit, and the simple failure is a real offset. The
port's simple cells respond about 0.1 more at non-preferred sf, which points at the model and its
training. The text below is the withdrawn draft, kept as the record.

Original status: draft v6, unsigned. Hamm (2026-10-13) asked for this draft after the v5 FAIL on T2.
**Post hoc:** declared after `P3_fig3_diagnostic.md` Q7, which scored T2 under many analysis
choices on the same runs. The v4 and v5 verdicts stand.

Everything not named here is `P3_fig3_v5.md` signed v5 verbatim:
- runs (`F:\warehouse\pcl\runs\fig3_cSTDP3\`, port 5d101eb3);
- cell sets: simple column (4,4), complex columns (2,2), (2,3), (3,2), (3,3);
- A1 = even phases 0, 2, 4, 6, 8;
- the `plot_fqcy` procedure, the reference curves, ε = 0.04, z = 2;
- T1, T3 and T4.

## Prior exposure

Under v5 on these runs:

| Set | Points within tolerance | SSE | Guard limit |
|---|---|---|---|
| Simple | 10/16 | 0.349 | 0.117 |
| Complex | 12/15 | 0.204 | 0.197 |

Q7 shows:
- the simple guard fails in every analysis choice;
- the complex pool passes with all phases;
- single complex columns range 0.14–1.09 in SSE.

Under the rules below, the simple fraction (10/16 = 0.625) is already known to be below 0.8, so
simple T2 is known to fail. Only complex T2 is computed for the first time under v6.

## Why the guard changes

- **Fig. 3c (simple).** A flat line at the figure's mean meets 15 of 16 points within tolerance.
  The panel carries no shape beyond flat that the digitization resolves, so a shape guard against
  flat cannot be met by any smooth curve.
- **Fig. 3d (complex).** The figure is one 32-cell column. A port that matched the population curve
  exactly would still differ from any single column by the column-to-column spread, which the v5
  guard ignored.

## T2 v6 criterion

**Simple (panel c), 16 scored points.**
- Points: v5's point test, with at least 80 % of points passing.
- Level: |mean_j ours_j − mean_j fig_j| ≤ ε.
- No shape guard.

**Complex (panel d), 15 scored points.**
- sd_col_j is the standard deviation (ddof 1) over the four port columns of each column's own
  `plot_fqcy` curve at point j. A bin missing from a column is left out of that point's SD. A point
  with fewer than 2 columns has sd_col_j = 0.
- Point j passes when |ours_j − fig_j| ≤ ε + z·√(se_fig_j² + se_ours_j² + sd_col_j²). At least
  80 % of points must pass.
- Shape guard: Σ(ours_j − fig_j)² − Σ sd_col_j² ≤ 0.5 · Σ(m − fig_j)². The subtracted term is the
  SSE a correct population curve is expected to show against one 32-cell column.

T2 passes for a set when all its parts hold. Scorer: `port/fig3_v5_t2.py --prereg v6`. The v5 path
must stay byte-identical, with a test on the v5 receipts.

## Outcome language

- All of T1–T4 pass for both sets: "Fig. 3 tuning reproduced qualitatively on the authors'
  trained network in the jaxfne port under amendment v6 (post hoc T2 criterion)".
- Otherwise: per set, per criterion, with numbers, stating that v6 was declared after Q7.
