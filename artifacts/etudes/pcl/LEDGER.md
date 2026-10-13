# PCL claim ledger

One row per scored criterion, pass or fail (B2). Source: N'dri et al. 2025.

| id | claim (paper) | prereg | set | verdict | value | reason | evidence |
|---|---|---|---|---|---|---|---|
| PCL-F3-T1s | simple cells are orientation tuned | P3 v4 | simple | reproduced | R_ori 3.57 | | `prereg/scores/fig3_v4_full.json` |
| PCL-F3-T1c | complex cells are orientation tuned | P3 v4 | complex | reproduced | R_ori 5.30 | | same |
| PCL-F3-T2s | simple cells are spatial-frequency tuned | P3 v4 | simple | not reproduced | R_sf 1.45 | above both nulls, under the ratio of 2 | same |
| PCL-F3-T2c | complex cells are spatial-frequency tuned | P3 v4 | complex | not reproduced | R_sf 1.46 | above both nulls, under the ratio of 2 | same |
| PCL-F3-T3s | simple cells are phase modulated | P3 v4 | simple | reproduced | median X 0.53 | | same |
| PCL-F3-T3c | complex cells are phase invariant | P3 v4 | complex | reproduced | median X −0.12 | | same |
| PCL-F3-T4 | complex cells frequency-double, simple cells do not | P3 v4 | both | not reproduced | median L simple −1.02, complex −0.19 | complex median not above 0; the set difference does beat its null (0.83 vs 0.42) | same |
| PCL-F3-T2s-v5 | simple sf curve matches Fig. 3c | P3 v5 | simple | not reproduced | 10/16 points; SSE 1.50 × flat | worse than a flat line; higher off-peak response at K/k 0.6–0.8, 1.3 | `prereg/scores/fig3_v5_t2_even.json` |
| PCL-F3-T2c-v5 | complex sf curve matches Fig. 3d | P3 v5 | complex | not reproduced | 12/15 points; SSE 0.52 × flat | shape guard missed by 0.02; passes with all 10 phases (S4, not scored) | same |
| PCL-F3-T4-v5 | complex cells frequency-double relative to simple (C3) | P3 v5 | both | reproduced | difference 0.83, null 99th 0.42 | v4 numbers, known before v5 | `prereg/scores/fig3_v4_full.json` |

Fig. 3 overall under v5: not reproduced (`prereg/P3_fig3_v5_result.md`), T2 fails both sets.
Fig. 3 overall under v4: not reproduced (`prereg/P3_fig3_result.md`). S1 disagrees with the full run,
so the map-order assumption is not excluded.
