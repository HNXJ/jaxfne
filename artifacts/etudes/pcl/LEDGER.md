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

Fig. 3 overall: not reproduced (`prereg/P3_fig3_result.md`). S1 disagrees with the full run,
so the map-order assumption is not excluded.
