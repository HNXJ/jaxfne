# CAMINOS claim ledger

One row per scored criterion, pass or fail (B2). Source: the CAMINOS reference model, Fig. 1.

| id | claim (paper) | prereg | verdict | value | reason | evidence |
|---|---|---|---|---|---|---|
| CAM-R1 | the network oscillates at about 35 Hz | P3 v6 | reproduced | f_pk 33–34 Hz, mean 33.8, seeds 1–5 | | `scores/score.json` |
| CAM-R2 | SOM fires after PV within each gamma cycle, about 8.5 ms | P3 v6 | reproduced | lag 6.9–7.3 ms, mean 7.04 | | same |
| CAM-R3 | rates order PV > SOM > E, E 1–3 Hz | P3 v6 | reproduced | E mean 1.11 Hz, order holds in every seed | | same |
| CAM-R4 | oscillations appear above a critical p (Fig. 1E) | P3 v6 | not reproduced | all 27 runs oscillate, p = 0.02 included | peak excess is largest at low p (seed means 81–83 at 0.02–0.03, 8 at 0.10); no onset in 0.02–0.10 | same |

Overall: not reproduced (`RESULT_P3.md`). S1 (float64) and S2 (dt 0.05 ms) stay within the
seed range.
