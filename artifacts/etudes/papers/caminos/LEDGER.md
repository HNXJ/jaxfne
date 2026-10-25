# CAMINOS claim ledger

One row per scored criterion, pass or fail (B2). Source: the CAMINOS reference model, Fig. 1.

| id | claim (paper) | prereg | verdict | value | reason | evidence |
|---|---|---|---|---|---|---|
| CAM-R1 | the network oscillates at about 35 Hz | P3 v6 | reproduced | f_pk 33–34 Hz, mean 33.8, seeds 1–5 | | `scores/score.json` |
| CAM-R2 | SOM fires after PV within each gamma cycle, about 8.5 ms | P3 v6 | finding: needs X | lag 6.9–7.3 ms, mean 7.04 (P3 criterion: positive, mean 6–11 ms, met) | against PLoS CB's 8.5 ms the lag is off (e 0.15–0.26, `CALIBRATION.md` §10, Hamm 2026-10-14); no residual fit | same |
| CAM-R3 | rates order PV > SOM > E, E 1–3 Hz | P3 v6 | reproduced | E mean 1.11 Hz, order holds in every seed | | same |
| CAM-R4 | oscillations appear above a critical p (Fig. 1E) | P3 v6 | not reproduced | all 27 runs oscillate, p = 0.02 included | peak excess is largest at low p (seed means 81–83 at 0.02–0.03, 8 at 0.10); no onset in 0.02–0.10 | same |
| CAM-R4v7 | oscillations appear above a critical p (Fig. 1E), drive p·NE | P3 v7 | reproduced | no seed oscillates at p = 0.02 (E 0.61–0.62 Hz); all seeds oscillate from p = 0.04 | rests on the assumption that the drive scales as p·NE; at p = 0.03, 2 of 3 seeds fail only the 20 Hz band-edge rule | `scores/score_v7.json` |

Overall: mixed. Reproduced under v7: R1, R3, R4v7 (`RESULT_P3.md`). The PV→SOM lag (R2) is a
finding that needs X (`CALIBRATION.md` §10). Under v6 alone R4 was not reproduced. S1 (float64) and S2 (dt 0.05 ms) stay within the seed range.
