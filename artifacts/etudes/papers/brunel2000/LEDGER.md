# brunel2000 claim ledger

One row per scored criterion, pass or fail (B2). Source: Brunel 2000, Fig. 8 points A–D.

| id | claim (paper) | prereg | verdict | value | reason | evidence |
|---|---|---|---|---|---|---|
| BRU-E1 | B (SI fast) cells fire at 60.7 Hz | P3 v4 | reproduced | 59.33 ± 0.22 Hz, e 0.023 (band 0.08) | | `score_p3.json` |
| BRU-E2 | B global oscillation near 180 Hz | P3 v4 | reproduced | 182.4 ± 4.4 Hz, e 0.013 (band 0.06) | | same |
| BRU-E3 | C (AI) cells fire at 37.7 Hz | P3 v4 | reproduced | 37.87 ± 0.07 Hz, e 0.0044 (band 0.01) | | same |
| BRU-E4 | D (SI slow) cells fire at 5.5 Hz | P3 v4 | reproduced | 5.63 ± 0.06 Hz, e 0.024 (band 0.18) | | same |
| BRU-E5 | D global oscillation near 22 Hz | P3 v4 | reproduced | 20.6 ± 2.9 Hz, e 0.064 (band 0.32) | | same |
| BRU-Q1 | A is synchronous regular | P3 v4 | reproduced | `saturated`, κ 1.00, CV ≤ 0.0006 | | same |
| BRU-Q2B | B is synchronous irregular | P3 v4 | reproduced | `oscillatory`, CV 0.78–0.83 | | same |
| BRU-Q2D | D is synchronous irregular | P3 v4 | reproduced | `oscillatory`, CV 0.63–0.64 | | same |
| BRU-Q3 | C is asynchronous irregular | P3 v4 | reproduced | spectral excess 2.95–3.27, CV 0.40–0.41 | reading (ii), the paper's wording; the classifier label `asynchronous` (declined reading (i)) fails at C | same |

Overall: close under signed v4 (`RESULT_P3.md`), seeds 4–8, float32; float64 at C agrees to
0.012% in rate. Sensitivity S1–S3 ran (`RESULT_P3.md`, unscored): S2 holds every label, S3 moves C
seed 4 to oscillatory, and S1 (N=6250, in-degrees held) moves D to irregular on all five seeds, so
BRU-Q2D holds at N=12500 only; no verdict changes.
