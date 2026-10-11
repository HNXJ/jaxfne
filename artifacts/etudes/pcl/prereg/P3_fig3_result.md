# L1-P3 result: PCL Fig. 3 (scored under signed v4)

Verdict: **FAIL.** Fig. 3 tuning is not reproduced on the authors' trained network in the
jaxfne port. T1 and T3 pass; T2 and T4 fail. Nothing was tuned after scoring.

- Prereg: `P3_fig3.md`, signed v4 (Hamm, 2026-10-11) on `pcl-port` cd18a693; scored at f6d9dc38.
- Runs: `F:\warehouse\pcl\runs\fig3_cSTDP3\` (full) and `fig3_cSTDP3_S1\` (S1), port at 5d101eb3.
- Scoring: `port/fig3_analysis.py --signed`, seed 0, 1000 null draws. Outputs copied to
  `scores/fig3_v4_full.json` and `scores/fig3_v4_S1.json`.
- ω = 1.981 Hz in every file; no file outside 1.9–2.1 Hz, no trial mismatch.

## Scored tests (full network)

| Test | Set | Statistic | N-label 99th | N-pref 99th | Criterion | Result |
|---|---|---|---|---|---|---|
| T1 | simple | R_ori 3.57 | 1.21 | 1.37 | ≥ 2, above both | PASS |
| T1 | complex | R_ori 5.30 | 1.27 | 1.45 | ≥ 2, above both | PASS |
| T2 | simple | R_sf 1.45 | 1.16 | 1.07 | ≥ 2, above both | FAIL (under 2) |
| T2 | complex | R_sf 1.46 | 1.19 | 1.10 | ≥ 2, above both | FAIL (under 2) |
| T3 | simple | median X 0.53 | | | ≥ 0.15 | PASS |
| T3 | complex | median X −0.12 | | | ≤ 0.5 × 0.53 | PASS |
| T4 | both | median L simple −1.02, complex −0.19; difference 0.83, null 99th 0.42 | | | L simple < 0 < L complex, difference above null | FAIL (L complex not > 0) |

Exclusions: T1–T2 simple 0/64, complex 10/128; T3 simple 0, complex 25 (19.5 %); T4 simple 0,
complex 18. No set lost more than half, so no test failed on exclusions.

T2 is above both nulls in both sets but under the ratio of 2. T4 orders the sets as
predicted (the difference beats the null), but the complex median is negative: the complex
cells are less phase-locked than the simple cells without frequency doubling.

## Sensitivity (reported, not scored)

S1, lateral and top-down maps zeroed:

| Test | simple | complex |
|---|---|---|
| T1 R_ori | 1.41 FAIL | 1.46 FAIL |
| T2 R_sf | 1.19 FAIL | 1.09 FAIL |
| T3 median X | 0.02 FAIL | −0.15 PASS |
| T4 median L | −1.60 | −0.30 (FAIL; difference 1.30, null 0.66) |

S1 does not agree with the full run: orientation tuning (T1) and simple-cell phase
modulation (T3) need the lateral and top-down maps. So the map-order assumption can matter
for Fig. 3, and it is not excluded by this test.

S2 (preferences fitted on all phases) and S3 (simple columns (3,3), (3,4), (4,3)) are not
run: `fig3_analysis.py` has no option for either yet.

## Ledger

Rows are in `../LEDGER.md` (B2: every scored criterion has a row, pass or fail).
