# L1-P3 result v5: PCL Fig. 3 (scored under signed v5)

Verdict: **FAIL.** Fig. 3 tuning is not reproduced on the authors' trained network in the
jaxfne port. T2 fails for both sets; T1, T3 and T4 pass. Nothing was tuned after scoring.

- Prereg: `P3_fig3_v5.md`, signed v5 (Hamm, 2026-10-13) on `pcl-port` 89e45191.
- Runs: the v4 runs, `F:\warehouse\pcl\runs\fig3_cSTDP3\` (port 5d101eb3).
- T2 scorer: `port/fig3_v5_t2.py --signed` (sha256 ea09b425…), reference
  `reference/fig3_freq_digitized.json` (sha256 fe0973e5…). Outputs:
  `scores/fig3_v5_t2_even.json` (scored), `_all.json` and `_0-4.json` (S4, S5).
- T1, T3: v4 definitions and numbers (`P3_fig3_result.md`). T4: v4 numbers under the v5
  criterion. All three were known before v5 was declared.

## Scored tests

| Test | Set | Statistic | Criterion | Result |
|---|---|---|---|---|
| T1 | simple / complex | R_ori 3.57 / 5.30 | as v4 | PASS / PASS |
| T2 | simple (64 cells, 0 skipped) | 10/16 points in tolerance (0.625); SSE 0.349 = 1.50 × flat | ≥ 0.8 and ≤ 0.5 × flat | FAIL (both parts) |
| T2 | complex (128 cells, 2 skipped) | 12/15 points (0.800); SSE 0.204 = 0.52 × flat | ≥ 0.8 and ≤ 0.5 × flat | FAIL (guard) |
| T3 | simple / complex | median X 0.53 / −0.12 | as v4 | PASS / PASS |
| T4 | both | median L complex −0.19 > simple −1.02; difference 0.83, null 99th 0.42 | complex above simple, above null | PASS |

T2 simple, by point (ours vs figure): the port's simple cells respond more off the preferred
frequency than the figure shows, most at K/k 0.6–0.8 (0.52, 0.45, 0.58 vs 0.36, 0.24, 0.37)
and 1.3 (0.58 vs 0.24). Their curve is worse than a flat line at the figure's mean.

T2 complex misses at K/k 0.6, 0.8 (ours 0.76, 0.82 vs 0.84, 0.92) and 1.9 (0.34 vs 0.73,
n = 11). The shape guard misses by 0.02.

## Sensitivity (reported, not scored)

| Phase set | simple: frac, SSE/flat | complex: frac, SSE/flat |
|---|---|---|
| even 0,2,4,6,8 (scored, A1) | 0.625, 1.50 FAIL | 0.800, 0.52 FAIL |
| all 0–9 (S4) | 0.688, 1.04 FAIL | 0.800, 0.49 PASS |
| 0–4 (S5) | 0.625, 1.61 FAIL | 0.733, 0.89 FAIL |

Each complex column alone (even phases, not scored): (2,2) 0.667, 1.14; (2,3) 0.867, 0.73;
(3,2) 0.933, 0.52; (3,3) 0.933, 1.48. None passes both parts.

The simple-cell failure holds under every phase mapping. The complex result depends on it.
S6 (K/k from measured periods) and S7 (window from 62.5 ms) are not implemented.
