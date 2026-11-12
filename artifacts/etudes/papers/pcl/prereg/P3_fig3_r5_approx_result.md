# PCL R5-approximate result under signed R4b

Protocol: `P3_fig3_r4b_signed.md` (Hamm 2026-11-03), R5-approximate. Scored 2026-11-12 from the frozen clone at
dev 07a90e70 (`manifest.json` `baseline_sha` in each run dir), Windows CPU, `--inhibition no-local`, learning off.
Receipts in `F:/warehouse/pcl/r5/`: core arm `approx_core/` (3168 outputs, driver `run_r5.sh`; final resume log
`approx_core_resume2.log` ends `summary done=1085 failed=0 pending_at_start=1085` at 2026-11-08T10:18:04Z), oracle
check arm `approx_numba/` (3168 outputs, `pending 0`). Scorer
`artifacts/etudes/papers/pcl/port/fig3_analysis.py --signed`, seed 0, 1000 null draws, executed at dev 342279db.

## Verdict: FAIL

Fig. 3 counts as reproduced only if T1-T4 pass for both sets (R4b, Outcome language). They do not.

| Test | Set | Statistic | Null 99th (n-label / n-pref) | Criterion | Result |
|---|---|---|---|---|---|
| T1 | simple | R_ori 38.11 | 1.74 / 2.07 | ≥ 2, above both | PASS |
| T1 | complex | R_ori 1.19 | 1.03 / 1.04 | ≥ 2, above both | FAIL |
| T2 | simple | R_sf 1.65 | 1.41 / 1.15 | as v5 | FAIL |
| T2 | complex | R_sf 1.13 | 1.03 / 1.15 | as v5 | FAIL |
| T3 | simple | median X 0.473 | | ≥ 0.15 | PASS |
| T3 | complex | median X −0.172 | | ≤ 0.5 × simple | PASS |
| T4 | both | median L simple −0.816, complex −0.981; difference −0.165 | 0.112 | complex above simple, above null | FAIL |

T2 fails as known before scoring (R4b, "What is known before scoring": the oracle under the C++ setting fails T2 for
both sets). T1 complex and T4 are the prospective failures. Exclusions: T1-T2 simple 0, complex 0.

## Arm check (R5-approximate, the oracle rerun)

| Quantity | Core | Numba oracle |
|---|---|---|
| Files | 3168 | 3168 (same set) |
| Event-exact files | 1246 | |
| Files differing | 1922 | |
| Simple spikes, total | 5 867 293 | 5 867 342 |
| Complex spikes, total | 109 436 568 | 109 439 469 |

Per R4b the differing files are listed and do not change the score; the largest differences reach 461 simple and
751 complex cells per file. The two arms agree to 49 of 5.87 M simple and 2901 of 109.4 M complex spikes.

## Both arms score the same

Scoring the oracle arm with the same scorer gives T1 simple 37.10, T1 complex 1.19, T2 simple 1.63, T2 complex 1.12,
T3 median X 0.47 / −0.17, T4 median L −0.82 / −0.95, T4 FAIL. Every pass/fail flag matches the core arm, so the
verdict does not rest on the core-vs-oracle divergence (P-055).

## The failure is the inhibition setting (inferred, not isolated)

Re-scoring the v4 run directory (`F:/warehouse/pcl/runs/fig3_cSTDP3`) with the same scorer and tree reproduces the
committed v4 score to three decimals on every statistic (T1 complex 5.297, T2 simple 1.450, T2 complex 1.461, T3
0.530 / −0.122, T4 −1.017 / −0.189), so scorer and tree drift do not explain the difference.

v4 ran the port's alias for `static` inhibition (local on); R5 runs the C++ setting (`no-local`). T1, T3 and T4
under `no-local` had never been computed (R4b). Observed: simple-cell orientation tuning is largely unaffected
(R_ori 38.1), and complex-cell orientation tuning collapses from 5.30 to 1.19, below the ratio criterion of 2.
Inferred cause: complex cells in this network depend on the local-inhibition pathway for their orientation tuning;
the alternative confound is the tree drift between port 5d101eb3 (v4) and dev 07a90e70 (R5), which is not isolated
by this run. Separating the two needs one v4-tree run under `no-local`.

## Ledger

Rows are in `../LEDGER.md` (B2). Scores: `scores/fig3_r5_approx.json` (core arm, scored) and
`scores/fig3_r5_approx_oracle.json` (oracle arm, control). Receipts for the arm comparison:
`artifacts/etudes/papers/pcl/prereg/scores/fig3_r5_approx_armcheck.json`.
