# PCL R5-exact result under signed R4b

Protocol: `P3_fig3_r4b_signed.md` (Hamm 2026-11-03), R5-exact. Runs 2026-11-03 from the frozen clone at dev 07a90e70
(`manifest.json` `baseline_sha` in each run dir), `run_tuning.py --subset pilot`, Windows CPU. Receipts in
`F:/warehouse/pcl/r5/`: `exact_numba/`, `exact_core/`, `negctl_core_full/`, driver `run_r5.sh` 116d99cf…,
scorer `score_exact.py` d7016444… with log `score_exact.log` 2791751f… and score `fig3_r5_exact.json` 6bff3936…
(copied to `scores/fig3_r5_exact.json`, read by `supplement.py write_s13`).

## Verdict: FAIL

- Pass criterion: on all 16 files, every cell's spike count and spike times equal in the core and numba arms.
- Observed: 4 of 16 files equal (0/0, 113/9, 68/9, 90/9 as ori/sf).
- Negative control (core with `--inhibition full` against numba under `no-local`): differs on 16 of 16 files, so
  the test could fail and is not void.

Outcome per R4b: no event-exact claim for the core on `test7`. R5-approximate runs as declared (the core outputs
are scored; the numba rerun is the check).

## Size of the differences (post hoc, descriptive)

`diff_exact.py` d4d6b2a9…, log `diff_exact.log` b2eb62ee…. Per file, cells whose spike count differs:

| Class | Files (ori/sf) | Simple cells differing | Complex cells differing |
|---|---|---|---|
| Equal | 0/0, 113/9, 68/9, 90/9 | 0 | 0 |
| Same counts, spike times or order differ (complex) | 113/0, 135/0, 45/0, 45/9, 68/0, 90/0 | 0 | 0 |
| Counts differ by 1-2 spikes (complex) | 158/0, 158/9, 23/9 | 0 | 1 |
| Large divergence | 0/9, 135/9, 23/0 | 317-352 | 586-653 |

The simple layer is exact on 13 of 16 files; every first divergence that is not in a large-divergence file is in
the complex layer. The non-scored pilot file (ori 0, sf 4, phase 1, trial 1) was equal (`CALIBRATION.md`).
Attribution is open (inferred candidate: floating-point order at near-threshold complex-cell updates; P-053
reports a related order effect in STDP updates).
