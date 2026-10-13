# L1-P3 preregistration v5: PCL Fig. 3 (simple and complex cell tuning)

Status: **DRAFT v5, unsigned.** Declared after the v4 FAIL (`P3_fig3_result.md`), which
stands. v5 is a new protocol on Hamm's calls of 2026-10-13: T2 scores the normalized
spatial-frequency curve against the published Fig. 3, and T4 tests the paper's claim C3 with a
criterion that drops the sign requirement. No v5 scoring runs before the signature.

## What is known before scoring

v5 reruns the scorer on the v4 runs (`F:\warehouse\pcl\runs\fig3_cSTDP3\`, port 5d101eb3):

- T1 and T3 keep the v4 definitions, so their v4 outcomes (both PASS) are known.
- T4 under the v5 criterion is decided by numbers already in the v4 result: median L simple
  −1.02, complex −0.19, difference 0.83 against a null 99th percentile of 0.42. T4 is therefore
  a re-declaration, not a prospective test.
- T2 under v5 has not been computed. It is the only prospective test in v5.

## Source claims

As v4 (C1–C3; the paper states them qualitatively). Fig. 3c,d frequency panels: mean over
64 simple and 32 complex cells, SE across cells, x axis K/k from 0.4 to 2.0.

## Reference curves

- `reference/fig3_freq_digitized.json`, written by `reference/digitize_fig3.py` from the
  publisher's full-size PNG. The image is CC BY-NC-ND 4.0 and is kept outside the repo
  (`F:\warehouse\pcl\paper\`); its SHA-256 is checked on every run.
- 17 points per panel, K/k = 0.4 … 2.0 in steps of 0.1. Panel d at K/k = 0.9 is covered by the
  example-cell lines and is null.
- Calibration checks: the 0.5 tick reads 0.499 in both panels; the example-cell minima in c
  read 0.006 (true 0); the mean at K/k = 1 reads 1.018 (c) and 0.968 (d, partly covered).
- Digitization error ε = 0.04, the largest calibration residual rounded up.
- se_fig per point is the band half-width minus half the stroke width; the stroke width is the
  panel's narrowest band away from K/k = 1.

## T2 procedure: the authors' `plot_fqcy` on the port's responses

From `PCL_Python_Analysis/tuningCurves.ipynb` (cells 5, 13, 14, 16), read as data:

1. Response r(ori, sf, phase) = spike count over the stimulus window, mean of the 2 trials.
2. Preference (`responses`): average r over phases; for each sf, the maximum over ori; sf* is
   the argmax of that maximum over sf, ori* the orientation that achieved it.
3. Simple cells: phase* = argmax over phases of r(ori*, sf*, phase); the curve uses phase*
   only. Complex cells: the curve averages all phases.
4. Per cell, g(i) = r(ori*, i) / r(ori*, sf*) for sf index i = 0 … 17; a cell with
   r(ori*, sf*) = 0 is skipped and counted.
5. K/k = (20 − sf*) / (20 − i): the notebook's `real_fqcyval` is 2π/(20 − i).
6. Pool all (K/k, g) pairs over the cells of a set; bin by round(K/k, 1); bin value is the
   mean and SE is std/√n (ddof 0), as in the notebook. Bins 0.4 … 2.0 are kept.

Declared assumptions (each with a sensitivity, reported and not scored):

| Id | Assumption | Sensitivity |
|---|---|---|
| A1 | The notebook's 5 phases are test7's even phases 0, 2, 4, 6, 8 (test7 has 10 distinct phases per period) | S4: all 10 phases; S5: phases 0–4 |
| A2 | Period ∝ 20 − i, as `real_fqcyval` states. The measured periods (9.67 px at index 0, 5.8 px at 9) agree within the 29 px profile's resolution; index 17 is aliased | S6: K/k from the measured periods |
| A3 | Response window is v4's (first to last event of the stimulus). The notebook starts at 62.5 ms | S7: window from 62.5 ms |

Cell sets as v4: simple column (4,4), 64 cells; complex columns (2,2), (2,3), (3,2), (3,3),
128 cells (the paper's 32 is one column; the four tie). Each complex column is reported alone,
not scored.

## T2 criterion (per set)

Scored points: K/k ≠ 1 (fixed at 1 by step 4) with a non-null figure value: 16 in c, 15 in d.

- Point j passes when |ours_j − fig_j| ≤ ε + z·√(se_fig_j² + se_ours_j²), with ε = 0.04 and
  z = 2. A bin missing from ours fails.
- **Fraction:** at least 80 % of scored points pass.
- **Shape guard:** Σ(ours_j − fig_j)² ≤ 0.5 · Σ(m − fig_j)², where m is the figure's mean
  over the scored points. The guard exists because the tolerance alone does not separate a flat
  curve: a flat line at m passes 10/16 (c) and 9/15 (d) points at se_ours = 0, and 15/16 (c) at
  se_ours = 0.06 (computed on the figure before scoring).

T2 passes for a set when both hold.

## T1, T3, T4

- T1 and T3: as v4, unchanged.
- T4: estimand as v4, L = log(A(2ω)/A(ω)). Pass when median L complex > median L simple and the
  difference exceeds the 99th percentile of the 1000-draw cell-permutation null. The sign
  requirement of v4 (L simple < 0 < L complex) is dropped (Hamm 2026-10-13: C3 is stated
  qualitatively, and the authors' notebook has no measure of it).

## Outcome language

- All of T1–T4 pass for both sets: "Fig. 3 tuning reproduced qualitatively on the authors'
  trained network in the jaxfne port (v5; T2 against the digitized published curves)." The
  report states that T1, T3 and T4 were known before v5 was declared.
- Any failure is reported with its numbers. v4's FAIL stays in the ledger beside v5.
