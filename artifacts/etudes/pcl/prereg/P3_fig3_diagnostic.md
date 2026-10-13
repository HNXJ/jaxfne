# PCL Fig. 3 FAIL diagnostic (unscored)

Approved by Hamm 2026-10-12, after the v4 verdict. Nothing here changes the verdict; any fix
it suggests is a new preregistration. Scripts: `../diag/` (read-only on the scored run;
numbers recomputed with the scorer's own loaders, R_sf matches the scored 1.450 and 1.461).

## What the authors measure (`tuningCurves.ipynb`, read as text)

- sf tuning (`plot_fqcy`, ipynb:709–836): response at each sf divided by the response at
  the preferred sf, plotted against the sf difference. No selectivity index:
  `tuningSelectivities` is imported (ipynb:34) and never called.
- No F1/F0 or frequency-doubling measure anywhere in the notebook or its modules.
- The sf axis is 2π/λ for λ = 20…3 px (ipynb:711), so index 0 is the lowest sf and the
  index steps are uneven in frequency.

So T2's ratio of 2 and T4's log ratio are our estimands, not the authors'. Fig. 3 shows
curves, and the prereg turned them into thresholds.

## T2 (R_sf about 1.45, criterion 2)

- The population curve is broad, not a narrow peak on a floor. Simple cells fall from
  123 at Δsf = 0 to 0.80, 0.74 and 0.62 of the peak at |Δsf| = 3, 6 and 12. Complex cells
  stay at or above the peak on the low side and fall steeply on the high side (0.48 at +6,
  0.25 at +9).
- Per-cell R_sf: simple median 1.42 (IQR 1.24–1.69), 8 of 64 cells ≥ 2; complex median
  1.51 (1.28–2.42), 39 of 118 ≥ 2.
- Complex preferences are bimodal: 25 of 118 cells (21 %) prefer index 0, the lowest
  tested sf, and most others prefer 11–14. Simple cells: 1 of 64 at an edge.
- Each Δsf point averages a different subset of cells (64 at Δsf = 0, 1 at −17), so the
  flank mixes cells with different coverage.

## T4 (complex median L −0.19, criterion > 0)

| Set | median A(ω) | median A(2ω) | median L | L, count above / below median |
|---|---|---|---|---|
| simple | 33.0 | 11.7 | −1.02 | −1.04 / −1.00 |
| complex | 9.33 | 8.11 | −0.19 | −0.15 / −0.34 |

Complex cells lose most of the F1 that simple cells have (A(ω) 9.3 against 33) but do not
gain a 2ω component above it. Higher-count cells are closer to 0 but still negative.

## Open

- Whether the notebook's cells z = 0..63 are column (4,4) is unconfirmed.
- The authors' own test data (`PCL_Nat_testing_counterphase_grating_tuningcurve.zip`) was
  not opened, so their curves were not recomputed here.

## v5 T2 sensitivities S6 and S7 (unscored, 2026-10-13)

Script: `artifacts/etudes/pcl/diag/q6_v5_s6_s7.py`. Unscored; the verdict, the scorer and `scores/` are unchanged. Even phases 0, 2, 4, 6, 8; reference and criterion from `port/fig3_v5_t2.py`. Steps 1 to 4 and 6 are the prereg's; only the K/k map differs.

Check: the default map reproduces `prereg/scores/fig3_v5_t2_even.json` (max curve gap simple 0.0e+00, complex 0.0e+00; frac, SSE and flat match).

### S6 periods (px per cycle; DFT over 2 to 20 px, step 1e-4 cycles/px)

| sf index | 20 - i | ori 0: period | width | edge | ori 90: period | width | edge |
|---|---|---|---|---|---|---|---|
| 0 | 20 | 10.21 | 29 | - | 20.00 | 29 | 20 px edge |
| 1 | 19 | 9.62 | 29 | - | 20.00 | 29 | 20 px edge |
| 2 | 18 | 9.01 | 29 | - | 20.00 | 29 | 20 px edge |
| 3 | 17 | 8.60 | 29 | - | 20.00 | 29 | 20 px edge |
| 4 | 16 | 7.94 | 28 | - | 20.00 | 29 | 20 px edge |
| 5 | 15 | 7.53 | 29 | - | 20.00 | 29 | 20 px edge |
| 6 | 14 | 7.04 | 29 | - | 20.00 | 29 | 20 px edge |
| 7 | 13 | 6.51 | 29 | - | 20.00 | 29 | 20 px edge |
| 8 | 12 | 5.98 | 29 | - | 20.00 | 29 | 20 px edge |
| 9 | 11 | 5.51 | 29 | - | 20.00 | 29 | 20 px edge |
| 10 | 10 | 5.01 | 29 | - | 20.00 | 29 | 20 px edge |
| 11 | 9 | 4.52 | 29 | - | 20.00 | 29 | 20 px edge |
| 12 | 8 | 3.98 | 29 | - | 20.00 | 29 | 20 px edge |
| 13 | 7 | 3.52 | 29 | - | 20.00 | 29 | 20 px edge |
| 14 | 6 | 3.00 | 29 | - | 20.00 | 29 | 20 px edge |
| 15 | 5 | 2.49 | 29 | - | 20.00 | 29 | 20 px edge |
| 16 | 4 | 2.00 | 29 | 2 px edge | 20.00 | 29 | 20 px edge |
| 17 | 3 | 3.00 | 29 | - | 20.00 | 29 | 20 px edge |

| Set | Sensitivity | frac (pass) | SSE/flat | T2 pass | cells (skipped) |
|---|---|---|---|---|---|
| simple | default | 0.625 (10/16) | 1.50 | False | 64 (0) |
| simple | S6 | 0.625 (10/16) | 1.42 | False | 64 (0) |
| simple | S7 | 0.688 (11/16) | 1.48 | False | 64 (0) |
| complex | default | 0.800 (12/15) | 0.52 | False | 128 (2) |
| complex | S6 | 0.800 (12/15) | 0.75 | False | 128 (2) |
| complex | S7 | 0.800 (12/15) | 0.48 | True | 128 (3) |

Simple, per point under S6 (K/k from measured periods):

| K/k | fig | ours | ok |
|---|---|---|---|
| 0.4 | 0.304 | 0.392 | False |
| 0.5 | 0.342 | 0.432 | True |
| 0.6 | 0.357 | 0.514 | False |
| 0.7 | 0.245 | 0.446 | False |
| 0.8 | 0.369 | 0.565 | False |
| 0.9 | 0.510 | 0.512 | True |
| 1.1 | 0.504 | 0.484 | True |
| 1.2 | 0.428 | 0.375 | True |
| 1.3 | 0.239 | 0.547 | False |
| 1.4 | 0.507 | 0.443 | True |
| 1.5 | 0.330 | 0.471 | True |
| 1.6 | 0.428 | 0.578 | True |
| 1.7 | 0.336 | 0.371 | True |
| 1.8 | 0.611 | 0.532 | True |
| 1.9 | 0.637 | 0.465 | True |
| 2.0 | 0.251 | 0.423 | False |

Simple, per point under S7 (window from 62.5 ms):

| K/k | fig | ours | ok |
|---|---|---|---|
| 0.4 | 0.304 | 0.382 | True |
| 0.5 | 0.342 | 0.423 | True |
| 0.6 | 0.357 | 0.509 | False |
| 0.7 | 0.245 | 0.454 | False |
| 0.8 | 0.369 | 0.585 | False |
| 0.9 | 0.510 | 0.545 | True |
| 1.1 | 0.504 | 0.471 | True |
| 1.2 | 0.428 | 0.427 | True |
| 1.3 | 0.239 | 0.586 | False |
| 1.4 | 0.507 | 0.443 | True |
| 1.5 | 0.330 | 0.489 | True |
| 1.6 | 0.428 | 0.545 | True |
| 1.7 | 0.336 | 0.336 | True |
| 1.8 | 0.611 | 0.496 | True |
| 1.9 | 0.637 | 0.739 | True |
| 2.0 | 0.251 | 0.422 | False |


Reading (dispatcher, 2026-10-13): the ori 0 periods follow (20 − i)/2 px to within 0.2 px from
index 0 to 16 (10.21 against 10, 5.51 against 5.5, 2.00 against 2), which supports A2: the
notebook's `real_fqcyval` ∝ 1/(20 − i) is the stimulus frequency. Index 17 is 1.5 px, under
2 px per cycle, and its DFT reads the alias at 3.00 px; S6 used that alias, so S6 is less
faithful than the scored map at index 17. The 9.67 and 5.8 px of A2 came from the coarser
v4 check. The ori 90 check is uninformative (an x-profile cannot see a horizontal grating).
The simple-set failure holds under S6 and S7; the complex result depends on the window (S7
passes at 0.48 against 0.5).
