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
