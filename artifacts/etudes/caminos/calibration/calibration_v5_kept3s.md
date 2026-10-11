# CAMINOS calibration gate v5

Spec: `artifacts/etudes/caminos/PREREG_P3.md`, "Calibration gate (v5)". Synthetic only; no network data.

NE=8000, dt=0.0001 s, kept 3 s, R0=1.2 Hz; draws=200 (power 100); theta grid (4.0, 5.0, 6.0, 8.0); FP <= 5 %; detection >= 95 % with |f_pk - 35| <= 2 Hz; realised within 10 %.

Gate: `Gate v5: FAIL (theta = 5; 11 of 14 detection rows fail): white Poisson, ar2: detection 0.915, realised PE 9.927 (target 10); AR(1) tau 8 ms, var 0.5x, ar2: detection 0.895, realised PE 9.906 (target 10); AR(1) tau 8 ms, var 2x, sinusoid: detection 0.940, realised PE 9.794 (target 10); AR(1) tau 8 ms, var 2x, ar2: detection 0.825, realised PE 9.936 (target 10); AR(1) tau 8 ms, var 8x, sinusoid: detection 0.930, realised PE 10.143 (target 10); AR(1) tau 8 ms, var 8x, ar2: detection 0.875, realised PE 10.020 (target 10); AR(1) tau 30 ms, var 0.5x, ar2: detection 0.900, realised PE 9.804 (target 10); AR(1) tau 30 ms, var 2x, sinusoid: detection 0.945, realised PE 9.778 (target 10); AR(1) tau 30 ms, var 2x, ar2: detection 0.905, realised PE 9.841 (target 10); AR(1) tau 30 ms, var 8x, sinusoid: detection 0.890, realised PE 10.011 (target 10); AR(1) tau 30 ms, var 8x, ar2: detection 0.805, realised PE 10.086 (target 10)`

## 1. Null floors: FP rate by theta (hits/draws)

| Null floor | Floor PSD at 35 Hz | FP theta 4 | FP theta 5 | FP theta 6 | FP theta 8 |
|---|---|---|---|---|---|
| white Poisson | 0.0002987 | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 8 ms, var 0.5x | 0.00618 | 0.025 (5/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 8 ms, var 2x | 0.02475 | 0.010 (2/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 8 ms, var 8x | 0.08152 | 0.025 (5/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 30 ms, var 0.5x | 0.002782 | 0.015 (3/200) | 0.005 (1/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 30 ms, var 2x | 0.01439 | 0.045 (9/200) | 0.005 (1/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 30 ms, var 8x | 0.0514 | 0.095 (19/200) | 0.035 (7/200) | 0.000 (0/200) | 0.000 (0/200) |

theta = 5: the smallest with every floor at FP <= 5 %.

## 2. Detection rows (gated): true PE = 2 theta = 10

| Null floor | Form | Scale | Component / floor (target 2 theta - 1) | Realised PE | Within 10 % | Detection (hits/draws) | Median f_pk (Hz) | Pass |
|---|---|---|---|---|---|---|---|---|
| white Poisson | sinusoid | 0.08981 | 9.0000 | 9.692 | yes | 0.995 (199/200) | 35.00 | PASS |
| white Poisson | ar2 | 0.0002562 | 9.0000 | 9.927 | yes | 0.915 (183/200) | 35.00 | FAIL |
| AR(1) tau 8 ms, var 0.5x | sinusoid | 0.4085 | 9.0000 | 9.952 | yes | 1.000 (200/200) | 35.00 | PASS |
| AR(1) tau 8 ms, var 0.5x | ar2 | 0.001159 | 9.0000 | 9.906 | yes | 0.895 (179/200) | 35.00 | FAIL |
| AR(1) tau 8 ms, var 2x | sinusoid | 0.8174 | 9.0000 | 9.794 | yes | 0.940 (188/200) | 35.00 | FAIL |
| AR(1) tau 8 ms, var 2x | ar2 | 0.002314 | 9.0000 | 9.936 | yes | 0.825 (165/200) | 35.00 | FAIL |
| AR(1) tau 8 ms, var 8x | sinusoid | 1.484 | 9.0000 | 10.143 | yes | 0.930 (186/200) | 35.00 | FAIL |
| AR(1) tau 8 ms, var 8x | ar2 | 0.004382 | 9.0000 | 10.020 | yes | 0.875 (175/200) | 35.00 | FAIL |
| AR(1) tau 30 ms, var 0.5x | sinusoid | 0.2741 | 9.0000 | 10.136 | yes | 0.985 (197/200) | 35.00 | PASS |
| AR(1) tau 30 ms, var 0.5x | ar2 | 0.0007832 | 9.0000 | 9.804 | yes | 0.900 (180/200) | 35.00 | FAIL |
| AR(1) tau 30 ms, var 2x | sinusoid | 0.6232 | 9.0000 | 9.778 | yes | 0.945 (189/200) | 35.00 | FAIL |
| AR(1) tau 30 ms, var 2x | ar2 | 0.001833 | 9.0000 | 9.841 | yes | 0.905 (181/200) | 35.00 | FAIL |
| AR(1) tau 30 ms, var 8x | sinusoid | 1.178 | 9.0000 | 10.011 | yes | 0.890 (178/200) | 35.00 | FAIL |
| AR(1) tau 30 ms, var 8x | ar2 | 0.003456 | 9.0000 | 10.086 | yes | 0.805 (161/200) | 35.00 | FAIL |

## 3. Power curve (not gated): detection at theta, hits/draws

Power rows use the first 100 of each floor's null draws; their realised PE includes that subset's floor sampling. Not gated.

| Null floor | Form | PE 2 | PE 3 | PE 4 | PE 6 | PE 8 |
|---|---|---|---|---|---|---|
| white Poisson | sinusoid | 0.00 (0/100) | 0.03 (3/100) | 0.25 (25/100) | 0.75 (75/100) | 0.99 (99/100) |
| white Poisson | ar2 | 0.02 (2/100) | 0.23 (23/100) | 0.51 (51/100) | 0.77 (77/100) | 0.93 (93/100) |
| AR(1) tau 8 ms, var 0.5x | sinusoid | 0.01 (1/100) | 0.19 (19/100) | 0.39 (39/100) | 0.86 (86/100) | 0.99 (99/100) |
| AR(1) tau 8 ms, var 0.5x | ar2 | 0.07 (7/100) | 0.30 (30/100) | 0.55 (55/100) | 0.68 (68/100) | 0.87 (87/100) |
| AR(1) tau 8 ms, var 2x | sinusoid | 0.02 (2/100) | 0.26 (26/100) | 0.43 (43/100) | 0.71 (71/100) | 0.86 (86/100) |
| AR(1) tau 8 ms, var 2x | ar2 | 0.13 (13/100) | 0.35 (35/100) | 0.52 (52/100) | 0.71 (71/100) | 0.81 (81/100) |
| AR(1) tau 8 ms, var 8x | sinusoid | 0.04 (4/100) | 0.30 (30/100) | 0.58 (58/100) | 0.81 (81/100) | 0.87 (87/100) |
| AR(1) tau 8 ms, var 8x | ar2 | 0.13 (13/100) | 0.41 (41/100) | 0.54 (54/100) | 0.75 (75/100) | 0.81 (81/100) |
| AR(1) tau 30 ms, var 0.5x | sinusoid | 0.01 (1/100) | 0.13 (13/100) | 0.43 (43/100) | 0.85 (85/100) | 0.95 (95/100) |
| AR(1) tau 30 ms, var 0.5x | ar2 | 0.04 (4/100) | 0.21 (21/100) | 0.56 (56/100) | 0.71 (71/100) | 0.87 (87/100) |
| AR(1) tau 30 ms, var 2x | sinusoid | 0.09 (9/100) | 0.37 (37/100) | 0.63 (63/100) | 0.82 (82/100) | 0.92 (92/100) |
| AR(1) tau 30 ms, var 2x | ar2 | 0.17 (17/100) | 0.42 (42/100) | 0.59 (59/100) | 0.76 (76/100) | 0.76 (76/100) |
| AR(1) tau 30 ms, var 8x | sinusoid | 0.23 (23/100) | 0.44 (44/100) | 0.56 (56/100) | 0.71 (71/100) | 0.80 (80/100) |
| AR(1) tau 30 ms, var 8x | ar2 | 0.18 (18/100) | 0.47 (47/100) | 0.61 (61/100) | 0.66 (66/100) | 0.70 (70/100) |

## 4. AR(2) component FWHM

Poles r exp(+/- i 2 pi 35 Hz dt) with r = 1 - pi 5 Hz dt, dt = 1e-4 s.
Measured FWHM: 5.189 Hz (fitted peak 34.77 Hz) on a 600 s realisation; target 5 Hz.

