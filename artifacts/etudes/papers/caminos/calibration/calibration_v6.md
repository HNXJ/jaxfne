# CAMINOS calibration gate v6

Spec: `artifacts/etudes/caminos/PREREG_P3.md`, "Calibration gate (v6)". Synthetic only; no network data.

NE=8000, dt=0.0001 s, kept 10 s, R0=1.2 Hz; draws=200 (power 100); theta grid (4.0, 5.0, 6.0, 8.0); FP <= 5 %; detection >= 95 % with |f_pk - 35| <= 2 Hz; realised within 10 %.

Gate: `Gate v6: PASS (theta = 4; 14 of 14 detection rows >= 95 %, realised within 10 %)`

## 1. Null floors: FP rate by theta (hits/draws)

| Null floor | Floor PSD at 35 Hz | FP theta 4 | FP theta 5 | FP theta 6 | FP theta 8 |
|---|---|---|---|---|---|
| white Poisson | 0.0002965 | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 8 ms, var 0.5x | 0.00621 | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 8 ms, var 2x | 0.02431 | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 8 ms, var 8x | 0.09246 | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 30 ms, var 0.5x | 0.002705 | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 30 ms, var 2x | 0.01358 | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |
| AR(1) tau 30 ms, var 8x | 0.1324 | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) | 0.000 (0/200) |

theta = 4: the smallest with every floor at FP <= 5 %.

## 2. Detection rows (gated): true PE = 2 theta = 8

| Null floor | Form | Scale | Component / floor (target 2 theta - 1) | Realised PE | Within 10 % | Detection (hits/draws) | Median f_pk (Hz) | Pass |
|---|---|---|---|---|---|---|---|---|
| white Poisson | sinusoid | 0.07891 | 7.0000 | 7.845 | yes | 1.000 (200/200) | 35.00 | PASS |
| white Poisson | ar2 | 0.0002287 | 7.0000 | 7.947 | yes | 0.990 (198/200) | 35.00 | PASS |
| AR(1) tau 8 ms, var 0.5x | sinusoid | 0.3611 | 7.0000 | 7.940 | yes | 1.000 (200/200) | 35.00 | PASS |
| AR(1) tau 8 ms, var 0.5x | ar2 | 0.001043 | 7.0000 | 7.882 | yes | 0.995 (199/200) | 35.00 | PASS |
| AR(1) tau 8 ms, var 2x | sinusoid | 0.7145 | 7.0000 | 7.982 | yes | 0.990 (198/200) | 35.00 | PASS |
| AR(1) tau 8 ms, var 2x | ar2 | 0.002046 | 7.0000 | 7.901 | yes | 0.965 (193/200) | 35.00 | PASS |
| AR(1) tau 8 ms, var 8x | sinusoid | 1.393 | 7.0000 | 8.032 | yes | 0.955 (191/200) | 35.00 | PASS |
| AR(1) tau 8 ms, var 8x | ar2 | 0.004109 | 7.0000 | 7.979 | yes | 0.950 (190/200) | 35.00 | PASS |
| AR(1) tau 30 ms, var 0.5x | sinusoid | 0.2384 | 7.0000 | 8.010 | yes | 1.000 (200/200) | 35.00 | PASS |
| AR(1) tau 30 ms, var 0.5x | ar2 | 0.0006798 | 7.0000 | 7.953 | yes | 0.995 (199/200) | 35.00 | PASS |
| AR(1) tau 30 ms, var 2x | sinusoid | 0.5341 | 7.0000 | 7.952 | yes | 0.980 (196/200) | 35.00 | PASS |
| AR(1) tau 30 ms, var 2x | ar2 | 0.001551 | 7.0000 | 7.982 | yes | 0.980 (196/200) | 35.00 | PASS |
| AR(1) tau 30 ms, var 8x | sinusoid | 1.668 | 7.0000 | 8.008 | yes | 0.980 (196/200) | 35.00 | PASS |
| AR(1) tau 30 ms, var 8x | ar2 | 0.004762 | 7.0000 | 8.010 | yes | 0.970 (194/200) | 35.00 | PASS |

## 3. Power curve (not gated): detection at theta, hits/draws

Power rows use the first 100 of each floor's null draws; their realised PE includes that subset's floor sampling. Not gated.

| Null floor | Form | PE 2 | PE 3 | PE 4 | PE 6 | PE 8 |
|---|---|---|---|---|---|---|
| white Poisson | sinusoid | 0.00 (0/100) | 0.05 (5/100) | 0.44 (44/100) | 1.00 (100/100) | 1.00 (100/100) |
| white Poisson | ar2 | 0.00 (0/100) | 0.11 (11/100) | 0.51 (51/100) | 0.97 (97/100) | 0.99 (99/100) |
| AR(1) tau 8 ms, var 0.5x | sinusoid | 0.00 (0/100) | 0.16 (16/100) | 0.64 (64/100) | 1.00 (100/100) | 1.00 (100/100) |
| AR(1) tau 8 ms, var 0.5x | ar2 | 0.00 (0/100) | 0.17 (17/100) | 0.64 (64/100) | 0.98 (98/100) | 0.98 (98/100) |
| AR(1) tau 8 ms, var 2x | sinusoid | 0.00 (0/100) | 0.13 (13/100) | 0.49 (49/100) | 0.93 (93/100) | 0.99 (99/100) |
| AR(1) tau 8 ms, var 2x | ar2 | 0.00 (0/100) | 0.25 (25/100) | 0.59 (59/100) | 0.90 (90/100) | 0.97 (97/100) |
| AR(1) tau 8 ms, var 8x | sinusoid | 0.02 (2/100) | 0.32 (32/100) | 0.68 (68/100) | 0.88 (88/100) | 0.97 (97/100) |
| AR(1) tau 8 ms, var 8x | ar2 | 0.04 (4/100) | 0.41 (41/100) | 0.66 (66/100) | 0.88 (88/100) | 0.95 (95/100) |
| AR(1) tau 30 ms, var 0.5x | sinusoid | 0.00 (0/100) | 0.11 (11/100) | 0.54 (54/100) | 1.00 (100/100) | 1.00 (100/100) |
| AR(1) tau 30 ms, var 0.5x | ar2 | 0.00 (0/100) | 0.14 (14/100) | 0.57 (57/100) | 0.96 (96/100) | 1.00 (100/100) |
| AR(1) tau 30 ms, var 2x | sinusoid | 0.04 (4/100) | 0.36 (36/100) | 0.66 (66/100) | 0.94 (94/100) | 0.99 (99/100) |
| AR(1) tau 30 ms, var 2x | ar2 | 0.03 (3/100) | 0.30 (30/100) | 0.63 (63/100) | 0.92 (92/100) | 0.97 (97/100) |
| AR(1) tau 30 ms, var 8x | sinusoid | 0.47 (47/100) | 0.82 (82/100) | 0.92 (92/100) | 0.98 (98/100) | 0.98 (98/100) |
| AR(1) tau 30 ms, var 8x | ar2 | 0.52 (52/100) | 0.79 (79/100) | 0.91 (91/100) | 0.95 (95/100) | 0.99 (99/100) |

## 4. AR(2) component FWHM

Poles r exp(+/- i 2 pi 35 Hz dt) with r = 1 - pi 5 Hz dt, dt = 1e-4 s.
Measured FWHM: 5.189 Hz (fitted peak 34.77 Hz) on a 600 s realisation; target 5 Hz.

