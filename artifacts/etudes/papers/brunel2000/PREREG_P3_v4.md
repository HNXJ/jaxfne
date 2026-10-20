# brunel2000 preregistration P3 v4

Status: **signed v4**, Hamm, 2026-10-19, on the text of v3 (`PREREG_P3_v3.md`,
dev ef1c8824) with his two choices fixed: band (b) and Q3 reading (ii). Any
later change to this file is a new amendment with its own signature. v1-v3
stay pinned. Scored runs check out the commit that carries this signed text.

Lane: brunel2000. Paper: Brunel N (2000), Dynamics of sparsely connected
networks of excitatory and inhibitory spiking neurons, J Comput Neurosci
8:183-208, doi:10.1023/A:1008925309027.

## Estimands (checklist P1)

Paper values are its simulations (`SOURCE_P1.md` §3), N=12500, CE=1000,
CI=250, Cext=1000, J=0.1 mV, D=1.5 ms, taurp=2 ms. z is the mean over seeds
4-8 of `brunel2000_api.score(spikes, dt_ms=0.1, burn_in_ms=500,
freq_res_hz=1.0)` on 2500 ms runs: regime labels on 128 ms Welch segments
(`LABEL_WINDOW_MS`), global frequency (`spectral_peak_hz`) on a 1 Hz grid. No
parameter is fitted: every model constant is the paper's, so seeds 1-3 stay
unused.

| Id | Fig. 8 point (g, nuext/nuthr) | y (paper sim; theory) | criterion |
|---|---|---|---|
| E1 | B (6, 4) cell rate | 60.7 Hz (55.8) | e <= 0.08 |
| E2 | B global frequency | ~180 Hz (190) | e <= 0.06 |
| E3 | C (5, 2) cell rate | 37.7 Hz (38.0) | e <= 0.01 |
| E4 | D (4.5, 0.9) cell rate | 5.5 Hz (6.5) | e <= 0.18 |
| E5 | D global frequency | 22 Hz (29) | e <= 0.32 |
| Q1 | A (3, 2) SR | synchronized, regular | label `saturated` or `oscillatory`, kappa >= 0.5, CV_ISI <= 0.1 |
| Q2 | B, D SI | oscillatory global, irregular cells | label `oscillatory`, CV_ISI > 0.1 |
| Q3 | C AI | stationary global, irregular cells | `spectral_peak_excess` < 5 on the 128 ms grid, CV_ISI > 0.1 |

e = |z - y| / |y| against the paper's simulation value. Each band is that
estimand's sim-theory gap in the paper (Hamm's choice (b); (a), every
e <= 0.05, declined). Q3 reads the paper's own words (Hamm's choice (ii)):
stationary global activity as `spectral_peak_excess` below the classifier's
oscillation floor, irregular cells as CV_ISI > 0.1 as in Q2. The classifier
label `asynchronous` (kappa < 0.1, CV_ISI in 0.7-1.5; choice (i), declined)
is reported beside it, unscored.

- Close: every E within its band and Q1-Q3 pass. Off: each failing item is
  reported as "needs X" with its value.

Disclosure: both seed-0 pilots were known when these choices were written
(dt=0.5 and dt=0.1, CALIBRATION). At dt=0.1: e = 0.020, 0.011, 0.0006, 0.093,
0.18 for E1-E5; Q1 passes (saturated, kappa 1.00, CV 0.0006); Q2 passes; Q3
(i) fails (CV 0.41, kappa 0.20), Q3 (ii) passes (excess 2.2, CV 0.41). Q3 (ii)
was offered after C failed (i); it restates the paper's claim (`SOURCE_P1.md`
§3). Neither band was tuned to the pilots.

## Numerics (gap G4)

- dt = 0.1 ms, set by the v2 rule: between the dt=0.5 and dt=0.1 pilots
  E1-E5 moved by 5.5%, 9.6%, 1.3%, 2.9% and 18%, above 2%. Delays 15 steps,
  refractory 20 steps.
- 2500 ms runs, the first 500 ms dropped; float32 on the CPU backend (the
  GPU kernel needs 13.4 GiB at 25000 steps, more than the card holds beside
  the desktop); the float64 check of checklist P3 on seed 4 at point C.

## Seeds (checklist P2)

Seed 0 is for pilots only. Score: held-out seeds 4-8. Seeds 1-3 stay unused
(no fit). The seed shown to change the output is named.

## Scored / unscored split

- Unscored (pilots, G0 checks): seed 0, timing runs; outputs never under a
  `scored/` path.
- Scored (needs `--signed`): seeds 4-8; one code SHA per output directory
  (enforced by `lane_kit.refuse_existing`).

## Sensitivity set (declared before scoring)

- S1 finite size: N=6250 with CE=1000, CI=250, Cext=1000 and J held
  (in-degrees fixed, N varies, epsilon 0.2), seeds 4-8; needs a lane option
  first (`build_spec` sets every in-degree to 0.1 N).
- S2 window: 5000 ms runs, the last 4500 ms scored, seeds 4-5.
- S3 dt: dt = 0.05 ms, seeds 4-5.
- Sensitivity runs never enter the score (checklist P6).

## Pilot (B8)

- Done on seed 0 at dt 0.5 and 0.1 (CALIBRATION). After signature at most one
  amendment; any further change is a new protocol version scored on fresh
  seeds.
