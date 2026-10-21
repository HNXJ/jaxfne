# brunel2000 preregistration P3 v2 (draft, UNSIGNED — no scored run may use it)

Supersedes the v1 template `PREREG_P3.md` (unfilled, kept byte-for-byte as a
pinned record).

Lane: brunel2000. Paper: Brunel N (2000), Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons, J Comput Neurosci 8:183-208, doi:10.1023/A:1008925309027. No scored run starts from this draft;
Hamm signs the frozen protocol before scoring (checklist P1).

## Estimands and criteria (checklist P1)

Paper values are its simulations (`SOURCE_P1.md` §3), N=12500, CE=1000, CI=250,
Cext=1000, J=0.1 mV, D=1.5 ms, taurp=2 ms. z is the mean over the scored seeds
of the lane scorer (`brunel2000_api.score`, `jaxfne.analysis.activity_regime`)
on the last 2000 ms of a 2500 ms run, called with `freq_res_hz=1.0`: labels on
the classifier's default Welch grid, global frequency (`spectral_peak_hz`) on
a 1 Hz grid. No parameter is fitted: every model
constant is the paper's, so seeds 1-3 stay unused and seeds 4-8 score.

| Id | Fig. 8 point (g, nuext/nuthr) | y (paper sim; theory) | z | e |
|---|---|---|---|---|
| E1 | B (6, 4) cell rate | 60.7 Hz (55.8) | | |
| E2 | B global frequency | ~180 Hz (190) | | |
| E3 | C (5, 2) cell rate | 37.7 Hz (38.0) | | |
| E4 | D (4.5, 0.9) cell rate | 5.5 Hz (6.5) | | |
| E5 | D global frequency | 22 Hz (29) | | |
| Q1 | A (3, 2) state SR | synchronized, regular | label `saturated` or `oscillatory` with kappa >= 0.5 and CV_ISI <= 0.1 | pass/fail |
| Q2 | B, D states SI | oscillatory global, irregular cells | label `oscillatory` with CV_ISI > 0.1 (not regular, the complement of Q1) | pass/fail |
| Q3 | C state AI | stationary global, irregular cells | label `asynchronous` | pass/fail |

Criterion (Hamm's choice before signature; both written before any scored run):

- (a) Template: close when every e <= 0.05 and Q1-Q3 pass; off otherwise, each
  failing estimand reported as "needs X".
- (b) Paper's own scale: close when each e is within that estimand's
  sim-theory gap in the paper (E1 0.08, E2 0.06, E3 0.01, E4 0.18, E5 0.32)
  and Q1-Q3 pass. The band comes from the paper, never from a jaxfne run.

Disclosure: the seed-0 full-scale pilot at dt=0.5 ms (CALIBRATION, 2026-10-19)
is known when this draft is written: e = 0.072, 0.096, 0.013, 0.062, 0.021 for
E1-E5, Q1 and Q2 pass, Q3 fails (label `irregular`, CV 0.39, kappa 0.196).
Neither option was tuned to it; a band of 0.10, which the pilot would just
pass, is not offered.

## Numerics (gap G4, declared before scoring)

- dt: the paper does not state it; its Fig. 8 bins global activity at 0.1 ms
  and the NEST reference implementation uses 0.1 ms. The seed-0 convergence
  pilot runs dt=0.5 and dt=0.1 ms. Scored runs use dt=0.1 ms unless every
  E1-E5 moves by less than 2% between the two, then dt=0.5 ms (a numerical
  rule: it reads the change between dts, never the distance to y).
- Duration 2500 ms, the first 500 ms dropped (`burn_in_ms=500`); float32 on
  the GPU, with the float64 check of checklist P3 on one scored seed at
  point C.

## Seeds (checklist P2)

Seed 0 is for pilots only. Fit (if needed): seeds 1-3. Score: held-out
seeds 4-8. The seed shown to change the output is named.

## Scored / unscored split

- Unscored (pilots, G0 checks): seed 0, timing runs; outputs never under a
  `scored/` path.
- Scored (needs `--signed`): seeds 1-8 per above; one code SHA per output
  directory (enforced by `lane_kit.refuse_existing`).

## Sensitivity set (declared before scoring)

- S1 finite size: N=6250 with CE=1000, CI=250, Cext=1000 and J held (the
  paper's sparse framework: in-degrees fixed, N varies, epsilon 0.2), seeds
  4-8; the paper ascribes its AI non-stationarity and SI irregularity to
  finite N (`SOURCE_P1.md` §3). Needs a lane option first: `build_spec`
  sets every in-degree to 0.1 N.
- S2 window: 5000 ms runs, the last 4500 ms scored, seeds 4-5.
- S3 dt: the dt the rule above did not pick, seeds 4-5.
- Sensitivity runs never enter the score (checklist P6).

## Pilot (B8)

- The full pipeline runs on unscored pilot seeds and on a known positive
  before this prereg is signed (project rule B8, Hamm 2026-10-18).
- The sensitivity set is declared in this prereg (see above); after
  signature at most one amendment, any further change is a new protocol
  version scored on fresh seeds.
