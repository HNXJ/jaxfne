# brunel2000 calibration record (RESIDUAL_PROTOCOL.md, Records)

Status: P5 (prereg v4 signed by Hamm 2026-10-19 and scored on seeds 4-8; `RESULT_P3.md`).
Verdict: reproduced (conditional): every estimand in band, Q3 on the paper's wording. Every number below is reopened from its JSON or receipt before
commit (checklist R2); verdicts use the status tags (checklist R3).

## Provenance

- Code SHA: c57ec220c395369528a93bf104a94c815409568a (pilot manifest
  `scratch/brunel2000_pilot/manifest.json`; one SHA per output directory,
  enforced by `lane_kit.refuse_existing` — observed refusing an overwrite on
  rerun).
- Backend, dtype, seeds: cpu, float32, seed 0 (manifest `backend`, `dtype`,
  `seed`; jaxfne 0.5.6, jax 0.10.1, platform win32, all in the manifest).
  Float64 appears only in the closed-form kernel test (checklist P3
  float64 check is due at scoring).

## P2 pilot (seed 0, unscored)

`brunel2000_api.run(0, "pilot-D", "scratch/brunel2000_pilot", point="D",
n=1250, T_s=0.5)`: N=1250 (NE=1000, NI=250), point D (g=4.5, eta=0.9),
external 900 Hz x 0.1 mV, 1000 steps at dt=0.5 ms.

| measure | value (manifest) |
|---|---|
| wall construct / simulate / total | 5.6 s / 1.6 s / 7.7 s |
| spikes | 0 (label `silent`, rate 0.0 Hz) |
| V | finite throughout |

Why silent (inferred, by construction): at N=1250 the external mean is
Cext*nuext*J*tau = 100*9*0.1*20 ms = 1.8 mV against threshold 20 mV; the
full-scale operating point needs Cext=1000 (18 mV). Small-N sparse nets sit
below threshold at fixed p — the pilot proves plumbing and timing, not
dynamics. No estimand claim follows.

Known-positive control (PREREG_P3 B8, same network, drive amplitude x10 =
1.0 mV, control only, not a paper point): 7473 spikes, rate 11.9568 Hz,
cv_isi 0.4633, kappa 0.0629, label `oscillatory`, V finite. Spikes propagate
through the recurrent delayed edges; the estimators run end to end.

Scale decision: full N=12500 projects ~15.6M edges (0.1 x 12500^2) and is
additionally barred for points A-C by the drive-rate cap (PORT_SPEC.md row
12; point B refused at N=1250 with p=2.00 per step, quoted in the pilot
log). Superseded 2026-10-19: the Cext-train drive (F3) lifts the rate cap,
and Hamm asked for Fig. 8 at full scale; the run below is a B8 pilot on
seed 0 (PREREG_P3 reserves seed 0 for pilots), so P3 stays unsigned and
its fit and held-out seeds 1-8 stay unused.

## P2 pilot at full scale, Fig. 8 A-D (seed 0, unscored, B8)

`brunel2000_api.run(0, "fig8<P>_n12500", out, point=P, n=12500, T_s=2.5,
burn_in_ms=500)` for P in A-D: NE=10000, NI=2500, CE=1000, CI=250 exact
(fixed in-degree), Cext=1000 trains at nuext=eta*nuthr, J=0.1 mV, D=1.5 ms,
t_ref=2 ms, 5000 steps at dt=0.5 ms, estimands over the last 2000 ms.
Code SHA d5689f6f3a356eeb4d8c44f5b4f42e06ecc56190 (rebased onto dev as
e891a6b5; `jaxfne/` and this lane equal dev 7bbb5f7c except a burn-in
refusal and `fields.signals_dipole`). Backend gpu (cuda:0, WSL2), float32,
jaxfne 0.5.6, jax 0.10.1; manifest, rasters, runner and log outside the repo in
`C:/workspace/jaxfne_receipts/brunel2000_fig8_full/` (manifest sha256 7079b4e3eb9d9495...).
Global frequency is `spectral_peak_hz` (Welch peak of the population rate
in 5-200 Hz) rescored from the saved rasters with `welch_nperseg=2048`
(0.98 Hz bins); the run-time default 256 (7.8 Hz bins) gave 195.3 Hz (B)
and 23.4 Hz (D).

| point (g, eta) | paper state | cell rate Hz (paper) | global Hz (paper) | CV_ISI | kappa | label |
|---|---|---|---|---|---|---|
| A (3, 2) | SR | 333.25 (none stated) | out of band: 3.0 ms period | 0.0006 | 1.000 | saturated |
| B (6, 4) | SI fast | 56.35 (60.7) | 197.3 (~180) | 0.530 | 0.285 | oscillatory |
| C (5, 2) | AI | 37.20 (37.7) | 146.5, excess 7.8 | 0.390 | 0.196 | irregular (256), oscillatory (2048) |
| D (4.5, 0.9) | SI slow | 5.84 (5.5) | 22.5 (22) | 0.641 | 0.075 | oscillatory |

Errors e against the paper: rate B 7.2%, C 1.3%, D 6.2%; global frequency
B 9.6%, D 2.1%. A fires regularly in lockstep, a 3.0 ms period of 2 ms
refractory plus 1 ms recharge (limit 500 Hz), the SR picture; its 333 Hz population
rhythm lies above the 200 Hz band, so the band peak (194 Hz) is not a
global frequency. C has the paper's rate, yet its CV (0.39) and kappa
(0.196) miss the classifier's asynchronous bounds (CV 0.7-1.5, kappa
below 0.1) and a weak 146 Hz peak appears on the fine grid: less irregular than an AI state (observed, one
seed); P3 decides whether this is a finding or a window effect.
Wall per point 520-858 s (construct 75-215 s, simulate 420-654 s).

## P2 pilot, dt convergence (gap G4; seed 0, unscored, B8)

The same four points at dt=0.1 ms (`run(..., dt_ms=0.1)`; delays 15 steps,
refractory 20 steps, 25000 steps). Code SHA
5080da5caa4b719843c30c852d211e6e443f9741, backend cpu (the GPU kernel asked
for 13.4 GiB at 25000 steps and ran out of memory beside the desktop's 10
GB), float32, jaxfne 0.5.6, jax 0.10.1; receipts in
`C:/workspace/jaxfne_receipts/brunel2000_fig8_dt01/` (manifest sha256
f322460cfe1e5c9e...). Both pilots rescored from their rasters with
`brunel_fig8_rescore2.py`: labels on 128 ms Welch segments (the classifier's
256 samples at dt=0.5 ms; at dt=0.1 ms the 256-sample default has 39 Hz bins
and read D `irregular`), global frequency on a 1 Hz grid.

| point | quantity | dt=0.5 | dt=0.1 | paper | e at dt=0.1 |
|---|---|---|---|---|---|
| A | rate Hz, CV, kappa, label | 333.25, 0.0006, 1.00, saturated | 333.27, 0.0006, 1.00, saturated | SR | |
| B | rate Hz | 56.35 | 59.46 | 60.7 | 0.020 |
| B | global Hz | 197 | 178 | ~180 | 0.011 |
| B | CV, kappa, label | 0.53, 0.29, oscillatory | 0.79, 0.30, oscillatory | SI | |
| C | rate Hz | 37.20 | 37.68 | 37.7 | 0.0006 |
| C | CV, kappa, excess, label | 0.39, 0.20, 2.1, irregular | 0.41, 0.20, 2.2, irregular | AI | |
| D | rate Hz | 5.84 | 6.01 | 5.5 | 0.093 |
| D | global Hz | 22 | 26 | 22 | 0.18 |
| D | CV, kappa, label | 0.64, 0.075, oscillatory | 0.62, 0.11, oscillatory | SI | |

dt moves E1-E5 by 5.5%, 9.6%, 1.3%, 2.9% and 18%: above the 2% rule of
PREREG_P3_v2, so scored runs use dt=0.1 ms. B converges onto the paper (rate
e 0.072 to 0.020, global 0.096 to 0.011, CV 0.53 to 0.79); D moves away on
both (one seed at 6 Hz: about 150k spikes in the 2 s window, so seed noise
is a candidate). C keeps the paper's rate and stays below the classifier's
asynchronous bounds at both dts: stationary global activity (excess 2.2,
below the floor of 5) with CV 0.41, a finding P3 settles. Wall per point
1049-1343 s on the CPU (construct 51-94 s).

## 1. Estimand table

e = |z - y| / |y| (absolute error for estimands near 0, with the scale named).

| Id | y (paper) | z at theta0 | z at theta* | e | Band |
|---|---|---|---|---|---|
| (fill) | | | | | |

## 2. Simplifications, fidelity fixes, X forms (checklist F1-F4)

- F1 hard refractory (2026-10-19, `brunel2000_api.py`, seed 0): the lane
  declares `t_ref_ms=2.0` (paper taurp); core `lif_step` holds V at reset
  and blocks spikes for `round(t_ref_ms/dt)` steps. Test
  `test_hard_refractory` fails on the code before the fix (no such term;
  cells re-fire the next step under strong drive).
- F2 exact fixed indegree (2026-10-19, `brunel2000_api.py`, seed 0): the
  lane declares `sampling="fixed_indegree"` with `in_degree=CE/CI`
  computed from the realized (NE, NI) counts; every post cell draws
  exactly that many distinct partners. Tests `test_exact_indegree` and
  `test_fixed_indegree_exact_totals` fail on the code before the fix
  (the baseline refuses `sampling="fixed_indegree"`).
- F3 external drive as Cext trains (2026-10-19, `brunel2000_api.py`,
  seed 0): the lane declares `poisson_drive={rate_hz: nuext, counts:
  Cext}`; each step draws `binomial(Cext, nuext*dt/1000)`, the exact sum
  distribution of Cext independent trains. Test
  `test_external_drive_counts` fails on the code before the fix (the
  baseline has no `counts` key; the old lane declared the summed rate
  with at most one event per step per cell).

## 3. D (degrees of freedom, declared before any fit)

- (fill: each parameter with its range and reason; never a parameter the
  paper states exactly, unless declared as a deviation with a reason)

## 4. Held-out result and claim (checklist P6)

- Fit on seeds 1-3, score on held-out seeds 4-8. Claim: "reproduced after
  calibration of [theta in D] (post hoc), held-out seeds 4-8" or "needs X:
  [form], |a| = ...".
