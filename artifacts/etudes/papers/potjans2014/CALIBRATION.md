# potjans2014 calibration record (RESIDUAL_PROTOCOL.md, Records)

Status: P2 (PORT_SPEC written, smoke pilot run, unscored). Every number
below is reopened from its JSON or receipt before commit (checklist R2);
verdicts use the status tags (checklist R3).

## 0. Smoke pilot, seed 0, unscored (B8)

Route runs end to end; no comparison claim beyond "runs".

- Output: `scratch/potjans2014_pilot/` (gitignored):
  `spikes_seed0_pilot.npz` + `manifest.json`. Code SHA
  `d7a7979a7c46892f9792e859be7b7e65774dd65b`, jaxfne 0.5.6, jax 0.10.1,
  backend cpu, dtype float32, seed 0.
- Scale: N=800 (214/61/227/57/50/11/149/31), 55 density rules, 30708
  edges, T=1000.0 ms, dt=0.5 ms, 2000 steps. Wall 42.8 s total
  (construct 37.0 s, simulate 3.5 s): under 2 minutes.
- Measured (reopened from `manifest.json`, `runs[0]`): 27785 spikes;
  overall rate_hz 34.73125, cv_isi 0.22980167397453996, kappa
  0.17703469343785727, active_fraction 1.0, label "irregular".
  Per population (rate_hz): L23E 33.14485981308411 (n=214), L23PV
  36.704918032786885 (61), L4E 33.82378854625551 (227), L4PV
  40.19298245614035 (57), L5E 39.9 (50), L5PV 39.0 (11), L6E
  32.69798657718121 (149), L6PV 38.32258064516129 (31). V finite;
  `poisson_drive` receipt carries counts 2000, rate_hz 8.0, seed 0.
- Verdict: in progress (smoke pilot, seed 0, unscored; rates above the
  paper C1 band, expected from the uniform 2000-train stand-in drive;
  scored comparison needs P3 with fresh seeds).

## Provenance

- Code SHA: (fill; one SHA per output directory, enforced by
  `potjans2014_api.write_run` via `lane_kit`).
- Backend, dtype, seeds: (fill per run; float32 default, one float64 check per
  scored result, checklist P3).

## 1. Estimand table

e = |z - y| / |y| (absolute error for estimands near 0, with the scale named).

| Id | y (paper) | z at theta0 | z at theta* | e | Band |
|---|---|---|---|---|---|
| (fill) | | | | | |

## 2. Simplifications, fidelity fixes, X forms (checklist F1-F4)

- (fill: each step with its file and seed; a fidelity fix carries a test
  that fails on the code before the fix)

## 3. D (degrees of freedom, declared before any fit)

- (fill: each parameter with its range and reason; never a parameter the
  paper states exactly, unless declared as a deviation with a reason)

## 4. Held-out result and claim (checklist P6)

- Fit on seeds 1-3, score on held-out seeds 4-8. Claim: "reproduced after
  calibration of [theta in D] (post hoc), held-out seeds 4-8" or "needs X:
  [form], |a| = ...".
