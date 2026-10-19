# vogels2011 calibration record (RESIDUAL_PROTOCOL.md, Records)

Status: P2 (smoke pilot, unscored). Every number below is reopened from its JSON or
receipt before commit (checklist R2); verdicts use the status tags
(checklist R3).

## Smoke pilot (P2, seed 0 only, unscored B8)

Route settled; no score against the paper (G-source: paper realized numbers
unknown; G-IE: I-to-E fixed at the D-IE stand-in).

- Output: `scratch/vogels2011_pilot/` (gitignored), manifest
  `manifest.json`, raster `spikes_seed0_pilot.npz`.
- Manifest (observed): git_sha `32f852b2`, backend cpu, dtype float32,
  jax 0.10.1, jaxfne 0.5.6, platform win32.
- Run (observed): n=200 (NE=160, NI=40), T=1.0 s, dt=0.1 ms, seed 0,
  10000 steps, 6900 spikes; wall_s construct 1.75 s, simulate 0.34 s.
- Regime (observed, `activity_regime`): rate_hz 34.5, cv_isi 0.029,
  kappa 0.194, active_fraction 0.995, spectral_peak_excess 1.446,
  label "irregular".
- Reading: far above the rho0=3 Hz target with the fixed D-IE stand-in and
  no plasticity; the pilot settles the route, not the rate (PORT_SPEC §18
  scored at P3).

## Provenance

- Code SHA: (fill; one SHA per output directory, enforced by
  `vogels2011_api.write_run` via `lane_kit`).
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
