# vogels2011 calibration record (RESIDUAL_PROTOCOL.md, Records)

Status: P2 (smoke pilot, unscored). Every number below is reopened from its JSON or
receipt before commit (checklist R2); verdicts use the status tags
(checklist R3).

## Smoke pilot (P2, seed 0 only, unscored B8)

Route settled; no score against the paper (G-source: paper realized numbers
unknown).

- Outputs: `scratch/vogels2011_pilot_segA/` (static segment A) and
  `scratch/vogels2011_pilot_segB/` (plastic segment B), gitignored; manifests
  `manifest.json`, rasters `spikes_seed0_segA.npz` / `spikes_seed0_segB.npz`.
- Manifests (observed): git_sha `c049151e`, backend cpu, dtype float32,
  jax 0.10.1, jaxfne 0.5.6, platform linux.
- Segment A, static (observed): n=200 (NE=160, NI=40), T=1.0 s, dt=0.1 ms,
  seed 0, 10000 steps, 10878 spikes; wall_s construct 1.89 s, simulate
  0.39 s. Regime (observed, `activity_regime`): rate_hz 54.39, cv_isi 0.017,
  kappa 0.343, active_fraction 1.0, spectral_peak_excess 1.681,
  label "irregular".
- Segment B, plastic from rest (observed): same grid, 10242 spikes;
  wall_s construct 0.17 s, simulate 0.29 s; 128 I-to-E edges plastic under
  `istdp_vogels2011`, w_ie_init 1e-10 nS, w_ie_final_mean 2.42 nS.
  Regime (observed): rate_hz 51.21, cv_isi 0.054, kappa 0.315,
  active_fraction 1.0, spectral_peak_excess 1.355, label "irregular".
- Reading: both segments run far above the rho0=3 Hz target after 1 s from
  rest with the paper 1e-10 init (near-zero initial inhibition); the pilot
  settles the two-segment route and shows I-to-E growth, not the rate
  (PORT_SPEC §18 scored at P3).

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
