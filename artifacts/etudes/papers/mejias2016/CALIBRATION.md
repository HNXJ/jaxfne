# mejias2016 calibration record (RESIDUAL_PROTOCOL.md, Records)

Status: P2 (smoke pilot, seed 0, unscored; B8). No comparison claim: the
pilot shows the route runs end to end only. Every number below is reopened
from `scratch/mejias2016_pilot/manifest.json` (checklist R2); verdicts use
the status tags (checklist R3).

## P2 smoke pilot (seed 0, unscored, B8)

Engine: lif stand-in S1 (`PORT_SPEC.md`; gap G1: no rate emitter). Route:
single-area two-layer module, n=20 per layer (40 cells), T=1.0 s, dt=0.5 ms
(2000 steps), CPU, float32, poisson 400 Hz / 2 mV, seed 0. Wall 3.2 s total
(construct 2.37 s, simulate 0.70 s); outputs under
`scratch/mejias2016_pilot/` (gitignored).

Measured values (reopened from the manifest): spikes 561; rate_hz 14.025;
cv_isi 0.6739; kappa 0.0726; active_fraction 1.0; spectral_peak_excess
1.2829; label "irregular"; layer_rates_hz L23 14.55, L56 13.5. V finite
(asserted in `run`). Verdict: runs (no score, no band).

## Rate smoke (seed 0, unscored; G1 closed)

Engine: core `rate` route (`jaxfne/emitters_rate.py`,
`jaxfne/_construct_rate.py`; PORT_SPEC.md row 1). Route: single-area
single-layer pair, 2 cells (1 E + 1 I), T=20.0 ms, dt=0.5 ms (40 steps),
CPU, float32, no Poisson drive, seed 0 (the `test_wilson_rate_emitter`
fixture, observed from this tree).

Measured values (observed from the run): rates shape (40, 2); finite True;
min 0.2962, max 4.5474, mean 3.0054; spikes 0 (the route emits none; rates
ride `V_m`); `metadata["emitter_family"] == "rate"`; n_edges 2,
n_floored_to_one_step 2 (no declared delay on the fixture rule).
Verdict: runs (no score, no band).

## Provenance

- Code SHA: (fill; one SHA per output directory, enforced by
  `mejias2016_api.write_run` via `lane_kit`).
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
