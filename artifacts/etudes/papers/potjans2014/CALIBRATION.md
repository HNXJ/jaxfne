# potjans2014 calibration record (RESIDUAL_PROTOCOL.md, Records)

Status: P1 (no runs yet). Every number below is reopened from its JSON or
receipt before commit (checklist R2); verdicts use the status tags
(checklist R3).

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
