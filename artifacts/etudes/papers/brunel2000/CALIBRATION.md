# brunel2000 calibration record (RESIDUAL_PROTOCOL.md, Records)

Status: P2 (PORT_SPEC written, seed-0 pilot done, no scored run). Verdict: in
progress. Every number below is reopened from its JSON or receipt before
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
log). No full-scale run precedes the P3 protocol decision.

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
