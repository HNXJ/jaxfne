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
- 2026-10-20: segment B ran with per-neuron traces, whose presynaptic trace summed arrivals over
  each I cell's outgoing edges, so its potentiation was scaled by the out-degree; the traces are
  per synapse since `changelog.d/istdp_vogels2011_traces.fixed.md`. w_ie_final_mean 2.42 nS is
  superseded; rerun segment B before P3.

## B8 convergence pilot (P2, seed 0, n 10 000, unscored, 2026-11-01)

- Run: `vogels2011_api.py pilot --seed 0 --n 10000 --T-s 120 --plastic-segment-s 5` at dceb49b0, WSL CUDA float32,
  32 min wall, 5.4 GB peak RSS. Receipts `F:/warehouse/vogels2011/conv_dceb49b0/s0_T120/` (`pilot_receipt.json`
  sha256 9eb55150…242784). Read with `read_convergence.py`.
- E rate course (observed): 76.5 Hz at 1 s, 8.2 at 10 s, 6.3 at 20 s, 4.8 at 60 s, 4.15 at 120 s. The 10 s-window
  slope first falls below 0.1 Hz/s at 24 s, while the rate falls a further 2.2 Hz from 20 s to 120 s.
- Fit r(t) = r_inf + A exp(-t/tau), t ≥ 20 s (derived): r_inf 4.17 ± 0.06 Hz, tau 31.7 ± 2.7 s, residual sd
  0.17 Hz. Last 1 s, E cells: median 4.0 Hz, 10/90 % 1/7 Hz; the top 10 % carry 21 % of spikes (unimodal).
- R-E at 120 s: z 4.15 Hz, e 0.38 (band 0.20); O-DROP ratio 38.7. The network converges above rho0 = 3 Hz.
- Ruling (Hamm 2026-11-01): T 150 s (about 5 tau), convergence redefined as |slope| < 0.01 Hz/s over 20 s; the
  baseline and both known positives (rho0 6 Hz, eta 0) rerun at T 150 s before the P3 v3 amendment.

## B8 T 150 s pilots (P2, seed 0, n 10 000, unscored, 2026-11-02)

- Runs at bafa4797, WSL CUDA float32, `--T-s 150 --plastic-segment-s 5`; walls 35:27 (base), 43:11 (rho0 6 Hz),
  24:24 (eta 0); peak RSS 5.4-5.6 GB. Receipts `F:/warehouse/vogels2011/T150_bafa4797/` (`pilot_receipt.json`
  sha256: base 522df0cd…, rho0 6 Hz 9918063c…, eta 0 0cdd5dcc…). Slope: least squares on the per-second E rate over
  130-150 s (`e_rate_per_s_hz`), with its standard error.

| Run | R-E z (y 3 Hz) | e | O-DROP | fit r_inf, tau (t ≥ 20 s) | slope 130-150 s (se) | \|slope\| < 0.01 |
|---|---|---|---|---|---|---|
| base | 4.241 Hz | 0.414 | 37.82 | 4.11 ± 0.04 Hz, 34.1 ± 2.3 s | −0.0019 (0.0079) Hz/s | yes |
| rho0 6 Hz | 6.928 Hz | 1.309 | 23.15 | 7.24 ± 0.04 Hz, 22.3 ± 3.5 s | −0.0157 (0.0097) Hz/s | no |
| eta 0 | 162.1 Hz | 53.0 | 0.99 | 161.6 Hz (flat) | +0.0066 (0.0103) Hz/s | yes |

- Known positives move as declared: rho0 6 Hz puts R-E outside the 0.20 band; eta 0 fails O-DROP (0.99 < 2).
- The baseline converges (slope rule met, r_inf 4.11 Hz) above y = 3 Hz: under the v2 band the pilot predicts an
  R-E FAIL (e 0.41).
- The slope's standard error over 20 one-second bins (0.008-0.010 Hz/s) equals the 0.01 Hz/s threshold, so the
  rule's pass on a converged run is not assured (inferred from the se; owner decision in P3 v3).

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
