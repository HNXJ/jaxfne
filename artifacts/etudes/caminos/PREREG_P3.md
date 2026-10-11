# L2-P3 preregistration: CAMINOS reference model (PLoS CB 2026, Fig. 1)

Status: draft v6 for Hamm's batch sign-off (batch per release, after a second-review PASS;
decided 2026-10-11). It revises v1 (PASS with notes), v2 (FAIL: the R1 null was vacuous), v3
(FAIL: the ratio criterion fails on sloped spectra; R4 was asymmetric; partial runs were
accepted), v4 (calibration gate infeasible by design) and v5 (gate failed: a 3 s window is too
short, and the PSD local-maximum rule rejects broad peaks), after the reviews of 2026-10-11. No scored run before the signature. G0 single-cell checks
and timing pilots are not scored.

## Source claims (PLoS CB 2026 Results, reference model)

- R1. The network oscillates at about 35 Hz (low gamma).
- R2. SOM fires after PV within each gamma cycle, with a delay of about 8.5 ms. The
  paper's method is that of its ref. 42 (the Cell Reports paper, unread). The estimator
  below differs and is ours.
- R3. Rates order as PV > SOM > E, with E at 1–3 Hz.
- R4. Oscillations appear above a critical connection probability p (Fig. 1E).

The bands below are ours. The paper gives point values without spread.

## Prior exposure

An unscored pilot ran before this draft: seed 0, 1.5 s, the first 0.5 s discarded. It
gave E 1.16, PV 11.8 and SOM 6.4 Hz, with an E-rate peak at 35.0 Hz. The bands in v1 were
written before that pilot, and v2 keeps them. Seed 0 is excluded from scoring.

## Runs

- Spec: `caminos/PORT_SPEC.md`; code `caminos_jax.py` at 06ed3734 or a later commit
  that changes only documentation or tests. Float32, Euler at dt = 0.1 ms, WSL2 CUDA.
- Connectivity: Bernoulli p = 0.07 with autapses removed. The source does not say; this is
  our choice.
- External drive: 560 independent 4 Hz Poisson sources per target cell. This reads "p·NE
  Poisson neurons" as private sources per cell, which is an assumption. S4 tests the shared
  reading.
- Initial state: V = EL, w = 0, g = 0, the same for every cell.
- Seeds: 1–5 for the network and drive.
- Duration: 12 s. The first 2 s are discarded (4 × τw), so that adaptation settles, and
  10 s are kept (v6; v1–v5 kept 3 s, see the calibration gate).
- Signal for the spectra: the E population rate in 0.1 ms bins, minus its mean.

## Estimands and criteria

### Oscillation criterion (v4)

v3's criterion was peak power divided by the median over 20–80 Hz. The third review showed
it calls a sloped asynchronous spectrum oscillating: P(ratio ≥ 5) was 0.5–0.9 for AR(1)
coloured noise. v4 measures the peak over a fitted aperiodic background instead.

1. **Spectrum.** Welch PSD of the E rate over the kept 10 s: 1 s Hann windows, 50 % overlap.
2. **Background.** Fit an aperiodic model, log10 P = a − log10(k + f^χ), by least squares
   on 10–150 Hz.
3. **Peak.** f_pk is the frequency of the largest residual in 20–80 Hz.
4. **Refit.** Refit the background with f_pk ± 8 Hz excluded.
5. **Peak excess.** PE = P(f_pk) / background(f_pk).
6. **Rule.** A run oscillates when all of these hold:
   - PE ≥ θ, the threshold set by the calibration gate below (v4 used 4);
   - f_pk is not within 2 Hz of 20 or 80 Hz.

   v4 and v5 also required f_pk to be a local maximum of the raw PSD. v6 drops that rule.
   f_pk is the largest residual in the band, so it is already a local maximum of the
   residual. On a broad, noisy peak the raw-PSD test rejected true peaks whose largest residual
   sits on a flank. It was the main cause of the narrowband misses (9 % of draws on the
   τ = 30 ms, 2× floor at true peak excess 8).

**Calibration gate (v6), run before signature on synthetic signals only.** It uses no
network data. Each row has 200 draws of a 10 s E rate at NE = 8000, R0 = 1.2 Hz, dt = 0.1 ms.

- **Null floors (7):** white Poisson; Poisson on a lognormal AR(1) rate with τ = 8 and
  30 ms, at variance 0.5×, 2× and 8× the Poisson count variance.
- **Threshold θ:** the smallest of 4, 5, 6 and 8 for which every null floor has a
  false-positive rate ≤ 5 %. If none qualifies, the fit is revised here before signature.
- **Detection rows (14):** each null floor plus an additive 35 Hz component, in two forms:
  a sinusoid with random phase, and a narrowband AR(2) bump with a 5 Hz full width at
  half maximum.
  - The component is scaled per floor so that the true peak excess at 35 Hz is 2θ. True
    peak excess is (component + floor) / floor in the mean Welch PSD over 200 null draws.
  - Required: detection ≥ 95 %, with f_pk within 2 Hz of 35 Hz.
- **Reported, not gated:** detection against true peak excess at 2, 3, 4, 6 and 8 on
  every floor, for both forms.

v4 set the detection rows at a fixed modulation depth of 0.1. On the AR(1) floors that
gives a true peak excess of only 1.04–2.66, below the threshold of 4, so no criterion could
have detected it. The v4 gate therefore failed by design: 6 of 7 detection rows (detection
0.5–23.5 %) and the τ = 30 ms, 8× null floor (false positives 9.0 %). v5 sets each
detection row by its true peak excess instead.

v5 history (tables in `caminos/calibration/`):
- At 3 s, θ = 5 and 11 of 14 detection rows fell below 95 % (0.805–0.945). A 3 s window gives
  5 Welch segments, so the PSD at one bin is too noisy.
- At 10 s, θ = 4 held 0 % false positives on every floor. All 7 sinusoid rows passed. The 7
  AR(2) rows reached only 0.875–0.915, mostly through the local-maximum rule.

The calibration table goes into this file. The unscored seed-0 pilot (12 s, p = 0.07) is
rescored under the calibrated θ and reported here before signature. The 5 s pilot under v5
gave E 1.04, PV 11.07 and SOM 6.55 Hz, with f_pk 35.0 Hz, PE 26.7 and a lag of 7.6 ms.

| Id | Estimand | Pass |
|---|---|---|
| R1 | f_pk per seed | every seed oscillates, with f_pk in 30–40 Hz |
| R2 | Lag of the PV→SOM rate cross-correlation peak, searched over −14 to +14 ms (under half a 35 Hz period), after 2 ms Gaussian smoothing; mean over seeds | lag positive (SOM after PV) in every seed; mean lag 6–11 ms |
| R3 | Mean rate per population over the kept 10 s | PV > SOM > E in every seed; mean E rate 1–3 Hz |
| R4 | Oscillation at p = 0.02, 0.03, …, 0.10, seeds 1–3 each | no seed oscillates at p = 0.02, and all seeds oscillate at p = 0.07; the smallest p at which all seeds oscillate is reported |

**Completeness.**

- R1–R3 need exactly seeds 1–5 at p = 0.07.
- R4 needs all 9 values of p × seeds 1–3.
- Every run needs T = 12 s, dt = 0.1 ms and one code SHA.
- Anything else scores as "incomplete", which counts as a fail.

When a population fires no spike in a run, its oscillation and lag are undefined. The run
counts as "does not oscillate" and is reported.

## Sensitivity (reported, not scored)

- S1: float64, seed 1. R1–R3 should move by less than the between-seed spread.
- S2: dt = 0.05 ms, seed 1, under the same rule.
- S3: fixed in-degree (pN) instead of Bernoulli.
- S4: shared external sources: 560 Poisson trains, each projecting to E and PV cells with
  probability p.
- S5: the step-order convention. Brian2 applies synaptic arrivals after the state update,
  so its effective delay is one step longer than this code's, and its refractory period is
  one step shorter. S5 runs with a 21-step ring and a 49-step refractory period.

## Outcome language

- All of R1–R4 pass: "The CAMINOS reference model reproduces in jaxfne (stage A) at the
  paper's operating point."
- A failure is reported with its numbers.
- Parameters are not tuned to pass. Any deviation from the spec is a new, separately
  preregistered run.
