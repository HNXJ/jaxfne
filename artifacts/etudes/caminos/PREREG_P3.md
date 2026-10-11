# L2-P3 preregistration: CAMINOS reference model (PLoS CB 2026, Fig. 1)

Status: draft v4 for Hamm's batch sign-off. It revises v1 (PASS with notes), v2 (FAIL:
the R1 null was vacuous) and v3 (FAIL: the ratio criterion fails on sloped spectra; R4 was
asymmetric; partial runs were accepted), after the reviews of 2026-10-11. No scored run before the signature. G0 single-cell checks
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
- Duration: 5 s. The first 2 s are discarded (4 × τw), so that adaptation settles.
- Signal for the spectra: the E population rate in 0.1 ms bins, minus its mean.

## Estimands and criteria

### Oscillation criterion (v4)

v3's criterion was peak power divided by the median over 20–80 Hz. The third review showed
it calls a sloped asynchronous spectrum oscillating: P(ratio ≥ 5) was 0.5–0.9 for AR(1)
coloured noise. v4 measures the peak over a fitted aperiodic background instead.

1. **Spectrum.** Welch PSD of the E rate over the kept 3 s: 1 s Hann windows, 50 % overlap.
2. **Background.** Fit an aperiodic model, log10 P = a − log10(k + f^χ), by least squares
   on 10–150 Hz.
3. **Peak.** f_pk is the frequency of the largest residual in 20–80 Hz.
4. **Refit.** Refit the background with f_pk ± 8 Hz excluded.
5. **Peak excess.** PE = P(f_pk) / background(f_pk).
6. **Rule.** A run oscillates when all of these hold:
   - PE ≥ 4;
   - f_pk is a local maximum of the PSD;
   - f_pk is not within 2 Hz of 20 or 80 Hz.

**Calibration gate, run before signature on synthetic signals only.** The criterion stands
when all of these hold:

| Synthetic signal (3 s, NE = 8000) | Required |
|---|---|
| White Poisson rate | false-positive rate ≤ 5 % |
| Poisson plus AR(1) fluctuations, τ = 8 and 30 ms, variance 0.5×, 2× and 8× the Poisson variance | false-positive rate ≤ 5 % |
| 35 Hz rate modulation at depth 0.1 on the white and AR(1) floors | detection ≥ 95 % |

The calibration table goes into this file. If the gate fails, the threshold or the fit is
revised here, with the table, before the signature. The calibration uses no network data.

The unscored seed-0 pilot is rescored under v4 and reported here before signature.

| Id | Estimand | Pass |
|---|---|---|
| R1 | f_pk per seed | every seed oscillates, with f_pk in 30–40 Hz |
| R2 | Lag of the PV→SOM rate cross-correlation peak, searched over −14 to +14 ms (under half a 35 Hz period), after 2 ms Gaussian smoothing; mean over seeds | lag positive (SOM after PV) in every seed; mean lag 6–11 ms |
| R3 | Mean rate per population over the kept 3 s | PV > SOM > E in every seed; mean E rate 1–3 Hz |
| R4 | Oscillation at p = 0.02, 0.03, …, 0.10, seeds 1–3 each | no seed oscillates at p = 0.02, and all seeds oscillate at p = 0.07; the smallest p at which all seeds oscillate is reported |

**Completeness.**

- R1–R3 need exactly seeds 1–5 at p = 0.07.
- R4 needs all 9 values of p × seeds 1–3.
- Every run needs T = 5 s, dt = 0.1 ms and one code SHA.
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
