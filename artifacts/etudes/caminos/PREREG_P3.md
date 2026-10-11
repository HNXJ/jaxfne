# L2-P3 preregistration: CAMINOS reference model (PLoS CB 2026, Fig. 1)

Status: draft v3 for Hamm's batch sign-off. It revises v1 (PASS with notes) and v2
(FAIL: the R1 null was vacuous) after the second reviews of 2026-10-11. No scored run before the signature. G0 single-cell checks
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

The spectral peak ratio is the Welch power at the peak in 20–80 Hz divided by the median
power in 20–80 Hz. Welch uses 1 s Hann windows with 50 % overlap over the 3 s kept.

A run "oscillates" when its peak ratio is at least 5. The second review measured about
2.6 for an asynchronous Poisson rate signal under the same Welch settings, so 5 is
roughly twice the asynchronous level. A phase-randomised surrogate null was dropped in
v3: it keeps the amplitude spectrum, so it cannot test a spectral peak.

| Id | Estimand | Pass |
|---|---|---|
| R1 | Peak frequency in 20–80 Hz, mean over seeds | 30–40 Hz, and every seed oscillates |
| R2 | Lag of the PV→SOM rate cross-correlation peak, searched over −14 to +14 ms (under half a 35 Hz period), after 2 ms Gaussian smoothing; mean over seeds | lag positive (SOM after PV) in every seed; mean lag 6–11 ms |
| R3 | Mean rate per population over the kept 3 s | PV > SOM > E in every seed; mean E rate 1–3 Hz |
| R4 | Peak ratio at p = 0.02, 0.03, …, 0.10, seeds 1–3 each. A p value "oscillates" when every seed oscillates | p = 0.02 does not oscillate and p = 0.07 does; the smallest oscillating p is reported |

When a population fires no spike in a run, its ratio and its lags are undefined. The run
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
