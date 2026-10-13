# L2-P6 preregistration: CAMINOS Fig. 1 through the jaxfne API

Status: **draft for Hamm's signature** (2026-10-13). Nothing here is run until it is signed.
The P5 verdicts (`PREREG_P3.md` signed v6, amendment v7; scores on `l2-caminos` 1e768f9d)
are the oracle. A signed P6 that fails leaves the P5 verdicts as they are.

## Question

Does the same network, built only from the jaxfne public API, give the P5 estimands within a
declared margin? Passing moves CAMINOS from tier T-methods to T-jaxfne
(`release_plan_0.5.6-0.5.10.md` §2).

## Model through the API

| Port element (`caminos_jax.py`) | API |
|---|---|
| AdEx E, PV, SOM cells (`PORT_SPEC.md` table) | `emitter(family="adex")`; `ADEX_CELL_TYPE_DEFAULTS` |
| Bernoulli connectivity at probability p | `connections(..., probability=p, sampling="bernoulli")` |
| Conductances Q, synaptic time constants, delay | `mechanisms()` and `connections(weight=, delay_ms=)` |
| Private 4 Hz Poisson drive, round(p·NE) sources per E and PV cell | `Simulation(poisson_drive=)` |
| NE 8000, NPV 750, NSOM 750; dt 0.1 ms; T 12 s, first 2 s discarded | `network(n=, cell_types=)`, `Simulation` |

The API draws connectivity and drive with its own keys, so spike trains differ from the port's
at equal seed. Equivalence is stated on the estimands.

## Before scoring (gates; a failure stops the run)

1. Parameter identity: a test reads the realized model (connection table, mechanism table, cell
   parameters, drive counts) and checks every value in the `PORT_SPEC.md` tables exactly.
2. Kernel identity: the existing single-cell and recurrent-step tests (rtol 1e-6, float64) pass.
3. Count format: the API spike output converts to the port's per-step population count arrays;
   the converter run on a port simulation reproduces its `.npy` element for element.
4. Pilot: one unscored 1 s run at p = 0.07, seed 0, to measure memory and time.

## Runs

- p = 0.07, seeds 1–5 (R1–R3); p = 0.02, 0.03, …, 0.10, seeds 1–3, drive p·NE (R4v7).
- WSL2 CUDA, float32, one GPU job at a time after a jchat claim (B5); one code SHA recorded per run.
- Analysis: `caminos_analysis.py` as at `l2-caminos` baca68e7, unchanged.

## Criteria

Primary, at p = 0.07, seeds 1–5. P5 means: f_pk 33.8 Hz, lag 7.04 ms, E 1.106 Hz, PV 11.37 Hz,
SOM 7.05 Hz.

| Id | Estimand | Pass |
|---|---|---|
| Q1 | f_pk, mean over seeds | within 1 Hz of P5 (the spectral bin) |
| Q2 | PV→SOM lag, mean over seeds | within 0.5 ms of P5 |
| Q3 | mean rate of E, PV and SOM | each within 5 % of P5 |
| Q4 | verdicts R1, R2, R3 (v6) and R4v7 | each equal to its P5 verdict |

P6 passes when Q1–Q4 all pass. Completeness: all runs above at T = 12 s; anything else scores
"incomplete" (fail).

## Margins: Hamm's choice

The margins are proposals. Seed-to-seed spread in P5 at p = 0.07: f_pk 33–34 Hz, lag 6.9–7.3 ms,
E 1.05–1.14 Hz, PV 11.26–11.42 Hz, SOM 6.60–7.32 Hz (SD 3.0 % of the mean for E, 4.1 % for SOM, 0.6 % for PV).

| Option | Q1 | Q2 | Q3 |
|---|---|---|---|
| A (proposed) | 1 Hz | 0.5 ms | 5 % |
| B (tighter) | 1 Hz | 0.3 ms | 3 % |
| C (seed-spread based) | 1 Hz | 2 SD of P5 seeds | 2 SD of P5 seeds |

R3's band puts E at 1–3 Hz and P5 sits at 1.05–1.14 Hz, so a 5 % lower E rate can turn R3's
verdict while Q3 passes; Q4 reports that case as a verdict mismatch.

## Outcome language

- Pass: "CAMINOS Fig. 1 (R1–R3, R4v7) is reproduced through the jaxfne public API within the
  declared margins (T-jaxfne)."
- Fail: reported with its numbers, criterion by criterion; the T-methods verdicts stand and the
  difference is diagnosed through the gates above, one at a time.
