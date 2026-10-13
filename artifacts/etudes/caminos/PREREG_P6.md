# L2-P6 preregistration: CAMINOS Fig. 1 through the jaxfne API

Status: **draft v2 for Hamm's signature** (2026-10-13; v1 fbf98698 revised after a second
review). Nothing here is run until it is signed. The P5 verdicts (`PREREG_P3.md` signed v6,
amendment v7; scores on `l2-caminos` 1e768f9d) are the oracle. A signed P6 that fails leaves
the P5 verdicts as they are.

## Question

Does the same network, built only from the jaxfne public API, give the P5 estimands within a
declared margin? Passing moves CAMINOS from tier T-methods to T-jaxfne
(`release_plan_0.5.6-0.5.10.md` §2).

## Prior exposure

All P5 results (v6 and v7 scores) were seen before this draft; the margins below are set from
the P5 seed spread. No API run of this network exists.

## Model through the API

| Port element (`caminos_jax.py`) | API |
|---|---|
| AdEx E, PV, SOM cells (`PORT_SPEC.md` table) | `emitter(family="adex", adex_decay="euler", adex_step_order="default")`; `ADEX_CELL_TYPE_DEFAULTS` |
| NE 8000, NPV 750, NSOM 750 | `network(n=9500, cell_types={"E": 8000/9500, "PV": 750/9500, "SOM": 750/9500})` (exact counts) |
| Bernoulli connectivity at probability p; seven blocks, none PV→SOM or SOM→SOM | seven `connections(..., probability=p, sampling="bernoulli")` rules |
| Conductances Q, synaptic time constants, delay | `mechanisms()` and `connections(weight=, delay_ms=)` |
| Private 4 Hz Poisson drive, round(p·NE) sources per E and PV cell; SOM undriven | `Simulation(poisson_drive={"n_sources": round(p·NE), "rate_hz": 4, "target_cell_types": ["E", "PV"], ...})` |
| Seed s | `connectivity(edge_seed=s)` (edges) and `poisson_drive["seed"]` (drive); `Simulation(seed=)` is refused on this route |
| dt 0.1 ms; T 12 s, first 2 s discarded | `Simulation(dt_ms=0.1, duration_ms=12000)` |

The API draws connectivity and drive with its own keys, so spike trains differ from the port's
at equal seed. Equivalence is stated on the estimands.

## Build prerequisite

B0. The adex route records V at every step (`_model_simulate.py:1455`, `record_v=True`): a 12 s
run holds about 4.6 GB of V plus 1.1 GB of spikes. P6 needs a way to run without recording V
(failing test first, second review) before any scored run. Hamm's choice: (a) a public
`Simulation` field every route consumes or refuses; (b) an adex-route option; (c) the ADVANCED
kernel `jaxfne.emitters.simulate_adex(record_v=False)` on the route's compiled arguments, which
leaves the public surface unchanged.

## Before scoring (gates; a failure stops the run)

1. Parameter identity: a test reads the realized model (mechanism table, cell parameters,
   connection rules, drive) and checks every value in the `PORT_SPEC.md` tables exactly.
2. Realized connectivity at p = 0.07, seed 1: each block's edge count within 4 SD of
   p·n_pre·n_post (binomial), zero autapses, zero PV→SOM and SOM→SOM edges.
3. Kernel identity: the existing single-cell and recurrent-step tests (rtol 1e-6, float64) pass.
4. Converter: API spikes become the port's per-step population count arrays plus the
   `manifest.json` that `caminos_analysis.score` reads (file, seed, p, T, dt, dtype, sizes,
   git_sha, `ext_sources: "pNE"`). Known-answer test: on a synthetic raster the counts equal the
   per-step population sums; `caminos_analysis.py` scores a converted run (no "not scored").
5. Pilot: one unscored full-length (12 s) run at p = 0.07, seed 0: memory and time.

## Runs

- p = 0.07, seeds 1–5 (Q1–Q3, R1–R3); p = 0.02, 0.03, …, 0.10, seeds 1–3, drive p·NE (R4v7).
  The sweep's p = 0.07 seeds 1–3 are the same runs as seeds 1–3 above.
- Drive control C1: p = 0.02, seed 1, with the v6 fixed drive of 560 sources, in its own
  directory, scored with `--ext-sources fixed`. P5 v6 found this network oscillating (peak
  excess 55–98 against threshold 4, f_pk 28–29 Hz); the API run must oscillate too, showing the
  route honours `n_sources`. C1 is read from `per_run[].oscillates` (that scoring prints
  "incomplete" for R1–R4 by design). Gates 1–3 cover decay, step order and parameters.
- WSL2 CUDA, float32, one GPU job at a time after a jchat claim (B5); one code SHA per run.
- Analysis: `caminos_analysis.py` as at `l2-caminos` baca68e7, unchanged.

## Criteria

Primary, at p = 0.07, seeds 1–5. P5 means: f_pk 33.8 Hz, lag 7.04 ms, E 1.106 Hz, PV 11.37 Hz,
SOM 7.05 Hz.

| Id | Estimand | Pass |
|---|---|---|
| Q1 | f_pk, mean over seeds | within the Q1 margin of P5 |
| Q2 | PV→SOM lag, mean over seeds | within the Q2 margin of P5 |
| Q3 | mean rate of E, PV and SOM | each within the Q3 margin of P5 |
| Q4 | verdicts R1, R2, R3 (v6) and R4v7 as v7 defines it (with the E ≥ 0.1 Hz floor at p = 0.02) | each equal to its P5 verdict |
| Q5 | smallest p at which all three seeds oscillate (drive p·NE) | within 0.01 of P5's 0.04 |
| C1 | control run oscillates | yes |

P6 passes when Q1–Q5 and C1 all pass. Completeness: all runs above at T = 12 s; anything else
scores "incomplete" (fail).

## Margins: Hamm's choice

P5 seed SDs at p = 0.07: E 3.0 %, PV 0.6 %, SOM 4.1 % of the mean; lag 0.17 ms; f_pk 0.45 Hz.
False-fail is the chance that an API identical to the port fails Q1–Q3 by seed noise alone
(both means over 5 seeds).

| Option | Q1 | Q2 | Q3 | False-fail |
|---|---|---|---|---|
| A | 1 Hz | 0.5 ms | 5 % | 6.4 % (mostly SOM) |
| C (recommended) | 1 Hz | 2 SD of the P5 seeds (0.34 ms) | 2 SD of the P5 seeds (E 6.1 %, PV 1.2 %, SOM 8.2 %) | about 0.8 % |

For scale: 1 Hz in f_pk is about 0.005 in p along the v7 sweep (1.85 Hz per 0.01).

## Outcome language

- Pass: "CAMINOS Fig. 1 (R1–R3, R4v7) is reproduced through the jaxfne public API within the
  declared margins (T-jaxfne)."
- Fail: reported with its numbers, criterion by criterion; the T-methods verdicts stand and the
  difference is diagnosed through the gates above, one at a time.
