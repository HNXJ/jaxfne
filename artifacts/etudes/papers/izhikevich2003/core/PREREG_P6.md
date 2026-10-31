# Izhikevich 2003 P6 preregistration: Fig. 3 through the jaxfne API (draft, UNSIGNED — no scored run may use it)

Status: draft (2026-10-27); seeds, Q5 and the margins wait on Hamm. The P5 verdicts (`PREREG_P3.md`
signed v7, amendment v8.2; `scores/p3_v7_score.json`) are the oracle. A signed P6 that fails leaves
them as they are.

## Question

Does the Fig. 3 network, built only through `Configuration`, `construct` and `Model.simulate`, give
the P5 estimands N1–N4 within a declared margin? Passing closes the E1 finding and moves Fig. 3 to
T-jaxfne (`RESIDUAL_PROTOCOL.md` R4).

## Prior exposure

All P5 results were seen. The unscored library pilot (`core/pilot2_unscored.json`, seeds 0 and 1,
10.2 s) was seen: E and I rates within 1.4 % of the kernel; gamma f_pk 39.5 and 35.0 Hz against the
kernel's 41.5 and 39.5 Hz; N2 and N3 were not computed. Before it, the same network at mechanism tau
5 ms saturated (2000 Hz). Tau moved to the instantaneous limit, the paper's coupling (P-051); nothing
else changed after any outcome.

## Model through the API (`core/run_izh_core.py`)

| P5 element (`PORT_SPEC.md`) | API |
|---|---|
| 800 E then 200 I | `network(kind="balanced_ei_population", n=1000, cell_types={"E": 0.8, "I": 0.2})` |
| Per-cell a, b, c, d from the draw r (`k_r`); v0 −65 mV, u0 = b·v0 | `cell_params` per type with the P5 draw; `runtime(random_v0=False)` |
| All-to-all S with self edges; W = 2S at dt 0.5 ms (E U[0, 1), I −U[0, 2)) | two rules at probability 1, `weight_range`, sign from the presynaptic I cells; `with_rule_allow_self_connections`; `connectivity_mode="explicit"`; `connectivity(edge_seed=s)` |
| Instantaneous coupling (`W @ prev_spikes`) | `edge_list` backend, mechanism tau 1e-3 ms: exp(−dt/tau) is exactly 0 in float32 at dt 0.5 ms |
| Noise 5·randn (E), 2·randn (I), × √2 at dt 0.5 ms | `Simulation(gaussian_drive={"std_by_cell_type": {"E": 5√2, "I": 2√2}})` |
| dt 0.5 ms, T 10.2 s, first 200 ms discarded | `Simulation(dt_ms=0.5, duration_ms=10200)` |

The API draws weights and noise with its own keys, so spike trains differ from P5 at equal seed.
Equivalence is stated on the estimands.

## Before scoring (gates; a failure stops the run)

1. Realization: `tests/test_izh2003_core_path.py` passes at the scoring SHA (cell parameters within
   1e-4 of the P5 draw, noise coefficients within 1e-6, 1e6 edges with 1000 self edges, signed
   weight blocks, receipt counts, zero decay, a seed change moves the spikes).
2. Scorer: a `score` command in `core/run_izh_core.py` computes N1–N4 with the estimators of
   `run_izh.score_network_seed` (θ and alpha band from `calibration/calibration_v7_15f36087.json`).
   Known-answer test: on a kernel raster it returns that seed's P5 numbers exactly.
3. Completeness: one code SHA; all five seeds at 10.2 s; anything else scores "incomplete" (fail).

## Runs

Seeds, Hamm's choice: (a, recommended) 2–6, none run on the library path; (b) 1–5, the P5 seeds,
seed 1 seen in the pilot. CPU, float32.

## Criteria

P5 over seeds 1–5, mean (SD): N1 rate 8.677 Hz (0.144); N2 median CV 0.4028 (0.0043); N3 f_pk
9.10 Hz (0.22), PE 8.01 (2.91); N4 f_pk 38.1 Hz (2.61), PE 5.83 (1.27); θ = 4.

| Id | Estimand, mean over seeds | Pass |
|---|---|---|
| Q1 | N1 mean rate | within 2 SD of P5 (0.29 Hz) |
| Q2 | N2 median ISI CV | within 2 SD of P5 (0.0085) |
| Q3 | N3 f_pk | within 1 Hz of P5 (2 SD is 0.45 Hz, under one 0.5 Hz bin) |
| Q4 | N4 f_pk | within 2 SD of P5 (5.2 Hz) |
| Q5 | band verdicts | option below |

P6 passes when Q1–Q5 pass. A 2 SD margin is 3.2 SD of the difference of two 5-seed means: a
network identical to P5 fails Q1, Q2 or Q4 by seed noise with probability about 0.5 % (normal
approximation).

## Q5: Hamm's choice

P5's own peak excess sits near θ. Under a normal approximation from the P5 seeds, one new seed
falls below θ with probability 0.084 (N3) and 0.075 (N4); "PE ≥ θ in every seed" then fails on
five new seeds with probability 0.36 and 0.32, about 0.56 together, for a network identical to P5.

| Option | Q5 | False-fail |
|---|---|---|
| A | N1–N4 verdicts equal to P5 (N1 pass, N2 fail, N3 pass, N4 pass) | about 56 % |
| B (recommended) | N1 and N2 verdicts equal to P5; N3 and N4 mean PE within 2 SD of P5 (5.8 and 2.5); per-seed band calls reported | under 1 % |

## Outcome language

- Pass: "Izhikevich 2003 Fig. 3 (N1–N4) is reproduced through the jaxfne public API within the
  declared margins (T-jaxfne); N2 stays a failed paper claim."
- Fail: reported with its numbers, criterion by criterion; the P5 verdicts stand.
