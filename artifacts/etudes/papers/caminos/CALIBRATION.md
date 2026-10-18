# CAMINOS calibration record (RESIDUAL_PROTOCOL.md, Records)

Status:

- R4: reproduced after fidelity fix (v7, signed). Step 2 (fidelity), not a fit.
- PV→SOM lag: **off** under both y values with the ref. 42 phase method, on seeds 1–3 (scored) and 4–8 (held-out). Recorded as a finding, "needs X", with no residual fit (section 10).
- Ruling (Hamm, 2026-10-14, relayed by the coordinator): PLoS CB is primary. The signed 2 ms model and y = 8.5 ms stay (section 10).
- Open fidelity item F1 (section 4): the port's 2 ms synaptic delay differs from the preprint's model. The 0.1 ms sensitivity (section 9) is recorded, not primary. True zero delay waits for core gap G12.

Tree baseline: `3dd02ed060390ecf711e04860a1d6d46e854ef83`. Changed: `artifacts/etudes/caminos/caminos_api.py` (gain keyword; ref. 42 estimator; `--delay-ms` override, default 2 ms; default path unchanged), new `test_caminos_api.py`, new `score_lag_ref42.py`, new `score_delay_sensitivity.py`, new `CALIBRATION.md`, new `docs/etudes/models/caminos.md`. No commit.

Inputs:

- Prereg and results: `PREREG_P3.md` (v6), `PREREG_P3_v7_R4.md` (v7; line 11 gives the Methods count as p·NE), `PREREG_P6.md` (v2), `RESULT_P3.md`, `RESULT_P6.md`.
- Scores: `scores/score.json` (v6), `scores/score_v7.json`, `scores/p6_v2_score.json`.
- Paper: PLoS CB 2026, DOI 10.1371/journal.pcbi.1014378. Local copy `F:\warehouse\tahvili\Tahvili2026_PLoSCB_PMC13286281.xml`, MD5 `0d98cd7384a63ce6c6f2411b8f168985` (matches PORT_SPEC.md). Table 2: τd = 2 ms, τE = 2 ms, τI = 8 ms.
- Ref. 42 method: Tahvili, Vinck, di Volo, bioRxiv 10.1101/2025.02.23.639719v1, Methods "Measuring spike phase differences" and "Population activity". Read by the dispatcher from the bioRxiv full text (ruling 2026-10-13). I could not reach the full text myself (section 3). The paraphrase below follows the ruling.

## 1. Estimand table

e = |z − y| / |y|. Bands per the protocol: exact (e = 0), close (every e ≤ 0.05), off (any e > 0.05). y is the spiking-network value (declared target, ruling 2026-10-13).

Engines are named per column: **core** = jaxfne core kernel `simulate_adex(record_v=False)` through `caminos_api.run` (B0 route c), P6 v2 code `a5539d34`, WSL2 CUDA; **oracle** = étude-local kernel `caminos_jax.py` through `run_caminos.py` (v6 code `e2501025`, v7 code `6d0fd783`).

| Estimand | y (paper) | z | engine of z | P5 (seeds 1–5) | engine of P5 | e | Band |
|---|---|---|---|---|---|---|---|
| Gamma frequency f_pk (Hz) | 35, "approximately 35 Hz" (spiking, Results) | 33.80 | core; `caminos_analysis.py` | 33.8 | oracle | 0.034 | close |
| PV→SOM lag, P3 estimator (ms) | 8.5 (spiking, Results) | 7.22 | core; `caminos_analysis.py` (P3 cross-correlation) | 7.04 | oracle | 0.151 | **off** |
| PV→SOM lag, ref. 42 method (ms) | 8.5 (spiking, Results) | 6.47 (seeds 1–3); 6.32 (seeds 4–8) | core counts; `caminos_api.ref42_phase_delay` | n/a | n/a | 0.239 (seeds 1–3); 0.257 (seeds 4–8) | **off** |
| E rate (Hz) | 1–3 (interval, Results) | 1.079 | core; `caminos_analysis.py` | 1.106 | oracle | n/a (interval) | inside interval |
| PV rate (Hz) | no number; highest of the three (Results) | 11.25 | core; `caminos_analysis.py` | 11.37 | oracle | n/a | order PV > SOM > E holds |
| SOM rate (Hz) | no number; second (Results) | 6.78 | core; `caminos_analysis.py` | 7.05 | oracle | n/a | order holds |
| R4 onset p | qualitative; Fig. 1E only (image not in the local copy) | 0.04 (seeds 1–3 all oscillate) | core; `caminos_analysis.py` | 0.04 (v7) | oracle (`6d0fd783`) | n/a | reproduced after fidelity fix (v7) |

Notes:

- z values from `scores/p6_v2_score.json`: f_pk 33.8 Hz, lag 7.22 ms (P3), E 1.07885 Hz, PV 11.2455 Hz, SOM 6.7818 Hz, onset 0.04.
- P5 means recomputed from the seeds 1–5 rates in `scores/score.json`: E 1.1060, PV 11.366, SOM 7.048. They match the constants in PREREG_P6.md.
- The P3 and ref. 42 lag rows differ in estimator, not in the model. Both are reported. Only the ref. 42 row follows the ruling.
- `caminos_jax.py` is the oracle. Its role is the P5 column, the v6/v7 sweep summaries in `RESULT_P3.md`, and the parity checks. It is not the source of any z value or any gain fit.

## 2. PV→SOM lag by the ref. 42 method

Method (ruling 2026-10-13, paraphrased):

1. LFP proxy: E spikes per 0.1 ms bin, convolved with a Gaussian of σ = 1 ms (the ruling's "1 ms kernel", read as σ). The first 2 s are dropped and 10 s are analysed.
2. Peak frequency f_pk: `caminos_analysis.spectral_peak` on that LFP window. The paper gives no band or peak rule; this choice is ours.
3. Band-pass: zero-phase 4th-order Butterworth at f_pk ± 5 Hz (declared; the paper leaves the filter unstated). Sensitivity at ± 3 and ± 8 Hz.
4. Hilbert transform of the band-passed LFP; phase at each spike.
5. Group phase: circular mean over all spikes of the group, weighted by per-step counts (a circular mean over a group's spikes equals the mean over its per-step counts, so no per-cell raster is needed).
6. Delay = wrap(φ_SOM − φ_PV) / (2π f_pk), in ms. Positive when SOM fires after PV.

Estimator: `artifacts/etudes/caminos/caminos_api.py` (`ref42_lfp`, `ref42_phase_delay`), reading the per-step group counts from the core raster (`spikes_to_counts`). The route-c run returns the raster, not a `Signals` object, so the estimator reads the raster reduction, not a `Signals` instance.

Runs (all at p = 0.07, T = 12 s, dt = 0.1 ms, float32, WSL2 CUDA, one GPU job at a time under `flock /tmp/jaxfne_gpu.lock`):

- Scored seeds 1–3: P6 v2 receipts `F:\warehouse\caminos\p6_scored_v2\` (code `a5539d34`).
- Held-out seeds 4–8: `F:\warehouse\caminos\calib\heldout_p007\` (code `3dd02ed0`, HEAD with the uncommitted gain keyword; default path unchanged).
- Identity check: seed 1 at HEAD in `calib\identity_p007\`; seeds 1, 4 and 5 at HEAD equal the receipts element for element (`numpy.array_equal`: all True). This shows the code SHA change left the counts unchanged for these seeds.

Results (declared band ± 5 Hz; f_pk from the LFP window):

| Seed | f_pk (Hz) | delay (ms) |
|---|---|---|
| 1 (scored) | 34 | 6.247 |
| 2 (scored) | 34 | 6.499 |
| 3 (scored) | 34 | 6.668 |
| 4 (held-out) | 33 | 6.700 |
| 5 (held-out) | 34 | 6.411 |
| 6 (held-out) | 34 | 5.912 |
| 7 (held-out) | 34 | 6.109 |
| 8 (held-out) | 35 | 6.459 |

| Set | Band | Mean (ms) | e vs 8.5 | band | e vs 7 (context) | band |
|---|---|---|---|---|---|---|
| Seeds 1–3 (scored) | ± 5 Hz (declared) | 6.471 | 0.239 | off | 0.076 | off |
| Seeds 1–3 | ± 3 Hz | 6.449 | 0.241 | off | 0.079 | off |
| Seeds 1–3 | ± 8 Hz | 6.622 | 0.221 | off | 0.054 | off |
| Seeds 4–8 (held-out) | ± 5 Hz (declared) | 6.318 | 0.257 | off | 0.097 | off |
| Seeds 4–8 | ± 3 Hz | 6.306 | 0.258 | off | 0.099 | off |
| Seeds 4–8 | ± 8 Hz | 6.483 | 0.237 | off | 0.074 | off |

- The lag is off under the declared band and both sensitivity bands, under both y values, on both seed sets.
- Phase: 6.47 ms at 34 Hz is about 79° of the 34 Hz cycle. The preprint's Results give "7 ms (80°)" for its spiking model (dispatcher reading). This agreement is an observation, not a test.
- Source: `F:\warehouse\caminos\calib\lag_ref42_p007.json` (per-run values, identity, summaries, declared parameters, code SHAs). Driver: `artifacts/etudes/caminos/score_lag_ref42.py`.

## 3. Ref. 42 access

The ruling says the dispatcher read the method from the bioRxiv full text. My own attempts failed:

- `doi.org` and `cell.com` for 10.1016/j.celrep.2025.116131: HTTP 403 "Just a moment..." (bot check), then a redirect page on the retry.
- Crossref: metadata only.
- Europe PMC: PMID 40783942, `isOpenAccess: N`, no PMCID.
- Europe PMC preprint PPR982122 (bioRxiv 10.1101/2025.02.23.639719): `isOpenAccess: N`; `fullTextXML` returned HTTP 500 on two tries.

## 4. Open fidelity item F1 (recorded, not changed)

- Port: a 2 ms synaptic delay on all synapses (PLoS CB Table 2, τd = 2 ms; stated in the text as "A synaptic delay of 2 ms was included for all synapses").
- Preprint (dispatcher reading): its model has no synaptic delay, and its spiking lag is about 7 ms (80°).
- Recorded as an open fidelity item, per the ruling. Not changed. A change would move a stated parameter.

## 5. Resolved and open by the rulings

| # | Item | Status |
|---|---|---|
| B1 | Optimizer | Spikes are non-differentiable, no surrogate is declared, so any residual fit uses `jaxfne.optim.agsdr`. No fit ran. |
| B2 | R4 target and fit | No a-fit for R4; no alternative X. |
| B3 | v7 p·NE drive | Fidelity fix (protocol step 2). Tests: `test_caminos_jax.py` lines 411–415; `test_caminos_api.py::test_no_gain_keeps_the_v7_drive`, `::test_gain_one_is_the_v7_drive`, `::test_core_realizes_the_gain_as_its_poisson_mean[None-640.0]`. |
| B4 | Lag y and estimator | y = 8.5 ms (declared). Estimator: ref. 42 method implemented and run. Lag off under both y. |
| B5 | Acceptance | Section 8: `71 passed in 316.69s (0:05:16)`, rc=0 (WSL, CPU). Earlier `65 passed in 688.89s (0:11:28)`. |
| B6 | Alternative X | None declared. |
| L1 | Ref. 42 method | Resolved by the dispatcher's reading (section 2). |
| L2 | Lag off after the estimator | Stop and report (this record). |
| L3 | Residual X for lag | Not run. The ruling's examples (the SOM-afferent delay, the SOM GABA τ) are both stated in Table 2 (τd, τI), so a residual at either would move a stated parameter. No other place was declared. |
| L4 | Lag y | 8.5 declared; 7 as context. Both reported in section 2. |

## 6. Steps taken

1. Simplify: not run.
2. Fidelity: v7 drive (R4) recorded as a fidelity fix. Lag: ref. 42 estimator applied, lag off (section 2). Open item F1 recorded (section 4).
3. Residual X for lag: not run (stop rule, section 5 L3). The gain realization exists (`drive_spec(..., gain=a)`, tested on CPU) and is not used.
4. Absorb a into θ: not run. D = ∅.
5. Remove X and re-run on held-out seeds 4–8: not run. No fitted value exists; the held-out lag in section 2 is an estimator result, not a fit.
6. Smallest |a|: n/a.

## 7. Claim

- R4: reproduced after fidelity fix (v7, signed), per the 2026-10-13 ruling.
- PV→SOM lag: off against 8.5 ms (e 0.239, seeds 1–3; e 0.257, seeds 4–8) and against 7 ms (e 0.076; e 0.097). No calibrated-reproduction claim for lag.

## 8. Run records and acceptance

- Tests: `test_caminos_api.py` 15 passed (WSL CPU, `JAX_PLATFORMS=cpu`) after the ref. 42 estimator was added (4 estimator tests failed before the change, `AttributeError`; the summary test failed before `score_lag_ref42.py` existed).
- Acceptance (three files, WSL CPU, `_temp_l3_accept.sh`, under `suite_lock`): `71 passed in 316.69s (0:05:16)`, rc=0 (`test_caminos_api.py` 21, `test_caminos_analysis.py` 48, `tests/test_script_repo_root_guard.py` 2). The run before the delay override and the comment edits gave `65 passed in 688.89s (0:11:28)`.
- GPU: seed 1 (identity) and seeds 4–8 at p = 0.07, T = 12 s, HEAD, WSL CUDA, one job at a time under `flock`. Every run returned rc=0. Manifests: `F:\warehouse\caminos\calib\identity_p007\`, `F:\warehouse\caminos\calib\heldout_p007\`. Scratch smoke run (T = 1 s): `scratchpad\smoke_T1` (outside the tree).
- Re-score: `F:\warehouse\caminos\calib\lag_ref42_p007.json`.
- Existing receipts used read-only: `F:\warehouse\caminos\p6_scored_v2\`, `F:\warehouse\caminos\p6_scored_v2_c1\`, `F:\warehouse\caminos\scored\v7_20261012\`.

## 9. Sensitivity: zero synaptic delay (preprint)

Status: **run as declared option (a): 0.1 ms, one step.** True zero delay is a core gap, G12: the AdEx route refuses 0 steps (`jaxfne/_construct_adex.py`, `_rule_delay_steps`). G12 is being closed in core separately, and true zero delay runs once it closes. The primary model is unchanged (2 ms, `DELAY_MS`).

Declared sensitivity: synaptic delay 0.1 ms on all seven connection rules, and drive delay 0.1 ms. The v7 drive was 2 ms, so it also goes to one step. Everything else matches the primary run: p = 0.07, T = 12 s, float32, seeds 1–8, one GPU run per `flock`. Code: `caminos_api.py --delay-ms` (uncommitted; the default path is unchanged and tested).

Requested: the preprint's model with delay 0 instead of 2 ms on all synapses, everything else at v6/v7 values, through `caminos_api` (route c). Seeds 1–3 and 4–8, p = 0.07, T = 12 s, under `flock`. Scoring: `ref42_phase_delay` (±5 Hz declared; ±3 and ±8 Hz), the P3 estimator, f_pk, E/PV/SOM rates, and the oscillation call.

Probe (CPU, `scratchpad\_temp_l3_delay0.py`, outside the tree), with the module delay `DELAY_MS = 0.0` that `build_spec` and `drive_spec` read:

- `construct` at p = 0.02, seed 1: OK. The config layer accepts delay ≥ 0.
- Connections (`adex_kernel_args`): refused. `connections(name='e_e') delay_ms=0.0 at dt_ms=0.1 is 0 steps; the adex route needs >= 1`.
- External drive (`adex_external_drive_kwargs`): refused. `delay_ms=0.0 at dt_ms=0.1 is 0 steps; the drive needs >= 1`.

Source of the refusals: `jaxfne/_construct_adex.py` (`_rule_delay_steps`, called from `adex_kernel_args`; `_drive_error`). `jaxfne/_config.py` accepts delay ≥ 0 at configuration time, so the refusal comes at compile time.

Option (a) was run, as the dispatcher instructed. Option (b), allowing zero delay in the core adex route, is a core change, closed separately and not done here.

Both y values, for any future delay run: 8.5 ms (PLoS CB Results, the declared target; read from the XML) and about 7 ms (preprint Results, dispatcher reading, not checked against the preprint by this lane). The PLoS CB text on the mean-field model says SOM cells fire "6–7 ms later" than E and PV (XML, read here); it is not a spiking-lag value and is not used as y.

Estimator declarations confirmed by the dispatcher (S2): the filter, the 1 ms Gaussian read as σ, and the f_pk choice stand. Reading counts is accepted.

### Results (observed; `score_delay_sensitivity.py`; `F:\warehouse\caminos\calib\sens_delay0p1_score.json`)

Both lags are shown against y = 8.5 ms (PLoS CB) and y = 7 ms (preprint, dispatcher reading). Each cell gives the mean, then e against 8.5 and against 7. Band: close when e ≤ 0.05.

| Run | Seeds | ref42 ±5 Hz | ref42 ±3 Hz | ref42 ±8 Hz | P3 | f_pk E (Hz, per seed) | E / PV / SOM (Hz, mean) | Oscillates |
|---|---|---|---|---|---|---|---|---|
| 2 ms (primary) | 1–3 | 6.47 (0.239 / 0.076) | 6.45 (0.241 / 0.079) | 6.62 (0.221 / 0.054) | 7.20 (0.153 / 0.029) | 34, 34, 34 | 1.08 / 11.24 / 6.74 | 3/3 |
| 0.1 ms (sensitivity) | 1–3 | 6.97 (0.180 / 0.004) | 6.95 (0.182 / 0.007) | 7.04 (0.172 / 0.005) | 7.10 (0.165 / 0.014) | 46, 46, 44 | 0.94 / 9.85 / 4.54 | 3/3 |
| 2 ms (primary) | 4–8 | 6.32 (0.257 / 0.097) | 6.31 (0.258 / 0.099) | 6.48 (0.237 / 0.074) | 7.12 (0.162 / 0.017) | 33, 34, 34, 34, 35 | 1.09 / 11.31 / 6.90 | 5/5 |
| 0.1 ms (sensitivity) | 4–8 | 7.07 (0.169 / 0.010) | 7.05 (0.170 / 0.007) | 7.09 (0.165 / 0.014) | 6.96 (0.181 / 0.006) | 46, 44, 44, 46, 43 | 0.95 / 9.84 / 4.60 | 5/5 |

- The E-rate f_pk and the LFP f_pk agree on every seed in both runs.
- Oscillation (`analyze_run`, PE ≥ 4, f_pk off the band edges): yes on all eight seeds in both runs.
- Band call against y = 8.5: off for every estimator and band. Against y = 7: close for the 0.1 ms run on every estimator and band, and off for the primary ref42 bands.
- The P3 lag is 7.1 ms on all three sensitivity seeds 1–3 (0.1 ms grid; the primary seeds 1–3 are 7.1, 7.2, 7.3). Seeds 4–8 range 6.9–7.1 ms.
- Reading: the 0.1 ms run moves the oscillation from about 34 Hz to 43–46 Hz, so it is a different network state. Its closeness to 7 ms is an observation. It is not a calibration, the primary model stays at 2 ms, and y = 7 is the dispatcher's preprint reading, not checked by this lane.

Receipts: `F:\warehouse\caminos\calib\sens_delay0p1_p007\` (seeds 1–8; the manifest records `delay_ms` 0.1 per run; git SHA 3dd02ed0, which is HEAD, so the `--delay-ms` code is uncommitted over that SHA). Sweep log: `scratchpad\_temp_l3_sens.log` (rc=0 on seeds 1–8, ALL_DONE). The unit tests (`test_caminos_api.py`, 21 passed on CPU) check the default 2 ms (20 steps on connections and drive), the 0.1 ms override (1 step), and that zero delay is refused by the core.

## 10. Ruling and final verdict (2026-10-14)

Ruling: Hamm, 2026-10-14, relayed to this lane by the coordinator.

1. PLoS CB is primary. The signed 2 ms model and y = 8.5 ms (PLoS CB Results, "around 8.5 ms") stay.
2. The lag miss is a finding: the ref. 42 lag is 6.3–6.5 ms (e 0.24–0.26 against 8.5 ms). It is reported as "needs X". No residual fit: both candidate X move stated parameters (section 5, L3).
3. The preprint no-delay model is a recorded sensitivity (section 9). At 0.1 ms it reproduces the preprint's about-7 ms lag (e ≤ 0.014), with f_pk at 43–46 Hz.
4. True zero delay will run once core gap G12 closes.
5. `caminos_api.py --delay-ms` keeps its default at 2 ms (`None` resolves to `DELAY_MS` = 2.0 in `drive_spec`, `build_spec` and the manifest).

Recorded discrepancy: the coordinator's relay gave f_pk 44–46 Hz. The scored JSON (`sens_delay0p1_score.json`) gives 43–46 Hz (seed 8 at 43 Hz). This record uses 43–46.

Final verdict (primary 2 ms model; z from the core route):

| Estimand | y | z | e | Band | Verdict |
|---|---|---|---|---|---|
| Gamma f_pk (Hz), seeds 1–5 | ≈ 35 (spiking, Results) | 33.80 | 0.034 | close | reproduced |
| E rate (Hz) | 1–3 (interval) | 1.079 | n/a | inside | reproduced |
| PV, SOM rate order (Hz) | PV > SOM > E | 11.25 / 6.78 / 1.08 | n/a | order holds | reproduced |
| R4 onset p | qualitative (Fig. 1E) | 0.04 | n/a | reproduced after fidelity fix (v7) | reproduced |
| PV→SOM lag, ref. 42, seeds 1–3 | 8.5 | 6.47 | 0.239 | off | finding: needs X |
| PV→SOM lag, ref. 42, seeds 4–8 | 8.5 | 6.32 | 0.257 | off | finding: needs X |
| PV→SOM lag, P3 estimator, seeds 1–5 | 8.5 | 7.22 | 0.151 | off | finding: needs X |

Recorded sensitivity (not the primary verdict; section 9):

| Estimand | y (context) | z at 0.1 ms | e | Band | Verdict |
|---|---|---|---|---|---|
| PV→SOM lag, ref. 42 ±5 Hz (seeds 1–3; 4–8) and P3 (1–3; 4–8) | ≈ 7 (preprint, dispatcher reading) | 6.97; 7.07 (ref. 42). 7.10; 6.96 (P3) | 0.004–0.014 | close | reproduces the preprint's lag; not primary; f_pk 43–46 Hz |

Corrections made in this step: the docs page said both estimators were off "at every band tested", but P3 has no band (now corrected). The docs labelled "≈ 7" as a PLoS CB mean-field value; the XML says "6–7 ms later" for SOM in the mean-field model, not "≈ 7" (label removed).

No code changed in this step. Acceptance remains section 8: `71 passed in 316.69s (0:05:16)`, rc=0.

## 11. True zero delay (ruling point 4; 2026-10-18)

Core gap G12 closed on dev in T-22: zero-delay edges as a declared intra-step fixed point
(`connections(..., intra_step_rounds=R)`: round 0 is the ordinary step, then R rounds in which
newly spiked cells add their delay-0 conductance and every cell not yet spiked is re-stepped from
the pre-step state; spikes are kept once fired), and a same-step Poisson drive
(`poisson_drive={..., "same_step": True}`: the count drawn for step t acts at step t).

`caminos_api.py`: a delay that rounds to 0 steps at dt 0.1 ms sets `intra_step_rounds` on all seven
connection rules and `same_step` on the drive, and the manifest records `intra_step_rounds` next to
`delay_ms`. The 2 ms default and the 0.1 ms sensitivity take the old path unchanged; a rounds
override with a delay of 1 step or more is refused.

R (observed, CPU, `scratchpad/cam_probe_R.py`; p = 0.07, seed 1, T = 0.5 s, float32, delay 0):

| R | E / PV / SOM spikes | raster sha256 | wall (s) |
|---|---|---|---|
| 1 | 4029 / 3490 / 3880 | bc1f3dbb9afe… | 114 |
| 2 | 3885 / 3361 / 3757 | d9d0462b0d58… | 189 |
| 4, 8, 16, 32 | 3885 / 3361 / 3757 | d9d0462b0d58… | 312, 548, 1004, 1920 |

R = 2 is the smallest R whose raster equals every larger R byte for byte, so `ZERO_DELAY_ROUNDS = 2`.
The kernel runs all R rounds every step, so cost grows linearly with R. The probe covers 0.5 s of
one seed; the scored run repeats seed 1 at R = 4 over the full 12 s, and a raster difference there
means R = 2 is not converged.

Runs (queued behind the GPU lock): p = 0.07, T = 12 s, float32, seeds 1–8, delay 0, R = 2, into
`F:\warehouse\caminos\calib\sens_delay0_p007\`; seed 1 at R = 4 into `sens_delay0_R4_p007\`.
Scoring as section 9 (`score_delay_sensitivity.py`).
