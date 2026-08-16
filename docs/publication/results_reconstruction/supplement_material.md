# Supplement Material — Methods (pre-authorization content, 2026-08-16)

Supplement-authority phase flag: `SUPPLEMENT_AUTHORIZED = YES` (set with the Methods seal,
`dev@e2f57ad`). This file holds the four named non-blocking items queued during the Phase-5
adversarial Methods review (`artifacts/developer/phase2_review/09_methods_adversarial_review.md`,
items M1.3, M3.2, M4.4, M4.2). Every numeric entry below was re-derived from the executed
receipts or from a byte-faithful replay of the estimator executor on 2026-08-16.

All values follow the manuscript discipline: time in ms, frequency in Hz, positions in mm
(arc-length convention), velocities in derived units; every magnitude not declared Absolute is
Relative. No p-values, no calibrated biological magnitudes.

---

## S.1 — RFFT bin spacing and frequency-quantization table (review items M4.4, M1.5)

The estimator's frequency step (`estimate_traveling_wave`) takes the argmax of the summed-site
rfft power over the preregistered band [8.0, 13.0] Hz; with the rectangular (unsmoothed)
analysis window the resolvable grid is fs/N = 2000 Hz / (duration_ms · 2 000 Hz/s · 0.001 s/ms).
A generated frequency that is an integer multiple of the bin spacing is recovered exactly;
an off-grid frequency is reported at the neighboring in-band bin.

| Duration (ms) | Samples N | Bin spacing (Hz) | A-2 S3 drive (Hz) | On-grid? | Stored result (f̂) | Stored ε_f |
|---|---|---|---|---|---|---|
| 250 | 500 | 4.0 | 10.0 | no (10/4 = 2.5) | 12.0 | 2.0 |
| 500 | 1,000 | 2.0 | 10.0 | yes (10/2 = 5) | 10.0 | 0.0 |
| 1,000 | 2,000 | 1.0 | 10.0 | yes | 10.0 | 0.0 |
| 2,000 | 4,000 | 0.5 | 10.0 | yes (10/0.5 = 20) | 10.0 | 0.0 |

Rows are the `stage_S3` cases of `p2v_a2_receipt.json` (n_sites 24/12/6 per duration all
identical; the S3 lattice is 4 durations × 3 samplings = 12 cases, all TRAVELING_WAVE).
Remarks:

- The S3 record's `frequency_hz` field stores the **estimate**; ground truth is 10.0 Hz in all
  12 cases (executor `scripts/p2v_a2_sensitivity_floor.py`).
- The 250 ms / 4.0 Hz spacing is the only grid on which 10.0 Hz is off-grid; ε_f = 2.0 there is
  structural quantization of the rectangular-window argmax, not estimator noise or a band-pass
  artifact. A byte-faithful replay of the exact executor code path (same Butterworth order-4
  band [8/1000, 13/1000] of Nyquist 1,000 Hz, filtfilt, summed-site rfft power, inclusive band
  mask) reproduced all 12 stored (f̂, ε_f) pairs exactly on 2026-08-16.
- A-1a's 48 positive controls use 2000 ms (0.5 Hz grid); the generated frequencies
  {8.5, 10.0, 12.5} Hz map to integer bins (17, 20, 25), so every A-1a ε_f = 0 as recorded.
- The A-2 S1 lattice also uses 2000 ms and records ε_f = 0 for all 40 cases.
- Consequence for the classification gates: the S3 stage's purpose is lane/cutoff set-pointing,
  and all 12 cases classify TRAVELING_WAVE at every duration; the 250 ms ε_f = 2.0 limit is
  reported, not hidden, and no Results claim quantifies frequency error below the S3 floor.

## S.2 — Units and magnitude convention table (review item M1.3)

| Quantity | Unit / convention | Status | Basis |
|---|---|---|---|
| Simulation time step dt | 0.5 ms (H4: 1.0 ms, declared inline) | Absolute | all executors |
| Duration T | ms | Absolute | specs |
| Ring radius R | 1.0 mm; contact arc a = 2πR/N | Absolute (geometry normalized to R = 1.0) | C3 spec |
| Positions x_i | mm along the arc | Absolute (convention) | C3 recording |
| Wave frequency, f̂ | Hz | Absolute (convention); ε_f floor 0.5 Hz at 2000 ms, 4 Hz at 250 ms | S.1 |
| Wave vector k̂ | rad/mm along the arc | Absolute (convention); sign convention k̂_raw = −k_true documented | estimator |
| Phase velocity v̂ | derived mm/s from ω̂/|k̂| | Relative (magnitude discipline), Absolute (unit arithmetic) | estimate_traveling_wave |
| Membrane voltage V_m | mV (Izhikevich-native units) | Relative (uncalibrated against biological mV) | all runs |
| Probe fields (LFP/CSD/EEG/MEG/EMM proxies) | Relative units | Relative | fields/proxy.py, probes.py |
| Drive amplitudes (e.g. 50 µA, 45 µA; A = 1.0 in A-1a/A-2 fields) | Relative drive units | Relative | protocol specs |
| Noise σ | Relative field units (σ_n = 0.0 across protocols) | Relative | specs |
| Coherence, R², bootstrap coverage | dimensionless [0, 1] | Absolute (mathematical) | estimator gates |
| HDP: H, W, τ_HK, κ_H, κ_W, λ_W | model-native Relative units; clamps H ∈ [0.1, 10], W ∈ [0.01, 10] | Relative | emitters, HDP presets |
| Synaptic state bound | |syn_state| ≤ 1e4 per edge | Relative | emitters `_bound_state` |
| E-matrix decoding (H4) | ridge-regularized identity decoding weights | Relative | h4_matrix |
| E5 contrast magnitudes (owner/non-owner differences) | Relative mV-magnitude differences | Relative | e5 interpretation receipt |

## S.3 — RNG usage note (review item M3.2)

- One `PRNGKey` per executed seed (protocols execute seeds 1001–1010, 11–13, 100–109, 200–209,
  and single-seed runs as declared). Within each run the simulator splits `key, noise_key =
  jax.random.split(key)` once per simulation and draws a single bulk Gaussian array of shape
  (n_steps, N_neurons) at scan start; the `scan` then indexes `bulk_noise[n]` at step n. There is
  no per-step re-keying.
- With σ_n = 0.0 (all protocols) the bulk draw is multiplied by zero; runs are deterministic for
  a given seed and reproduce bit-identically across machines under the pinned stack (S.4).
- The A-2 S2 stage draws no noise (σ_n = 0.0) and replays frozen C3 cells; the frozen reason
  strings of the carrier cells are preserved in the A-2 receipt.
- float32 is the JAX default; float64 is declared in the estimator spectral arithmetic and in
  the A-2 S2/S3 estimator input path (spec `note`).

## S.4 — C0-registered versus implemented estimator decision branches (review item M4.2)

The preregistered C0 rule set names two UNRESOLVED triggers; the implemented decision tree
(`jaxfne/protocol_c/estimator.py`, SHA 684859a…) registers a strict subset that is a superset
in practice for every executed input. The manuscript's §7 disclosure stands; this table is the
full reconciliation.

| Trigger | C0 registration (frozen doc) | Implementation (code) | Executed by any condition? | Consequence |
|---|---|---|---|---|
| Non-finite spectral/phasor input | — | `finite_status` gate → UNRESOLVED | No (all executed fields finite) | Latent guard; exercised in unit suite only |
| Fewer than 4 qualifying band sites | — | site-count gate → UNRESOLVED | No | Latent guard |
| R² < 0.35 AND coherence < 0.55 | UNRESOLVED (registered) | not a distinct branch: the output is NO_WAVE via `structured_but_fails` when gates fail | No C-series, A-series, D3, or W3b input produced it | Registration-only condition; never policy-determinative |

Basis: C0 spec (waves appendix), estimator source, and the executed receipts (C3 60/60,
A-1a 53, A-2 S1 40 + S2 3×γ-lattices + S3 12, A-1b 45, D3, E5, W3b). No Results claim routes
through the third row.

---

End of Supplement pre-material. Revision control: this file is tracked with the Methods seal
commit lineage; any later rearrangement belongs to final Supplement assembly, which is
explicitly not begun.