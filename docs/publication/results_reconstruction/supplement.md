# Supplement — final assembly (consolidation draft, 2026-08-16)

**Authority phase:** `SUPPLEMENT_AUTHORIZED = YES` (Methods seal, `dev@e2f57ad`). This document is
the final Supplement assembly checkpoint: it consolidates already-authorized material and adds
no new analysis. Every quantitative entry below is transcribed from the frozen executed
receipts or from the sealed Results/Methods drafts; section-level traceability and automated
re-derivation are maintained in `supplement_traceability_map.md` and
`scripts/audit_supplement_draft.py`. Any numeric discrepancy between this document and its
cited receipt resolves to the receipt.

**Evidence boundary:** the scientific evidence boundary is closed. New analysis is not part of
this checkpoint; if assembly had revealed an unsupported manuscript proposition, it would be
reported, not repaired here.

**Assembly corrections to the pre-authorization material**
(`supplement_material.md`, tracked with the Methods seal lineage, retained unmodified):

1. S.1 (below) corrects a factual error in the pre-material's remark: the A-2 S3 lattice is
   12 cases, of which **10 classify TRAVELING_WAVE and 2 classify NO_WAVE** (n_sites = 6 at
   500 ms and at 2000 ms). The pre-material's "all 12 cases TRAVELING_WAVE" was inconsistent
   with the frozen A-2 receipt and is not repeated here.

**Manuscript discipline:** time in ms, frequency in Hz, positions in mm (arc-length
convention), velocities in derived units; every magnitude not declared Absolute is Relative.
No p-values, no calibrated biological magnitudes. Relative/absolute language follows the
truth gates of the Methods and Results.

---

## S.1 — RFFT bin spacing and frequency-quantization table (review items M4.4, M1.5)

The estimator's frequency step (`estimate_traveling_wave`) takes the argmax of the summed-site
rfft power over the preregistered band [8.0, 13.0] Hz; with the rectangular (unsmoothed)
analysis window the resolvable grid is fs/N = 2000 Hz / (duration_ms · 2000 Hz/s · 0.001 s/ms).
A generated frequency that is an integer multiple of the bin spacing is recovered exactly;
an off-grid frequency is reported at the neighboring in-band bin.

| Duration (ms) | Samples N | Bin spacing (Hz) | A-2 S3 drive (Hz) | On-grid? | Stored result (f̂) | Stored ε_f |
|---|---|---|---|---|---|---|
| 250 | 500 | 4.0 | 10.0 | no (10/4 = 2.5) | 12.0 | 2.0 |
| 500 | 1,000 | 2.0 | 10.0 | yes (10/2 = 5) | 10.0 | 0.0 |
| 1,000 | 2,000 | 1.0 | 10.0 | yes | 10.0 | 0.0 |
| 2,000 | 4,000 | 0.5 | 10.0 | yes (10/0.5 = 20) | 10.0 | 0.0 |

Basis: `artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json`, stage_S3.

Remarks (corrected):

- The S3 record's `frequency_hz` field stores the **estimate**; ground truth is 10.0 Hz in all
  12 cases (executor `scripts/p2v_a2_sensitivity_floor.py`).
- The 250 ms / 4.0 Hz spacing is the only grid on which 10.0 Hz is off-grid; ε_f = 2.0 there is
  structural quantization of the rectangular-window argmax, not estimator noise or a band-pass
  artifact. A byte-faithful replay of the exact executor code path (same Butterworth order-4
  band [8/1000, 13/1000] of Nyquist 1,000 Hz, filtfilt, summed-site rfft power, inclusive band
  mask) reproduced all 12 stored (f̂, ε_f) pairs exactly on 2026-08-16.
- **Classification outcome of the 12 S3 cases: 10 TRAVELING_WAVE, 2 NO_WAVE.** The NO_WAVE
  cases are n_sites = 6 at 500 ms (R² = 0.0768, coherence 0.9049) and n_sites = 6 at 2000 ms
  (R² = 0.3588, coherence 0.9756). Every other case classifies TRAVELING_WAVE, including all
  three n_sites at 250 ms and at 1000 ms (S.8, full table).
- A-1a's 48 positive controls use 2000 ms (0.5 Hz grid); the generated frequencies
  {8.5, 10.0, 12.5} Hz map to integer bins (17, 20, 25), so every A-1a ε_f = 0 as recorded.
- The A-2 S1 lattice also uses 2000 ms and records ε_f = 0 for all 40 cases.
- Consequence for the classification gates: the S3 stage's purpose is lane/cutoff set-pointing;
  the 250 ms ε_f = 2.0 limit is reported, not hidden, and the n_sites = 6 classification
  failure at 500/2000 ms is reported, not hidden. No Results claim quantifies frequency error
  below the S3 floor, and no claim requires TRAVELING_WAVE at n_sites = 6 for any duration.

## S.2 — Units and magnitude convention table (review item M1.3)

| Quantity | Unit / convention | Status | Basis |
|---|---|---|---|
| Simulation time step Δt | 0.5 ms (H4: 1.0 ms, declared inline) | Absolute | all executors |
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
  a given seed and reproduce bit-identically across machines under the pinned stack (S.15).
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

## S.5 — Consolidated protocol and regime definitions

### S.5.1 Ring protocols (C, A-series, H4, A-3)

N = 24 neuron emitters on a planar ring of radius R = 1.0 mm, directed one-neighbor edges
`i → (i + 1) mod N` (K-neighbor variants `i → (i + m) mod N`, m = 1..K, for A-1b and the H4
lattice), every edge weight w = 6.0 (dimensionless, Relative), synaptic time constant
τ = 3.0 ms, all-excitatory E labels, weights fixed across conditions and seeds. Arc-length
embedding: θ_i = 2πi/N, r_i = (R cos θ_i, R sin θ_i), analysis axis x_i = R·θ_i (mm); the
estimator consumes the (N, 1) column vector of arc-length positions. Geometry-shuffled
condition: Fisher–Yates permutation of the neuron-index-to-position assignment, preserving the
edge pre/post topology, edge weight multiset, and (in the delay-shuffled condition) the delay
multiset; shuffle seed = base_geometry_seed + condition_index·9973 (base 4242). Constructed
states default to v₀ = −65.0 (mV, Relative), u₀ = 0.0 per neuron, empty synaptic state;
drives are applied as added native uncalibrated current. Integration is explicit forward
Euler, Δt = 0.5 ms (H4 cell-level config Δt = 1.0 ms).

| Protocol | Duration | Seeds | Cells | Specifics |
|---|---|---|---|---|
| C3 | 2000 ms | 1001–1010 | 60 | 6 conditions = geometry {ordered, shuffled} × delay policy {uniform (4 steps), geometry-derived, delay-shuffled}; drive 1-ms pulses A = 50 at 200 ms (neuron 0), A = 45 at 800 ms (neuron 6); σ_n = 0.0 |
| A-1a | — | — | 53 | estimator-only synthetic controls (48 positive + 5 negative), no neural simulation |
| A-1b | 2000 ms | 1001–1003 | 45 | 15 lattice points = v_c {0.033, 0.065, 0.131, 0.262, 0.524} mm/ms × K {1, 2, 4}; delays from Eq. 6; anchor point v_c = 0.131, K = 1 must reproduce frozen C3 bitwise |
| A-2 S1 | 2000 ms | — | 40 | estimator-only sensitivity grid, mode {1, 2} × amplitude {0.05, 0.1, 0.2, 0.5, 1.0} × σ {0.0, 0.1, 0.25, 0.5} |
| A-2 S2 | 2000 ms | 1001–1003 | 3 × 12 | frozen C3 cells with injected wave γ {0.0, 0.25, 0.5, 1.0, 2.0, 4.0} × φ0 {0.0, 1.57} rad |
| A-2 S3 | 250–2000 ms | — | 12 | duration {250, 500, 1000, 2000} ms × n_sites {24, 12, 6} |
| A-3 | 2000 ms | 1001–1003 | 6 | DEFAULT_HDP / DEFAULT_HDP_DESYNC presets, K_w_ctrl = 0.0 (kernel default) |
| H4 | 80 steps @ 1.0 ms | train 100–109, test 200–209 | 4 | short (3 neurons) vs long (12 neurons) × uniform (4 steps) vs heterogeneous ({2, 8}) delays; identity-decoding assay, lags Δ {2, 5, 10, 20, 35}, 8 shuffle nulls, ridge λ = 0.01 |

Basis: Methods §2, §4, §10, §11; `c3_neural_experiment_spec.json`; A-series specs and receipts.

### S.5.2 D3 (adaptation/recovery phenotype)

36 cells = 3 seeds {11, 12, 13} × 4 arms × 3 recovery intervals. Paradigm: baseline (100 ms,
no drive) → repeated stimulation (6 identical pulses, amplitude 15 (Relative), duration 40 ms,
onset-to-onset ISI 60 ms; train block ends at 440 ms) → recovery interval
T_recovery ∈ {50, 100, 250} ms (2·τ_A / τ_K / 2.5·τ_K, prospective from D2b timescales
τ_A = 25, τ_K = 100 ms) → rechallenge pulse (identical) → post window; total 1000 ms.
Arms: N0 classical emitter (no RBS), N1 static H_K = 1, N2 κAK = 0 (D2a-equivalent),
D full D2b activity-writing. Primary response R_j = spike count in the fixed 80 ms post-onset
window of pulse j; A_adapt = 1 − R_late/R_early with R_early = mean over pulses 1–2 and
R_late = mean over pulses 5–6 (defined only when R_early > 0);
R_recovery = (R_rechallenge − R_late)/(R_early − R_late) (secondary). Classification:
ADAPTATION iff A_adapt > θ_A = 0.15 and mean H_K over late train pulses > 1 + θ_H = 1.01 and
signal quality sufficient (min mean R_early ≥ 1.0); NO_ADAPTATION is a valid outcome;
UNRESOLVED for insufficient signal (silence is never classified as adaptation).

Basis: Methods §11; `d3_adaptation_recovery_phenotype_spec.json`, `d3_execution_receipt.json`,
`d3_interpretation_receipt.json`.

### S.5.3 E-series (hierarchy, delays, RBS, observation, causal perturbation)

Two-area laminar network (A1, A2; cortical_eig preset; E drive 8.0, PV drive 6.0, Relative;
duration 1000 ms @ Δt 0.5 ms; seeds {11, 12, 13}). E1 identity/bio-identity hierarchy with
typed FF/FB ownership; E2 typed provenance-class delays (zero delays recover E1); E3 sparse
D1/D2a-type H_K RBS owned by A2:L5:E (flat indices 70–76, Γ_H: H_K → b_eff = H_K·b,
τ_K = 100 ms, H_K0 = 1.2 on owners at t₀⁺); E4 downstream observation chain per the frozen
workflow; E5 causal perturbation with arms N0 (Γ_H = I, dot H_K = 0), N1 (technical: H_K
trajectory identical to D on owners; Γ_H = I), D (Γ_H = H_K·b active on owners). Contrast
D − N1 isolates Γ_H expression, not hidden-state presence alone. All readouts post-hoc on one
simulated trajectory per arm/seed.

Basis: Methods §11; `e5_causal_perturbation_spec.json`, `e5_interpretation_receipt.json`.

### S.5.4 Experiment A (operator factorization)

40-neuron column, seed 7, 2000 ms @ 0.5 ms; layers L2/3 : L4 : L5 = 0.33 : 0.34 : 0.33,
E 70% / PV 30%, cortical_eig; drives E 8.0, PV 8.0 (Relative); X = {V_m, spikes}, H = ones
(HDP off), Q = canonical relative source; F operators: lfp_ref (project_laminar_sources,
16 contacts, width 0.10) plus 4 declared F variants; P probes: lfp_contact_shallow {0.2}
(width 0.10) and 5 declared P variants; field geometry laminar column, mean-zero Neumann
boundary, mean-zero gauge; spectral burn-in 200 ms excluded from summaries.

Basis: Methods §11; `artifacts/etudes/experiment_a/`.

### S.5.5 Evidence regimes (frozen vocabulary, Methods §12)

1. deterministic property; 2. parameter-domain characterization; 3. seed robustness;
4. model generalization; 5. empirical inference. No statistical tool computes or reports
p-values. Reported spread values are ranges or exact extremal values (min–max) across the
enumerated seeds/cells; uncertainties are declared as ranges, not standard errors. Unresolved
classes are never collapsed into negatives (UNRESOLVED → NO_WAVE forbidden; N_S = 0 ≠ negative
without N_X = 0).

## S.6 — C3 per-condition, per-seed classification (60 cells)

All 60 cells classify **NO_WAVE**. f̂ = 8.5 Hz on every cell (the argmax bin of the summed-site
band power; no cell reaches the traveling gates). The three ordered conditions
(ordered_uniform, ordered_geometry_derived, ordered_delay_shuffled) produce **bit-identical
estimator outputs cell-for-cell per seed** — identical classifications, frequencies, wave
vectors, and reasons — despite their different delay policies; a provenance property of the
frozen lineage, not a result of estimator noise.

| Condition | Seed | R² | Coherence | Null score | Reason |
|---|---|---|---|---|---|
| ordered_uniform | 1001 | 0.1169 | 0.4340 | 0.5660 | standing_or_flipping_spatial_gradient |
| ordered_uniform | 1002 | 0.1054 | 0.4579 | 0.5421 | synchronous_oscillation_k_near_zero |
| ordered_uniform | 1003 | 0.0631 | 0.4288 | 0.5712 | synchronous_oscillation_k_near_zero |
| ordered_uniform | 1004 | 0.0876 | 0.4423 | 0.5577 | synchronous_oscillation_k_near_zero |
| ordered_uniform | 1005 | 0.0590 | 0.4272 | 0.5728 | synchronous_oscillation_k_near_zero |
| ordered_uniform | 1006 | 0.0125 | 0.4796 | 0.5204 | synchronous_oscillation_k_near_zero |
| ordered_uniform | 1007 | 0.0415 | 0.3667 | 0.6333 | synchronous_oscillation_k_near_zero |
| ordered_uniform | 1008 | −0.0022 | 0.4171 | 0.5829 | synchronous_oscillation_k_near_zero |
| ordered_uniform | 1009 | 0.0757 | 0.4343 | 0.5657 | structured_but_fails_traveling_gates |
| ordered_uniform | 1010 | −0.0000 | 0.3476 | 0.6524 | synchronous_oscillation_k_near_zero |
| ordered_geometry_derived | 1001–1010 | identical to ordered_uniform per seed | | | identical reasons per seed |
| ordered_delay_shuffled | 1001–1010 | identical to ordered_uniform per seed | | | identical reasons per seed |
| shuffled_uniform | 1001 | 0.0597 | 0.4242 | 0.5758 | synchronous_oscillation_k_near_zero |
| shuffled_uniform | 1002 | 0.0249 | 0.4021 | 0.5979 | synchronous_oscillation_k_near_zero |
| shuffled_uniform | 1003 | 0.0569 | 0.3992 | 0.6008 | synchronous_oscillation_k_near_zero |
| shuffled_uniform | 1004 | 0.0310 | 0.3801 | 0.6199 | synchronous_oscillation_k_near_zero |
| shuffled_uniform | 1005 | 0.0968 | 0.4398 | 0.5602 | standing_or_flipping_spatial_gradient |
| shuffled_uniform | 1006 | 0.1341 | 0.4476 | 0.5524 | synchronous_oscillation_k_near_zero |
| shuffled_uniform | 1007 | 0.1070 | 0.3909 | 0.6091 | structured_but_fails_traveling_gates |
| shuffled_uniform | 1008 | 0.0790 | 0.4240 | 0.5760 | synchronous_oscillation_k_near_zero |
| shuffled_uniform | 1009 | 0.0276 | 0.4384 | 0.5616 | synchronous_oscillation_k_near_zero |
| shuffled_uniform | 1010 | 0.0790 | 0.3794 | 0.6206 | synchronous_oscillation_k_near_zero |
| shuffled_geometry_derived | 1001–1010 | 0.0097, 0.0407, 0.0179, 0.0096, 0.0012, −0.0028, 0.0016, 0.0046, 0.0893, 0.0157 (by seed) | 0.3547–0.4578 | 0.5422–0.6453 | synchronous_oscillation_k_near_zero on all 10 |
| shuffled_delay_shuffled | 1001–1010 | 0.0486, 0.0112, 0.0616, 0.0112, 0.0196, −0.0019, 0.0535, 0.0169, −0.0051, 0.0066 (by seed) | 0.3568–0.4410 | 0.5590–0.6432 | synchronous_oscillation_k_near_zero on all 10 |

Reason distribution across all 60 cells: 52 × synchronous_oscillation_k_near_zero,
4 × standing_or_flipping_spatial_gradient (ordered_uniform 1001, ordered_geometry_derived
1001, ordered_delay_shuffled 1001, shuffled_uniform 1005),
4 × structured_but_fails_traveling_gates (ordered_uniform 1009, ordered_geometry_derived
1009, ordered_delay_shuffled 1009, shuffled_uniform 1007). Outcome letter C: sufficient-quality neural activity yields predominantly
NO_WAVE; observed Δp_W = 0.0 for the directional conjecture
p_W(ordered, geometry_derived) > p_W(ordered, uniform), conjecture direction not supported.

Basis: `c3_execution_receipt.json`, `c3_condition_summary.json`, `c4_interpretation_receipt.json`.

## S.7 — Wave-estimator validation on synthetic controls (A-1a, A-1b)

### S.7.1 A-1a recovery tolerances (preregistered)

| Quantity | Tolerance |
|---|---|
| Relative frequency error | 0.05 |
| Absolute frequency error | 0.5 Hz |
| Relative k-norm error | 0.1 |
| Direction error | 15° |
| Relative velocity error | 0.1 |
| Phase-fit R² minimum | 0.6 |
| Spatial coherence minimum | 0.55 |

Basis: `p2v_a1a_receipt.json` `recovery_tolerances`.

### S.7.2 A-1a positive controls (48/48 recovered)

Frequency family × mode × sign × phase/noise corner = 48 cases; all 48 classify
TRAVELING_WAVE with f̂ = ground truth (ε_f = 0.0), R² ≥ 0.9986, coherence ≥ 0.98496 (the
minimum coherence case is pos_f10.0_m2_s1_p17_n25, 10.0 Hz, m = 2, s = +1). Recovery
error ranges per family:

| Family (f, m, s) | Cases | ε_k range | ε_v range |
|---|---|---|---|
| 8.5 Hz, m = 1, s = ±1 | 8 | 0.0001–0.0065 | 0.0001–0.0064 |
| 8.5 Hz, m = 2, s = ±1 | 8 | 0.0002–0.0043 | 0.0002–0.0043 |
| 10.0 Hz, m = 1, s = ±1 | 8 | 0.0002–0.0074 | 0.0002–0.0073 |
| 10.0 Hz, m = 2, s = ±1 | 8 | 0.0002–0.0026 | 0.0002–0.0026 |
| 12.5 Hz, m = 1, s = ±1 | 8 | 0.0002–0.0083 | 0.0002–0.0082 |
| 12.5 Hz, m = 2, s = ±1 | 8 | 0.0001–0.0059 | 0.0001–0.0060 |

(All 48 per-case rows, with ε_f/ε_k/ε_v/R²/coherence per case, are stored in the receipt;
the audit re-derives every row.)

### S.7.3 A-1a negative controls (5/5 classified NO_WAVE)

| Case | Expected | Observed reason | R² | Coherence |
|---|---|---|---|---|
| sync_oscillation | NO_WAVE | synchronous_oscillation_k_near_zero | 1.0000 | 1.0000 |
| standing_wave | NO_WAVE | standing_or_flipping_spatial_gradient | 0.0101 | 0.8261 |
| random_spatial_phases | NO_WAVE | structured_but_fails_traveling_gates | 0.0130 | 0.1935 |
| noise_only | NO_WAVE | noise_only_power_below_floor | 0.0708 | 0.0118 |
| shuffled_coordinates_true_wave | NO_WAVE | standing_or_flipping_spatial_gradient | 0.0028 | 0.0481 |

Basis: `p2v_a1a_receipt.json` (summary: all_positives_pass, all_negatives_pass, a1a_pass).

### S.7.4 A-1b dynamic search (15 points × 3 seeds = 45 cells)

Domain outcome: **NO_POSITIVE_DOMAIN_IN_TESTED_RANGE**. Every point outcome is NEGATIVE with
0/3 traveling-wave cells and 0 invalid cells (full grid: v_c ∈ {0.033, 0.065, 0.131, 0.262,
0.524} mm/ms × K ∈ {1, 2, 4}). Anchor verification: v_c = 0.131, K = 1 reproduces the frozen
C3 ordered-uniform construction and V_m bitwise for all 3 seeds (max |ΔV_m| = 0.0, bitwise =
true). no_adaptive_extension_observed = true. Estimator module SHA `684859a98da51de79887ec26ba8d7134e2fa0e97`.

Per-cell reason distribution across the 45 cells: 27 × standing_or_flipping_spatial_gradient,
11 × synchronous_oscillation_k_near_zero, 7 × structured_but_fails_traveling_gates;
0 traveling-wave cells and 0 invalid cells (no cell reached the traveling gates). The two
flanking endpoints (0 of 24 neurons fired, max rate 0.5 Hz, near-silent) and the
no-adaptive-extension observation are recorded in the receipt; the interpretation constraint
honored in Results: no claim beyond "no TRAVELING_WAVE classification in the tested delay
family by the same estimator that classified all of A-1a and C3". No point, criterion, or
decision rule was added, removed, or redefined after outcomes were observed.

Basis: `p2v_a1b_receipt.json` (point_outcomes, anchor_identity, domain_outcome).

## S.8 — A-2 sensitivity floor (S1 amplitude/noise grid, S2 γ-embedding, S3 duration/sites)

### S.8.1 S1: 40-case grid, mode {1, 2} × amplitude {0.05, 0.1, 0.2, 0.5, 1.0} × σ {0.0, 0.1, 0.25, 0.5}

38 of 40 cases classify TRAVELING_WAVE. The 2 NO_WAVE cases are both at amplitude = 0.05,
σ = 0.5 (mode 1 and mode 2): the amplitude/noise sensitivity floor of the estimator for
10.0 Hz on the 2000 ms grid. All 40 cases record ε_f = 0.0.

### S.8.2 S2: wave-injection γ threshold on frozen C3 cells (3 cells × 12 cases)

Frozen C3 carrier cells (ordered conditions, seeds 1001, 1002, 1009 — the three seeds whose
frozen reason strings span the reason vocabulary) with injected wave of relative
amplitude γ and phase φ0: classifications by γ:

| γ | φ0 = 0.0 | φ0 = 1.57 rad | Flip point |
|---|---|---|---|
| 0.0 | NO_WAVE | NO_WAVE | — |
| 0.25 | NO_WAVE | NO_WAVE | — |
| 0.5 | NO_WAVE | NO_WAVE | — |
| 1.0 | TRAVELING_WAVE | TRAVELING_WAVE | γ* = 1.0 |
| 2.0 | TRAVELING_WAVE | TRAVELING_WAVE | — |
| 4.0 | TRAVELING_WAVE | TRAVELING_WAVE | — |

Frozen reason strings of the carrier cells are preserved in the receipt. The flip threshold
γ* = 1.0 (the lowest injected γ classified TRAVELING_WAVE) is a set-point of the estimator's
lane, not a biological amplitude calibration.

### S.8.3 S3: duration × site-count limits (12 cases)

| Duration (ms) | n_sites = 24 | n_sites = 12 | n_sites = 6 |
|---|---|---|---|
| 250 | TRAVELING_WAVE (f̂ = 12.0, ε_f = 2.0, R² = 0.9575, coh = 0.9641) | TRAVELING_WAVE (f̂ = 12.0, ε_f = 2.0, R² = 0.9654, coh = 0.9089) | TRAVELING_WAVE (f̂ = 12.0, ε_f = 2.0, R² = 0.9895, coh = 0.8020) |
| 500 | TRAVELING_WAVE (f̂ = 10.0, R² = 0.9965, coh = 0.9828) | TRAVELING_WAVE (f̂ = 10.0, R² = 0.9971, coh = 0.9583) | **NO_WAVE** (R² = 0.0768, coh = 0.9049) |
| 1000 | TRAVELING_WAVE (f̂ = 10.0, R² = 0.9994, coh = 0.9911) | TRAVELING_WAVE (f̂ = 10.0, R² = 0.9996, coh = 0.9787) | TRAVELING_WAVE (f̂ = 10.0, R² = 0.9999, coh = 0.9518) |
| 2000 | TRAVELING_WAVE (f̂ = 10.0, R² = 0.9998, coh = 0.9956) | TRAVELING_WAVE (f̂ = 10.0, R² = 0.9999, coh = 0.9892) | **NO_WAVE** (R² = 0.3588, coh = 0.9756) |

Summary: 10 TRAVELING_WAVE / 2 NO_WAVE (n_sites = 6 at 500 ms and 2000 ms). The duration/site
limit of reliable classification under this estimator is n_sites ≥ 12 at every duration, and
n_sites = 6 only at 250/1000 ms durations in this grid.

Basis: `p2v_a2_receipt.json` (stage_S1, stage_S2, stage_S3).

## S.9 — HDP boundedness classification (A-3, 2 presets × 3 seeds)

Run domain: 24-neuron C3 ring anchor, weight 6.0, τ 3.0, delay 4; 2000 ms; seeds 1001–1003;
K_w_ctrl = 0.0 (kernel default). All runs: `all_hard_bound_invariants_pass = true`,
`no_tuning_observed = true`.

| Preset | Seed | H_min_obs | H_max_obs | w_min_obs | w_max_obs | w_abs_growth_ratio | max\|V_m\| | Mean rate (Hz) |
|---|---|---|---|---|---|---|---|---|
| DEFAULT_HDP | 1001 | 1.0000002 | 1.0008069 | 6.0 | 6.0 | 1.0 | 82.27 | 0.0417 |
| DEFAULT_HDP | 1002 | 1.0000002 | 1.0008069 | 6.0 | 6.0 | 1.0 | 82.27 | 0.0417 |
| DEFAULT_HDP | 1003 | 1.0000002 | 1.0008069 | 6.0 | 6.0 | 1.0 | 82.27 | 0.0417 |
| DEFAULT_HDP_DESYNC | 1001 | 1.0000098 | 1.0310019 | 5.8692 | 6.1684 | 1.00004 | 82.27 | 0.0417 |
| DEFAULT_HDP_DESYNC | 1002 | 1.0000098 | 1.0310019 | 5.8692 | 6.1684 | 1.00004 | 82.27 | 0.0417 |
| DEFAULT_HDP_DESYNC | 1003 | 1.0000098 | 1.0310019 | 5.8692 | 6.1684 | 1.00004 | 82.27 | 0.0417 |

Scoped statement (receipt): trajectories remained bounded over the tested parameter and time
domain (24-neuron ring, DEFAULT_HDP / DEFAULT_HDP_DESYNC, 2000 ms, seeds 1001–1003,
K_w_ctrl = 0.0): every state component stayed within its per-step hard bounds for every step.
Classification reference (C-HDP):

| ID | Claim | Class |
|---|---|---|
| C-HDP-1 | H_i is bounded | bounded_by_implementation (jnp.clip to [0.1, 10.0] every step, unconditional) |
| C-HDP-2 | synaptic weights w are bounded | bounded_by_implementation (every-step clip to [w_floor, w_ceiling]; with K_w_ctrl > 0 an additional restoring term contracts |w| toward wmag_baseline) |
| C-HDP-3 | DEFAULT_HDP with K_w_ctrl = 0.0 is safe with no weight-magnitude restoration | **not_established** (long-horizon drift behavior with K_w_ctrl = 0.0 is not established) |
| C-HDP-4 | with K_w_ctrl > 0 the law is analytically bounded | analytically_bounded |
| C-HDP-5 | shipped-preset trials remain the verified domain | numerically_bounded_over_tested_domain |
| C-HDP-6 | any claim beyond the tested presets/domain | **not_established** |
| C-HDP-7 | closed-loop HDP memory claim on the W3b map | **not_established** |
| C-HDP-8 | |syn_state| bound | bounded_by_implementation |

Basis: `p2v_a3_hdp_boundedness/p2v_a3_receipt.json` (runs, all_hard_bound_invariants_pass,
scoped_statement, classification_reference). The known-fragility note in `AGENTS.md` is
consistent: K_w_ctrl = 0.0 permits unbounded weight drift on long/custom HDP runs outside the
verified presets.

## S.10 — D3 per-arm, per-seed, per-interval evidence (36 cells)

### S.10.1 Classification and endpoint table

Arms: N0 (classical emitter), N1 (static H_K = 1), N2 (κAK = 0), D (full D2b activity-writing).
R_early = 3.5 and R_late = 2.5 on every cell (all arms, all seeds, all intervals); A_adapt =
0.2857 on every cell — the observable attenuation is identical across arms by construction of
the paradigm; the D arm's A_adapt does not differ from the null arms (D − N2 null: A_adapt
0.2857 vs 0.2857 at seed 11/short). Classification: 27 ADAPTATION (all N0/N1/N2 cells) /
9 NO_ADAPTATION (all D-arm cells) / 0 UNRESOLVED. No facilitation (n_facilitation = 0).

| Arm | Seeds | Intervals | Classification | A_adapt | R_train (identical across arms) |
|---|---|---|---|---|---|
| N0 | 11, 12, 13 | short, medium, long | ADAPTATION (9/9) | 0.2857 | [4, 3, 3, 3, 3, 2] |
| N1 | 11, 12, 13 | short, medium, long | ADAPTATION (9/9) | 0.2857 | [4, 3, 3, 3, 3, 2] |
| N2 | 11, 12, 13 | short, medium, long | ADAPTATION (9/9) | 0.2857 | [4, 3, 3, 3, 3, 2] |
| D | 11, 12, 13 | short, medium, long | NO_ADAPTATION (9/9) | 0.2857 | [4, 3, 3, 3, 3, 2] |

R_train = [4, 3, 3, 3, 3, 2] on every cell (all arms, all seeds, all intervals): R_early =
mean(pulses 1–2) = 3.5, R_late = mean(pulses 5–6) = 2.5, A_adapt = 1 − 2.5/3.5 = 0.2857 on
every cell. The observable attenuation is identical across arms by construction of the
paradigm; the D − N2 contrast on A_adapt is 0.2857 vs 0.2857 on all 9 pairwise cells
(n_D_adaptation_N2_not = 0). Recovery-block numbers (S.10.2 Q3) are D-arm scoped.

### S.10.2 Mechanism and phenotype questions (D arm, 9 cells)

- Q1 (mechanism, M1/M2): n_D_cells = 9, n_M1_pass = 9, n_M2_pass = 0,
  n_mechanism_ok = 0, fraction_mechanism_ok = 0.0 → **partial**: activity writes the intended
  RBS state (M1) but the late-train mean H_K does not exceed 1 + θ_H (M2 fails on every D cell);
  formal ADAPTATION therefore never fires.
- Q2 (phenotype): ADAPTATION 0 / NO_ADAPTATION 9 / UNRESOLVED 0 → **no_adaptation**.
- Q3 (recovery): hidden state decays with rest —
  mean |H_K − 1| at rechallenge: short 0.00551, medium 0.00355, long 0.00080
  (T_rec up ⇒ H_K closer to 1, all seeds); observable response does not recover —
  mean R_rechallenge 2.0 (short) / 3.0 (medium, long) vs R_early 3.5, R_recovery = −0.5 / 0.5 / 0.5.

Headline (receipt): observable attenuation without formal ADAPTATION — hidden-state writing
is detectable on D but does not differentiate the preregistered spike-count phenotype from N2.

Basis: `d3_execution_receipt.json`, `d3_interpretation_receipt.json`.

## S.11 — H4 geometry-memory matrix (4 cells, identity-decoding assay)

### S.11.1 Per-cell decoding table

Identity-decoding assay: ridge-regularized identity decoding (λ = 0.01), lags
Δ ∈ {2, 5, 10, 20, 35} steps, 8 shuffle nulls, train seeds {100…109}, test seeds {200…209}.
M_X(Δ) = decoding accuracy of perturbation-state identity from activity; chance levels differ
by class count (P_chance = 1/12 = 0.0833 for long rings, 1/3 = 0.3333 for short rings).
M_H = 1.0 at every lag on every cell (hidden state perfectly decodable from hidden state —
machinery sanity check).

| Cell | P_chance | M_X at Δ = 2 | Δ = 5 | Δ = 10 | Δ = 20 | Δ = 35 | M_X_shuffle (max) | M_X_area |
|---|---|---|---|---|---|---|---|---|
| long_uniform | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0 |
| long_heterogeneous | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0 |
| short_uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.0 |
| short_heterogeneous | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3292 (Δ = 20) | 0.0521 |

The only positive M_X_area is short_heterogeneous (0.0521), produced by the Δ = 20 shuffle
depression (M_X_shuffle 0.3292 vs M_X 0.3333), not by any M_X excess peak
(M_X_excess_peaks = [] and M_X_secondary_peaks = [] on all cells).

### S.11.2 Factorial point estimates and interpretation

| Term | Point estimate |
|---|---|
| μ | 0.0 |
| α_length | 0.0 |
| α_heterogeneity | +0.0521 |
| α_interaction | −0.0521 |

- α_length = 0: no positive length effect detected by the preregistered identity-decoding
  assay; the H4 conclusion is the absence of a positive length effect in the tested domain,
  not statistical evidence of no effect.
- α_heterogeneity = +0.0521 arises entirely from the short+heterogeneous cell; not a general
  heterogeneity effect. **Bootstrap confidence intervals were not computed; +0.0521 is a
  prospective point estimate only, not statistically established.**
- α_interaction cancels the short-hetero signal in the long-hetero cell.
- Directional conjecture M_X^{long,hetero} > M_X^{short,uniform}: supported = false.

### S.11.3 Confound material and methodological limitations (receipt)

1. Cross-N identity decoding is confounded by different class counts (P_chance = 1/3 vs 1/12),
   sample complexity, and feature dimensionality.
2. Do not conclude "length has no effect on memory"; only that no positive length effect was
   detected by the preregistered H4 assay.
3. Future topology-specific protocols may use matched binary perturbation decoding or
   information-normalized metrics.
4. Recurrence diagnostic: loop-aligned secondary peaks detected = false; extending Δ_max
   would be H5/new protocol, not an H4 correction.

Basis: `h4_matrix/h4_matrix_receipt.json`, `h4_matrix/h4_interpretation_receipt.json`
(status FROZEN_NEGATIVE_RESULT; repository SHA at run 8f01ae75…).

## S.12 — E5 causal perturbation, per-seed contrasts (3 seeds)

Interpretation rule: HIERARCHICAL_PROPAGATION requires G_O true (owner gate, threshold 1e-6)
plus ≥ 1 of {G_A1, G_Q, G_Y}; LOCAL_EXPRESSION = owner expression without sufficient
downstream propagation; NO_EFFECT = typed expression failed under assay; UNRESOLVED = quality
gates insufficient. Contrast D − N1 isolates Γ_H expression (G1: H_K(N1) == H_K(D) bit-exact
3/3 seeds; G2: N0 ≡ N1 at H_K = 1).

| Seed | Classification | Owner Δ (mV, spikes) | A2 non-owner Δ (mV, spikes) | A1 Δ (mV, spikes) | Q L2 | Q |Δt·| | Y L2 | Y |Δt·| | Depth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | HIERARCHICAL_PROPAGATION | 9.2634, +7 | 2.4295, 0 | 3.1617, +9 | 702.29 | 379.09 | 96691.5 | 202.09 | Y |
| 12 | HIERARCHICAL_PROPAGATION | 9.2634, +7 | 2.4295, 0 | 3.1617, +9 | 702.29 | 379.09 | 96691.5 | 202.09 | Y |
| 13 | HIERARCHICAL_PROPAGATION | 9.2634, +7 | 2.4295, 0 | 3.1617, +9 | 702.29 | 379.09 | 96691.5 | 202.09 | Y |

Every-seed HIERARCHICAL_PROPAGATION (3/3). Permissible A1 statement: "local A2 RBS
perturbation propagated through existing hierarchical connectivity into A1". Forbidden A1
statement: "feedback suppresses/enhances a particular frequency band" — no spectral/functional
FF/FB claim exists (same language as Results Q30). Values are Relative deviations (mean abs
V_m deviation in Relative mV, spike-count differences, V_m time-integral differences).

Basis: `e5_interpretation_receipt.json` (per_seed, interpretation_rules, causal_contrast,
permissible/forbidden A1 statements, propagation_depth_diagnostic).

## S.13 — W3b parameter-domain map (HDP, 2187 cells)

Frozen lattice: κ_H {0.02, 0.05, 0.1} × κ_W {0.5, 1.0, 2.0} × λ_W {0.05, 0.1, 0.2} ×
τ_H {60, 80, 120} ms × τ_W {80, 100, 150} ms × I_tonic {0, 5, …, 40} = 243 × 9 = 2187 cells.
Frozen gates: active_syn_threshold 0.001, robust margin m_F > 0.02 (ρ excluding multipliers
with |λ−1| < 0.05), L_HDP > 1e-6, r_tau = (τ_W/λ_W)/τ_H > 1, period ≥ 2 steps with a spike in
the period window, mean synaptic activity > 1e-3.

| Regime | Meaning | Count |
|---|---|---|
| D | dormant/vanishing feedback | 243 |
| S | robustly stable active HDP | 0 |
| C | near-critical (0 < m_F ≤ 0.02) | 0 |
| U | unstable (m_F ≤ 0); negative evidence | 0 |
| X | active but stability-unresolved (X ≠ U) | 1944 |
| N_S | useful-domain count (active ∩ stable ∩ nontrivial ∩ timescale) | 0 |
| N_X | unresolved-active count | 1944 |

Interpretation (receipt): no demonstrated robust active domain; active regimes
stability-unresolved (X ≠ U). N_S = 0 is the count of S-regime points only; with N_X = 1944 >
0 the useful domain is not claimed scientifically empty. No operating point is selected from
the map (selected_operating_point = null; selection rule = max m_F among S-regime points with
L_HDP > L_min and r_tau > 1, fixed before the memory experiment). No closed-loop HDP memory
claim rests on this map (W3 remains unauthorized); next step is orbit characterization (W3c),
not law redesign. The dimensionless coordinates r_tau and Γ_HDP = |2κ_W b_HW/(a_H λ_W)| are
explanatory only; regime labels use the implementation-faithful Floquet analysis.

Basis: `w3b_parameter_domain/w3b_domain_receipt.json` (frozen_lattice, frozen_gates,
regime_counts, aggregate_quantities, interpretation, selection_rule),
`w3b_parameter_domain/w3b_interpretation_receipt.json` (status FROZEN_UNRESOLVED;
outcome_classification = unresolved_not_negative; counts; three_level_interpretation).

## S.14 — Consolidated negative/unresolved evidence register

| Item | Polarity (evidence index) | Verdict | Where established |
|---|---|---|---|
| H4 topology-memory extension | NEGATIVE | no positive length effect in tested domain (α_length = 0; +0.0521 point estimate not established) | S.11, Methods §11, Results H4 |
| C3 traveling-wave conjecture | NEGATIVE | 60/60 NO_WAVE; Δp_W = 0.0; directional conjecture not supported | S.6, Results C3 |
| D3 adaptation attribution | NEGATIVE | observable attenuation identical across null arms; formal ADAPTATION 0/9 on D | S.10, Results D3 |
| W3b useful HDP domain | UNRESOLVED | N_S = 0 with N_X = 1944; X ≠ U; no operating point selected | S.13, Methods §12, Results W3b |
| E5 hierarchical propagation | POSITIVE | HIERARCHICAL_PROPAGATION 3/3 seeds with owner gate and downstream levels | S.12, Results E5 |

Declared-not-established register (never collapsed into negatives): H4 bootstrap CIs; W3b
stability classification of X points; standing-pattern reason parity (D2 parity at
500/2000 ms measured separately); C-HDP-3/6/7 (K_w_ctrl = 0.0 long-horizon drift; claims
beyond tested presets; closed-loop HDP memory claim).

Basis: `artifacts/publication/publication_evidence_index.json` (evidence_summary,
polarity_axis), the per-protocol receipts, Results/Methods.

## S.15 — Reproducibility, environment, and provenance

### S.15.1 Environment (executed)

Python 3.13.7; JAX 0.10.1 (CPU) with float32 default (float64 where declared: A-2 S2/S3 estimator
inputs, estimator spectral arithmetic); NumPy; SciPy == 1.17.1; matplotlib >= 3.10.9, <3.11
(byte-identity pin: darwin + matplotlib 3.10.9, enforced by the equivalence gate
`byte_identity_pinned()`); jaxfne at baseline `dev@8cc60a6…` (each frozen receipt records its
own git SHA; figures frozen at fig05 SHA f7f9a494…; estimator module SHA
684859a98da51de79887ec26ba8d7134e2fa0e97).

### S.15.2 Determinism

σ_n = 0.0 in all protocol runs; per-seed `jax.random.PRNGKey(seed)`; float32 arithmetic;
identical trajectories across runs verified bitwise where declared (sequencing, continuation,
replay). The frozen figure set was rendered with matplotlib >= 3.10.9, <3.11 and
scipy == 1.17.1; either pin drift makes the equivalence gate fail on byte-identity (verified
by clean-room bisection, 2026-08-15).

### S.15.3 Provenance (receipt index)

| Checkpoint | Receipt | Status |
|---|---|---|
| C0/C1/C2 | `artifacts/protocol_c/c0_wave_protocol_receipt.json`, `c1_protocol_receipt.json`, `c1_synthetic_validation_receipt.json`, `c2_protocol_receipt.json`, `c2_delay_continuation_receipt.json` | FROZEN |
| C3/C4 | `artifacts/protocol_c/c3_execution_receipt.json`, `c3_condition_summary.json`, `c4_interpretation_receipt.json` | FROZEN |
| A-1a | `artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json` | FROZEN |
| A-1b | `artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json` | FROZEN |
| A-2 | `artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json` | FROZEN |
| A-3 | `artifacts/protocol_c/p2v_a3_hdp_boundedness/p2v_a3_receipt.json` | FROZEN |
| D0–D3 | `artifacts/protocol_d_biological_rbs/d0_protocol_receipt.json` … `d3_interpretation_receipt.json` | FROZEN |
| E0–E5 | `artifacts/protocol_e_integration/e0_protocol_receipt.json` … `e5_interpretation_receipt.json` | FROZEN |
| H4 | `artifacts/protocol_h_rbd/h4_matrix/h4_matrix_receipt.json`, `h4_interpretation_receipt.json` | FROZEN_NEGATIVE_RESULT |
| W3b | `artifacts/protocol_w/w3b_parameter_domain/w3b_domain_receipt.json`, `w3b_interpretation_receipt.json` | FROZEN_ANALYSIS / FROZEN_UNRESOLVED |
| Figures 1–7 | `artifacts/publication/fig0X_generation_receipt.json` + semantic audits; SHAs: fig01 707c3fec…, fig02 e65ad1f3…, fig03 ffa0a542…, fig04 b7646a51…, fig05 f7f9a494…, fig06 547e36c7…, fig07 fa74ddf2… | FROZEN |
| Equivalence | `artifacts/publication/equivalence_report.json` (seam equivalence 7/7 byte-identity) | FROZEN |

Canonical source: `artifacts/etudes/experiment_a/canonical_source.npz` is a tracked frozen
publication-evidence exception (`.gitignore` exception, added 2026-08-16); its SHA-256 is
recorded in `b1_canonical_receipt.json`; regeneration verification remains available via
`python3 scripts/run_experiment_a.py` (verification-first: receipts compared, never rewritten).

Basis: Methods §14; `REVIEW_NAVIGATION.md`; `artifacts/publication/publication_evidence_index.json`.

## S.16 — Material deliberately displaced from Results/Methods, and not-established register

The following material is assembled here because the Results and Methods drafts keep
summary-level reporting only; nothing in this section adds a claim beyond the cited receipts
or the sealed drafts:

1. Full per-condition/per-seed estimator tables (S.6, S.7, S.8) — Results reports aggregate
   counts and gate outcomes only.
2. Full per-arm/per-seed D3 endpoints and mechanism questions (S.10) — Methods §11 reports
   the aggregate 27/9/0 classification.
3. H4 confound register and methodological limitations (S.11.3) — the Methods §11 protocol
   carries a compact confound disclosure (class-count asymmetry, single-cell point estimate);
   the full receipt-backed register with interpretation rules and future-protocol
   suggestions is assembled here.
4. A-3 HDP boundedness classification reference C-HDP-1..8 (S.9) — Methods reports the
   scoped statement only.
5. W3b lattice gates, counts, and three-level interpretation (S.13) — Methods §12 reports
   counts and the X ≠ U discipline.
6. E5 interpretation rules, permissible/forbidden A1 statements, depth diagnostic (S.12).
7. The environment/determinism/provenance block (S.15) — displaced from Results Appendix
   (which keeps the four-row A-series table) and Methods §14.
8. The S.1 correction (this header): the pre-material's "all 12 S3 cases TRAVELING_WAVE"
   remark is corrected to 10/2 per the frozen A-2 receipt; the pre-material file itself is
   retained unmodified as a historical artifact of the Methods seal lineage.

Declared-not-established register (never collapsed into negatives): H4 bootstrap CIs; W3b
stability classification of X points; standing-pattern reason parity (D2 parity at
500/2000 ms measured separately); C-HDP-3/6/7 as classified in S.9. Each entry states exactly
what was not established; none is reported as a negative result.

## S.17 — Claim × evidence-regime classification (21 claims)

The full claim × regime classification block, verbatim from the publication claim ledger
(`artifacts/publication/publication_claim_ledger.json`, schema
`jaxfne.publication.claim_ledger.v1`), with each claim mapped to the five-regime taxonomy
of Methods §12. CL-01 (representational document), CL-04 (method-boundary statement), and
CL-21 (method reproduction) are non-quantitative classifications outside the five
quantitative regimes.

| Claim | Ledger evidence regime (verbatim) | Methods §12 family |
|---|---|---|
| CL-01 | representational_document | — (Fig. 1 grammar map; no empirical quantity) |
| CL-02 | deterministic_numerical_property | 1 · deterministic property |
| CL-03 | deterministic_numerical_property | 1 |
| CL-04 | method_boundary_statement | — (method-boundary statement) |
| CL-05 | deterministic_numerical_property | 1 |
| CL-06 | parameter_space_characterization | 2 · parameter-domain characterization |
| CL-07 | deterministic_numerical_property | 1 |
| CL-08 | deterministic_numerical_property (classification of simulated cells) | 1 |
| CL-09 | deterministic_numerical_property + parameter_space_characterization | 1 + 2 |
| CL-10 | deterministic_numerical_property (factorial point estimates) | 1 (point estimates; the H4 conclusion carries the regime-5 caveat) |
| CL-11 | parameter_space_characterization (lattice scan, deterministic gates) | 2 |
| CL-12 | deterministic_numerical_property | 1 |
| CL-13 | deterministic_numerical_property | 1 |
| CL-14 | deterministic_numerical_property | 1 |
| CL-15 | deterministic_numerical_property | 1 |
| CL-16 | deterministic_numerical_property | 1 |
| CL-17 | deterministic_numerical_property | 1 |
| CL-18 | deterministic_numerical_property across seeds | 1 + 3 · seed robustness |
| CL-19 | parameter_space_characterization | 2 |
| CL-20 | deterministic_numerical_property | 1 |
| CL-21 | method_reproducibility | — (reproduction verification) |

Basis: `publication_claim_ledger.json` (all 21 rows verbatim from the `evidence_regime`
field); family mapping follows Methods §12.

---

End of final Supplement assembly draft. Section-level traceability: `supplement_traceability_map.md`;
automated re-derivation: `scripts/audit_supplement_draft.py` (every quantitative table value
re-derived from the cited frozen receipt).