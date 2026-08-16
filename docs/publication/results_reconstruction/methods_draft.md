# Methods — Reconstruction Draft

**Status:** DRAFT (Phase 5, Methods Reconstruction), authoritative baseline `dev@8cc60a677217a54b8bad321ac2bdfd479e3f9c13`.
**Central invariant:** the mathematics stated here ≡ the operators implemented in `jaxfne` ≡ the configurations executed in the frozen/post-freeze evidences ≡ the results reported in `results_draft.md` (same freeze lineage).
**Evidence language:** every quantity is stated as **Relative** or **Absolute**; no value below is claimed to be a calibrated physical measurement. All field/readout outputs are proxy readouts from a linear-solver scaffold, not a PDE/volume-conductor solve and not empirically calibrated.
**Figures:** all figure numbers refer to the same Figures 1–6 described in the Results; no figure is produced in this document.

---

## 1. Operator factorization of the simulated system

All simulations follow a single frozen causal architecture (Experiment A lineage; identical factorization in every protocol):

```text
(X, H) → S → Q → F → Φ → P → Y
```

where X are neural state trajectories, H hidden (Relative Biophysical State, RBS) coordinates, S the source operator, Q the canonical relative source, F the field operator, Φ the field-time proxy, P the probe operator, and Y the final readout. In operator terms:

- **S_ψ**: `(x_t, u_t, ξ_t) ↦ q_t` — the source operator maps emitter state to the canonical relative source Q at each time step (layer 3, Eq. 4).
- **E_θ**: `(x_t, u_t, ξ_t) ↦ x_{t+1}` — the evolution operator; for HDP runs the hidden coordinate H becomes part of the carried state (layer 5).
- **F_γ**: `q_t ↦ φ_t` — the field operator maps the stacked source to field-coordinate proxies Φ (layer 6).
- **P_η**: `φ_t ↦ y_t` — the probe operator maps field quantities to readout Y (layer 6).

These operators are not assumed linear, and F, P are applied **post-hoc on frozen trajectories**: exactly one neural simulation is executed per seed, after which X, H, and Q are frozen and every observation operator runs on that single frozen trace (anti-drift rule). Differences between readouts therefore arise from (F, P), never from re-simulated neural trajectories. Nothing computed by a downstream estimator may feed upstream into S, F, the emitters, or the simulation state (upstream-feed prohibition); the estimator is a pure function of its inputs.

## 2. Circuit specification and construction path

Models are constructed through a single dispatch `construct()` in `jaxfne`, with two call forms that converge on the same object grammar `Config → Net → Paradigm → Objective → Trainer → Signals → Vis/Export`:

- **Configuration tier** (flat fluent builder): used for the ring-based protocols (C, A-series, H4, A-3). Example path: `laminar_cortex_config` / `build_laminar_column` → `construct` → `simulate`.
- **NeuronalTensor tier** (structured Areas × Layers × NeuronTypes): used for the multi-area protocols (E-series, Experiment A) with explicit 3-D geometry per layer and area pose. The two tiers are separate, not a hierarchy; neither wraps the other, and only the downward conversion `neuronal_tensor_to_configuration` exists.

**Ring topology (protocols C, A-1a/A-1b/A-2/A-3, H4 short/long cells):** N = 24 neuron emitters on a planar ring of radius R = 1.0 mm, directed one-neighbor edges `i → (i + 1) mod N` (K-neighbor variants `i → (i + m) mod N`, m = 1..K, for A-1b and the H4 lattice), every edge weight w = 6.0 (dimensionless, Relative), synaptic time constant τ = 3.0 ms, all-excitatory E labels, weights fixed across conditions and seeds. Arc-length embedding: θ_i = 2πi/N, r_i = (R cos θ_i, R sin θ_i), analysis axis x_i = R·θ_i (mm); the estimator consumes the (N, 1) column vector of arc-length positions.

Geometry-shuffled condition: Fisher–Yates permutation of the neuron-index-to-position assignment, preserving the edge pre/post topology, edge weight multiset, and (in the delay-shuffled condition) the delay multiset; shuffle seed = base_geometry_seed + condition_index·9973 (base 4242).

Constructed states default to v₀ = −65.0 (mV, Relative), u₀ = 0.0 per neuron, empty synaptic state; drives are applied as added native uncalibrated current.

## 3. Neural dynamics (emitter kernel)

The emitter is the reduced Izhikevich fast-subsystem model with a sparse exponential-synapse recurrence (all protocol runs, `edge_list` recurrent backend):

```text
dv/dt = 0.04 v² + 5 v + 140 − u + I(t)                    (1)
du/dt = a (b v − u)                                        (2)
```

with I(t) = I_drive(t) + I_syn(t) + σ_n ξ(t), where I_drive is the constant cell drive plus any scheduled pulse, I_syn is the recurrent synaptic current, σ_n the stochastic-current coefficient (default 0.5; every protocol run in this study sets σ_n = 0.0, making runs deterministic given the RNG key), and ξ(t) a standard-normal increment drawn once per step per neuron from the seed-keyed RNG.

Integration is explicit forward Euler with fixed Δt = 0.5 ms (H4 cell-level config uses Δt = 1.0 ms):

```text
v_{t+1} = v_t + Δt·(dv/dt),   u_{t+1} = u_t + Δt·(du/dt),
```

followed by the conditional spike rule and reset:

```text
spike: v_{t+1} ≥ V_thr = +30.0 (Relative mV), and neuron not silenced
reset: v → c, u → u + d
```

Synaptic state is carried per edge with exponential decay:

```text
x_e(t+1) = x_e(t)·exp(−Δt/τ_e) + s_{pre(e)}(t)            (3)
I_syn,i(t) = Σ_e: post(e)=i  w_e·x_e(t)                    (4)
```

A silence mask (when active) replaces v with c for masked neurons and suppresses their spikes; the mask is applied before the threshold test.

**Canonical relative source (source operator S_ψ):** the recorded source signal is

```text
Q_t = s_scale · (I_native,t + g_spk·s_t),   g_spk = 20.0 (Relative)   (5)
```

with s_t the binary spike vector, I_native the summed drive + recurrent + stochastic current, and s_scale a per-population relative scale. Q is the substrate consumed by all field operators; Q is **Relative**, never a calibrated physical current.

## 4. Finite conduction-delay coupling

Edges may carry integer delay steps `delay_steps` ≥ 0; delay_steps = 0 selects exactly the instantaneous kernel of layer 3 (verified bit-equivalence, zero-delay regression). For positive delays the kernel maintains a per-neuron spike-history ring of length D_max + 1 (D_max = max delay_steps); the presynaptic increment delivered to edge e at step t is the spike indicator of pre(e) at step t − delay_steps(e) (`spike_hist[(t − delay_steps) mod D_max+1]`), written into the ring at slot t mod (D_max + 1) after reading. The decay and summation follow Eq. 3–4 with this delayed presynaptic increment.

**Delay grid (Protocol D0 semantics):** delay_steps are aligned from milliseconds by `round(ms/Δt)` with tolerance 1e-9 at Δt = 0.5 ms. Protocol C's uniform condition uses delay_steps = 4 (= 2.0 ms). The A-1b dynamic-search family parameterizes delays by an axonal conduction velocity proxy:

```text
d_m = ⌈ m·a / (v_c·Δt) ⌉,   a = 2πR/N = 0.261799... mm, Δt = 0.5 ms, v_c in mm/ms   (6)
```

for skip m (m = 1..K), with the convention that a skip-m edge is the arc of length m·a. d_m is a dimensionless integer ≥ 1; all signature vectors were computed and verified identically in float64 and float32 arithmetic before any outcome was observed, and the 15 signature vectors are embedded verbatim in the frozen A-1b spec. Quantization plateaus (distinct (v_c, K) points mapping to equal integer vectors) are recorded in the pre-freeze evidence, not collapsed. Supporting data: the anchor point v_c = 0.131 mm/ms, K = 1 gives d_1 = ⌈0.261799/(0.131·0.5)⌉ = ⌈3.9977⌉ = 4, i.e. exactly the frozen C3 uniform delay; at this point the A-1b construction reproduces the frozen C3 ordered-uniform trajectories bitwise (max |ΔV_m| = 0.0 across 3/3 seeds), which is the anchor criterion (a construction-identity check, not a classification claim).

Segmented continuation (multi-segment runs) requires the full continuation state including the delay ring (`delay_state`), the per-edge synaptic state, voltages, recovery variables, and — for HDP runs — H and w; `Model.simulate`'s segmented path was validated against uninterrupted runs (Protocol C2).

## 5. Relative Biophysical State (RBS) and HDP dynamics

### 5.1 Hidden-state coordinates

Protocols D, E, and W operate on typed hidden coordinates H (per-neuron, Relative). Two families are used:

- **Static/autonomous H_K (Protocols D1, D2a, D3, E3–E5):** H_K modifies the recovery drive through an effective parameter, `b_eff = H_K·b`, entering Eq. 2 as `du/dt = a (H_K·b·v − u)`; no other parameter changes (parameter locality). D1 holds H_K constant per neuron for the full run (dH_K/dt = 0, expression gate H_K=1 must return the classical trajectory bit-exactly); D1 was characterized on an isolated single neuron (500 ms, Δt 0.5 ms, step pulse amplitude 15 at t = 50 ms, seeds 11–13) over H_K ∈ {0.8, 1.0, 1.2} (step 0.2), with the bidirectional-sensitivity gate (no expected direction); D2a relaxes owners toward 1 with `τ_K·dH_K/dt = 1 − H_K` on the owner set only (τ_K = 100.0 ms); D2b adds activity writing via an activity-coordinate H_A (τ_A = 25 ms) with coupling κAK (window-structure κAK = 0 defined as the D2a-equivalent null). H_K = 1 and Γ_H = I reproduce the classical emitter exactly for v, u, and spikes (G1_containment, verified).
- **HDP (Protocols A-3, W-series):** per-neuron state H_i (default scalar, h_state_dim = 1) with the following implemented law (τ_i = τ_0·size³ by cell type, cube law):

```text
τ_i·dH_i/dt = α·I_syn,i + β − γ·H_i·r_i − δ·W_i + ρ_passive/H_i² + K_ctrl·(1 − H_i) − dC/dH_i   (7)
```

with r_i the previous-step spike indicator, W_i = Σ_e: pre(e)=i |w_e| the outgoing weight burden, and the asymmetric safety barrier potential C(H) = barrier_c/(H − H_min) + barrier_d/(H_max − H); the drive term entering I_native is additionally multiplied per neuron by the activity boost 1 + H_boost_gain·max(0, 1 − H_i) (H_boost_gain = 4.0 in both shipped presets; the boost is ≈ 1 wherever H_i ≈ 1, and it scales only drive + scheduled current, never the recurrent or noise terms); barrier denominators are floored at barrier_eps = 1e-3. On the neuron's own spike, H is additionally drained discretely by C_spike (C_spike = 0.0 in every shipped preset, hence inert in all runs reported here).

### 5.2 HDP weight plasticity

Weight-magnitude updates (per edge, postsynaptic-indexed; m = |w|, ΔH = H_post − H_pre):

```text
signed_linear:      dm/dt = +K_HDP·(H_post − H_pre)·m    (E edges),  −K_HDP·(H_post − H_pre)·m (I edges)
signed_quadratic:   dm/dt = ±K_HDP·(H_post − H_pre)·|H_post − H_pre|·m
hebbian_product:    dm/dt = ±K_HDP·(H_pre·H_post)·m
```

plus the two-sided weight-magnitude restoration `dw_w_ctrl = K_w_ctrl·(wmag_baseline − m)` (sign-agnostic, applied to the magnitude before the E/I sign is reapplied). The update is multiplicative in m and uses the updated (step-t+1) H values.

### 5.3 Boundedness and clamps

Every carried HDP variable is hard-clamped at every integration step (`implementation-bounded`):

```text
H ∈ [H_min, H_max] = [0.1, 10.0]
|w| ∈ [w_floor, w_ceiling] = [0.01, 10.0]  (BASE_HDP scaffold; kernel defaults 0.001/50.0)
v ∈ [−150, 100] (mV, Relative), |u| ≤ 2000, |syn_state| ≤ 1e4 per edge (spikes binary)
```

Clamp statements supersede all model behavior at the boundary: H, w, v, u, syn cannot diverge in float32 regardless of parameters (A-3 invariant checks). Classification of each manuscript-facing boundedness claim (C-HDP set, A-3):

| Claim | Class | Statement allowed in the manuscript |
|---|---|---|
| H_i bounded | implementation-bounded | H is hard-clamped to [0.1, 10.0] at every integration step regardless of parameters |
| |w| bounded | implementation-bounded (per-step clamp; with K_w_ctrl > 0 an additional restoring term pulls |w| toward its baseline) |
| DEFAULT_HDP (K_w_ctrl = 0.0) safe on arbitrary horizons | **not established** | with the default K_w_ctrl = 0.0 the weight magnitude has no active restoration; boundedness over untested horizons is not established |
| Default presets' verified runs | numerically bounded over the tested domain | trajectories remained bounded over the exact tested parameter/time domain (see receipts) |
| |v|, |u|, |syn| finiteness | implementation-bounded | hard-bounded every step, so HDP runs cannot overflow/underflow in float32 |

The two shipped presets used in executed runs: DEFAULT_HDP {K_HDP = 0.01, τ_0 = 200 ms, K_ctrl = 5.0, ρ_passive = 0.0, barrier_c = barrier_d = 0.01} (K_w_ctrl at the kernel default 0.0) and DEFAULT_HDP_DESYNC {K_HDP = 0.01, τ_0 = 5 ms, K_ctrl = 0.15, α = 0.05, γ = 0.5, C_spike = 0.0, barrier 0.01/0.01}, both folded over the BASE scaffold (α = 0.01, H_boost_gain = 4.0, or explicitly overridden). τ_i scales by size³ with per-cell-type sizes from the canonical size table.

A-3 executed both presets × seeds {1001, 1002, 1003} on the frozen C3 ring (duration 2000 ms, Δt 0.5 ms, drive as C3, K_w_ctrl = 0.0) and recorded extremal invariants per run (H range, |w| growth factor, |v| ≤ 150, |u| ≤ 2000, |syn| ≤ 1e4); no parameter was tuned and no instability inference made. Observed bounds (Relative) {Q22}: DEFAULT_HDP H ∈ [1.0000, 1.0008], w = 6.0 unchanging (growth 1.00000); DEFAULT_HDP_DESYNC H ∈ [1.0000, 1.0310], w ∈ [5.87, 6.17] (growth 1.00004). All reported statements are scoped to the tested domain ("trajectories remained bounded over the tested parameter and time domain"); no claim of unbounded divergence, and no claim of stability beyond the tested domain.

## 6. Field and probe operators (observations)

All field operators are **linear-solver proxies**, not PDE/volume-conductor solves (field_solver_status = linear_solver; field_claim_level = proxy_readout; physical_amplitude_calibrated = False). The only PDE-family solver in the repository (`experimental_poisson_1d`) is experimental and was **not used** for any figure or protocol reported here.

- **Laminar field proxy (F, Fig. 2–4 lineage) {Q01}:** sources Q_t (T, N) at positions with relative laminar depth z ∈ [0, 1] are projected onto n_contacts = 16 Gaussian contacts of width 0.10 (relative depth): `Φ_c(t) = Σ_i Q_i(t)·exp(−(z_c − z_i)²/(2·0.10²))` with density-preserving normalization (the default; the alternative row-normalize mode erases source density and is not used in any recorded protocol). Outputs: phi_e_proxy (LFP-like), csd_proxy (second-derivative/divergence-style spatial operator on phi_e, sign convention positive = extracellular source), kernel, contact depths.
- **LFP-proxy probe (P):** samples phi_e at probe contact depths by interpolation (`depth_interpolation_on_phi_e_proxy`).
- **CSD-proxy probe:** reports the precomputed csd_proxy field; `second_derivative_or_divergence_proxy`.
- **EEG/MEG-proxy probes:** linear leadfield projections `Y = Q @ Lᵀ` with a fixed [C, K] leadfield supplied per protocol; pure linear instantiations of the observation operator. MEG uses `source_oriented` data by convention.
- **EMM-proxy:** a normalized activity/source/field cost readout with unit weights λ = 1.0 (analysis-only; not promoted to a physical claim).

All probe outputs carry semantic status: native (direct simulate output), relative_proxy (computational proxy, amplitude_semantics = relative), calibrated_proxy (none in this study), calibrated_physical (excluded), analysis_only (comparison only). The canonical source Q and all derived readouts are **Relative**; spectral summaries (e.g., PSD normalizations) are computed on relative units and reported as relative power.

## 7. Traveling-wave estimator (Ŵ)

All wave classifications in the C-series (60 cells {Q09, Q10}) and A-series use the single estimator `estimate_traveling_wave` (module SHA 684859a…; late revision disclosure below), configured exactly with the preregistered parameters: band [8.0, 13.0] Hz, minimum spatial coherence 0.55, R²-phase-traveling threshold 0.6, coherence-traveling threshold 0.55, noise band-power floor 1e-4. Pipeline per condition:

1. **Preprocessing:** per-site order-4 Butterworth band-pass (zero-phase filtfilt) in the band.
2. **Frequency estimate f̂:** summed-per-site spectral power via rfft; f̂ = argmax over in-band bins of the total (summed) power spectrum. The summed-site form is the D1 repair (see below); it equals the mean-trace spectrum for synchronous fields and remains well-defined for integer ring modes whose spatial mean cancels.
3. **Phasor:** per-site complex phasor via narrow DFT projection at f̂.
4. **Coherence:** mean resultant length |mean_i exp(i·Δφ_i)| of the wrapped nearest-neighbor phase differences (time-pooled), from the per-site Hilbert analytic phase (the preregistered `hilbert_analytic_phase_per_channel` method; the complex phasor at f̂ is used for the wave-vector fit and stability score); ≥ 0.55 required.
5. **Wave-vector fit:** least-squares linear fit of wrapped phase vs centered arc position; slope k̂ with fit R²; ω̂ = 2πf̂; v̂ = ω̂/|k̂|; direction = k̂/|k̂| (degrees from the depth/arc axis). Raw k̂ carries the sign convention k̂_raw = −k_true; sign canonicalization (direction error = arc-cos of the canonicalized unit vectors) is applied before error reporting.
6. **Noise floor:** if the mean in-band power falls below 1e-4 → NO_WAVE (noise_only).
7. **Stability gate:** cosine similarity between k̂ estimated on the first and second time halves; negative value flags standing/flipping structure (see D2 limitation below).
8. **Classification:** TRAVELING_WAVE requires coherence ≥ 0.55 and R² ≥ 0.6; otherwise structured_but_fails. NO_WAVE labels: synchronous oscillation (|k̂| < k_min), random phase (coherence below null), noise only (band power below floor), standing/flipping spatial gradient (k-stability < 0). UNRESOLVED is returned for non-finite estimates or fewer than 4 spatial samples. The C0 registration additionally lists "R² < 0.35 and coherence < 0.55" as an unresolved condition; the implemented decision tree treats that combination as structured_but_fails → NO_WAVE, and the distinction is not exercised by any executed condition in this study.
9. **Null score:** null_score = max(0, 1 − coherence).
10. **Output contract per condition:** classification, frequency_hz, wave_vector, direction, phase_velocity, phase_fit_r2, spatial_coherence, null_score, quality_reasons, finite_status.

**D1 repair disclosure (estimator revision, 2026-08-15):** the frozen C1-era estimator estimated frequency from the spatial-mean trace, which cancels for integer ring modes (vanishing spatial mean), returning f̂ = 9.5 Hz and k̂ = −0.987 for a true m = 1, f = 10.0 Hz synthetic wave. The defect was demonstrated on a ring-mode synthetic before any change; the repair (summed-site power argmax) was applied only to the demonstrated defect and the module was re-registered with SHA 684859a…; regression on the frozen corpora preserved the original classifications (60/60 C3 cells and 7/7 C1 cases identical before/after), and the freeze-era figure bytes are unchanged (fig05 SHA f7f9a494…). All A-series estimates below use the repaired module with unchanged thresholds.

**D2 limitation (standing-wave gate parity):** with an even number of spatial sites and Hermitian-equal ±ω sidebands, no single-frequency phasor method distinguishes standing from traveling patterns; the stability gate can fire on phasor degeneracy (off-resonance residue) rather than a standing signature. In every tested configuration the standing pattern still classifies NO_WAVE (possibly with reason standing_or_flipping_or_structured_but_fails); the coherence/R² gates are the backstop. Accordingly, the A-1a standing negative asserts classification == NO_WAVE; the reason-level expectation is recorded as a finding, not a gate. Effective estimator resolution floor: |k̂| ≥ k_min = (π/extent)/8, with extent = max_i |x_i − x̄| + 1e-9.

## 8. Estimator validation on synthetic fields (A-1a)

Synthetic fields are injected directly at the estimator input (no simulation): `Φ_i(t) = A·cos(k·x_i − ωt + φ₀) + ε_i(t)`, A = 1.0 relative, k = sign·m/R (rad per mm of arc length), ε ~ N(0, σ²) per site per step with seeded RNG, on the C3 ordered arc positions (24 sites, R = 1.0 mm, Δt 0.5 ms, 2000 ms).

- **Positive lattice:** 48 = 3×2×2×2×2 cases over frequency {8.5, 10.0, 12.5} Hz, mode m ∈ {1, 2}, sign ∈ {+1, −1}, φ₀ ∈ {0.0, 1.7} rad, noise σ ∈ {0.0, 0.25} (relative to A); fiducial case f = 10 Hz, m = 1, sign +, φ₀ = 0, σ = 0.
- **Recovery tolerances:** pass iff classification == TRAVELING_WAVE and finite and relative frequency error ≤ 0.05 (or absolute ≤ 0.5 Hz, whichever is looser), relative k-norm error ≤ 0.1, direction error ≤ 15°, relative velocity error ≤ 0.1, R² ≥ 0.6, coherence ≥ 0.55.
- **Negative controls (5 families):** sync_oscillation (k = 0), standing_wave, random_spatial_phases, noise_only, shuffled_coordinates_true_wave; each expected NO_WAVE.
- **Pass criterion:** all 48 positives and all negatives must classify as prescribed.
- **Measured performance (Relative error envelope at the tested grid) {Q02, Q03, Q04, Q05, Q06, Q07}:** ε_f = 0 across cases; ε_k̂ ≤ 0.00831 (relative), ε_θ = 0, ε_v ≤ 0.00824; the only sensitivity loss appears at amplitude A = 0.05 with σ = 0.5 (2/40 cases in the A-2 S1 grid below). A-1a cosmetizes only the estimator; the dynamical question of whether the network generates waves is A-1b.

## 9. Estimator sensitivity floor (A-2)

Estimator-only characterization on the same arc positions/dt; no thresholds or bands are altered.

- **S1 — amplitude × noise grid:** 40 = 5×4×2 cases over A ∈ {0.05, 0.1, 0.2, 0.5, 1.0} × σ ∈ {0, 0.1, 0.25, 0.5} × m ∈ {1, 2} (f = 10 Hz, sign +, φ₀ chosen), recording which gate fails first (band-power floor, k̂ < k_min, stability, coherence/R²). Result: all cases with amplitude ≥ 0.05 and σ ≤ 0.25 detected ("all 38 with amplitude ≥ 0.05 and σ ≤ 0.25 were detected {Q16}; only two failures at A = 0.05, σ = 0.5").
- **S2 — C3-regime embedding:** for 3 representative frozen NO_WAVE C3 cells (condition ordered_uniform; seeds 1001, 1002, 1009, covering all three frozen NO_WAVE reason classes: standing/flipping, synchronous, structured-but-fails), embed a known traveling wave: `Φ_mix = Φ_c3_bp + γ·rms_c3·cos(mx − ωt + φ₀)` with rms_c3 = √mean(Φ_c3_bandpassed²) computed from the cell's own frozen V_m before embedding, γ ∈ lattice up to 4.0, φ₀ ∈ {0.0, 1.57} (36 = 3×6×2 cases). Flip threshold γ* = smallest γ with TRAVELING_WAVE; ceiling reported as NO_FLIP_WITHIN_LATTICE. Result: γ* = 1.0 at both φ₀ for all 3 cells {Q17} — a genuine traveling component of in-band RMS amplitude is not hidden by the frozen C3 field structure.
- **S3 — duration × site count:** clean m = 1, f = 10 Hz wave over duration {250, 500, 1000, 2000} ms × sites {24, 12, 6} (uniform stride subsampling) = 12 cases {Q18}; classification reliability floor reported (NO_WAVE at 6 sites for 500 and 2000 ms durations, standing-gate parity limitation). Note on frequency resolution: the rfft bin spacing is fs/N = 2000/duration (0.5 Hz at full duration, 4 Hz at 250 ms), so at short durations the frequency estimate is quantized (e.g. 2 Hz off-bin at 250 ms) while the coherence/R² gates still classify correctly; frequency-error reporting in the S3 stage reflects this bin quantization, not estimator noise.

## 10. Dynamic search for a positive regime (A-1b)

- **Design:** 15 = 5×3 lattice points over v_c {Q11} ∈ {0.033, 0.065, 0.131, 0.262, 0.524} mm/ms × K ∈ {1, 2, 4}; × 3 seeds {1001, 1002, 1003} = 45 cells; delays from Eq. 6; all else identical to frozen C3 ordered conditions (same construction, same drive n0 pulse A = 50 at 200–201 ms and n6 pulse A = 45 at 800–801 ms, duration 2000 ms, Δt 0.5 ms, σ_n = 0.0, enable_hdp false).
- **Anchor:** v_c = 0.131, K = 1 must reproduce the frozen C3 ordered-uniform edges and emitter bitwise (construction identity) and the V_m traces bitwise {Q14}; any failure relabels the point as a nearby control with anchor flag false (criteria/outcomes unchanged). Verified: 3/3 seeds, max |ΔV_m| = 0.0.
- **Cell → point → domain decision tree (all preregistered, no adaptive extension):** cell invalid if NaN/Inf or max|V_m| > 150 (Relative mV) or non-finite estimator outputs; point POSITIVE if ≥ 2/3 cells TRAVELING_WAVE, MARGINAL if exactly 1/3, NEGATIVE if 0/3, UNRESOLVED if ≥ 1/3 invalid; domain POSITIVE_DOMAIN_FOUND (≥ 1 POSITIVE point), NO_POSITIVE_DOMAIN_IN_TESTED_RANGE (0 POSITIVE, 0 UNRESOLVED), UNRESOLVED (0 POSITIVE, ≥ 1 UNRESOLVED). MARGINAL points are reported but do not determine the outcome.
- **Outcome:** NO_POSITIVE_DOMAIN_IN_TESTED_RANGE {Q13} (27/11/7 point reasons distributed MARGINAL/NEGATIVE/… {Q15}; per-cell activity: 2 of 24 neurons fired, max rate 0.5 Hz {Q15} — near-silent, no adaptive extension observed). Interpretation constraint honored in Results: no claim beyond "no TRAVELING_WAVE classification in the tested delay family by the same estimator that classified all of A-1a and C3". No point, criterion, or decision rule was added, removed, or redefined after outcomes were observed.
- The background near-silence is a stated limitation of the tested range, and is consistent with the frozen C3 no-wave outcome; the tested range is a subset of delay-parameter space, not evidence about it.

## 11. Protocol-specific interventions and simulation policies

**Protocol C (C3, 6 conditions, 10 seeds {1001…1010}):** factors geometry {ordered, shuffled} × delay policy {uniform (4 steps), geometry-derived, delay-shuffled}; duration 2000 ms @ Δt 0.5 ms, float32, σ_n = 0.0, enable_hdp/homeostasis/rbd false, record Q (sources), no fields for C3 classification. Drive: 1-ms pulses A = 50 at t = 200 ms on neuron 0 and A = 45 at t = 800 ms on neuron 6 (Relative current). Result classes per condition: NO_WAVE; estimator failure points (D2 parity at 500/2000 ms measured separately).

**Protocol D3 (adaptation/recovery phenotype, 36 cells = 3 seeds {11, 12, 13} × 4 arms × 3 recovery intervals):** paradigm: baseline (100 ms, no drive) → repeated stimulation (6 identical pulses, amplitude 15 (Relative), duration 40 ms, onset-to-onset ISI 60 ms; train block ends at 440 ms) → recovery interval T_recovery ∈ {50, 100, 250} ms (2·τ_A / τ_K / 2.5·τ_K, prospective from D2b timescales τ_A = 25, τ_K = 100 ms) → rechallenge pulse (identical) → post window; total 1000 ms. Arms: N0 classical emitter (no RBS), N1 static H_K = 1, N2 κAK = 0 (D2a-equivalent), D full D2b activity-writing. Primary response R_j = spike count in the fixed 80 ms post-onset window of pulse j; A_adapt = 1 − R_late/R_early with R_early = mean over pulses 1–2 and R_late = mean over pulses 5–6 (defined only when R_early > 0); R_recovery = (R_rechallenge − R_late)/(R_early − R_late) (secondary). Classification: ADAPTATION iff A_adapt > θ_A = 0.15 and mean H_K over late train pulses > 1 + θ_H = 1.01 and signal quality sufficient (min mean R_early ≥ 1.0); NO_ADAPTATION is a valid outcome; UNRESOLVED for insufficient signal (silence is never classified as adaptation). Result: 27 ADAPTATION / 9 NO_ADAPTATION, with the D arm uniformly NO_ADAPTATION and the D − N2 null indistinguishable (A_adapt 0.2857 vs 0.2857 at seed 11/short) {Q20, Q25}; gates evaluated where H_K is written (D arm) {Q20}. A_adapt is a relative difference index, not a rate difference.

**Protocol H4 (geometry-memory matrix, 4 cells):** short rings (3 neurons) vs long rings (12 neurons) × uniform (4 steps) vs heterogeneous ({2, 8} steps) delays, dt 1.0 ms, n_steps 80, weight 6.0, τ_syn 3.0; H3-style RBS config (β_h 0.5, τ_h 80 ms, κ_h 0.0, family f1); identity-decoding assay with lags Δ ∈ {2, 5, 10, 20, 35} steps, 8 shuffle nulls, ridge λ = 0.01, train seeds {100…109}, test seeds {200…209}. Endpoint per cell: M_X_area = ∫ [M_X(Δ) − M_X_shuffle(Δ)]₊ dΔ {Q19} (substrate identity of perturbation state, decoded from activity). Factorial estimates {Q29}: μ = 0, α_length = 0, α_heterogeneity = +0.0521, α_interaction = −0.0521 (point estimates; bootstrap confidence intervals were not computed, and +0.0521 is a prospective point estimate only, not statistically established). The endpoint is a Relative decoding-area without absolute memory-capacity semantics.

**Protocol E-series (hierarchy, delays, RBS, observation, causal perturbation):** two-area laminar network (A1, A2; cortical_eig preset; E drive 8.0, PV drive 6.0, Relative; duration 1000 ms @ Δt 0.5 ms; seeds {11, 12, 13}); E1 identity/bio-identity hierarchy with typed FF/FB ownership; E2 typed provenance-class delays (zero delays recover E1); E3 sparse D1/D2a-type H_K RBS owned by A2:L5:E (flat indices 70–76, Γ_H: H_K → b_eff = H_K·b, τ_K = 100 ms, H_K0 = 1.2 on owners at t₀⁺); E4 downstream observation chain per the frozen workflow; E5 causal perturbation with arms N0 (Γ_H = I, dot H_K = 0), N1 (technical: H_K trajectory identical to D on owners; Γ_H = I), D (Γ_H = H_K·b active on owners). Contrast D − N1 {Q23}; owner-primary gate threshold 1e-6 (any owner contrast above float32 numerical noise); HIERARCHICAL_PROPAGATION requires owner gate plus ≥ 1 of {X_A1, Q, Y} levels; produced every-seed HIERARCHICAL_PROPAGATION (3/3 seeds) with owner contrast (mean abs V_m deviation 9.26 mV (Relative), +7 spikes across seeds 11–13) and downstream contrasts A2 non-owner 2.43 mV / 0 spikes, A1 3.16 mV / +9 spikes {Q24, Q30}; measurements reported as Relative deviations (mean abs V_m deviation in Relative mV, spike-count differences, V_m time-integral differences). All readouts post-hoc on one simulated trajectory per arm/seed; A1 effects are structural propagation along the frozen A2 → A1 feedback pathway only — no spectral/functional FF/FB claim (same language as the Results, Q30 row).

**Experiment A (operator factorization, 40-neuron column, seed 7, 2000 ms @ 0.5 ms):** N = 40, layers L2/3: L4: L5 = 0.33 : 0.34 : 0.33, E 70% / PV 30%, cortical_eig; drives E 8.0, PV 8.0 (Relative); X = {V_m, spikes}, H = ones (HDP off), Q = canonical relative source; F operators: lfp_ref (project_laminar_sources, 16 contacts, width 0.10) plus 4 declared F variants; P probes: lfp_contact_shallow {0.2} (width 0.10) and 5 declared P variants; field geometry laminar column, mean-zero Neumann boundary, mean-zero gauge; spectral burn-in 200 ms excluded from summaries.

## 12. Statistics and generalization regimes

No statistical tool in this study computes or reports p-values; all quantitative statements belong to one of five declared evidence regimes (frozen vocabulary):

1. **deterministic property** — bit-exact/structural comparisons at fixed configuration (e.g. H_K=1 → classical trajectory identical, G1-G2; estimator regression 60/60; E5 G-gates);
2. **parameter-domain characterization** — full-grid statements over a preregistered lattice (A-1b domain outcome, W3b counts, A-2 S1/S3 grids);
3. **seed robustness** — consistency across 3–10 fixed seeds (C3 10 seeds, D/E/A-series 3 seeds);
4. **model generalization** — statements about the steward-defined family of configurations (never claimed beyond the tested family);
5. **empirical inference** — prospective hypothesis tests where effect size ≠ 0 is claimed (H4 factorial point estimates, with the explicit caveat that +0.0521 is a point estimate without bootstrap CIs; the H4 conclusion is the absence of a positive length effect in the tested domain, not statistical evidence of no effect).

Reported spread values are ranges or exact extremal values (min–max) across the enumerated seeds/cells; uncertainties are declared as ranges, not standard errors. Where an endpoint could not be computed (e.g., W3b stability classification of X points; H4 bootstrap CIs; standing-pattern reason parity), the manuscript states exactly what was not established, and unresolved classes are never collapsed into negatives (UNRESOLVED → NO_WAVE forbidden; N_S = 0 ≠ negative without N_X = 0) {Q21}.

**W3b parameter-domain map (the one fully executed lattice):** the HDP parameter law (Eq. 7 + weight rules) was analyzed on a frozen 3×3×3×3×3 factorial lattice over (κ_H, κ_W, λ_W, τ_H, τ_W) × 9 operating points I_tonic ∈ {0, 5, …, 40}: κ_H ∈ {0.02, 0.05, 0.1}, κ_W ∈ {0.5, 1.0, 2.0}, λ_W ∈ {0.05, 0.1, 0.2}, τ_H ∈ {60, 80, 120} ms, τ_W ∈ {80, 100, 150} ms = 243 parameter combinations × 9 drive points = 2187 cells, all seeded and deterministic. Regime labels per cell: D dormant/vanishing feedback; S robustly stable active HDP (Floquet margin m_F = 1 − ρ_nonneutral > 0.02, ρ excluding multipliers with ||λ|−1| < 0.05, period ≥ 2 with a spike in the period window, mean synaptic activity > 1e-3, nontrivial HDP activity L_HDP > 1e-6, timescale r_tau = (τ_W/λ_W)/τ_H > 1); C near-critical; U unstable (negative evidence); X active but stability-unresolved (X ≠ U; never convertible to a negative). N_S counts the useful domain D_useful = active ∩ stable ∩ nontrivial ∩ timescale (a subset of S); the counts are D = 243, S = 0, C = 0, U = 0, X = 1944 {Q21}. N_S = 0 is a count of zero S-points; with N_X = 1944 > 0 the useful domain is not claimed scientifically empty, no operating point is selected from the map, and no closed-loop HDP memory claim rests on it (W3 remains unauthorized). The dimensionless coordinates r_tau and Γ_HDP = |2κ_W b_HW/(a_H λ_W)| are explanatory only; regime labels use the implementation-faithful Floquet analysis as the classifier. This analysis is a parameter-domain characterization, not a parameter tuning.

## 13. Quality gates and decision rules (cross-cutting)

- **Estimator honesty gates:** same estimator + thresholds across C3/A-1a/A-1b/A-2; upstream-feed prohibition; frozen parameters before outcomes.
- **Construction gates:** bit-exact reproduction of the frozen lineage by any derived construction (A-1b anchor; A-2 S2 uses frozen C3 replay; A-3 uses the C3 ring construction).
- **Mechanism-vs-observable separation:** D3 classification requires both the observable (A_adapt > θ_A) and the mechanism condition (H_K late mean > 1 + θ_H); E5 requires owner expression to be propagated through declared gates 1e-6.
- **Arm-isolation gates:** E5 G1 (N1 and D share H_K trajectories on owners; only Γ_H differs) — verified bit-exact H_K(N1) == H_K(D) 3/3 seeds; G2 N0 ≡ N1 at H_K = 1 (bit-exact V_m/spikes/Q); G3 owner contrast measured per seed with declared metrics {Q30}; all G1–G10 pass; single source of truth (one simulate per arm/seed).
- **Overclaim guards:** structural feedback → "structural propagation only" language; near-silent activity reported as 2/24 neurons at ≤ 0.5 Hz in A-1b; no phenotype (spectral/FB-FF/adaptation/memory) claims beyond the tested gates.

## 14. Reproducibility, environment, and provenance

- **Environment (executed):** Python 3.13.7; JAX CPU with float32 default (float64 used where declared: A-2 S2/S3 estimator inputs, estimator spectral arithmetic); NumPy; SciPy == 1.17.1; matplotlib >= 3.10.9, <3.11; jaxfne at baseline `dev@8cc60a6…` (each frozen receipt records its own git SHA; figures frozen at fig05 SHA f7f9a494…; estimator module SHA 684859a…).
- **Determinism:** σ_n = 0.0 in all protocol runs; per-seed `jax.random.PRNGKey(seed)`; float32 arithmetic; identical trajectories across runs verified bitwise where declared (sequencing, continuation, replay).
- **Audit trail:** every protocol has a frozen spec (schema-versioned, write-once status) plus execution/interpretation receipts at declared paths; the Results claim ledger maps each claim to its receipt; the Methods traceability map (companion document) maps each method element to its code authority and configuration/evidence authority with verification status.
- **Clean-room reproduction:** a separate clean-room rerun (frozen env, 15/15 checks) reproduces the protocol-C figure pipeline — C4-adjacent codes and classifications reproduce outside the authoring environment.
- **Language control:** public-facing guard `scripts/audit_public_docs_language.py --check` enforces the Relative/Absolute statement discipline on this document and its companions.