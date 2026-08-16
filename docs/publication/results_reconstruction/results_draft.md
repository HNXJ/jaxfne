# Results

> Reconstruction checkpoint: `publication_results_reconstruction` — authorizing baseline `dev@e91582ec`.
> Paragraph tags `[P1]…[P21]` and quantitative markers `{Q01}…` are audited by
> `scripts/audit_results_draft.py` (quantitative + forbidden-overclaim checks) — see `traceability_map.md`.

## Results

**[P1] A single object grammar and one frozen source support typed readouts.**

jaxfne is a JAX-based neural simulator whose computational pipeline follows a single object
grammar — Configuration → Net → Paradigm → Objective → Trainer → Signals →
Visualization/Export — with a single `construct()` dispatch entry point unifying three
configuration tiers (flat `Configuration`, structured `NeuronalTensor`, and canonical-column
`HDPColumnConfig`) onto one `Model`. Fig. 1 is a representational grammar map; it reports no
empirical quantity {CL-01; Fig. 1; `artifacts/publication/fig01_grammar_spec.json`}.

**[P2] All observation figures derive from a single canonical relative source.** The
emitter-source (Fig. 2), local-observation (Fig. 3) and multiscale-boundary (Fig. 4) figures
are rendered from one canonical source realization Q, with the cross-figure invariant
hash(Q_Fig2) = hash(Q_Fig3) = hash(Q_Fig4) {Q01} {CL-02; Fig. 2–4;
`artifacts/publication/fig02_04_experiment_a_spec.json`}. This is a deterministic numerical
property of the frozen evidence set; no physical calibration is claimed.

**[P3]** Local field-potential and current-source-density traces are relative computational
readouts of that same frozen source Q, not experimental recordings {Q01} {CL-03; Fig. 3}.
The toy EEG/MEG leadfields of Fig. 4 are analysis-only computational readouts for which
physical amplitude calibration was not performed {CL-04; Fig. 4}. No proxy-equals-experiment
equivalence is claimed at any point.

**[P4] A validated traveling-wave estimator.**

The traveling-wave estimator is applied to membrane-potential trajectories in the 8–13 Hz
band, requiring a spatial coherence of at least 0.55, a phase-fit R² of at least 0.6, a noise
floor of 1e-4, and an explicit null-score check
{`artifacts/protocol_c/c0_wave_protocol_spec.json`, module SHA
`684859a98da51de79887ec26ba8d7134e2fa0e97`}. Estimator validity is established on synthetic
controls only {CL-05}.

**[P5] Post-freeze validation of the estimator (A-1a).** *The runs in this paragraph were
executed after the 0.4.17 feature freeze as reviewer-motivated validation
(`protocol_c_p2v_a1a`); they are reported separately from the frozen experiments below.* On
48 synthetic positive cases spanning the preregistered band — 24-neuron ring geometry, ring
radius 1.0 mm, frequencies 8.5–12.5 Hz, phase velocities 53.4→39.3 mm/s {Q08} — the estimator
detected all 48 {Q02}, with zero frequency error {Q03}, wave-number error bounded by 0.00831
relative {Q04}, zero direction error {Q05}, and phase-velocity error bounded by 0.00824
relative {Q06}; all 5 synthetic negative controls were correctly rejected {Q07}
{CL-05; A-1a; `artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json`}.

**[P6] Preregistered null coverage of ring regimes.** The frozen protocol screened
60 cells — 6 conditions (ordered/shuffled × uniform/geometry-derived/delay-shuffled delay
policies) × 10 seeds — on the 24-neuron ring, 2000 ms at dt = 0.5 ms {Q09}. All 60 cells were
classified NO_WAVE {Q10}, by three quality reasons: 52 synchronous oscillations with
near-zero wave number, 4 standing/flipping spatial gradients, 4 structured-but-fails
traveling gates {Q10}. A scope qualifier applies: geometry-derived delays collapsed to the
same four-step delay as the uniform condition on this ring, so the experiment does not test
genuinely distance-heterogeneous conduction {CL-06; Fig. 5;
`artifacts/protocol_c/c3_execution_receipt.json`; `artifacts/publication/fig05_generation_receipt.json`}.

**[P7] Post-freeze predeclared dynamic search (A-1b).** *Post-freeze reviewer-motivated
validation (`protocol_c_p2v_a1b`).* To widen the tested regime beyond the frozen grid, a
predeclared search enumerated 15 points — five phase velocities (0.033–0.524 mm/ms) × three
neighbor skips K ∈ {1, 2, 4} — each with its exact multi-step delay signature embedded in the
frozen specification; each point was run on 3 seeds (45 cells total) {Q11}. The anchor point
(vc0.131_k1) is bitwise identical to the frozen ring for all 3 seeds (max |ΔV_m| = 0.0)
{Q14}. All 45 cells were classified NO_WAVE with 0 invalid cells {Q12}, yielding the domain
outcome NO_POSITIVE_DOMAIN_IN_TESTED_RANGE {Q13}. The failure reasons mirror the frozen grid
(27 standing/flipping, 11 synchronous, 7 structured-but-fails) {Q15}; per cell only 2
neurons spiked (max rate 0.5 Hz), i.e. the activity is a drive-echo regime, and no adaptive
extension of the estimator's integration window was observed {Q15}
{CL-06; A-1b; `artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json`}.

**[P8] Post-freeze sensitivity floor (A-2).** *Post-freeze reviewer-motivated validation
(`protocol_c_p2v_a2`).* Across 40 amplitude/noise cases (amplitudes 0.05–1.0, noise σ
0–0.5) {Q16}, detection is reliable for amplitude 0.05 and σ ≤ 0.25; the only failures occur
at the corner of lowest amplitude with highest noise (A = 0.05, σ = 0.5). Across the three
frozen-grid carrier cells, the C3-embedding flip threshold γ* = 1.0 on all three cells and
for both phase initializations {Q17}. Across 12 duration/site cases {Q18}, detection is
reliable for ≥ 12 recording sites at all tested durations (250–2000 ms); 6 sites is
parity-dependent (NO_WAVE at 500 ms and 2000 ms)
{CL-05, CL-06; A-2; `artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json`}.

**[P9]** Taken together, the validated traveling-wave estimator detected no traveling waves
across the preregistered neural geometry/delay conditions (Fig. 5, outcome letter C). No
claim is made about regimes outside the tested geometry/delay domain, and no claim of wave
absence outside this simulator is made {CL-05, CL-06; Fig. 5}.

**[P10] Typed vector RBS extends the classical emitter.** Within one mechanism subsystem,
the typed vector resting-state (RBS) extension writes hidden states (H_K, H_A) into model
parameters; setting H_K = 1 exactly restores the base emitter {CL-07; Fig. 6A/B}. The
extension adds no new biological mechanism and no new neuron type — it is a
deterministic numerical property of the same Izhikevich machinery {CL-07}.

**[P11]** The RBS state retains perturbation history across time, and with the H1c coupling
(β_H) that history is decodably expressed in neural activity (P1/P2). This is a
machinery-level claim: recurrent geometry and delays do not extend this memory (see H4
below), and sign-symmetry is not presented as memory {CL-09; Fig. 6A/B}.

**[P12] H4: longer recurrent length and heterogeneous delays do not produce a positive
memory-extension effect.** Under the preregistered H4 identity-decoding assay on the
24-neuron ring, the four-factorial endpoint measures are M_X = 0.0 (short, uniform), 0.0521
(short, heterogeneous), 0.0 (long, uniform), 0.0 (long, heterogeneous) {Q19}; the length
effect estimate α_length = 0.0 {Q19}. The only nonzero point, short+heterogeneous +0.052, is
exploratory and single-cell: bootstrap confidence intervals were not computed, and it is not
presented as a confirmed heterogeneity effect. The conclusion is limited to "no positive
effect detected by the preregistered assay"; no generalised claim about recurrent length is
made
{CL-10; Fig. 6C; `artifacts/protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json`}.

**[P13]** A written parameter ω, frozen during expression, changes neural dynamics through W
in the minimal A↔B pair (W1/W2) {CL-12; Fig. 6D/E}. Closed-loop feedback (W3) is not
demonstrated, and no memory persistence through closed-loop HDP is claimed {CL-12}.

**[P14] D3: observable spike attenuation without formal adaptation.** Across 36 cells × 4
arms (N0, N1, N2, D; seeds 11–13 × recovery intervals) {Q25}, the D arm shows observable
spike-count attenuation with A_adapt > 0.15 in 9 of 9 cells {Q20}; formal ADAPTATION
classification failed in 0 of 9 cells because the joint mechanism gates are not satisfied
(M1 mechanism pass 9/9; M2, H_K_late > 1 + θ_H, pass 0/9) {Q20}. The all-arms classification
counts are N0/N1/N2: 9 ADAPTATION each; D: 9 NO_ADAPTATION {Q20}. Observably, attenuation is
identical across arms (D minus N2 null on A_adapt, 0.2857 vs 0.2857) {Q20}; therefore
attenuation is not attributable to hidden-state (H_K) writing, and the phenotype is recorded
as NO_ADAPTATION {CL-08; Fig. 6G}. Attenuation is presented as attenuation, not adaptation
{CL-08}.

**[P15] W3b: the active HDP stability region remains unresolved.** Across the frozen W3b
lattice of 2187 points {Q21}, 243 points are classified D (actively forgetting), 0 as
robust-active S, 0 as C, 0 as U, and 1944 points remain active-unresolved X {Q21}. With
N_S = 0, no robust-active point was classified {Q21}, and no useful robust-active HDP domain
is claimed; the stability classification of the active region remains open — this is not a
negative result {CL-11, CL-19; Fig. 6F;
`artifacts/protocol_w/w3b_parameter_domain/w3b_domain_receipt.json`}.

**[P16] Post-freeze boundedness of the tested HDP trajectories (A-3).** *Post-freeze
reviewer-motivated validation (`protocol_c_p2v_a3`).* On the C3 ring anchor with the default
HDP kernel, H stays within [1.0000, 1.0008] and |w| is fixed at 6.0, with absolute growth
ratio 1.00000; with the DESYNC preset, H ∈ [1.0000, 1.0310], |w| ∈ [5.8692, 6.1684], growth
ratio 1.00004 {Q22}. All per-step hard-bound invariants pass — H ⊂ [0.1, 10], |w| ⊂ [0.01,
10], |V| ≤ 150, |u| ≤ 2000, |syn| ≤ 1e4 — and no weight-tuning was observed under the kernel
defaults (K_ctrl = 5.0, K_w_ctrl = 0.0) {Q22}. These trajectories remained bounded over the
tested parameter and time domain; this does not resolve the W3b stability classification of
the active region {CL-19; A-3; `artifacts/protocol_c/p2v_a3_hdp_boundedness/p2v_a3_receipt.json`}.

**[P17] Hierarchy, delays, and local RBS compose.** A minimal two-area laminar hierarchy
supports typed feedforward/feedback ownership with identity round-trip {CL-13; Fig. 7A};
no functional spectral claim and no biological specialization claim is made {CL-13}. Typed
pathway delays compose with the hierarchy: setting the delay to zero recovers the E1
semantics exactly {CL-14; Fig. 7B}. Sparse, owned H_K RBS composes with the hierarchy and
delays: setting H = H* recovers the E2 semantics exactly; no HDP/plasticity claim is made in
this composition {CL-15; Fig. 7C}.

**[P18]** Observations are pure downstream compositions of the frozen substrate state:
disabling observation recovers the substrate state exactly, and observations never feed back
into the substrate (no closed loop in this configuration) {CL-16; Fig. 7D}.

**[P19] Causal disablement and propagation.** With expression disabled (Γ_H = I),
perturbing H_K leaves X unchanged bit-exactly — the negative control of the propagation
result {Q23} {CL-17; Fig. 7E}. When the same local H_K perturbation is expressed through its
typed Γ_H coupling, it propagates through the existing hierarchical connectivity: the E5
protocol (3 arms N0/N1/D × 3 seeds = 9 trajectories) {Q24} passes its sanity gates
(N0 ≡ N1 bit-exactly on V_m, spikes and Q across all 3 seeds {Q24}; arm-isolation gate
H_K(N1) ≡ H_K(D) bit-exactly {Q24}) and classifies every seed as HIERARCHICAL_PROPAGATION
(3/3) {Q24} {CL-18; Fig. 7E/F;
`artifacts/protocol_e_integration/e5_execution_receipt.json`}. No spectral-effect statement
of any kind is made for this propagation, and no cognition framing is invoked {CL-18}.

**[P20] Provenance and reproducibility of the evidence set.** All main figures were rendered
from a single frozen evidence set with per-file SHA-256 provenance: each generation receipt
records a figure SHA-256, and all seven declared hashes match the shipped figure bytes
(7/7) {Q27} {CL-20}. The recorded equivalence gate confirms byte equality for 7/7 figure
cases across the frozen and re-rendered artifacts {Q26}
{`artifacts/publication/equivalence_report.json`}. The figure regeneration path is
reproducible from a clean checkout per the documented entrypoint, with pinned dependencies
(matplotlib ≥ 3.10.9, < 3.11; scipy == 1.17.1) {Q28}. No claim of external byte-for-byte
reproduction in every downstream environment is made beyond the recorded equivalence gate
{CL-20, CL-21}.

## Appendix A — Evidence ledger used (machine-readable)

The quantitative markers above are verified by `scripts/audit_results_draft.py`, which loads
each cited receipt and asserts the quoted values; the sentence-level mapping
paragraph → claim → artifact is maintained in `traceability_map.md`.

## Appendix B — Post-freeze validation summary

| Tag | Protocol | Checkpoint | Receipt |
|-----|----------|------------|---------|
| A-1a | estimator synthetic control (53 cases) | post-freeze | `artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json` |
| A-1b | predeclared dynamic search (15 points × 3 seeds) | post-freeze | `artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json` |
| A-2 | sensitivity floor (S1/S2/S3) | post-freeze | `artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json` |
| A-3 | HDP trajectory boundedness | post-freeze | `artifacts/protocol_c/p2v_a3_hdp_boundedness/p2v_a3_receipt.json` |

All four checkpoints were executed after the 0.4.17 feature freeze (authorizing baseline
`dev@e91582ec`), against the frozen specs and the repaired estimator module (SHA
`684859a98da51de79887ec26ba8d7134e2fa0e97`); their results are reported separately from the
frozen experiments in the paragraphs above and never change the frozen claims.