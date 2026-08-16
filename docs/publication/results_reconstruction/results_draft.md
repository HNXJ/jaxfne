# Results

## One object grammar

jaxfne is a JAX-based neural simulator whose computational pipeline follows a single object
grammar — Configuration → Net → Paradigm → Objective → Trainer → Signals →
Visualization/Export — with a single `construct()` dispatch entry point unifying three
configuration tiers onto one `Model`. Fig. 1 is a representational grammar map of this
pipeline; it reports no empirical quantity.

## One frozen canonical source

The emitter-source (Fig. 2), local-observation (Fig. 3) and multiscale-boundary (Fig. 4)
figures are rendered from one canonical source realization Q, with the cross-figure
invariant hash(Q_Fig2) = hash(Q_Fig3) = hash(Q_Fig4) {Q01}. This is a deterministic
numerical property of the frozen evidence set; no physical calibration was performed.

## Relative computational readouts

Local field-potential and current-source-density traces are relative computational readouts
of that same frozen source Q, not experimental recordings {Q01}. The toy EEG/MEG leadfields
of Fig. 4 are analysis-only computational readouts for which physical amplitude calibration
was not performed. These quantities are defined on the same evidence regime as Q; no
equivalence with experimental recordings exists within this paper.

## A validated traveling-wave estimator

The traveling-wave estimator operates on membrane-potential trajectories in the 8–13 Hz
band with preregistered coherence, phase-fit, noise-floor and null-score gates
(Methods). Its validity is established on
synthetic controls only . The estimator module underwent one revision after the
primary analyses were registered (final module revision SHA
`684859a98da51de79887ec26ba8d7134e2fa0e97`; Appendix): the published NO_WAVE figure bytes
and the registered classification record are unchanged since original registration {Q27},
and all recovery and search results below were re-derived with the final revision
{Appendix}.

## Estimator recovery bounds (secondary registered validation)

These runs were executed as secondary, independently registered validation after the
primary protocol suite was registered, and are reported separately (internal tag A-1a;
Appendix). On 48 synthetic travelling-wave cases spanning the preregistered band — 24-neuron
ring geometry, ring radius 1.0 mm, frequencies 8.5–12.5 Hz, phase velocities 53.4 to 39.3
mm/s {Q08} — the estimator recovered all 48 {Q02}, with zero frequency error {Q03},
wave-number error bounded by 0.00831 relative {Q04}, zero direction error {Q05}, and
phase-velocity error bounded by 0.00824 relative {Q06}. Five synthetic negative controls —
synchronous oscillation, standing wave, random spatial phases, noise-only, and a true
travelling plane with shuffled sensor coordinates — were all correctly rejected {Q07}
.

## Preregistered null coverage of ring regimes

The frozen protocol screened 60 cells — 6 conditions (ordered/shuffled geometry ×
uniform/geometry-derived/delay-shuffled delay policies) × 10 seeds — on the 24-neuron ring,
2000 ms at dt = 0.5 ms {Q09}. All 60 cells were classified NO_WAVE {Q10}, for three
reasons: 52 synchronous oscillations with near-zero wave number, 4 standing/flipping
spatial gradients, 4 structured-but-fails traveling gates {Q10}. A scope qualifier applies:
geometry-derived delays collapsed to the same four-step delay as the uniform condition on
this ring, so this leg does not exercise genuinely distance-heterogeneous conduction
.

## Predeclared dynamic search (secondary registered validation)

Secondary, independently registered validation (internal tag A-1b; Appendix): a predeclared
search enumerated 15 points — five phase velocities (0.033–0.524 mm/ms) × three neighbor
skips K ∈ {1, 2, 4} — each with its exact multi-step delay signature embedded in the frozen
specification; each point was run on 3 seeds (45 cells total) {Q11}. The anchor point
(vc0.131_k1) is bitwise identical to the frozen ring for all 3 seeds (max |ΔV_m| = 0.0)
{Q14}. All 45 cells were classified NO_WAVE with 0 invalid cells {Q12}, yielding the
predeclared outcome NO_POSITIVE_DOMAIN_IN_TESTED_RANGE {Q13}; failure reasons mirror the
grid (27 standing/flipping, 11 synchronous, 7 structured-but-fails) {Q15}. Because only 2
of 24 neurons spiked per cell (max rate 0.5 Hz), this leg exercised the (velocity ×
neighbour-skip) delay family under near-silent activity: together with the grid leg's
delay collapse, no tested combination of active dynamics and genuinely heterogeneous
delays was available, and no result is claimed outside the tested combinations
.

## Estimator sensitivity floor (secondary registered validation)

Secondary, independently registered validation (internal tag A-2; Appendix). Of the 40
tested amplitude/noise cases (amplitudes 0.05–1.0, noise σ 0–0.5) {Q16}, all 38 with
amplitude ≥ 0.05 and σ ≤ 0.25 were detected; the only two failures occur at the lowest-
amplitude/highest-noise corner (A = 0.05, σ = 0.5). Across the three grid carrier cells,
the C3-embedding flip threshold γ* = 1.0 on all three cells for both tested phase
initializations {Q17}. Across 12 duration/site cases {Q18}, detection is reliable for ≥ 12
recording sites at all tested durations (250–2000 ms); 6 sites is parity-dependent (NO_WAVE
at 500 ms and 2000 ms) .

## No traveling waves in the tested regimes

Taken together, the validated traveling-wave estimator detected no traveling waves across
the preregistered neural geometry/delay conditions (Fig. 5, outcome letter C): the frozen
60-cell grid, the predeclared 45-cell delay-family search, and the sensitivity floor all
return NO_WAVE or grid-defined positive detection only. No claim is made for regimes
outside the tested geometry/delay domain, and none outside this simulator.

## Typed vector RBS extends the classical emitter

Within one mechanism subsystem, the typed vector resting-state (RBS) extension writes
internal state variables (H_K, H_A) into model parameters; setting H_K = 1 exactly restores
the base emitter . The extension re-parameterises the existing Izhikevich machinery:
it introduces no additional biological mechanism and no new neuron type.

## State writing and expression

The RBS state retains perturbation history across time, and with the H1c coupling (β_H)
that history is decodably expressed in neural activity. This is a machinery-level claim:
recurrent geometry and delays do not extend this memory (below), and sign-symmetry is not
presented as memory.

## H4: no positive length effect on decoded state memory

Under the preregistered H4 identity-decoding assay on the 24-neuron ring, the four-factorial
endpoint measures are M_X = 0.0 (short, uniform), 0.0521 (short, heterogeneous), 0.0 (long,
uniform), 0.0 (long, heterogeneous) {Q19}; factorial estimates are α_length = 0.0,
α_heterogeneity = +0.0521, α_interaction = −0.0521 {Q29}. The interaction cancels the
short-heterogeneous signal in the long arm, so no general heterogeneity effect survives.
The +0.052 point is exploratory and single-cell: bootstrap confidence intervals were not
computed, and the conclusion is limited to "no positive effect detected by the preregistered
assay" .

## Expression through a written parameter

A written parameter ω, frozen during expression, changes neural dynamics through W in the
minimal A↔B pair (W1/W2) . Closed-loop feedback (W3) is not demonstrated, and no
memory persistence through closed-loop dynamics is claimed.

## D3: observable spike attenuation without formal adaptation

Across 36 cells × 4 arms (N0, N1, N2, D; 3 seeds × 3 recovery intervals) {Q25}, the D arm
shows observable spike-count attenuation with A_adapt > 0.15 in 9 of 9 cells {Q20}; formal
ADAPTATION classification failed in 0 of 9 D cells because the joint mechanism gates are
not satisfied — the state-formation gate passes 9/9 while the late-state gate (H_K_late >
1 + θ_H) fails 0/9 {Q20}. The all-arms classification counts are N0/N1/N2: 9 ADAPTATION
each; D: 9 NO_ADAPTATION {Q20}; the mechanism gates are evaluated where H_K is written (the
D arm), so the null arms' ADAPTATION labels reflect matching observables without the gate,
not a differing observable. Observably, attenuation is identical across arms (D minus N2
null on A_adapt, 0.2857 vs 0.2857) {Q20}; therefore attenuation is not attributable to
H_K writing, and the D arm is classified and reported as NO_ADAPTATION — attenuation is
reported as attenuation, not adaptation.

## W3b: the active stability region remains unresolved

Across the frozen W3b lattice of 2187 points {Q21}, 243 are classified D (dormant/vanishing
feedback), 0 robustly stable S, 0 near-critical C, 0 unstable U, and 1944 remain active but
stability-unresolved X {Q21}. With N_S = 0, no robustly stable active point was classified
{Q21}, and no useful robust-active HDP domain is claimed; the stability classification of
the active region remains open — this is not a negative result
.

## Boundedness of the tested HDP trajectories (secondary registered validation)

Secondary, independently registered validation (internal tag A-3; Appendix). On the C3 ring
anchor with the default HDP kernel, the trajectories stay deep inside the per-step clamp
bounds: H ∈ [1.0000, 1.0008] with |w| fixed at its initial 6.0 (growth ratio 1.00000); with
the DESYNC preset H ∈ [1.0000, 1.0310], |w| ∈ [5.8692, 6.1684], growth ratio 1.00004 {Q22}.
No tested state approaches a clamp boundary — H ⊂ [0.1, 10], |w| ⊂ [0.01, 10], |V| ≤ 150,
|u| ≤ 2000, |syn| ≤ 1e4 — and no adaptive weight tuning occurred (control gains at kernel
defaults; K_w_ctrl = 0.0) {Q22}. These trajectories remained bounded over the tested
parameter and time domain; this does not resolve the W3b stability classification of the
active region .

## Hierarchy, delays, and local RBS compose

A minimal two-area laminar hierarchy supports typed feedforward/feedback ownership with
identity round-trip ; no functional spectral claim and no biological specialization
claim is made. Typed pathway delays compose with the hierarchy: setting the delay to zero
recovers the E1 semantics exactly . Sparse, owned H_K RBS composes with the
hierarchy and delays: setting H = H* recovers the E2 semantics exactly; no plasticity claim
is made in this composition .

## Observations are downstream

Observations are pure downstream compositions of the frozen substrate state: disabling
observation recovers the substrate state exactly, and observations never feed back into the
substrate .

## Causal disablement and propagation

With expression disabled (Γ_H = I), perturbing H_K leaves X unchanged bit-exactly — the
negative control of the propagation result {Q23} . When the same local H_K
perturbation is expressed through its typed Γ_H coupling, the E5 protocol — 3 arms
(N0/N1/D) × 3 seeds = 9 trajectories {Q24} — passes its sanity gates (N0 ≡ N1 bit-exactly
on V_m, spikes and Q across all 3 seeds {Q24}; H_K identical between the N1 and D arms
{Q24}) and classifies every seed as HIERARCHICAL_PROPAGATION (3/3) {Q24}. The perturbation
response is measurable and localised: the owner cell shows mean |ΔV_m| 9.26 mV with 7
additional spikes, a non-owner cell of the same area 2.43 mV with 0 additional spikes, and
the other area (A1) 3.16 mV with 9 additional spikes {Q30}. Propagation here is structural
— over the existing A2→A1 feedback pathway — with no spectral or functional
feedforward/feedback claim .

## Provenance and reproducibility

All main figures were rendered from a single frozen evidence set with per-file SHA-256
provenance: each generation receipt records a figure SHA-256, and all seven declared hashes
match the shipped figure bytes (7/7) {Q27}. The recorded equivalence gate confirms byte
equality for 7/7 figure cases across the frozen and re-rendered artifacts {Q26}. The figure
regeneration path is reproducible from a clean checkout per the documented entrypoint, with
pinned dependencies (matplotlib ≥ 3.10.9, < 3.11; scipy == 1.17.1) {Q28}. Byte-for-byte
reproduction is claimed only within the recorded equivalence gate.

## Appendix — Evidence provenance

Registration: primary protocol suite and figures frozen at the pre-revision baseline
(2026-08); the estimator module revision SHA is `684859a98da51de79887ec26ba8d7134e2fa0e97`
(see Results above). Secondary, independently registered validation runs (executed after
registration; results reported separately in the paragraphs above; never alter the
registered claims):

| Internal tag | Content | Artifact |
|--------------|---------|----------|
| A-1a | estimator recovery, 48 positive + 5 negative synthetic cases | `artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json` |
| A-1b | predeclared dynamic search, 15 points × 3 seeds (45 cells) | `artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_receipt.json` |
| A-2 | sensitivity floor (S1 amplitude/noise; S2 γ*; S3 duration/sites) | `artifacts/protocol_c/p2v_a2_sensitivity_floor/p2v_a2_receipt.json` |
| A-3 | HDP trajectory boundedness (2 presets × 3 seeds) | `artifacts/protocol_c/p2v_a3_hdp_boundedness/p2v_a3_receipt.json` |

Sentence/paragraph-level traceability (claims, receipts, forbidden-language mapping) is
maintained in `docs/publication/results_reconstruction/traceability_map.md`; quantitative
markers {Qnn} are verified by `scripts/audit_results_draft.py`.