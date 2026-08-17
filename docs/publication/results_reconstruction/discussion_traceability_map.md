# Discussion — Traceability Map

Companion to `discussion_draft.md` (1329 words, DRAFT Phase 7). Maps every
Discussion proposition to its classification (DEMONSTRATED / INTERPRETATION /
BIOLOGICALLY_COMPATIBLE / LIMITATION / FUTURE_DIRECTION / UNSUPPORTED), to its
manuscript sentence, and to its evidence authority. No Discussion sentence
introduces a claim whose category exceeds its authority; the UNSUPPORTED table
lists phrasings that were deliberately excluded.

## Proposition architecture (movement -> propositions -> authority)

| Movement | Propositions | Manuscript sentences | Authority |
|----------|--------------|----------------------|-----------|
| 1 Principal contribution | D1 typed E/S/F/P contracts; D4 relative/calibrated semantics; D3 reduction identities; D2 computational latency; anti-drift one-sim-per-seed | Para 1 ("The framework contributes..."; "each stage boundary carries an explicit ownership and disablement contract"; "the contribution is not automatic differentiation itself"; "every extension ... defined by a reduction identity"; "can be kept computationally latent"; "one executed simulation per seed") | CL-01,02,03,07,14,15,16,17; Methods L1/4/6; Intro para 4-5 |
| 2 What experiments reveal | D5 synthetic-only estimator validity; D6 no-wave in tested regimes; I3 bounded-negative reading (delay collapse + near-silence); D7 NO_ADAPTATION; D8 H4 no positive length effect; D9 E5 propagation; D12 W3b unresolved; I4 detector-blindness bounds (S2 gamma* = 1.0; D2 parity) | Para 2-3 ("The same executable framework produced supported, negative, and unresolved outcomes"; "all 48 synthetic traveling waves ... rejection of all five negative control families"; "no traveling-wave classification in the tested regimes"; "bounded negative result — not a statement about untested regimes, and not a neural positive control"; "parity-dependent at low site counts — constraints on the instrument"; "attenuation, not adaptation"; "no positive length effect by the preregistered test"; "bit-exact leakage controls across all seeds"; "2187 tested lattice points ... 1944 active points remain classified as stability-unresolved") | CL-05,06,08,10,18,11,19; A-1a/A-1b/A-2/C3/D3/H4/E5/W3b receipts; Methods L7/8/9/10/11/12 |
| 3 Relative to existing methods | Complementarity; typed-contract NOT-FOUND gap; design-intent interchangeability (not demonstrated); no supersession | Para 4 ("used alongside, not instead of"; "the typed contract layer this paper demonstrates"; "could be replaced ... neither replacement is demonstrated here"; "no claim that these systems are superseded") | Intro para 1-2; R2 landscape notes; Phase-6 ledger refs 1-8,13,14 |
| 4 Biological/physical boundaries | L1 proxy boundary; HDP relative/latent state; clamps = finiteness not stability; K_w_ctrl = 0.0 horizon; B1 compatibility not validation | Para 5 ("Every readout reported here is a relative computational proxy"; "linear projection/proxy operators, not volume-conductor or partial-differential-equation solves"; "implementation-bounded by hard clamps that guarantee finiteness, which is not dynamical stability"; "no active restoration to connection magnitudes"; "compatibility is not biological validation") | Methods L6/5.3; CL-03,04; A-3 receipt; Results HDP paragraphs |
| 5 Limitations | L1-L9 with consequences | Para 6-7 ("The scope of the paper is bounded in ways that each carry a consequence"; proxy; finite domain; deterministic; D2 parity; wave-domain absence; HDP implementation-level; post-freeze validation; H4 point estimate) | Full limitation list below |
| 6 Future directions | F1-F6 downstream of limitations; developmental operator = future possibility only, unnamed | Para 8 ("Calibrated source bridges"; "validated field solvers"; "gradient-based inverse problems"; "Broader dynamical families"; "wave-positive regime search"; "a developmental operator could generate a realized circuit specification upstream of the existing pipeline; this is a future possibility, not a capability of the present work") | L1, L8, L2, L5, L2 (each cited direction maps to its limitation) |

## UNSUPPORTED (excluded phrasings; grep-verified absent)

| Phrasing | Why excluded | Check |
|----------|--------------|-------|
| Networks cannot generate waves | U1: no claim outside tested domains | absent |
| A-1b as positive neural control | U2: bounded negative search | absent (explicitly negated) |
| Global/arbitrary-horizon HDP stability | U3: K_w_ctrl = 0.0; N_S = 0 | absent |
| Calibrated physical fields / solver evidence | U4: proxy operators only | absent |
| Biological mechanism identity (adaptation/memory/waves) | U5: compatibility only | absent |
| Individual novelty of AD or single stages | U6: explicitly disclaimed | absent |
| Supersession of mature simulators | U7: complementarity | absent |
| "Robust" vocabulary | U8: no noise run | 0 occurrences |
| JDNA / new framework as current contribution | U9: future possibility only | absent |
| first/unique/unprecedented/novel | Phase-6 banned words | 0 occurrences |

## Terminology consistency (vs finalized Introduction/Results/Methods)

| Term in Discussion | Authority | Status |
|--------------------|-----------|--------|
| typed stages (dynamics, source, field, probe) | Intro para 4; Results grammar | MATCH |
| ownership and disablement contract | Intro para 4 (P4) | MATCH |
| relative computational proxy / readout | Results "Relative computational readouts"; Methods L6 | MATCH |
| reduction identity | Intro para 5; Methods reduction ladder | MATCH |
| computationally latent | Intro para 4; CL-17 | MATCH |
| pure downstream compositions | Intro para 4; Results "Observations are downstream" | MATCH |
| one executed simulation per seed | Methods anti-drift rule (L1) | MATCH |
| tested regimes / tested domain | Results CL-06 carve; Methods L10 | MATCH |
| reported as such / reported as unresolved | Results negative/unresolved discipline | MATCH |
| no positive domain in the tested range | Methods L10 A-1b outcome (quoted token) | MATCH |
| stability-unresolved | Results W3b paragraph; Methods L12 (X class) | MATCH |
| attenuation, not adaptation | Results D3 paragraph; CL-08 | MATCH |
| no positive length effect by the preregistered assay | Results H4 paragraph; CL-10 | MATCH |
| structural propagation / existing feedback pathway | Results "Causal disablement and propagation"; CL-18 | MATCH |
| implementation-bounded | Methods L5.3 classification table | MATCH |
| no active restoration to connection magnitudes | Methods L5.3 (K_w_ctrl = 0.0) | MATCH |

## Citation usage (Discussion body)

| Ref | Used in | Verified |
|-----|---------|----------|
| 1-2 LFPy/TVB | Para 1 (pipeline contract semantics context) | Phase-6 ledger VERIFIED |
| 3-5 Jaxley/BrainPy/TVB-Optim | Para 1, 4 (differentiable fitting) | VERIFIED |
| 6-8 NEURON/NEST/Brian | Para 1, 4 (dynamics infrastructure) | VERIFIED |
| 13-14 waves reviews | Para 2 (cortical signature context) | VERIFIED |
| 9-12 | Not cited in Discussion body (Introduction-owned); listed in shared reference list | VERIFIED |

## R3/R4 disposition

See `artifacts/developer/phase2_review/12_discussion_adversarial_review.md`:
0 FATAL, 2 MAJOR (N1 W3b counts; N2 interchangeability), 4 MINOR (E1 PRUNE,
E2 REWIRE, N3 SPECIALIZE, E3), 1 REJECTED; all repairs applied in G2 with
ADAPT vocabulary; Final Review gates all PASS.

## Remaining Discussion debt

1. Body length 1329 words (above the ~1200 planning target); further PRUNE
   deferred to full-manuscript reconciliation (B-block phase).
2. Ref 5 journal-version re-check before submission (inherited from Phase-6).
3. B-2 (regime table), B-3 (H4 confound paragraph in Methods), B-4, B-5, B-6
   remain MUST_BEFORE_SUBMISSION items outside the Discussion scope.