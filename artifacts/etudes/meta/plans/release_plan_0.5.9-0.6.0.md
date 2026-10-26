# Release plan 0.5.9–0.6.0: the paper's figures, submission, acceptance

Requested by Hamm, 2026-10-26: write the 0.5.9 packages from the arXiv plan sequence; 0.6.0 is the
post-peer-review and acceptance release, a hard and complete one. This file replaces §4 of
`artifacts/etudes/meta/release_plan_0.5.7-0.5.10.md`; that file's §5 (execution protocol) and §6 (exit gate per release)
stay in force. Paper authority: `artifacts/programme/design/plans/arxiv_paper_plan.md`. Each package
names its figures (F, S) and checklist items (C). Tags, main and PyPI still need Hamm's word per release.

Baseline: dev 99a8a3b4, 0.5.8 published 2026-10-26.

## 1. Order

| Release | Theme | Figures and checklist |
|---|---|---|
| 0.5.9 | Language, H-state and the replication figures | F1, F3, F5, F6, F7; S1, S2, S5, S8–S14; C3, C5, C8 |
| 0.5.10 | Geometry, compute and extras | F2, F4; S3, S4, S6, S7, S16; C4, C6, C7 |
| 0.5.11 | Submission | F8, S15; manuscript, refreeze, review, arXiv; C1, C2, C9–C12 at 100 |
| 0.6.0 | Acceptance | peer-review revisions on every figure and claim; the hard exit gate of §5 |

Re-homed from `release_plan_0.5.7-0.5.10.md` §4: Potjans 2014 to 0.5.9 (F6); AT-10-R6 (RSS flat over
chunks, XLA cache cleared) and the LIF edge kernel within the GPU budget (25000 steps × 12500 cells)
to 0.5.10 (F2, F4); Mejias 2016 (G6 constants and the φ(0) normalization first) to 0.5.10 (S16).
Lee, Pennartz and Mejias 2025, the Mazzoni 2015 LFP proxies and EEG validity name no figure; they
wait for an arXiv plan amendment (S16 is their likely home).

## 2. 0.5.9 work packages

| Id | Package | Items | Feeds | Acceptance |
|---|---|---|---|---|
| 9A | Language | The 8 open rows of `docs/theory/doctrine/definition_map.md`: picks among candidates for D17, D26, D35; public names D21 `resolved_edge_weight` and D34 `rbd_recurrent_gain`; D28 (`ValueTag` lacks B, R, E, T); D31 and D32 need a code path for a_eff and the AdEx clip. F1 and S2 generators (frozen fig01 is the F1 candidate) | F1, S1, S2, C3 | every row outside `INTERNAL_ONLY` names a public symbol guarded by a test that reads the map; F1, S1, S2 regenerate with one command each |
| 9B | H-state and HDP | Reduction identity exists for every live registered rule (`tests/test_hdp_reduction_identity.py`, with a registry coverage test). Add the typed coupling check on D34: H reaches E, S, F or P only through a declared map. S5 per rule; F3 (frozen fig06 is the candidate) | F3, S5, C5 | a mutant that bypasses the coupling map fails the check; F3 and S5 regenerate with one command each |
| 9C | Emitter validation | Brunel S1 finite size (N = 6250, `sampling='fixed_indegree'` exists in the core) and the S8 sensitivity panel. Izhikevich 2003 E1: per-edge uniform weights and per-cell Gaussian noise on the construct path (failing test first), then the P6-style library-path prereg; S9 cell classes, eight criteria | F5, S8, S9, C8 | `check_replications.py --check` loses the izhikevich2003 E1 finding; scored runs only under signed preregs |
| 9D | Circuit validation | Potjans P3 from the unsigned draft `potjans2014/PREREG_P3.md`, seeds 1–8; S10 rates, CV, synchrony. CAMINOS S2: rerun the scored P3 sweeps on the verified core route; S11 shows the PV-SOM lag finding (mixed) | F6, S10, S11, C8 | the caminos S2 findings leave the known list; Potjans scored under its signed prereg |
| 9E | Plasticity and fit to data | Vogels rerun with the per-synapse rule (85199ade), then P3 from the unsigned draft; S12. PCL: the read-only R2/R4 design study, R2 on identical inputs, R3 diagnosis, a new R4 prereg signed before confirmatory runs, R5 scored as exact and approximate separately; S13 keeps the signed v5 T2 failure. GSDR: B8 pilot with MCDP on (`mcdp_hdp_step`), prereg, P3; S14 with held-out seeds | F7, S12–S14, C8 | F7 shows passing criteria only, each failure in its S figure; the pcl S2 findings close at the signed R4 prereg |
| 9F | Release mechanics and reproducibility | JUnit parity classifier by marker or skip source; figure lock file; clean-checkout figure job; read the `publication` job's first CI run | C2, C9 | the rc gate runs end to end on the 0.5.9 candidate and writes its attestation |
| 9G | Carried, lower priority | jnwb adoption (b) JSON Schema with SHA patterns and (c) receipt sha256 on evidence items, both checked against the dashboard data (C8); (a) dark twins, (d) real-page test, interactive theme, docs figure leftovers name no figure | C8 for (b), (c) | none blocks 0.5.9 |

Order: 9F and the agent parts of 9A, 9B, 9C, 9E (Izhikevich E1 core capability, S-figure generators,
PCL design study, Vogels and GSDR pilots, CAMINOS rerun) start now; the signed-prereg runs follow
Hamm's signatures. At most 3 lanes at once.

Decisions for Hamm that gate 0.5.9 (recommendation first):

| Id | Decision | Recommendation |
|---|---|---|
| H1 | F1–F8 against frozen fig01–fig07 (C9) | F1 from fig01, F3 from fig06, S15 from fig05; fig02–fig04 and fig07 placed when 9A and 9B rebuild the generators |
| H2 | Public names and picks for the 9A rows | D21 `resolved_edge_weight`, D34 `rbd_recurrent_gain`; picks proposed in the 9A diff |
| H3 | Signatures | Potjans and Vogels P3 after review; GSDR and Destexhe after their pilots (D2); Izhikevich E1 library path; PCL-R4 |
| H4 | C1 claim numbering, CL-22 to CL-26 | rule before the 0.5.11 manuscript cites them |
| H5 | LICENSE name, data and code statement, external data licences (C9) | any time before 0.5.11 |

## 3. 0.5.10 outline (packages written at the 0.5.9 seal)

- F2, S3, S4, C4: geometry from one neuron to 20 areas (AT-10-N20), NeuronalTensor schema; the turn
  invariance radius cutoff settled with the C4 wording; AT-10-R6.
- F4, S6, S7, C6, C7: benchmark prereg (protocol is Hamm's call), hash-pinned receipts, Brian2, NEST
  and Jaxley baselines where the model class applies; the benchmark `build_config` geometry repair;
  the LIF GPU kernel; the augmentation ensemble study (two seeds change output, a seed reproduces).
- S16: Mejias 2016, Destexhe 2009, kernel LFP (`cl-klfp` 6eb3c6b8, 44cd0111, ff45364b), Jansen-Rit.

## 4. 0.5.11 outline: submission

- F8 and S15: claim coverage by status tier and the fit-to-data summary; S15 negatives match the
  dashboard data (C8).
- Manuscript rewrite: at most 80 references with Crossref DOIs; every figure passes the visual-style
  audit (C10); 0 orphan claims (C1).
- Refreeze at the paper tag; every figure regenerates from a fresh clone (C2, C9).
- Second reviewer, then an external reader (C11); arXiv TeX Live build, category, endorsement,
  ancillary files (C12).
- Exit adds: C1–C12 at 100 by their checks; submission on Hamm's word.

## 5. 0.6.0: acceptance

Proposed exit gate, for Hamm to amend: the per-release gate plus every peer-review point answered
and recorded; C1–C12 rechecked on the accepted text; 0 open issues; `check_replications.py --check`
with no finding outside Hamm's exceptions; every xfail and skip with a recorded reason; the rc gate
end to end with 0 unexplained parity differences; every figure, the docs and the dashboard regenerate
from a clean checkout of the tag.

## 6. Execution

One Haiku 5.5 subagent per lane (Hamm 2026-10-25), each in its own clone or worktree; the
dispatcher keeps science, semantics, verdicts, every integration and every push. Heavy runs on Colab
(Hamm 2026-10-21). Scored runs only under signed preregs. Main figures carry no negative results;
each negative, mixed or unresolved verdict sits in an S figure and is named in the main text. Work
that misses its release moves to the next with its reason.
