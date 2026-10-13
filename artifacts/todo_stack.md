# Remaining work

Top = next. Prerequisites sit above dependents. `[A]` agent can do it, `[H]` needs Hamm. Done items
are deleted (history: git; the pre-2026-09-27 stack and the sealed 0.5.0–0.5.5 sections are under
`artifacts/archive/0.5.x/`). Rules: `artifacts/AGENTS.md`. Facts: `artifacts/fact_stack.md` (Hamm
edits). Open issues: `artifacts/issue_log/ISSUE_LOG.md`. Other editors change this file: re-read
first, amend in place.

## 1. Decisions waiting on Hamm (they unblock items below)

Open: the manuscript tree and other archive moves (T-17). Decided 2026-10-13: T-12, T-13,
T-14, T-17 (calls in each item); T-1 (a) landed in e3a7a41b; T-2
bundle; Izhikevich 2003 and CAMINOS P6 move to 0.5.7 (reasons in
`artifacts/etudes/REPLICATIONS.md`).

- T-3 [H] PCL Fig. 3 stays in progress (Hamm 2026-10-13: it must replicate). v4 FAIL stands
  (`pcl-port` 5640ce18: T1, T3 pass; T2, T4 fail). Next: a new declared prereg (v5), then P6
  (B1). Hamm's calls for v5 (2026-10-13): T2 scores the normalized Δsf tuning curve against the
  authors' curve with a declared tolerance per Δsf point; T4 tests the paper's claim C3
  (frequency doubling, stated qualitatively; the authors' notebook has no measure of it) and
  counts toward "reproduced", with our log A(2ω)/A(ω) estimand and a qualitative criterion
  (complex median above simple median, plus the permutation null); the reference is the published
  Fig. 3 curves, digitized with a declared digitization error (the authors' archives hold the
  grating stimuli as DVS events and the trained weights, no network responses). Cells z = 0–63
  of `neuron_id = 2560` are column (4,4): cSTDP3 `weights/layout_0.npy` (9×9×64) holds ids
  2560–2623 at (4, 4, 0–63). Hamm signs v5. Closes when: a signed prereg reproduces Fig. 3 and the P6
  attempt is recorded.

## 2. 0.5.6 release (G-5; Hamm 2026-10-12: cleaning, organizing, docs, dashboard)

- R2 narrowing met at 33413a12 (2268 files, `budget_files` 2269 in
  `scripts/tree_size_allowlist.json`). Further narrowing waits: most files in `tutorial_figures`,
  `_static/gallery` and `docs/assets` drift on regeneration, which is a declared figure change.
- T-7 [A] R5: every paper row in `REPLICATIONS.md` carries a verdict or a reason (2026-10-13).
  Closes when: `python scripts/build_dashboard_data.py --check` passes on the pushed SHA.
- T-8 [A] Each new score: `dashboard/sources.json` (lane SHA, stage artifacts), regenerate,
  push dev. Standing item. Closes when: `python scripts/build_dashboard_data.py --check` passes
  on each push.
- T-9 [H] R6 release gate: rc gate, CI green on the SHA, second review, receipt; tag, main and
  PyPI are Hamm's. Closes when: the v0.5.6 tag exists and its release CI run succeeded.

## 3. 0.5.6–0.5.10 programme

Goals in plain words (0.5.x): a faster jaxfne that runs the full Atlas, from one neuron to twenty
areas, and a manuscript built on it; old results change only under an authorized correctness
repair; new abilities are switched on explicitly; every claim has evidence. Analysis and
visualization move to jnwb wherever jnwb has the operation (human, 2026-09-29), additively, with
existing outputs bit-identical; jaxfne tracks the latest jnwb (extra `jnwb>=0.2.9`).

Rules B1–B7 (adopted 2026-10-11):
- [A] B1 no paper counts as replicated without a recorded library-path attempt (index row).
- [A] B2 negative verdicts go in the claim ledger with their reason (one row per criterion).
- [H] B3 one emitter per release, authorized by Hamm, gated by a P4 test perturbing every
  accepted parameter.
- [A] B4 narrowing, one class per commit with its gate; generated assets regenerate at the docs
  build (T-4; plan `artifacts/etudes/narrowing_plan.md`).
- [A] B5 GPU: one job at a time, claimed in jchat. B6 each accepted binding becomes a row here the
  same session. B7 calibrate on synthetic nulls and known positives before a signature.

Decisions in force (2026-09-23…2026-10-11, Atlas caveats AT-10-R5, AT-10-R6, AT-05-R2,
AT-01-R6/AT-00-R4): `artifacts/archive/0.5.x/decisions_and_caveats_2026-10-12.md`. Newest: T1
option A (meeting 3790d96e), manuscript sections 1–7 freeze on 0.5.5 evidence, 0.5.6–0.5.10 fill
section 8 citing a replication only at a signed-prereg verdict; JAX runs above about 1000 neurons
run on WSL2 CUDA.

- T-10 [A] G-1 (meet 204dfdaa): port jnwb's `changelog.d`, `assemble_changelog.py`,
  `release_body.py` and claim check; release body as tables. Closes when: `release_body.py --tag <next>` passes.
  Then one newline policy for docs (check: `docs/changelog.md` has 0 LF-only lines), the errors
  page, a proxy-claim wording gate, AST-hash freezing of validated kernels
  (`artifacts/harness/jnwb_cross_learning.md`).
- T-11 [A] G-3: CAMINOS p = 0.07 seed 1 as an NWB excerpt plus provenance for jnwb `tests/`
  (waits on O-2). Closes when: the excerpt's recovery test passes in jnwb `tests/`.
- T-12 [A] `cross_area_coherence` Welch path (the periodogram fallback is refused since
  Hamm's call, 2026-10-13): `nperseg = min(1024, T)` gives one segment for T < 1536 samples, so
  coherence is 1 at the native bins there too. Independent white noise, dt 0.5 ms
  (`C:/workspace/pytest_temp/coh_single_segment.py`): median coherence 0.81 at T = 1000, 0.40 at
  1536 (2 segments), 0.22 at 2048, 0.10 at 4000. AT-09 (`artifacts/atlas/at08_at09_054.py`, 1000 ms,
  T = 2000) reads it in 8–25 Hz with about 2 segments. Hamm 2026-10-13: shrink nperseg so Welch
  has at least 8 half-overlapping segments, refuse when that leaves nperseg < 64. Landed with
  tests; AT-09 C_12 falls from 0.50/0.45/0.45 to 0.105/0.110/0.111 (noise level, no arm
  difference), recorded in `artifacts/programme/composition_054.md`; the AT-09 verdict stands.
  Closes when: the change is on dev with the broad gate green.
- T-16 [A] C7: parametrize the protocol_e spec tests; merge the 8 `test_atlas_at*_05x` files
  after a per-test coverage run. CI dedupe (examples list and wheel smoke in three places).
  Closes when: `python scripts/run_test_gate.py broad` passes with coverage unchanged.
- T-17 [H] P3 audit, remaining owner calls: the manuscript tree and other archive moves. Four
  unreferenced scripts were archived 2026-10-13 (table in `artifacts/README.md`). Five more have
  no reference but write tracked or frozen receipts that docs cite, so they stay:
  `hdp_k_w_ctrl_sweep.py`, `p2v_a1b_dynamic_search.py`, `p2v_a1b_prefreeze_evidence.py`,
  `p2v_a3_hdp_boundedness.py`, `p2v_a4_claim_language.py`. Closes when: each has Hamm's call.
- T-18 [A] Inhibitory-type lists that omit `I`, `Inl`, `Ing`, `SOM` (F6 review, 57f40ddf):
  `tutorial_utils.py:1564` (`simulate_laminar_trials` zeroes same-layer input only onto
  PV/SST/VIP, a model rule), `tutorial_utils.py:800` (default `cell_signs`), `util.py:712`
  (diagnostic grouping). Callers checked (b0a12a36): every caller of `make_laminar_column_config`
  and `simulate_laminar_trials` (4 notebooks, `tests/test_tutorial_utils.py`, `tune_laminar_agsdr`)
  passes only E/PV/SST/VIP, so no realized run is affected at b0a12a36. A change to 1564 is a model
  change for Hamm. Closes when: each site reads the emitter table's sign or carries a stated scope.
- Frozen, do not move: `artifacts/publication/frozen_manifest.json`,
  `artifacts/release/current_release_authorities.json`, hash-pinned skills and `AGENTS.md`;
  `artifacts/legacy/AGENTS.md` kept by Hamm 2026-10-11.

## 4. Directions recorded by Hamm, open

- Augmentation science runs (2026-10-01, `artifacts/programme/augment_science_runs_plan.md`): an
  opt-in "balanced" preset with a measured asynchronous-irregular regime, canonical 1000n column
  bit-identical; runs 1k → 2k → 5k, then a 10k decision.
- PCL: `pcl_stdp_h2` for new work, `pcl_stdp` stays default for K1 and the replication; a
  learned-tuning A1 test needs a declared criterion (`artifacts/etudes/pcl/README.md`).
- Structural HDP (2026-09-29, `artifacts/programme/structural_hdp_design.md`): existence over a
  declared candidate set, opt-in; API and hard-vs-graded existence open.
- Primate primary-connection mode (2026-09-29): label FF/FB defaults uncalibrated; a macaque-only
  evidence table (direct / anatomical-only / functional / unknown, cited), SLN in [0,1],
  heavy-tailed FLN with a distance rule; unknown edges refused.
- Theory programme (2026-10-08, `artifacts/programme/theory_programme_proposal.md`): next family
  F3, F4 or F5 is Hamm's pick; T3 Lean proofs of L1 and L3 need Lean.
- Python 3.15 in CI and the classifier once 3.15.0 ships and pyyaml, h5py and zarr publish cp315
  wheels.

## 5. Deferred, each with its trigger

1. AT-10-R6: RSS growth (+80–98 MB over 4 chunks of the 20-area run): resolve against the XLA cache
   before any run over 100 s (0.5.8 laminar models on GPU). WSL2 CUDA 20-area receipts
   (`scratch/wsl_cuda_2026-10-09/`, gitignored): one unexplained failed GPU run. Closes when:
   RSS is flat over chunks with the XLA cache cleared.
2. A8 gh-pages policy for D1 or Atlas figures (the dashboard is at `/dashboard/` only).
3. Architecture split (`emitters/`, `tfne.py`; `artifacts/audit/deep_review_89_2026-10-04.md` §5)
   only as a measured bottleneck.
4. C5–C7 numerical deferrals: need a signed-zero/NaN exactness contract.
5. UNTESTED-exact refusal tail; placeholder notebooks and artifact-gated skips; post-0.4.14 aliases.
6. `scripts/` lint (P-001, ruff 171): first `audit_w3_broad_handlers.py` `_OVERRIDES` (running it
   rewrites `artifacts/audit/w3_broad_handler_tally.json`): re-key by content or retire.
7. S27 population definition family, when CTX-01 cannot express one.
8. Agent-native step 10: MCP or another transport.
9. From 0.5.4: item 1b (S20.1 ordering, needs Hamm) and item 0 (`X[k]` frontier override).
10. 2026-09-20 audit owner calls: `compile_step_fn **hdp_kwargs` unknown keys;
    `validate_hdp_params` non-dict pass; frozen protocols recording JAX versions (P-003).
11. Field Approximation Atlas ladder (deep review 89 §7).
12. Rule proposal (deep review 89 §12): new mechanism ⇒ reference model, discriminator and
    reduction test; adoption is Hamm's.
13. 0.5.7 (Hamm 2026-10-13): Izhikevich 2003 P3 prereg draft v7 (calibration passed at θ = 4,
    `artifacts/etudes/izhikevich2003/calibration/calibration_v7_15f36087.json`) for signature,
    then S1 reference arm `net_reference.py` and the scored runs. CAMINOS P6 prereg (Fig. 1
    through `emitter(family="adex")`, `connections(sampling="bernoulli")`,
    `Simulation(poisson_drive=)`; F1–F6 landed) for signature and its equivalence test.

Tracks (standing directions): P performance (optimize profiled components only) · T testing
(cheapest gate per defect class) · S skills · I integration (one implementation, many entrances) ·
A architecture (factor only with measured benefit) · V validation (configured → realized →
executed → observed) · U inspection (provenance queries).
