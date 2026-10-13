# Remaining work

Top = next. Prerequisites sit above dependents. `[A]` agent can do it, `[H]` needs Hamm. Done items
are deleted (history: git; the pre-2026-09-27 stack and the sealed 0.5.0–0.5.5 sections are under
`artifacts/archive/0.5.x/`). Rules: `artifacts/AGENTS.md`. Facts: `artifacts/fact_stack.md` (Hamm
edits). Open issues: `artifacts/issue_log/ISSUE_LOG.md`. Other editors change this file: re-read
first, amend in place.

## 1. Decisions waiting on Hamm (they unblock items below)

- T-1 [H] Etude drift. First bad commit 9cc45659 ("refuse inert Configuration.cell_type_drives"):
  the frozen multiscale and heterogeneous etudes ran with an inert E/PV drive of 8.0; the
  migration to `.drive(baseline_drive_by_cell_type=...)` makes it act
  (`scripts/run_multiscale_observation_etude.py:128`, `scripts/run_heterogeneous_emitters_etude.py:103`),
  so `rerun_etude_figures.py` aborts for both. (a, recommended) restore the realized frozen
  protocol by removing the drive call, with a note; (b) declare a new protocol and regenerate.
  Unblocks T-4. Closes when: `python scripts/rerun_etude_figures.py --etude multiscale` and
  `--etude heterogeneous` exit 0.
- T-2 [H] `artifacts/.lab` (108 Labyrinth JSON nodes, read only by `scripts/lab_balance.py`, which
  crashes at `build_edges`): bundle into one file, retire, or keep. Unblocks part of T-4.
  Closes when: Hamm's choice is recorded here and `scripts/check_tree_size.py --check` passes.
- T-3 [H] PCL Fig. 3 after the diagnostic (v4 FAIL stands, `pcl-port` 5640ce18): new prereg for
  T2/T4, the P6 library path anyway (B1), or move on with a stated reason (T-7). Closes when:
  the choice is recorded in `artifacts/etudes/REPLICATIONS.md`.

## 2. 0.5.6 release (G-5; Hamm 2026-10-12: cleaning, organizing, docs, dashboard)

- T-4 [A] R2 narrowing to -10 % files and bytes against the 0.5.5 tag (2522 files, 125982465
  bytes; targets 2269 and 113384218). At b9e55903: 2391 files, 88960885 bytes; bytes done, 122
  files to go. One class per commit, generator wired into the docs build first, then untracked.
  Closes when: `scripts/check_tree_size.py --check` passes with the lowered budget.
  - Largest lever: `.lab` bundled into one file (T-2) removes 107; with the 13 multiscale and
    heterogeneous etude HTML files (T-1) that is 120 of 122.
  - Not ready: of 53 files in `tutorial_figures`, `_static/gallery` and `docs/assets` only 3
    regenerate byte-identical (tutorials 08, 09, circuit geometry); 36 drift, 6 are copies no
    script writes, 1 gallery PNG is never written, 7 have no generator. Regenerating them is a
    declared figure change.
- T-5 [A] dev → main (Hamm 2026-10-12): broad gate on a fresh clone of the head, push dev, release
  CI with slow tests on dev, `gh pr create --base main --head dev`, required checks, merge; then
  main CI, the Read the Docs latest build (first `pre_build` atlas build) and the Dashboard
  workflow. Closes when: `gh run list -R HNXJ/jaxfne --branch main -L 1` shows success on the
  merge SHA.
- T-6 [A] F6 (G-2): "SOM" in the AdEx table and `fields/proxy.py:775` (unknown types default to
  +1); failing test first, second review. Then the CAMINOS P6 prereg for Hamm's signature.
  The F1 connection table does not yet record `sampling`. Closes when: the F6 test passes and
  fails on the pre-fix code.
- T-7 [A] R5: each paper ships at its P5 verdict; stated reasons for PCL Fig. 3 (after T-3) and
  Izhikevich 2003 in `artifacts/etudes/REPLICATIONS.md`. Izhikevich 2003: P3 prereg (batch
  signature, S1 reference arm `net_reference.py`), calibration gate first (B7). Closes when:
  every paper row in `REPLICATIONS.md` carries a verdict or a reason and
  `python scripts/build_dashboard_data.py --check` passes.
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
- T-12 [A] jnwb item 2 (plain Welch done, dev 541f7604): remaining spectra at `vis/spectra.py:75`,
  `vis/fields.py:162,866`, `vis/tutorial_panels.py:305`, `fields/proxy.py:913,918,1106`,
  `tutorial_utils.py:69`, `protocol_c/estimator.py:74`, `analysis/spectral.py:23,86`; each equals
  `jnwb.compute_psd` within float tolerance or is declared. Closes when: one equivalence test
  per site passes.
- T-13 [H] jnwb item 3: `analysis/metrics.py:226,122,19` (Fano of the population sum,
  `burst_index` ignoring `bin_ms`, unbinned pairwise correlation), pinned by
  `tests/test_analysis_metrics.py:27-51`; needs a declared definition change. Closes when: the
  declaration is signed and those tests pin the new definitions.
- T-14 [H] jnwb item 4: no remaining family is a drop-in (band power, CSD, metrics, coherence);
  each move is a declared change or keeps jaxfne's (`artifacts/programme/jnwb_inventory.md`).
  Closes when: every inventory row is marked moved, declared or kept.
- T-15 [A] `docs/quickstart.md` `connections()` snippet after the dense refusal (0d2b1018):
  check `model.simulate` vs `jtfne.simulate`. Closes when: a test runs the snippet
  and passes.
- T-16 [A] C7: parametrize the protocol_e spec tests; merge the 8 `test_atlas_at*_05x` files
  after a per-test coverage run. CI dedupe (examples list and wheel smoke in three places).
  Closes when: `python scripts/run_test_gate.py broad` passes with coverage unchanged.
- T-17 [H] P3 audit owner calls: unreferenced scripts left by `f4397bd6`
  (`make_delta_test_01_report.py`, `generate_mechanism_tutorials.py`, `repair_notebooks.py`),
  benchmark and sync scripts, unwired gates, the manuscript tree, archive moves. Closes when:
  each has Hamm's call recorded here.
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

Tracks (standing directions): P performance (optimize profiled components only) · T testing
(cheapest gate per defect class) · S skills · I integration (one implementation, many entrances) ·
A architecture (factor only with measured benefit) · V validation (configured → realized →
executed → observed) · U inspection (provenance queries).
