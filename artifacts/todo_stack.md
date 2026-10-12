# Remaining work

Remaining work only. Done work lives in git, receipts and results; the full pre-2026-09-27 stack (rationale, sealed 0.5.0–0.5.4 stacks, AT-10 narrative) is archived byte-for-byte at `artifacts/archive/0.5.x/todo_stack_2026-09-27.md`.

Rules: `artifacts/AGENTS.md`. Facts (human edit only): `artifacts/fact_stack.md`. Open issues: `artifacts/issue_log/ISSUE_LOG.md`. Atlas source: `artifacts/project_sources/8_atlas.md`; requirement rows: `artifacts/programme/atlas_coverage.json`. Other editors also change this file: re-read before editing, amend in place, keep diffs you did not make.

## Now

- T-1 fix P-2: `connections(sampling="bernoulli")` with the `probability` docstring corrected (G-2 F1, branch `f1-bernoulli`). Closes when: `pytest -q tests/test_connections_bernoulli_sampling.py` passes on dev. Blocked by: none.

## Goals in plain words (0.5.x)

Want: a faster JaxFNE that runs the full Atlas, from one neuron to twenty
brain areas, and a manuscript built on it. Need: old results never change
unless an authorized correctness repair says so; new abilities are switched
on explicitly; every claim has evidence.

0.5.5 (current, last release of the programme): one Atlas showing the same
model language works from 1 neuron to 20 areas, and the manuscript. Needs:
the 20-area simulation (done: `AT-10-N20`); one shared list of measurements;
what survives each simplification; generated figures; every manuscript claim
tied to evidence.

Starting now (human, 2026-09-29): jaxfne analyses and visualizes its results
through jnwb wherever jnwb has the operation. jnwb: data/signal → processing
→ analysis → results. jaxfne: model/signal → implementation → simulation →
results. The move is gradual, and simulation results pass to jnwb through
one seam that is independent of the rest of jaxfne, ideally with no delay
(item 0d below). The seam is additive: existing outputs stay bit-identical.

Short list for the jnwb move (human, 2026-09-30), in order:
1. jnwb 0.2.8 carries lane D (`compute_psd(nperseg=)` and the four spike
   measures); jaxfne pins the jnwb release that has them. Lane D built,
   bundled, and independently verified by a jnwb agent (jchat #78,
   2026-09-30: 6100 passed, 22/22 gates). 2026-10-06 (human): track the latest
   jnwb; extra pinned `jnwb>=0.2.9` (0.2.9 on PyPI, installed, jnwb tests pass).
2. Spectra go through jnwb: jaxfne's spectrum equals `jnwb.compute_psd`
   with its own `nperseg` within float tolerance; any other change is
   declared and listed; frozen receipts are untouched.
3. The three `analysis/metrics.py` defects (Fano of the population sum,
   `burst_index` ignoring `bin_ms`, unbinned pairwise correlation) are
   logged, then fixed by delegating to jnwb or declared.
4. The rest of `artifacts/programme/jnwb_inventory.md` moves in order;
   `to_jnwb` joins the top-level surface when that settles.

## Decisions in force

Dated decisions 2026-09-23…2026-10-11 and the Atlas caveats (AT-10-R5, AT-10-R6, AT-05-R2,
AT-01-R6/AT-00-R4) are in `artifacts/archive/0.5.x/decisions_and_caveats_2026-10-12.md`; still in
force. Newest: T1 option A (meeting 3790d96e): sections 1–7 freeze on 0.5.5 evidence, 0.5.6–0.5.10
fill section 8 citing a replication only at a signed-prereg verdict; JAX runs above about 1000
neurons run on WSL2 CUDA.

## 0.5.6–0.5.10 programme rules (B1–B7, adopted 2026-10-11)

- [A] B1 no paper counts as replicated without a recorded library-path attempt. Check: the replication index row.
- [A] B2 negative verdicts go in the claim ledger with their reason. Check: one ledger row per scored criterion.
- [H] B3 one emitter per release, authorized by Hamm, gated by a P4 test perturbing every accepted parameter. Check: that test fails on a stored-but-unconsumed parameter.
- [A] B4 narrowing, one class per commit with its gate; asset moves wait for Hamm. Check: `scripts/check_tree_size.py --check`.
- [A] B5 GPU: one job at a time, claimed in jchat. Check: one open GPU claim.
- [A] B6 each accepted binding becomes a row here the same session.
- [A] B7 calibrate before signature on synthetic nulls and known positives. Check: the prereg's calibration table.
- [A] jnwb item 2: plain Welch done (dev 541f7604); remaining spectra at `vis/spectra.py:75`, `vis/fields.py:162,866`, `vis/tutorial_panels.py:305`, `fields/proxy.py:913,918,1106`, `tutorial_utils.py:69`, `protocol_c/estimator.py:74`, `analysis/spectral.py:23,86`.
- [H] jnwb item 3 (logged, not fixed): `analysis/metrics.py:226,122,19`, pinned by `tests/test_analysis_metrics.py:27-51`; needs a declared definition change.
- [H] jnwb item 4: no remaining family is a drop-in (band power, CSD, metrics, coherence differ in definition); each move is a declared change or keeps jaxfne's.

## 0.5.5 stack

Legend: [A] an agent can run it now · [H] needs a human decision · [B] blocked.

- [A] Next release owned by the cursor agent (hub main #335); quiet-matrix timing run waits for an idle machine.
- [A] Seal notes for the receipt: H1 (e) 9958a72 (0 → 173 inter-area edges at defaults); P-017 runs whose drive changed (f8960a0); `build_laminar_column` geometry keys now written; AT-02…AT-06 records carry the pre-correction spec digest (regenerate; `tests/test_agent_bench_055.py`).
- [A] ATLAS 8 `artifacts/perf/matrix_055.json`: quiet-machine rerun before the seal; add `T_test` and per-cell drive provenance; null-drive `output_checksum` equals `matrix_051.json` (only `mech_hdp` may differ). Items 9–14 take deltas from it.
- [A] AT-10-R6 RSS growth (+80–98 MB over 4 chunks): resolve leak vs XLA cache before any run over 100 s.
- [A] `_meta.jaxfne_version` reads 0.5.0 in `artifacts/atlas/results/at01_055.json`, `at05_055.json`, `at07_055.json`, `at_reduction_055.json`. Blocked by: the version bump.
- Coverage (checked 2026-10-12): 48 rows, 42 VALIDATED, 6 OUT_OF_SCOPE. Seal not met; no tag. PR #101 (dev → main, Hamm 2026-10-07) not re-checked.

## MANUSCRIPT (ends 0.5.5), open

- [A] Item 9 coverage matrix 100% mapped or marked out of scope. Blocked by: human scope marks.
- [A] Item 10 figures via `scripts/publication_figures/` from generated Atlas data. Blocked by: item 9.
- [H] Item 11 claim ledger `artifacts/programme/claim_ledger_055.md` (48 rows): Hamm confirms each `kind` and fills 4 PLANNED rows.
- [A] Item 12 methods quoted from manifests. Blocked by: item 10.
- [A] Item 13 citations primary-verified; NMDA, GABA_B and AMPA owed.
- [H] Item 14 text consistent with the ledger; submission is Hamm's. Blocked by: items 9–13.

## ACCEPTANCE (0.5.5 seal = end of programme), open

- [H] Canonical configurations bit-identical to the frozen baseline except authorized repairs (P-014), listed in the seal receipt; `python scripts/run_test_gate.py release`, then `rc`. Tag, main merge and release are Hamm's.
- [A] AT-01…AT-10 and AT-10-N20 regenerate from manifests; coverage 100%; deltas vs v0.4.25 measured (T_simulation, M_peak, T_test, T_agent_task, E_semantic = 0); every claim traces to a PASS ledger row, negatives reported.

## Compaction track (audit `6a99cc12`, 2026-10-06)

- [A] C1: `rerun_etude_figures.py --etude hdp_mcc3` never run end to end (heavy).
- [H] C4 remainder: unreferenced scripts not removed in `f4397bd6` (e.g. `make_delta_test_01_report.py`, `generate_mechanism_tutorials.py`, `repair_notebooks.py`): P3 owner call.
- [H] C6: generated HTML (about 27 MB) → release asset or doc-build time.
- [A] C7: parametrize protocol_e spec tests; merge the 8 `test_atlas_at*_05x` files; first a per-test coverage run.
- Frozen: `artifacts/publication/frozen_manifest.json`, `artifacts/release/current_release_authorities.json`, hash-pinned skills and `AGENTS.md`. C3 (`artifacts/legacy/`) and C5 (`artifacts/.lab/`) skipped: live references.

## Open work outside the release stacks (after 0.5.5 unless a trigger fires)

Replications: index `artifacts/etudes/REPLICATIONS.md`, dashboard
https://hnxj.github.io/jaxfne/dashboard/ (`dashboard/DESIGN.md`), programme
`artifacts/etudes/replication_programme.md`. CAMINOS passes under amendment v7 (`l2-caminos`
1e768f9d; p = 0.07 identical to v6 in seeds 1–5); PCL Fig. 3 v4 FAIL stands (`pcl-port` 5640ce18).
- [H] PCL Fig. 3 after the diagnostic: new prereg for T2/T4, P6 library path anyway (B1), or move on.
- [A] G-2 features, one at a time, failing test first, second review (design: CAMINOS port spec
  and the 2026-10-12 code map): F1 `connections(sampling="bernoulli")` (branch `f1-bernoulli`);
  F2–F4 `emitters_adex` module (AdEx, conductance synapses and delay ring inside its kernel,
  ADVANCED tier, no root export); F5 `poisson_drive` independent sources per target cell type;
  F6 "SOM" in the AdEx table and `fields/proxy.py:775` (unknown types default to +1). Then the
  CAMINOS P6 prereg for Hamm's signature. Shared-drive variant dropped (Hamm 2026-10-12).
- [A] Each new score: update `dashboard/sources.json` (lane SHA), regenerate, push to dev.
- [A] G-1 (meet 204dfdaa): port jnwb's `changelog.d`, `assemble_changelog.py`, `release_body.py`
  and claim check; release body as tables. Check: `release_body.py --tag <next>` passes.
- [A] G-3: CAMINOS p = 0.07 seed 1 as an NWB excerpt plus provenance for jnwb `tests/` (jnwb 12-21).
- [A] `docs/quickstart.md` `connections()` snippet after the dense refusal (0d2b1018): check
  `model.simulate` vs `jtfne.simulate` and tell the docs lane.
- [H] P3 audit owner calls: benchmark and sync scripts, unwired gates (`scripts/lab_balance.py`
  crashes at `build_edges`), docs pages, manuscript tree, archive moves, 50.7 MB generated assets.
- [A] CI dedupe (examples list and wheel smoke in three places).
- Kept by Hamm 2026-10-11: `artifacts/legacy/AGENTS.md`.
- WSL2 CUDA 20-area GPU test after the 0.5.5 seal (2026-10-09; receipts `scratch/wsl_cuda_2026-10-09/`,
  gitignored): CSD agreement, peak VRAM, one unexplained failed GPU run.
- Python 3.15 (Hamm 2026-10-10): add 3.15 to CI and the classifier once 3.15.0 final ships and
  pyyaml, h5py and zarr publish cp315 wheels.

Directions recorded by Hamm, open:
- Augmentation science runs (2026-10-01, `artifacts/programme/augment_science_runs_plan.md`):
  canonical 1000n column unchanged; an opt-in "balanced" preset with a measured
  asynchronous-irregular regime (preset carries per-cell-type drive and noise; absent fields keep
  the canonical column bit-identical); then runs 1k → 2k → 5k, then a 10k decision.
- PCL: `pcl_stdp_h2` is the rule for new work, `pcl_stdp` stays default for K1 and the
  replication; a learned-tuning A1 test needs a new declared criterion (`artifacts/etudes/pcl/README.md`).
- Structural HDP (2026-09-29): existence over a declared candidate set, after the seal, opt-in
  (`artifacts/programme/structural_hdp_design.md`); API and hard-vs-graded existence open.
- Primate primary-connection mode (2026-09-29): label current FF/FB defaults uncalibrated now; then
  a macaque-only evidence table (edge tagged direct / anatomical-only / functional / unknown, with
  citation), SLN in [0,1], heavy-tailed FLN with a distance rule; unknown edges refused, never defaulted.
- jnwb as the analysis layer (2026-09-29; `artifacts/programme/jnwb_inventory.md`): jaxfne keeps
  only views jnwb cannot express; raise the extra to `jnwb>=0.2.8` so the `compute_psd`
  bit-identity test runs; then migrate in inventory order.
- Theory programme (2026-10-08, `artifacts/programme/theory_programme_proposal.md`): next family
  F3, F4 or F5 is Hamm's pick; T3 Lean proofs of L1 and L3 need Lean.

Deferred, each with its trigger:
1. A8 gh-pages policy for D1 or Atlas figures (the dashboard is at `/dashboard/` only).
2. Architecture split (`emitters/`, `tfne.py`; `artifacts/audit/deep_review_89_2026-10-04.md` §5) only as a measured bottleneck.
3. C5–C7 numerical deferrals: need a signed-zero/NaN exactness contract.
4. UNTESTED-exact refusal tail; placeholder notebooks and artifact-gated skips; post-0.4.14 aliases.
5. `scripts/` lint (P-001, ruff 171): first `audit_w3_broad_handlers.py` `_OVERRIDES` (duplicate keys, drifted line keys; running it rewrites `artifacts/audit/w3_broad_handler_tally.json`): re-key by content or retire with the tally.
6. S27 population definition family, when CTX-01 cannot express one.
7. Agent-native step 10: MCP or another transport.
8. From 0.5.4: item 1b (S20.1 ordering, needs Hamm's authorization) and item 0 (`X[k]` frontier override).
9. Owner decisions from the 2026-09-20 audit: `compile_step_fn **hdp_kwargs` unknown keys; `validate_hdp_params` non-dict pass; frozen protocols recording JAX versions (P-003).
10. Field Approximation Atlas ladder after the seal (deep review 89 §7).
11. Post-0.5.5 rule proposal (deep review 89 §12): new mechanism ⇒ reference model, discriminator and reduction test; adoption is Hamm's.

## Tracks (standing directions, not tasks)

P performance (optimize only profiled components) · T testing (cheapest gate
per defect class) · S skills (task-shaped, benchmarked) · I integration (one
semantic implementation, many entrances) · A architecture (factor only with
measured benefit) · V validation (configured → realized → executed →
observed per capability) · U inspection (origin/provenance queries).
Full text in the archive.
