# Remaining work

Remaining work only. Done work lives in git, receipts and results; the full
pre-2026-09-27 stack (rationale, sealed 0.5.0–0.5.4 stacks, AT-10 narrative)
is archived byte-for-byte at `artifacts/archive/0.5.x/todo_stack_2026-09-27.md`.

Rules: `artifacts/AGENTS.md`. Facts (human edit only): `artifacts/fact_stack.md`.
Open issues: `artifacts/issue_log/ISSUE_LOG.md`. Atlas source:
`artifacts/project_sources/8_atlas.md`; requirement rows:
`artifacts/programme/atlas_coverage.json`. Other editors also change this
file: re-read before editing, amend in place, keep diffs you did not make.

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

## Decisions in force

- 2026-09-23 (human): the Atlas manuscript is new, in
  `artifacts/publication/atlas/`, write-once per figure; the frozen Figure
  1–7 snapshot stays untouched.
- 2026-09-25 (human): `y_schema.py` stays an artifact module; `build_atlas`
  keeps its simulate fallback, the Atlas generator uses a view-only entry;
  data bundles are in memory; AT-07-R4 (B as input) OUT_OF_SCOPE.
- 2026-09-25 (agent): `build_atlas` given signals records seed/duration only
  when passed; network-spec digests stay separate from `at_manifest.spec_digest`.
- 2026-09-26 (human): P-014 refuse + migrate (done); G_20 synthetic
  hierarchy, 50 neurons per area, noise + H01 pulses, 10 s phases, AT-04-R2
  assay (done: `artifacts/atlas/at10_n20_055.py`).
- 2026-09-27 (human): harness work before ATLAS 7 (items H1–H4 below); docs
  = one tutorial track with visuals (D1); docs stay in the repo, gh-pages
  publishing decided separately (A8).
- 2026-09-27 (agent): AT-10-N20 is a new registry id beside the toy AT-10,
  which the frozen agent benchmark pins.
- 2026-09-27 (agent): descriptive labels nothing reads stay labels
  (`connectivity` route labels such as `feedforward`, `feedback`, `kind`,
  `mode`, `e_to_all`; `network(kind=)`; declared probe `modes`), following
  the human decision to keep field proxy labels.
- 2026-09-28 (human): P-017, fix the paradigm semantics: inject only events
  that carry a stimulus, for their own duration, at the declared amplitude
  and targets; refuse a multi-condition `Paradigm`; `evoked_l4` reads its
  onset and amplitude and targets L4; rebuild pages 08/10 on it; changed
  `ParadigmCondition` runs go in the seal receipt. P-018, wire `drive()` on
  the plain route. Wire `edge_seed` (in-repo outputs stay bit-identical).
  Probe kernels refuse `position`/`reference`/`filter_spec` they do not
  apply. H1 (e): construct refuses layer tables and column layers the route
  drops; `default_cortical_column_config` stops declaring them (outputs
  bit-identical; its config hash changes and pinned hashes are regenerated).
- 2026-09-28 (agent): probe kernels accept `reference`/`filter_spec="none"`,
  the `Configuration.probe` rule. The plain route leaves cell-type fractions
  within 1e-9 of unit mass unnormalized: dividing by a mass of 1 ± ulp moves
  n·frac off a .5 rounding boundary (n=10, 0.35) and would change existing
  labels.
- 2026-09-28 (agent): receipts made while a declaration was dropped stay as
  made, and their configurations declare what ran: `mcc3_config()` keeps the
  default drives of its checkpoint and etude (as `_mcc3_model` under P-014),
  `tests/test_equiv01_table.py` the drives of its bounds (P-019);
  `artifacts/perf/matrix_051.json` stays the 0.5.0 receipt (see ATLAS 8).
- 2026-09-28 (human): H1 (e), `default_complete_configuration` drops
  `uniform3d()` and builds a laminar column, so its declared L2/3→core map
  builds edges; its outputs and config hash change and the seal receipt
  lists them. P-016 and P-020: bisect each to the commit that moved it,
  then re-freeze in a new receipt that names the cause, or revert a buggy
  commit; the human signs off each outcome; nothing is re-tuned. P-013: add
  declarative per-task `arm_definitions` (inputs only) with a schema test;
  frozen `results_055` stays; the human reviews the definitions before the
  next freeze. Docs style pass over `docs/` and README
  (`match-writing-style`): cut padding only, soften overclaims and list
  each for review, no truth-status headers on public pages.
- 2026-09-28 (agent): P-012, exempt paths `git check-ignore` reports and
  skip the v033 check on an empty directory. P-019, bound the jit/eager gap
  per drive regime on the P-018 path.
- 2026-09-28 (human): agy (Antigravity CLI) takes high-load, low-complexity
  lanes: quality sweeps of docs, markdown and code against the canonical
  style skills, and upkeep of `artifacts/skills/jaxfne-*`. Every diff is
  reviewed before it is integrated.

## 0.5.5 stack

HARNESS (human, 2026-09-27)
- Seal note (H1 e, 9958a72): `default_complete_configuration` is laminar;
  its outputs and config hash changed (0 -> 173 inter-area edges at
  defaults). Other callers kept their outputs.
- Seal note (P-017): list the runs whose drive changed in the seal receipt
  (commit f8960a0 message: stimulus-less marker events now silent; stimulus
  events use their own duration_ms).
- Seal note: AT-02…AT-06 run records (`artifacts/publication/atlas/`) carry
  the pre-correction spec digest (AT-02…AT-05 recorded `n_contacts` 4,
  executed 16; AT-06 recorded a common-average reference and an 8–25 Hz
  band-pass that were never applied); regenerate them with the other Atlas
  records. The frozen agent task set keeps the old digests
  (`tests/test_agent_bench_055.py` allows exactly these corrections).

ISSUES (order; decisions above)
- Broad gate executes no notebook, so 0.5.5 refusals broke three release
  notebooks unseen (P-021, fixed): add one fast notebook-execution check.
- P-012 test hygiene, then P-019 drive-swept bounds (one packet each).
- P-016, P-020 bisect packet; each re-freeze waits for human sign-off.
- P-013 `arm_definitions` packet; human reviews before the next freeze.

DOCS (human, 2026-09-27)
- D0 Style pass landed (b278f70, eddfa1b, 42e0e2f). Human: review the 23
  softened overclaims and decide the 17 flagged items listed in
  `artifacts/audit/docs_style_pass_2026-09-28.md`.
- `build_laminar_column(geometry='laminar')` ignores `radius_mm` and
  `height_mm` (the laminar route reads `column_radius_mm` metadata only):
  apply or refuse them.
- D0b `jaxfne/` docstring and comment pass done by hand (12 padding and
  overclaim edits; skills restate no style rules). agy is not installed
  (`agy.exe` missing), so its lanes wait: `scripts/` comments and the
  markdown outside `docs/` (188 files). Human: stale "v0.2.x" notes at
  `_signals.py:1445,1453`, `validation.py:1230` and
  `experimental_hpc/physical_field_solver_v040.py:55` are flagged, not edited.
- D1b Pages 08/10 (`08_jaxfne_suite_no_2_evoked_l4_drive.md`,
  `10_v0313_omission_oddball.md`) claim figures and results their notebooks
  never produce: the evoked-L4 notebook passes a full `Paradigm`, which
  simulate now refuses (P-017), and the omission notebook runs one plain
  simulation and plots two time steps under condition titles. Rebuild them
  on the P-017 semantics (`paradigm.condition(...)`).
- D2 Études pages ("studios"): embed each page's own figure beside its
  text, as the tutorial track does (`scripts/generate_docs_visuals.py`
  pattern: still tab + interactive iframe).

ATLAS
7. Reduction/scale matrix: for each transition M_i → M_(i+1) (1N → 2N →
   population → 2A → 20A), which observations survive within the
   predeclared tolerance and which do not; failures stay in the matrix.
   Existing row: `at01_at06_052.run_reduction` (HH → reduced → population).
8. Performance/reduction map: T_compute and M_compute per AT beside
   E_reduction. `artifacts/perf/matrix_051.json` ran the
   `benchmark_050_baseline` models at the default drives; their declared
   drives run since P-018, so measure them anew here.

MANUSCRIPT (ends 0.5.5)
9. Atlas coverage matrix: every section, figure, simulation and claim in the
   Atlas source → todo item → produced artifact; 100% mapped or marked out of
   scope by the human; mechanical check over `atlas_coverage.json`. AT-10-R*
   rows point at `AT-10-N20` evidence (results, figures, caveats: selection
   status, control-window dependence, no-stimulus null).
10. Figures via the existing generator seam (`scripts/publication_figures/`
    pattern), consuming generated Atlas data only; columns structure →
    dynamics → state/plasticity → source → field → observation → computation.
11. Claim ledger: each claim with its V(claim) and evidence path; package
    capability claims apart from scientific-result claims.
12. Methods from manifests: units, calibration level, tolerances, seeds and
    versions quoted from generated artifacts.
13. Citations primary-verified (H10) before inclusion.
14. Manuscript text complete and consistent with the claim ledger;
    submission is a separate human action.

ACCEPTANCE (0.5.5 seal = end of programme)
- Existing canonical configurations bit-identical to the frozen baseline,
  except outputs changed by authorized repairs (P-014: suite2 presets,
  experiment_a, protocol_e now run their declared drives), each listed in
  the seal receipt. Run the release-tier baseline check before the seal.
- AT-01…AT-10 (and AT-10-N20) regenerate from manifests; Atlas and figures
  reproduce.
- Coverage matrix 100% (item 9).
- Programme deltas vs v0.4.25 measured: T_simulation, M_peak, T_test,
  T_agent_task, E_semantic = 0, C_scientific retained.
- Every manuscript claim traces to a ledger row with PASS evidence; negative
  and failed results reported as such.

## Open work outside the release stacks (after 0.5.5 unless a trigger fires)

1. A8 gh-pages publishing policy. Trigger: publishing D1 or Atlas figures
   to the public site.
2. `units.py` and `pynwb_compat` unwired: wire or remove (owner decision).
3. Architecture candidates (`emitters.py` variant split, entry
   fragmentation, dual manifests, same-named builders); enter only as a
   measured bottleneck.
4. C5–C7 numerical deferrals; need a signed-zero/NaN exactness contract.
5. UNTESTED-exact refusal tail; PLACEHOLDER_NOTEBOOKS and artifact-gated
   skips; post-0.4.14 compatibility aliases.
6. P-001 `scripts/` legacy lint cleanup.
7. S27 population/`P_{l,c}` definition family. Trigger: a population
   definition CTX-01 cannot express.
8. Agent-native step 10: MCP or other transport.
9. Deferred from 0.5.4: item 1b (S20.1 ordering; individual human
    authorization required) and item 0 (`X[k]` frontier override, language
    decision).
10. Deferred owner decisions from the 2026-09-20 deep audit:
    `compile_step_fn **hdp_kwargs` unknown-key policy; `validate_hdp_params`
    non-dict non-strict silent pass; frozen protocols record JAX/lib
    versions (P-003).

## Tracks (standing directions, not tasks)

P performance (optimize only profiled components) · T testing (cheapest gate
per defect class) · S skills (task-shaped, benchmarked) · I integration (one
semantic implementation, many entrances) · A architecture (factor only with
measured benefit) · V validation (configured → realized → executed →
observed per capability) · U inspection (origin/provenance queries).
Full text in the archive.
