# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to `artifacts/archive/0.5.x/issues_closed.md`.

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

Entries P-036 to P-042 come from the whole-project scan of 2026-10-18 (six Haiku workers on
dev 1764f4b0, read-only; every line below was re-opened and confirmed by the dispatcher unless
marked inferred).

### P-036

- Date: 2026-10-18. Type: BUG. Area: core public API. Severity: high.
- `Simulation.plasticity` (`jaxfne/_signals.py:87`, `with_plasticity` :147) is accepted, never
  refused, and only copied into metadata (`_model_simulate.py:1271` `plasticity_gain`): a
  stored-but-unconsumed parameter (P-014 class).
- `Model.tune` fall-through (`jaxfne/_model_tune.py:404-422`) returns
  `acceptance_decision: "ACCEPT_CANDIDATE"` with `best_parameters={}`, `best_score=inf`,
  `history=[]`; `tests/test_optim_tune.py:229` accepts either verdict.
- `generations or 8`, `population_size or 6` (`_model_tune.py:228-229`) turn an explicit 0 into
  the default; seed 0 likewise is replaced (`:220-222`).
- `RuntimeConfig.backend="gpu"/"tpu"` on a host without it runs on the default device
  (`_runtime_config.py:310-319`); only `runtime_report` says `backend_enforced=False`.
- Expected: each refused or consumed. Possible change: refuse non-zero `plasticity` (or wire
  it), make the fall-through REVISE, `is not None` defaults, refuse an unavailable backend.
- Fixed 2026-10-18: the fall-through reports REVISE; explicit `generations`/`population_size`
  below 1 raise (`tests/test_tune_explicit_values_p036.py`, 4 tests fail on the old code).
  Plasticity label and backend downgrade are documented designs: Hamm's call (O-47).
- Closed 2026-10-19 by Hamm's ruling of 2026-10-18: non-zero `Simulation(plasticity=)` and an
  absent gpu/tpu backend are refused (fragments `simulation_plasticity_refused.fixed.md`,
  `absent_backend_refused.fixed.md`). Moves to the closed archive at its next re-pin.

### P-037

- Date: 2026-10-18. Type: BUG. Area: tests and gates. Severity: high (tests pass vacuously).
- `tests/conftest.py:22-31` turns x64 off before every test (except `test_protocol_w_*`), so
  `test_closure_hp_reconciliation.py:49-52` always skips and the float64 asserts in
  `test_optim_jax_native_audit.py:94-97,118-121` (inside `except Exception: pass`) never run.
- Skips on gitignored inputs: `test_agent_api_catalog.py:36` (6 tests, catalog gitignored),
  `test_notebook_structure_v0330.py:57`; warehouse-dependent skips in CI
  (`test_pcl_stage_c.py:57`, `test_gsdr2026_labels.py:167`).
- Exact float pins on optimizer output: `test_param_handles.py:496-497`.
- cwd-dependent relative paths: `test_experiment_a_b3_receipt.py:13,26`,
  `test_experiment_a_b1_canonical.py:18,52` (from another cwd: one error, two silent skips).
- The 11 lane test files under `artifacts/etudes/papers/**` are in no gate and no CI workflow
  (`pyproject` `testpaths=["tests"]`); they pass today (226 passed, 1 xfailed).
- `scripts/harness/tests/test_harness_adversarial.py` is reached by no gate; `networkx`
  (`test_network3d_visualize.py:175`) is declared nowhere.
- 2026-10-20 (release plan A1): optimizer pins in `test_param_handles.py` use `assert_allclose`;
  the float64 asserts, the `__file__` paths and the `networkx` importorskip already held at
  81b602bd. Open: lane tests in a gate need one process per lane (todo queue item 5); the
  harness tests moved to `scripts/archive/`; skips on gitignored inputs stay (listed in A1).

### P-038

- Date: 2026-10-18. Type: DOC/BUG. Area: stale pointers and ignored flags. Severity: medium.
- `docs/evidence/evidence_checklist.json` moved to `docs/theory/publication/` (72eafbbd), but
  `scripts/evidence_figures/ed05_manifest_hashes.py:51`, `ed10_release_archive_receipt.py:54`,
  `fig07_reproducibility_artifacts.py:44` still cite the old path, and
  `jaxfne/vis/evidence_manifest.py:114-134` records `source_files` unchecked.
- `artifacts/runs/mcc3_10s_checkpoint/mcc3_10s_manifest.json` names `mcc3_10s_metrics.json`,
  `mcc3_10s_weights.csv`, `mcc3_10s_H.csv`, absent since the 2026-10-06 compaction.
- Flags parsed and never read: `build_manuscript.py:437` `--verify` (it always verifies),
  `generate_gallery.py:123`, `generate_replications_page.py:63`, `harness/sync_skills.py:138`
  `--check` (siblings of O-15).
- `docs/reference/api/core/core.md:154` documents `simulate(...) -> Signals`; with
  `return_state=True` it returns `(signals, continuation)` (`_model_simulate.py:1337-1338`).
- `.readthedocs.yaml` builds without `fail_on_warning`, and the quick docs gate skips the strict
  build; RTD regenerates atlases only, so étude pages publish stand-ins.
- 2026-10-20 (release plan A2): the evidence checklist path has one home (the fallback to
  `docs/evidence/` is gone, `tests/test_evidence_checklist_path.py`); `simulate` documents its
  `return_state=True` tuple. Already fixed at the A2 baseline: the three script pointers, the
  `source_files` check, `build_manuscript --verify`, `generate_replications_page --check`.
  `generate_gallery --check` names the default mode (harmless). Open: `sync_skills --check`
  (hash-pinned harness file), the mcc3 manifest's three absent files, the RTD stand-ins (O-41).

### P-039

- Date: 2026-10-18. Type: SCIENCE. Area: GSDR lane. Severity: high.
- Fit manifests record `"mcdp": True` (`gsdr2026/fit/fit_gsdr.py:208`, `fit_gsdr_500.py:189`),
  but `jaxfne/` has no MCDP (no match for `mcdp` or shifted correlation in `jaxfne/`), and
  `run/run_gsdr.py:264` `mcdp_factors` has no caller. The 350- and 500-generation fits ran
  without the authors' MCDP step, while their manifests say otherwise.
- Possible change: correct the manifests' wording (record `mcdp: false` going forward), and add
  MCDP to the core `gsdr` before any further fit (replications run on the core).
- 2026-10-20 (release plan A3): GSDR CALIBRATION §6.3 states the 350- and 500-generation fits
  ran without MCDP; the fit drivers already record `mcdp_applied`; steps 4-6 read as unscored
  pilots. Open: the authors' `"mcdp": True` block in the fit drivers (Hamm's call).

### P-040

- Date: 2026-10-18. Type: SCIENCE. Area: Destexhe 2009 lane. Severity: high.
- `destexhe2009/PREREG_P3.md:1` reads "draft, UNSIGNED — no scored run may use it";
  `CALIBRATION.md:3-4` scores E1-E4 against it (header "Verdict: OFF"), and its final section
  says E1 passes (:427) and scores the reserved seeds 4-8. `REPLICATIONS.md:12` says "no verdict
  yet" and "28 passed" (30 pass today).
- 2026-10-20 (release plan A3): CALIBRATION header and steps 1-6 marked unscored pilots (ruling
  2026-10-18), body verdicts too; REPLICATIONS row agrees. Closes with the lane's next prereg.

### P-041

- Date: 2026-10-18. Type: DOC/SCIENCE. Area: PCL lanes. Severity: high.
- `docs/etudes/pcl.md:3` says "Fig. 2 reproduced standalone and on the HDP kernel; central
  claim reproduced in a small column", while its own K0 row (:34) and A1 rows (:33, :36, :37)
  are FAIL and `REPLICATIONS.md` lists Fig. 2 in progress.
- `pcl/pcl_column.py:114,124,131` and `pcl/pcl_fig2.py:58,87` implement STDP and LIF locally
  (no jaxfne import) although the core has both (`jaxfne/hdp_rules_pcl.py:35`);
  `pcl_h/pcl_h_rule.py:48,100` defines rules outside the core, used by `pcl/k1h_nf_gate.py:39`,
  with no prereg, docs page or REPLICATIONS row.
- PCL v5 T4 PASS rests on a criterion changed after the v4 numbers (`prereg/P3_fig3_v5.md:7`,
  `P3_fig3_result.md:23`); `REPLICATIONS.md:8` gives no qualifier.
- 2026-10-20 (release plan A3): `docs/etudes/pcl.md` opener follows its own gate rows; the
  REPLICATIONS T4 qualifier was already present. Open: the local STDP/LIF in `pcl_column.py` and
  `pcl_fig2.py` and the `pcl_h` rules go to the core with release plan B2.

### P-042

- Date: 2026-10-18. Type: DOC. Area: replication records. Severity: medium.
- `REPLICATIONS.md:9` lists CAMINOS under 0.5.6 while the todo stack (:14-15) moved it to
  0.5.7 "reasons in REPLICATIONS.md", which holds no reason.
- REPLICATIONS omits scored-off results against its own B2 rule (`REPLICATIONS.md:3`): GSDR
  steps 4 to 6.13 (row says P1), Destexhe E1.
- 16 scripts are referenced by nothing (no gate, workflow, doc, test or script; several cited
  only in receipts, e.g. `benchmark_054_scaling.py`, `run_protocol_h_h4_matrix.py`); 8 audit
  scripts are reached by no gate. Deleting any is Hamm's call.
- 2026-10-20 (release plan A3): REPLICATIONS gives the CAMINOS P6 release reason; the GSDR and
  Destexhe pilot clauses were already present. Open: the 16 + 8 scripts (release plan D1, Hamm).

### P-044

- Date: 2026-10-20. Type: BUG. Area: `Configuration.population` (`jaxfne/_config.py`), layer
  allocation in `jaxfne/_construct_population.py`. Severity: medium.
- Observation: `neurons=` keys that are not names in `layers=` are dropped without a message.
  When no key names a layer, counts fall back to layer thickness; when some do, layers missing
  from `neurons` get zero cells.
- Reproduction: N = 100, layers A (0.0-0.2) and B (0.2-1.0). `{"A": 9, "B": 1}` gives 90/10;
  `{"E": 0.8, "I": 0.2}` gives 20/80 (thickness); `{"A": 9, "I": 1}` gives 100/0.
- Callers passing cell-type keys (shares ignored, thickness used): `jaxfne/experiment_a/canonical.py`,
  `tests/mcc_fixtures.py`, `scripts/generate_meg_eeg_figures.py`,
  `scripts/run_heterogeneous_emitters_etude.py`, `scripts/run_multiscale_observation_etude.py`.
- Expected: a key outside `layers` is refused, and an omitted layer is refused or reported.
- Possible change: refuse unknown keys in `population()`, and change the five callers to layer
  shares equal to their thicknesses, so that their outputs stay bit-identical. This changes the
  public API and touches protocol and fixture code: Hamm's call.

### P-043

- Date: 2026-10-19. Type: SCIENCE. Area: Izhikevich cell-type presets. Severity: medium.
- `IZHIKEVICH_CELL_TYPE_DEFAULTS` (`jaxfne/emitters.py`): `VIP` (b = -0.10, drive 3.0), and `I`,
  which aliases VIP, sit far below rheobase. Constant input gives a stable rest while
  `(5 - b)^2 > 0.16 (140 + I)`, so VIP needs I > 22.56; E and PV need I > 4.0, SST I > 1.02.
- Reproduction: near-uncoupled column (all type gains 1e-6, default noise), one drive for all
  types, 1 s after a 250 ms burn. VIP: 0 Hz up to drive 22, 4.2 Hz at 23, 11.8 Hz at 25. PV: 0 Hz at
  the default 3.0. SST: 42 Hz at 3.0, 51 Hz at 4.0.
- Consequence: a column built on the defaults has silent VIP (and `I`) cells and SST-dominated
  dynamics; in a 1050-neuron, 21-bin laminar column, SST fired at 48 Hz while E and PV fired once at onset and
  then stopped.
- Closed 2026-10-19 (Hamm 2026-10-19: keep the presets, document them): the rheobase per label and
  the advice to set per-type drives explicitly are in `docs/reference/api/model/emitters.md`
  (Default drives and rheobase); defaults unchanged. Stays here until the hash-pinned
  `archive/0.5.x/issues_closed.md` is re-pinned with its next batch.

### P-035

- Date: 2026-10-18. Type: BUG. Area: docs build (Read the Docs). Severity: high (published docs
  stale since 2026-10-13).
- Observation: every Read the Docs build since 2026-10-14 01:46Z fails (builds 35131933 on
  413fb4a2 through 35176753 on b24e09ef, the v0.5.6 tag, on both `latest` and `stable`); the last
  success is 35110238 on ff89b24b, 2026-10-13.
- Reproduction: `python scripts/build_docs_assets.py --build` on the RTD image (Python 3.13.14,
  `pip install ".[viz]"`, CPU).
- Expected: rc 0. Actual: rc 1 at `docs/_static/etudes/hdp_controllability_reachability`:
  `condition-B population_rate_hz drift: 10.561121940612793 vs 10.62124252319336`, raised by the
  exact float equality at `scripts/rerun_etude_figures.py:548-552` against the frozen checkpoint.
  The tuned theta check just above it passes. Atlases 25 ok, the other 3 études ok.
- Evidence: RTD API v2 build logs (`/api/v2/build/<id>/`). Reproduced in WSL on CPU
  (`rerun_etude_figures.py --etude hdp_mcc3`: ff89b24b passes, 413fb4a2 gives the same 10.5611).
  `git bisect` over the 154 commits: first bad commit 5090af24 (D1, Hamm 2026-10-13: Izhikevich
  noise made dt-invariant, `noise_coef = noise_scale * sqrt(0.5 / dt_ms)`), an intended model
  change. The frozen digest `artifacts/runs/mcc3_10s_checkpoint/mcc3_10s_metrics_digest.json`
  holds the pre-D1 firing; the tuned theta still matches.
- Possible change: keep the pre-D1 digest as the historical record and pin a post-D1 digest
  beside it as a declared regeneration (reason: D1), which the generator checks; the docs panels
  then show the post-D1 run. The docs gate (`run_test_gate.py docs`) does not run the generators, so CI
  stays green while RTD fails.
- Fix 2026-10-18: `mcc3_10s_post_d1_digest.json` (condition B after D1: population 10.5611 Hz,
  1063 spikes; theta unchanged) beside the pre-D1 digest; `mode_hdp_mcc3` checks theta against
  the frozen digest and condition B against the post-D1 one. Moves to closed when a Read the
  Docs build succeeds.
- Closed 2026-10-19: on main by PR 110 (merge cf46d6d3, fix-only, Hamm 2026-10-18); Read the
  Docs `latest` built green on cf46d6d3; `stable` follows the next tag. Stays here until the
  hash-pinned `archive/0.5.x/issues_closed.md` is re-pinned with its next batch.
